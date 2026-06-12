"""Live web search grounding for MedBrief AI — PubMed + Wikipedia, no API key required."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

import httpx


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str
    source: str  # "pubmed" | "wikipedia"


@dataclass
class WebSearchContext:
    results: list[SearchResult] = field(default_factory=list)
    query: str = ""
    used: bool = False

    def as_context_text(self) -> str:
        if not self.results:
            return ""
        lines = ["Current medical knowledge from authoritative sources:"]
        for i, r in enumerate(self.results, 1):
            lines.append(f"\n[Source {i}: {r.source.upper()} — {r.title}]")
            if r.snippet:
                lines.append(r.snippet)
        return "\n".join(lines)

    def to_serializable(self) -> list[dict[str, str]]:
        return [{"title": r.title, "snippet": r.snippet, "url": r.url, "source": r.source} for r in self.results]


_PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_WIKI_BASE = "https://en.wikipedia.org/api/rest_v1"
_HEADERS = {"User-Agent": "MedBriefAI/1.0 (medbriefai.vercel.app; medical-education)"}
_SEARCH_TIMEOUT = 7.0


async def _pubmed_search(query: str, max_results: int = 3) -> list[SearchResult]:
    try:
        async with httpx.AsyncClient(timeout=_SEARCH_TIMEOUT, headers=_HEADERS) as client:
            search_resp = await client.get(
                f"{_PUBMED_BASE}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmax": max_results,
                    "retmode": "json",
                    "sort": "relevance",
                },
            )
            search_resp.raise_for_status()
            ids: list[str] = search_resp.json()["esearchresult"]["idlist"]
            if not ids:
                return []

            fetch_resp = await client.get(
                f"{_PUBMED_BASE}/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(ids[:max_results]),
                    "retmode": "xml",
                    "rettype": "abstract",
                },
            )
            fetch_resp.raise_for_status()
            xml = fetch_resp.text
    except Exception:
        return []

    results: list[SearchResult] = []
    articles = re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml, re.DOTALL)
    for article in articles[:max_results]:
        title_m = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", article, re.DOTALL)
        abstract_m = re.search(r"<AbstractText[^>]*>(.*?)</AbstractText>", article, re.DOTALL)
        pmid_m = re.search(r"<PMID[^>]*>(\d+)</PMID>", article)
        year_m = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", article, re.DOTALL)
        if not title_m:
            continue
        title = re.sub(r"<[^>]+>", "", title_m.group(1)).strip()
        snippet = re.sub(r"<[^>]+>", "", abstract_m.group(1)).strip()[:500] if abstract_m else ""
        pmid = pmid_m.group(1) if pmid_m else ""
        year = year_m.group(1) if year_m else ""
        display_title = f"{title} ({year})" if year else title
        results.append(SearchResult(
            title=display_title,
            snippet=snippet,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            source="pubmed",
        ))
    return results


async def _wikipedia_search(query: str) -> list[SearchResult]:
    try:
        async with httpx.AsyncClient(timeout=_SEARCH_TIMEOUT, headers=_HEADERS) as client:
            # Try direct page first, then opensearch fallback
            slug = query.replace(" ", "_")
            resp = await client.get(f"{_WIKI_BASE}/page/summary/{slug}")

            if resp.status_code == 404:
                sr = await client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={"action": "opensearch", "search": query, "limit": 1, "format": "json"},
                )
                sr.raise_for_status()
                titles: list[str] = sr.json()[1]
                if not titles:
                    return []
                slug = titles[0].replace(" ", "_")
                resp = await client.get(f"{_WIKI_BASE}/page/summary/{slug}")

            if not resp.is_success:
                return []
            data = resp.json()
    except Exception:
        return []

    extract = data.get("extract", "")[:500]
    if not extract:
        return []
    return [SearchResult(
        title=data.get("title", query),
        snippet=extract,
        url=data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        source="wikipedia",
    )]


_MEDICAL_SEARCH_TERMS = (
    "symptom", "disease", "condition", "treatment", "medication", "drug", "diagnosis",
    "syndrome", "disorder", "infection", "cancer", "pain", "therapy", "surgery",
    "inflammation", "chronic", "acute", "itis", "ology", "emia", "osis", "pathy",
    "heart", "lung", "kidney", "liver", "brain", "blood", "bone", "nerve",
    "diabetes", "hypertension", "stroke", "allergy", "vaccine", "antibiotic",
    "vitamin", "hormone", "immune", "genetic", "anxiety", "depression",
)


def should_search(text: str) -> bool:
    """Return True if the query warrants a live web search."""
    lower = text.lower()
    return any(term in lower for term in _MEDICAL_SEARCH_TERMS)


def build_search_query(text: str) -> str:
    """Extract a clean search query from user text."""
    text = re.sub(r"[^\w\s\-]", " ", text.lower())
    stopwords = {"what", "is", "are", "does", "do", "can", "how", "why", "when", "where",
                 "the", "a", "an", "and", "or", "i", "me", "my", "you", "your", "it",
                 "tell", "explain", "describe", "please", "help", "about", "know"}
    words = [w for w in text.split() if w and w not in stopwords]
    query = " ".join(words[:8])
    return query.strip() or text[:80]


async def search_for_query(query: str, max_results: int = 4) -> WebSearchContext:
    """Run PubMed + Wikipedia searches concurrently."""
    clean_query = build_search_query(query)

    pubmed_task = asyncio.create_task(_pubmed_search(clean_query, max_results=min(3, max_results)))
    wiki_task = asyncio.create_task(_wikipedia_search(clean_query))

    pubmed_res, wiki_res = await asyncio.gather(pubmed_task, wiki_task, return_exceptions=True)

    combined: list[SearchResult] = []
    if isinstance(pubmed_res, list):
        combined.extend(pubmed_res)
    if isinstance(wiki_res, list):
        combined.extend(wiki_res)

    return WebSearchContext(
        results=combined[:max_results],
        query=clean_query,
        used=bool(combined),
    )
