from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from array import array
from collections import Counter
from pathlib import Path
from random import Random

from bpe import BPETokenizer


PSYCH_KEYWORDS = (
    "anxious",
    "anxiety",
    "depressed",
    "feeling",
    "overwhelmed",
    "struggling",
    "therapist",
    "panic",
    "burnout",
    "lonely",
)
HEALTH_KEYWORDS = (
    "symptom",
    "medication",
    "diagnosis",
    "doctor",
    "pain",
    "condition",
    "treatment",
    "hospital",
    "medical",
    "health",
)
NARRATIVE_KEYWORDS = ("story", "character", "journey", "memory", "scene", "narrative")
QA_KEYWORDS = ("what", "how", "why", "when", "where", "explain", "define", "describe")
CRISIS_KEYWORDS = (
    "kill myself",
    "want to die",
    "end my life",
    "suicide",
    "suicidal",
    "hurt myself",
    "self harm",
    "self-harm",
    "better off dead",
    "not worth living",
)
MEDICATION_KEYWORDS = (
    "sertraline",
    "prozac",
    "fluoxetine",
    "lexapro",
    "zoloft",
    "adderall",
    "xanax",
    "ibuprofen",
    "acetaminophen",
    "tylenol",
    "dose",
    "dosage",
    "milligrams",
    "mg",
)

DOMAIN_TAGS = {
    "psych": "[PSYCH]",
    "health": "[HEALTH]",
    "narrative": "[NARRATIVE]",
    "qa": "[QA]",
    "general": "[GENERAL]",
}


def load_data(file_path: str | Path) -> str:
    return Path(file_path).read_text(encoding="utf-8", errors="ignore")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def advanced_clean_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_documents(text: str) -> list[str]:
    return [chunk.strip() for chunk in re.split(r"\n\s*\n+", text) if chunk.strip()]


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = {token for token in re.findall(r"[a-z]{3,}", left.lower())}
    right_tokens = {token for token in re.findall(r"[a-z]{3,}", right.lower())}
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def deduplicate_documents(documents: list[str], near_dup_threshold: float = 0.85) -> list[str]:
    unique_docs: list[str] = []
    seen_hashes: set[str] = set()
    for document in documents:
        normalized = normalize_text(document)
        if not normalized:
            continue
        fingerprint = text_hash(normalized)
        if fingerprint in seen_hashes:
            continue
        tagged_instruction = normalized.startswith(("[PSYCH]", "[HEALTH]", "[NARRATIVE]", "[QA]", "[GENERAL]"))
        should_apply_near_dup = (not tagged_instruction) and len(normalized.split()) >= 80 and len(normalized) >= 500
        if should_apply_near_dup and any(
            len(existing.split()) >= 80
            and len(existing) >= 500
            and jaccard_similarity(normalized, existing) > near_dup_threshold
            for existing in unique_docs[-100:]
        ):
            continue
        seen_hashes.add(fingerprint)
        unique_docs.append(normalized)
    return unique_docs


def filter_quality(documents: list[str]) -> list[str]:
    filtered: list[str] = []
    for document in documents:
        words = document.split()
        tagged_instruction = document.startswith(("[PSYCH]", "[HEALTH]", "[NARRATIVE]", "[QA]", "[GENERAL]"))
        minimum_words = 25 if tagged_instruction else 60
        if len(words) < minimum_words:
            continue
        non_alpha = sum(1 for char in document if not char.isalpha() and not char.isspace())
        if len(document) and non_alpha / len(document) > 0.3:
            continue
        sentences = [segment for segment in re.split(r"[.!?]+", document) if segment.strip()]
        if not sentences:
            continue
        avg_len = sum(len(sentence.split()) for sentence in sentences) / len(sentences)
        if avg_len < 4 or avg_len > 60:
            continue
        filtered.append(document)
    return filtered


def detect_domain(text: str) -> str:
    lowered = text.lower()
    scores = {
        "psych": sum(keyword in lowered for keyword in PSYCH_KEYWORDS),
        "health": sum(keyword in lowered for keyword in HEALTH_KEYWORDS),
        "narrative": sum(keyword in lowered for keyword in NARRATIVE_KEYWORDS),
        "qa": sum(keyword in lowered for keyword in QA_KEYWORDS),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] else "general"


def detect_mode(text: str) -> str:
    lowered = text.lower()
    if any(keyword in lowered for keyword in CRISIS_KEYWORDS):
        return "crisis"
    if any(keyword in lowered for keyword in PSYCH_KEYWORDS):
        return "psych"
    if any(keyword in lowered for keyword in HEALTH_KEYWORDS):
        return "health"
    if "medbrief" in lowered or "portfolio" in lowered or "project" in lowered or "website" in lowered:
        return "portfolio"
    return "general"


def add_domain_tag(text: str, domain: str) -> str:
    return f"{DOMAIN_TAGS.get(domain, '[GENERAL]')} {normalize_text(text)}"


def create_train_val_split(documents: list[str], val_ratio: float = 0.05, seed: int = 17) -> tuple[list[str], list[str]]:
    items = list(documents)
    Random(seed).shuffle(items)
    split_index = max(1, int(len(items) * (1.0 - val_ratio)))
    return items[:split_index], items[split_index:]


def save_tokenized_data(data: list[int], filename: str | Path) -> None:
    values = array("I", data)
    with Path(filename).open("wb") as handle:
        values.tofile(handle)


def load_tokenized_data(filename: str | Path) -> list[int]:
    path = Path(filename)
    values = array("I")
    with path.open("rb") as handle:
        values.fromfile(handle, path.stat().st_size // values.itemsize)
    return list(values)


def load_tokenizer(vocab_path: str | Path = "vocab.json", merges_path: str | Path = "merges.pkl") -> BPETokenizer:
    return BPETokenizer().load(vocab_path, merges_path)


def encode(text: str, tokenizer: BPETokenizer, add_bos: bool = False, add_eos: bool = False) -> list[int]:
    return tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)


def decode(indices: list[int], tokenizer: BPETokenizer) -> str:
    return tokenizer.decode(indices)


def clean_response(text: str) -> str:
    text = text.replace("<eos>", "").replace("[EOS]", "")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    last_punct = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last_punct > len(text) * 0.6:
        text = text[: last_punct + 1]
    return text.strip()


def is_low_quality_response(text: str) -> bool:
    words = text.split()
    if len(words) < 12:
        return True
    lowered = text.lower()
    if any(marker in lowered for marker in ("http://", "https://", "www.", "%20", "[assistant]", "[user]", "[mode:")):
        return True
    alpha_tokens = [word for word in words if re.search(r"[a-zA-Z]", word)]
    if len(alpha_tokens) / max(len(words), 1) < 0.75:
        return True
    noisy_tokens = [
        word for word in words
        if re.search(r"\d", word)
        or word.count("/") >= 1
        or word.count("&") >= 1
        or len(word) > 22
    ]
    if len(noisy_tokens) / max(len(words), 1) > 0.12:
        return True
    four_grams: Counter[str] = Counter()
    for index in range(len(words) - 3):
        chunk = " ".join(words[index : index + 4]).lower()
        four_grams[chunk] += 1
        if four_grams[chunk] > 2:
            return True
    return False


def response_completeness(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and stripped[-1] in ".!?"


def looks_like_medication_dosing(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in MEDICATION_KEYWORDS) and any(
        cue in lowered for cue in ("mg", "milligrams", "dose", "dosage", "how much")
    )


def add_mode_tag(text: str, mode: str) -> str:
    return f"[MODE:{mode.upper()}] {text}".strip()


def chunk_words(text: str, words_per_chunk: int = 12) -> list[str]:
    words = text.split()
    return [" ".join(words[index : index + words_per_chunk]) for index in range(0, len(words), words_per_chunk)]


def calculate_perplexity(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
