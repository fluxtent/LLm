"""Safety and response-quality rules for MedBrief AI."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .constants import (
    CRISIS_DISTRESS_SIGNALS,
    CRISIS_KEYWORDS,
    CRISIS_RESOURCE_BLOCK,
    CRISIS_RESPONSE,
    DEGRADED_MODE_RESPONSE,
    HIGH_CONFIDENCE_CRISIS_KEYWORDS,
    MEDICAL_EMERGENCY_KEYWORDS,
    LOW_QUALITY_FALLBACK,
    MEDICATION_DOSING_RESPONSE,
    MEDICATION_KEYWORDS,
    PRIVACY_DISCLAIMER,
    PROFANITY_TERMS,
)
from .medical_ontology import emergency_medical_request, scrub_unapproved_drugs
from .schemas import ModeLiteral


DOSING_PATTERN = re.compile(
    r"\b(\d+\s?(mg|mcg|g|ml|milligram|milligrams|tablet|tablets|capsule|capsules)|"
    r"how much should i take|what dose should i take|what dosage should i take|"
    r"should i increase my dose|safe dose)\b",
    re.IGNORECASE,
)

ARTIFACT_PATTERN = re.compile(r"<[^>]+>")
ARTIFACT_MARKERS = ("[inst]", "[/inst]", "<|user|>", "<|assistant|>", "human:", "assistant:")


def _matched_terms(text: str, terms: tuple[str, ...]) -> set[str]:
    lowered = " ".join(text.lower().split())
    return {term for term in terms if term in lowered}


@dataclass(frozen=True)
class SafetyDecision:
    allow_model: bool
    response_text: str | None = None
    safety_flag: str | None = None


def is_crisis(text: str) -> bool:
    high_confidence_matches = _matched_terms(text, HIGH_CONFIDENCE_CRISIS_KEYWORDS)
    if high_confidence_matches:
        return True

    matched_keywords = _matched_terms(text, CRISIS_KEYWORDS)
    matched_distress = _matched_terms(text, CRISIS_DISTRESS_SIGNALS)
    return len(matched_keywords) >= 2 or (len(matched_keywords) >= 1 and len(matched_distress) >= 1)


def is_medical_emergency(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in MEDICAL_EMERGENCY_KEYWORDS) or emergency_medical_request(lowered)


def is_medication_dosing_request(text: str) -> bool:
    lowered = text.lower()
    mentions_medication = any(keyword in lowered for keyword in MEDICATION_KEYWORDS)
    return mentions_medication and bool(DOSING_PATTERN.search(lowered))


def evaluate_request(mode: ModeLiteral, user_text: str) -> SafetyDecision:
    if is_crisis(user_text) or mode == "crisis":
        return SafetyDecision(
            allow_model=False,
            response_text=CRISIS_RESPONSE,
            safety_flag="crisis_intercept",
        )
    if is_medical_emergency(user_text):
        return SafetyDecision(
            allow_model=False,
            response_text=(
                "This could be a medical emergency. Please call 911 or your local emergency services now, "
                "or go to the nearest emergency department, especially if symptoms are severe, sudden, or getting worse."
            ),
            safety_flag="medical_emergency_redirect",
        )
    if is_medication_dosing_request(user_text):
        return SafetyDecision(
            allow_model=False,
            response_text=MEDICATION_DOSING_RESPONSE,
            safety_flag="medication_dosing_refusal",
        )
    return SafetyDecision(allow_model=True)


def clean_response_text(text: str) -> str:
    cleaned = text.replace("<eos>", "").replace("</s>", "").strip()
    cleaned = ARTIFACT_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    last_punctuation = max(cleaned.rfind("."), cleaned.rfind("!"), cleaned.rfind("?"))
    if last_punctuation > len(cleaned) * 0.6:
        cleaned = cleaned[: last_punctuation + 1]
    elif cleaned and cleaned[-1] not in ".!?" and last_punctuation >= max(20, int(len(cleaned) * 0.35)):
        cleaned = cleaned[: last_punctuation + 1]

    return cleaned.strip()


def is_low_quality_response(text: str) -> bool:
    words = text.split()
    if len(words) < 12:
        return True
    lowered = text.lower()
    if any(marker in lowered for marker in ("http://", "https://", "www.", "%20", "[assistant]", "[user]", "[mode:")):
        return True
    if any(marker in lowered for marker in ARTIFACT_MARKERS):
        return True
    if "you are medbrief ai" in lowered:
        return True
    alpha_tokens = [word for word in words if re.search(r"[a-zA-Z]", word)]
    if len(alpha_tokens) / max(len(words), 1) < 0.75:
        return True
    unique_words = len({word.lower() for word in words})
    if unique_words / max(len(words), 1) < 0.4:
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

    seen_ngrams: dict[str, int] = {}
    for index in range(len(words) - 3):
        ngram = " ".join(words[index : index + 4]).lower()
        seen_ngrams[ngram] = seen_ngrams.get(ngram, 0) + 1
        if seen_ngrams[ngram] > 2:
            return True
    return False


def ensure_crisis_resources(text: str) -> str:
    if "988" in text:
        return text
    return f"{text}\n\n{CRISIS_RESOURCE_BLOCK}".strip()


def apply_profanity_filter(text: str) -> str:
    filtered = text
    for term in PROFANITY_TERMS:
        filtered = re.sub(rf"\b{re.escape(term)}\b", "[redacted]", filtered, flags=re.IGNORECASE)
    return filtered


def postprocess_health_response(text: str) -> tuple[str, bool]:
    return scrub_unapproved_drugs(text)


def inject_privacy_disclaimer(text: str) -> str:
    if PRIVACY_DISCLAIMER in text:
        return text
    return f"{text}\n\n{PRIVACY_DISCLAIMER}".strip()


def fallback_response() -> str:
    return LOW_QUALITY_FALLBACK


def degraded_mode_response() -> str:
    return DEGRADED_MODE_RESPONSE
