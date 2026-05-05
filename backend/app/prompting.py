"""Prompt assembly helpers for the MedBrief gateway."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .constants import (
    HEALTH_KEYWORDS,
    MODE_INSTRUCTIONS,
    PORTFOLIO_KEYWORDS,
    PSYCH_KEYWORDS,
    SUPPORTED_MODES,
    SYSTEM_PROMPT,
)
from .medical_ontology import detect_medical_context
from .schemas import ChatCompletionRequest, ModeLiteral, UserProfile
from .safety import is_crisis


MEMORY_SUMMARY_BLOCKLIST = (
    "ignore previous",
    "disregard",
    "forget",
    "new instruction",
    "system:",
    "assistant:",
    "[inst]",
    "[/inst]",
)

TONE_INSTRUCTIONS = {
    "supportive": "Tone: warm and grounded.",
    "clinical": "Tone: calm and medically literate.",
    "direct": "Tone: direct and clear.",
}

LENGTH_INSTRUCTIONS = {
    "concise": "Keep the answer tight.",
    "balanced": "Keep the answer useful but not long.",
    "detailed": "Add a bit more explanation when it helps.",
}

MODE_TAG_PATTERN = re.compile(r"^\s*\[mode:[a-z]+\]\s*", re.IGNORECASE)
DEFINITION_REQUEST_PATTERN = re.compile(
    r"^\s*(what is|what's|who is|define|explain|what does .+ mean)\b",
    re.IGNORECASE,
)

DIRECT_RESET_PHRASES = (
    "lets be serious",
    "let's be serious",
    "be serious",
    "answer the question",
    "stop being generic",
    "stop dodging",
    "just answer",
    "be direct",
)


@dataclass(frozen=True)
class PromptBundle:
    mode: ModeLiteral
    latest_user_text: str
    upstream_messages: list[dict[str, str]]


def detect_mode(text: str) -> ModeLiteral:
    lowered = _strip_mode_tag(text).lower()
    if is_crisis(lowered):
        return "crisis"
    if any(keyword in lowered for keyword in PSYCH_KEYWORDS):
        return "psych"
    if any(keyword in lowered for keyword in HEALTH_KEYWORDS):
        return "health"
    if any(keyword in lowered for keyword in PORTFOLIO_KEYWORDS):
        return "portfolio"
    return "general"


def _reduce_history(messages: list[dict[str, str]], mode: ModeLiteral = "general") -> list[dict[str, str]]:
    conversational = [message for message in messages if message["role"] in {"user", "assistant"}]
    keep = {
        "crisis": 10,
        "psych": 14,
        "health": 8,
        "portfolio": 10,
        "general": 12,
    }.get(mode, 12)
    if len(conversational) <= keep:
        return conversational
    return conversational[-keep:]


def _count_tokens_rough(text: str) -> int:
    return max(1, len(text.split()))


def _coerce_profile(request: ChatCompletionRequest) -> UserProfile | None:
    raw = request.metadata.get("user_profile")
    if not isinstance(raw, dict) or not raw.get("user_id"):
        return None
    try:
        return UserProfile.model_validate(raw)
    except Exception:
        return None


def _strip_mode_tag(text: str) -> str:
    return MODE_TAG_PATTERN.sub("", text).strip()


def _is_definition_request(text: str) -> bool:
    return bool(DEFINITION_REQUEST_PATTERN.search(_strip_mode_tag(text)))


def is_definition_request(text: str) -> bool:
    return _is_definition_request(text)


def _needs_direct_reset(text: str) -> bool:
    lowered = _strip_mode_tag(text).lower()
    return any(phrase in lowered for phrase in DIRECT_RESET_PHRASES)


def _recent_history_mentions_crisis(request: ChatCompletionRequest) -> bool:
    earlier_user_text = " ".join(
        _strip_mode_tag(message.content)
        for message in request.messages[:-1]
        if message.role == "user"
    )
    assistant_history = " ".join(
        message.content.lower()
        for message in request.messages[:-1]
        if message.role == "assistant"
    )
    return is_crisis(earlier_user_text) or "988" in assistant_history


def _sanitize_memory_summary(summary: str) -> str:
    cleaned = summary.strip()
    lowered = cleaned.lower()
    if any(pattern in lowered for pattern in MEMORY_SUMMARY_BLOCKLIST):
        return "[Memory summary removed due to content policy]"
    return cleaned[:800]


def _truncate_history(messages: list[dict[str, str]], max_prompt_tokens: int) -> list[dict[str, str]]:
    kept = list(messages)
    while len(kept) > 2 and sum(_count_tokens_rough(message["content"]) for message in kept) > max_prompt_tokens:
        kept.pop(0)
    return kept


def _merge_system_messages(
    request: ChatCompletionRequest,
    mode: ModeLiteral,
    profile: UserProfile | None,
    latest_user_text: str,
) -> list[str]:
    extra_system = [message.content for message in request.messages if message.role == "system"]
    definition_request = _is_definition_request(latest_user_text)
    merged = [SYSTEM_PROMPT, MODE_INSTRUCTIONS[mode]]
    if _is_definition_request(latest_user_text):
        merged.append(
            "Start with the answer in the first sentence. Keep it to 2 or 3 sentences unless more detail is clearly needed."
        )
    if mode == "psych" and "injur" not in latest_user_text.lower() and any(
        keyword in latest_user_text.lower() for keyword in ("hurt", "hurting", "pain")
    ):
        merged.append(
            "Assume the user likely means emotional pain unless they clearly mention a physical injury. "
            "Do not shift into bodily injury or medical-care language unless the user brings that up."
        )
    if _needs_direct_reset(latest_user_text):
        merged.append(
            "The user is frustrated with generic answers. Acknowledge that briefly, then answer directly."
        )
        if _recent_history_mentions_crisis(request):
            merged.append(
                "The conversation just involved a safety response. Do not fall back to generic wellness language."
            )
    if profile and not definition_request:
        if profile.preferences.tone != "supportive":
            merged.append(TONE_INSTRUCTIONS[profile.preferences.tone])
        if profile.preferences.response_length != "balanced":
            merged.append(LENGTH_INSTRUCTIONS[profile.preferences.response_length])
        if profile.preferences.terminology == "professional":
            merged.append("Use medically literate language when it helps.")
        if profile.display_name and mode == "psych":
            merged.append(f"Use the name {profile.display_name} naturally if helpful.")
        if mode == "psych" and profile.patterns:
            merged.append(f"Known pattern: {', '.join(profile.patterns[:2])}.")
        if mode == "psych" and profile.recurring_topics:
            merged.append(f"Recurring topic: {', '.join(profile.recurring_topics[:2])}.")
        if mode == "health" and profile.medical_context:
            merged.append(f"Medical context: {', '.join(profile.medical_context[:3])}.")
    merged.extend(extra_system)
    if request.memory_summary and not definition_request and mode in {"psych", "general"}:
        sanitized_summary = _sanitize_memory_summary(request.memory_summary)
        merged.append(f"Continuity note if relevant: {sanitized_summary}")
    return merged


def build_prompt_bundle(request: ChatCompletionRequest) -> PromptBundle:
    latest_user_text = next(
        _strip_mode_tag(message.content) for message in reversed(request.messages) if message.role == "user"
    )
    mode: ModeLiteral = request.mode or detect_mode(latest_user_text)
    profile = _coerce_profile(request)
    medical_context = detect_medical_context(latest_user_text)

    upstream_messages: list[dict[str, str]] = []
    upstream_messages.append(
        {
            "role": "system",
            "content": "\n\n".join(_merge_system_messages(request, mode, profile, latest_user_text)),
        }
    )
    conversational = [
        {
            "role": message["role"],
            "content": _strip_mode_tag(message["content"]) if message["role"] == "user" else message["content"],
        }
        for message in _reduce_history([message.model_dump() for message in request.messages], mode=mode)
    ]
    if medical_context.has_medical_signal:
        ontology_bits = [
            "Use any medical terms as context only, not as a diagnosis."
        ]
        if medical_context.symptoms:
            ontology_bits.append(f"Symptoms: {', '.join(medical_context.symptoms[:3])}")
        if medical_context.drugs:
            ontology_bits.append(f"Medications mentioned: {', '.join(medical_context.drugs[:3])}")
        if medical_context.diagnoses:
            ontology_bits.append(f"Diagnosis terms mentioned by the user: {', '.join(medical_context.diagnoses[:3])}")
        if medical_context.procedures:
            ontology_bits.append(f"Procedures mentioned: {', '.join(medical_context.procedures[:3])}")
        if ontology_bits:
            upstream_messages.append({"role": "system", "content": " ".join(ontology_bits)})
    upstream_messages.extend(_truncate_history(conversational, max_prompt_tokens=1800))

    return PromptBundle(
        mode=mode,
        latest_user_text=latest_user_text,
        upstream_messages=upstream_messages,
    )


def supported_modes() -> list[ModeLiteral]:
    return list(SUPPORTED_MODES)
