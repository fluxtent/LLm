"""Universal conversation understanding and response planning for MedBrief."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable

from .prompting import detect_mode
from .schemas import ChatCompletionRequest, ChatMessage, ModeLiteral, UserProfile


STOPWORDS = {
    "a",
    "about",
    "again",
    "all",
    "am",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "can",
    "could",
    "do",
    "does",
    "dont",
    "don't",
    "for",
    "from",
    "get",
    "have",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "just",
    "know",
    "like",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "should",
    "so",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "why",
    "with",
    "you",
    "your",
}

EMOTION_LEXICON = {
    "confused": {"confused", "lost", "stuck", "unclear", "unsure", "overthinking"},
    "overwhelmed": {"overwhelmed", "too much", "so much pressure", "cant think", "can't think", "spiral", "shut down", "shutdown"},
    "self_hate": {"hate myself", "hate my life", "fuck life", "worthless", "useless", "failure", "failed", "ruined", "ashamed", "shame"},
    "anger": {"angry", "mad", "furious", "pissed", "annoyed"},
    "lonely": {"lonely", "alone", "isolated", "ignored", "abandoned"},
    "grief": {"grief", "grieving", "loss", "lost", "miss them"},
    "anxiety": {"anxious", "anxiety", "panic", "worried", "scared", "afraid"},
    "sad": {"sad", "depressed", "miserable", "empty", "numb", "giving up", "give up"},
    "frustrated": {"frustrated", "not helping", "do you even", "do you not", "care either", "listen"},
}

INTENT_PATTERNS = (
    ("assistant_repair", re.compile(r"\b(do you|you)\b.*\b(care|listen|understand|help|repeat|repeating)|\bnot helping\b|\bwhat are you talking about\b|\bsimplify\b", re.I)),
    ("next_step", re.compile(r"\b(what do i do|what should i do|dont know what to do|don't know what to do|help me)\b", re.I)),
    ("explanation", re.compile(r"^\s*(what is|what's|why|how does|explain|define)\b", re.I)),
    ("planning", re.compile(r"\b(plan|organize|structure|roadmap|steps|schedule)\b", re.I)),
    ("comparison", re.compile(r"\b(compare|versus|vs\.?|tradeoff|which is better|choose between)\b", re.I)),
    ("creation", re.compile(r"\b(write|draft|create|make|build|generate)\b", re.I)),
    ("debugging", re.compile(r"\b(error|bug|fix|broken|failing|doesn't work|not working)\b", re.I)),
)

FOLLOWUP_REFERENCES = {
    "that",
    "this",
    "it",
    "either",
    "again",
    "same",
    "still",
    "they",
    "them",
    "he",
    "she",
    "there",
}

GENERIC_MARKERS = (
    "the cleanest way to approach",
    "separate what is factual",
    "what is interpretive",
    "what kind of answer would actually help",
    "multiple possibilities before committing",
    "anchor this in one real example",
    "tell me a bit more about what you mean",
    "you're asking:",
    "the answer should be",
    "based on the context i have",
    "the useful focus is",
    "i am reading this as",
    "the useful response is",
    "practical layer and a conceptual layer",
    "break the question into its smaller working parts",
    "clarifying what kind of answer would actually help",
    "valid ways to frame this",
    "not a canned framework",
    "the next thing to do is a direct answer",
    "the important part is",
    "tell me what outcome you want",
    "shape the answer around that instead of guessing",
    "i hear you expressing",
    "your current experience",
    "various lenses",
    "philosophical, spiritual, psychological",
)


@dataclass(frozen=True)
class PersonalizationContext:
    latest_user: str
    previous_user_messages: list[str]
    previous_assistant_messages: list[str]
    memory_summary: str | None
    profile: UserProfile | None


@dataclass(frozen=True)
class ConversationUnderstanding:
    latest_user: str
    mode: ModeLiteral
    topic: str
    direct_question: str
    user_intent: str
    implied_need: str
    emotional_state: str
    distress_level: int
    references_prior_context: bool
    escalation_detected: bool
    frustration_at_assistant: bool
    repeated_unanswered_need: bool
    continuity_notes: list[str] = field(default_factory=list)
    relevant_memory: list[str] = field(default_factory=list)
    previous_assistant_messages: list[str] = field(default_factory=list)
    memory_candidates: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class ResponsePlan:
    mode: ModeLiteral
    tone: str
    direct_opening: str
    must_address: list[str]
    prior_context: list[str]
    relevant_memory: list[str]
    avoid: list[str]
    response_shape: str
    should_use_personalized_responder: bool
    understanding: ConversationUnderstanding

    def to_system_prompt(self) -> str:
        parts = [
            "Universal personalization plan:",
            f"- Latest user meaning: {self.understanding.topic}",
            f"- Intent: {self.understanding.user_intent}",
            f"- Implied need: {self.understanding.implied_need}",
            f"- Emotional state: {self.understanding.emotional_state} (distress {self.understanding.distress_level}/4)",
            f"- Tone: {self.tone}",
        ]
        if self.must_address:
            parts.append(f"- Must directly address: {'; '.join(self.must_address)}")
        if self.prior_context:
            parts.append(f"- Current-chat context to use: {'; '.join(self.prior_context)}")
        if self.relevant_memory:
            parts.append(f"- Relevant saved memory to use gently: {'; '.join(self.relevant_memory)}")
        parts.append(f"- Response shape: {self.response_shape}")
        if self.understanding.distress_level >= 2:
            parts.append(
                "- Emotional-distress handling: acknowledge the exact pressure in the user's words, "
                "do not ask what outcome they want, keep it to 2 short sentences, give one immediate stabilizing move, and ask a direct safety check if the wording suggests giving up."
            )
        if "purpose" in self.understanding.latest_user.lower():
            parts.append(
                "- Purpose-question handling: first sentence must give a direct answer, e.g. purpose is the reason something matters or the direction it serves. Do not start by listing lenses or perspectives."
            )
        if self.understanding.frustration_at_assistant:
            parts.append(
                "- Assistant-repair handling: own the miss briefly, then answer the latest message instead of explaining the failure."
            )
        parts.append(f"- Avoid: {'; '.join(self.avoid)}")
        parts.append(
            "Write to this individual user. Do not give an abstract framework unless their actual request calls for one."
        )
        parts.append(
            "Do not mention this plan, do not say \"You're asking\", and do not restate the user's sentence as a formula."
        )
        return "\n".join(parts)


@dataclass(frozen=True)
class ResponseQuality:
    ignored_latest_user: bool = False
    generic_response: bool = False
    duplicate_response: bool = False
    missed_continuity: bool = False
    missed_assistant_repair: bool = False

    @property
    def should_override(self) -> bool:
        return any(
            (
                self.ignored_latest_user,
                self.generic_response,
                self.duplicate_response,
                self.missed_continuity,
                self.missed_assistant_repair,
            )
        )

    @property
    def flags(self) -> list[str]:
        return [
            name
            for name, value in (
                ("ignored_latest_user", self.ignored_latest_user),
                ("generic_response", self.generic_response),
                ("duplicate_response", self.duplicate_response),
                ("missed_continuity", self.missed_continuity),
                ("missed_assistant_repair", self.missed_assistant_repair),
            )
            if value
        ]


def build_personalization_context(
    request: ChatCompletionRequest,
    profile: UserProfile | None,
) -> PersonalizationContext:
    conversational = [message for message in request.messages if message.role in {"user", "assistant"}]
    latest_user = next(message.content for message in reversed(conversational) if message.role == "user")
    previous = conversational[:-1]
    return PersonalizationContext(
        latest_user=_strip_mode_tag(latest_user),
        previous_user_messages=[_strip_mode_tag(message.content) for message in previous if message.role == "user"],
        previous_assistant_messages=[message.content for message in previous if message.role == "assistant"],
        memory_summary=request.memory_summary,
        profile=profile,
    )


def build_response_plan(context: PersonalizationContext, requested_mode: ModeLiteral | None = None) -> ResponsePlan:
    understanding = understand_conversation(context, requested_mode=requested_mode)
    tone = _select_tone(understanding, context.profile)
    direct_opening = _direct_opening(understanding)
    must_address = _must_address(understanding)
    prior_context = understanding.continuity_notes[:3]
    avoid = [
        "Do not repeat a prior assistant answer.",
        "Do not answer with generic abstractions before addressing the user's actual words.",
        "Do not ask for clarification when a concrete first response is possible.",
    ]
    if understanding.distress_level >= 2:
        avoid.append("Do not turn emotional pain into a detached decision framework.")
    if understanding.frustration_at_assistant:
        avoid.append("Do not defend the assistant or ignore the user's anger at the assistant.")

    return ResponsePlan(
        mode=understanding.mode,
        tone=tone,
        direct_opening=direct_opening,
        must_address=must_address,
        prior_context=prior_context,
        relevant_memory=understanding.relevant_memory,
        avoid=avoid,
        response_shape=_response_shape(understanding),
        should_use_personalized_responder=(
            understanding.distress_level >= 2
            or understanding.frustration_at_assistant
            or understanding.repeated_unanswered_need
        ),
        understanding=understanding,
    )


def understand_conversation(
    context: PersonalizationContext,
    requested_mode: ModeLiteral | None = None,
) -> ConversationUnderstanding:
    latest = context.latest_user
    lowered = _normalize(latest)
    previous_user = context.previous_user_messages[-4:]
    previous_assistant = context.previous_assistant_messages[-4:]
    mode = requested_mode or detect_mode(latest)
    emotional_state, distress_level = _detect_emotion(lowered)
    if distress_level >= 1 and mode == "general":
        mode = "psych"

    user_intent = _detect_intent(latest)
    topic = _extract_topic(latest, previous_user)
    direct_question = _extract_direct_question(latest)
    references_prior = _references_prior_context(latest)
    frustration_at_assistant = user_intent == "assistant_repair" or _assistant_frustration(lowered)
    repeated_unanswered = _repeated_unanswered_need(latest, previous_user, previous_assistant)
    escalation = _escalation_detected(distress_level, previous_user)
    continuity = _continuity_notes(
        latest=latest,
        previous_user=previous_user,
        previous_assistant=previous_assistant,
        references_prior=references_prior,
        escalation=escalation,
        frustration_at_assistant=frustration_at_assistant,
        repeated_unanswered=repeated_unanswered,
    )
    relevant_memory = _relevant_memory(latest, context.profile, context.memory_summary)

    return ConversationUnderstanding(
        latest_user=latest,
        mode=mode,
        topic=topic,
        direct_question=direct_question,
        user_intent=user_intent,
        implied_need=_implied_need(user_intent, emotional_state, distress_level),
        emotional_state=emotional_state,
        distress_level=distress_level,
        references_prior_context=references_prior,
        escalation_detected=escalation,
        frustration_at_assistant=frustration_at_assistant,
        repeated_unanswered_need=repeated_unanswered,
        continuity_notes=continuity,
        relevant_memory=relevant_memory,
        previous_assistant_messages=previous_assistant,
        memory_candidates=_memory_candidates(latest, emotional_state, topic),
    )


def evaluate_response_quality(text: str, plan: ResponsePlan) -> ResponseQuality:
    lowered = _normalize(text)
    understanding = plan.understanding
    latest_tokens = _keywords(understanding.latest_user)
    response_tokens = _keywords(text)
    overlap = len(latest_tokens & response_tokens)
    personal_or_specific = overlap >= 1 or any(
        marker in lowered
        for marker in (
            "you said",
            "you're asking",
            "you are asking",
            "what you wrote",
            "what stands out",
            "right now",
            "i hear",
            "you were",
        )
    )
    if "purpose" in _normalize(understanding.latest_user) and any(
        marker in lowered for marker in ("purpose", "meaning", "reason", "aim", "value", "matters", "exists", "essence")
    ):
        personal_or_specific = True
    ignored_latest = bool(latest_tokens) and not personal_or_specific
    generic = any(marker in lowered for marker in GENERIC_MARKERS)
    if understanding.distress_level >= 1 and any(
        marker in lowered for marker in ("factual", "interpretive", "tractable", "possibilities")
    ):
        generic = True

    duplicate = any(_similarity(text, prior) > 0.88 for prior in plan.understanding.previous_assistant_messages[-3:])
    missed_continuity = (
        understanding.references_prior_context
        and not any(marker in lowered for marker in ("that", "this", "earlier", "before", "again", "what you said", "the previous"))
    )
    missed_repair = (
        understanding.frustration_at_assistant
        and not any(marker in lowered for marker in ("you're right", "i missed", "sorry", "failed", "not helping"))
    )

    return ResponseQuality(
        ignored_latest_user=ignored_latest,
        generic_response=generic,
        duplicate_response=duplicate,
        missed_continuity=missed_continuity,
        missed_assistant_repair=missed_repair,
    )


def apply_memory_updates(profile: UserProfile, understanding: ConversationUnderstanding) -> UserProfile:
    next_profile = profile.model_copy(deep=True)
    candidates = understanding.memory_candidates
    next_profile.themes = _merge_limited(next_profile.themes, candidates.get("themes", []), 18)
    next_profile.patterns = _merge_limited(next_profile.patterns, candidates.get("patterns", []), 12)
    next_profile.recurringStressors = _merge_limited(
        next_profile.recurringStressors,
        candidates.get("recurringStressors", []),
        12,
    )
    next_profile.goals = _merge_limited(next_profile.goals, candidates.get("goals", []), 12)
    next_profile.recurring_topics = _merge_limited(
        next_profile.recurring_topics,
        candidates.get("themes", []) + candidates.get("recurringStressors", []),
        16,
    )
    return next_profile


def _strip_mode_tag(text: str) -> str:
    return re.sub(r"^\s*\[mode:[a-z]+\]\s*", "", text, flags=re.I).strip()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _keywords(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z']{2,}", _normalize(text))
        if token not in STOPWORDS
    }


def _detect_emotion(lowered: str) -> tuple[str, int]:
    scores: dict[str, int] = {}
    for label, terms in EMOTION_LEXICON.items():
        scores[label] = sum(1 for term in terms if term in lowered)
    label, score = max(scores.items(), key=lambda item: item[1])
    if score == 0:
        return "neutral", 0
    distress = 1
    if label in {"overwhelmed", "self_hate", "sad", "lonely", "grief"}:
        distress = 2
    if label == "self_hate" or any(term in lowered for term in ("it's all over", "its all over", "can't do this", "cant do this")):
        distress = 3
    if any(term in lowered for term in ("want to die", "kill myself", "end my life", "suicide")):
        distress = 4
        label = "crisis"
    return label, distress


def _detect_intent(text: str) -> str:
    for intent, pattern in INTENT_PATTERNS:
        if pattern.search(text):
            return intent
    if text.strip().endswith("?"):
        return "question"
    if _detect_emotion(_normalize(text))[1] > 0:
        return "emotional_disclosure"
    return "statement"


def _extract_topic(latest: str, previous_user: list[str]) -> str:
    latest_clean = _strip_mode_tag(latest)
    keys = list(_keywords(latest_clean))
    if keys:
        fragment = _key_fragment(latest_clean, keys)
        if fragment:
            return fragment
    if _references_prior_context(latest_clean) and previous_user:
        prior = previous_user[-1]
        prior_keys = list(_keywords(prior))
        fragment = _key_fragment(prior, prior_keys)
        if fragment:
            return f"{latest_clean} in relation to {fragment}"
    return latest_clean[:160].strip() or "the user's current message"


def _key_fragment(text: str, keys: list[str]) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= 120:
        return cleaned
    lowered = cleaned.lower()
    positions = [lowered.find(key) for key in keys if lowered.find(key) >= 0]
    if not positions:
        return cleaned[:120].rstrip(" ,.;:")
    start = max(0, min(positions) - 35)
    end = min(len(cleaned), max(positions) + 80)
    return cleaned[start:end].strip(" ,.;:")


def _extract_direct_question(text: str) -> str:
    cleaned = _strip_mode_tag(text)
    if "?" in cleaned:
        return cleaned[: cleaned.find("?") + 1]
    if re.match(r"\s*(what|why|how|when|where|who|should|can|could)\b", cleaned, flags=re.I):
        return cleaned[:180]
    return ""


def _references_prior_context(text: str) -> bool:
    tokens = set(re.findall(r"[a-z']+", text.lower()))
    return bool(tokens & FOLLOWUP_REFERENCES)


def _assistant_frustration(lowered: str) -> bool:
    return any(
        phrase in lowered
        for phrase in (
            "do you not care",
            "do you even care",
            "you dont care",
            "you don't care",
            "not helping",
            "you keep",
            "are you listening",
        )
    )


def _repeated_unanswered_need(latest: str, previous_user: list[str], previous_assistant: list[str]) -> bool:
    latest_need = _detect_intent(latest) in {"next_step", "assistant_repair", "emotional_disclosure"}
    previous_need = any(_detect_intent(message) in {"next_step", "emotional_disclosure"} for message in previous_user[-3:])
    generic_assistant = any(any(marker in _normalize(message) for marker in GENERIC_MARKERS) for message in previous_assistant[-3:])
    return latest_need and previous_need and generic_assistant


def _escalation_detected(distress_level: int, previous_user: list[str]) -> bool:
    previous_levels = [_detect_emotion(_normalize(message))[1] for message in previous_user[-4:]]
    return bool(previous_levels) and distress_level > max(previous_levels)


def _continuity_notes(
    *,
    latest: str,
    previous_user: list[str],
    previous_assistant: list[str],
    references_prior: bool,
    escalation: bool,
    frustration_at_assistant: bool,
    repeated_unanswered: bool,
) -> list[str]:
    notes: list[str] = []
    if previous_user:
        notes.append(f"Previous user message: {previous_user[-1][:160]}")
    if references_prior and previous_user:
        notes.append(f"Latest message appears to refer back to: {previous_user[-1][:160]}")
    if escalation:
        notes.append("The user's distress appears to be escalating compared with earlier turns.")
    if frustration_at_assistant:
        notes.append("The user is reacting to the assistant's failure to respond personally.")
    if repeated_unanswered:
        notes.append("The user has expressed a need that prior assistant replies did not answer.")
    if previous_assistant and any(any(marker in _normalize(message) for marker in GENERIC_MARKERS) for message in previous_assistant[-2:]):
        notes.append("Recent assistant output was generic and should not be repeated.")
    return notes


def _relevant_memory(latest: str, profile: UserProfile | None, memory_summary: str | None) -> list[str]:
    if profile is None and not memory_summary:
        return []
    latest_tokens = _keywords(latest)
    memory_items: list[str] = []
    if profile is not None:
        for label, values in (
            ("recurring topic", profile.recurring_topics),
            ("theme", profile.themes),
            ("pattern", profile.patterns),
            ("stressor", profile.recurringStressors),
            ("goal", profile.goals),
        ):
            for value in values:
                if not _meaningful_memory(value):
                    continue
                value_tokens = _keywords(value)
                if value_tokens and (latest_tokens & value_tokens or len(latest_tokens) <= 3):
                    memory_items.append(f"{label}: {value}")
    if memory_summary and (latest_tokens & _keywords(memory_summary)):
        memory_items.append(f"recent summary: {memory_summary[:220]}")
    return _dedupe(memory_items)[:4]


def _implied_need(intent: str, emotion: str, distress_level: int) -> str:
    if intent == "assistant_repair":
        return "repair trust by answering the user's actual words and acknowledging the miss"
    if distress_level >= 3:
        return "one immediate stabilizing action and one concrete next detail, not a life-wide solution"
    if distress_level >= 1:
        return f"a response that names the {emotion.replace('_', ' ')} and gives one grounded next move"
    if intent == "next_step":
        return "a practical next step that can be done now"
    if intent == "planning":
        return "a usable plan with the next concrete action"
    if intent == "comparison":
        return "a comparison based on the user's real decision criteria"
    if intent == "creation":
        return "the requested artifact or a concrete first draft"
    if intent == "debugging":
        return "isolate the failure and identify the smallest next diagnostic step"
    if intent == "explanation":
        return "a direct explanation before extra context"
    return "a direct answer grounded in the current message"


def _select_tone(understanding: ConversationUnderstanding, profile: UserProfile | None) -> str:
    if understanding.frustration_at_assistant:
        return "accountable, direct, and personally responsive"
    if understanding.distress_level >= 3:
        return "steady, immediate, and non-abstract"
    if understanding.distress_level >= 1:
        return "warm, specific, and grounded"
    if profile and profile.preferences.tone == "direct":
        return "direct and clear"
    if profile and profile.preferences.tone == "clinical":
        return "calm and precise"
    return "natural, specific, and useful"


def _direct_opening(understanding: ConversationUnderstanding) -> str:
    if understanding.frustration_at_assistant:
        return "You're right to call that out."
    if understanding.distress_level >= 3:
        return "I hear this as immediate pain, not an abstract problem."
    if understanding.distress_level >= 1:
        return f"What stands out is the {understanding.emotional_state.replace('_', ' ')} in this."
    if understanding.direct_question:
        return f"You're asking {understanding.direct_question}"
    return f"You're talking about {understanding.topic}."


def _must_address(understanding: ConversationUnderstanding) -> list[str]:
    items = [understanding.latest_user[:180]]
    if understanding.direct_question:
        items.append(understanding.direct_question)
    if understanding.references_prior_context:
        items.append("the latest message's reference to prior context")
    if understanding.frustration_at_assistant:
        items.append("the user's frustration with the assistant")
    if understanding.escalation_detected:
        items.append("the escalation in emotional intensity")
    return _dedupe(items)


def _response_shape(understanding: ConversationUnderstanding) -> str:
    if understanding.frustration_at_assistant:
        return "repair acknowledgment, direct reading of user's need, one concrete next step"
    if understanding.distress_level >= 3:
        return "specific acknowledgment, immediate stabilization, one narrow next question"
    if understanding.distress_level >= 1:
        return "specific reflection, grounded interpretation, one useful next move"
    if understanding.user_intent in {"planning", "comparison", "debugging", "creation"}:
        return "direct output-oriented help, not generic preamble"
    return "direct answer first, then concise reasoning"


def _memory_candidates(latest: str, emotion: str, topic: str) -> dict[str, list[str]]:
    lowered = _normalize(latest)
    themes = _extract_keyphrases(latest)
    patterns: list[str] = []
    stressors: list[str] = []
    goals: list[str] = []

    if emotion not in {"neutral", "crisis"}:
        patterns.append(emotion.replace("_", " "))
    for label, terms in {
        "work": ("work", "job", "boss", "career"),
        "school": ("school", "class", "exam", "college", "grade"),
        "relationships": ("friend", "partner", "relationship", "family"),
        "health": ("health", "symptom", "doctor", "medication"),
        "future": ("future", "year", "life", "plan"),
        "project": ("project", "app", "code", "website", "llm"),
    }.items():
        if any(term in lowered for term in terms):
            stressors.append(label)
            themes.append(label)
    if any(term in lowered for term in ("want to", "need to", "trying to", "goal")):
        goals.append(topic[:80])

    return {
        "themes": _dedupe(themes)[:6],
        "patterns": _dedupe(patterns)[:4],
        "recurringStressors": _dedupe(stressors)[:4],
        "goals": _dedupe(goals)[:3],
    }


def _extract_keyphrases(text: str) -> list[str]:
    cleaned = _normalize(text)
    phrases = re.findall(
        r"\b(?:my|this|the|a|an)\s+([a-z][a-z\s]{2,40}?)(?:\.|,|;|!|\?|$|\s+is|\s+feels|\s+keeps)",
        cleaned,
    )
    keyword_phrases = []
    words = [word for word in re.findall(r"[a-z][a-z']{2,}", cleaned) if word not in STOPWORDS]
    for index in range(0, max(len(words) - 1, 0)):
        pair = " ".join(words[index : index + 2])
        if len(pair) >= 8:
            keyword_phrases.append(pair)
    return [item for item in _dedupe([*phrases, *keyword_phrases]) if _meaningful_memory(item)][:6]


def _merge_limited(existing: list[str], additions: Iterable[str], limit: int) -> list[str]:
    merged = _dedupe([*existing, *[item for item in additions if item]])
    return merged[-limit:]


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        cleaned = re.sub(r"\s+", " ", str(item)).strip(" ,.;:")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def _clean_fragment(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= 180:
        return cleaned
    return cleaned[:177].rstrip(" ,.;:") + "..."


def _user_facing_fragment(text: str) -> str:
    cleaned = _clean_fragment(text)
    cleaned = re.sub(r"^\s*but\s+", "", cleaned, flags=re.I)
    lowered = cleaned.lower()
    if re.match(r"^i\s+(just\s+)?(dont|don't)\s+know\s+what\s+to\s+do\b", lowered):
        return "not knowing what to do right now"
    if re.match(r"^i\s+(just\s+)?hate\s+my\s+life\b", lowered):
        return "feeling like you hate your life"
    need_match = re.match(
        r"^i\s+need\s+(?:a\s+)?(?:plan|help|advice|support|answer)?\s*(?:for|with|to)?\s+(.+)$",
        lowered,
    )
    if need_match and need_match.group(1).strip():
        return need_match.group(1).strip()
    cleaned = re.sub(r"^my\b", "your", cleaned, flags=re.I)
    return cleaned


def _meaningful_memory(value: str) -> bool:
    cleaned = _normalize(value)
    if cleaned in {"dont know", "don't know", "know what", "just dont", "just don't", "what do", "should do"}:
        return False
    tokens = _keywords(cleaned)
    if len(tokens) == 0:
        return False
    return any(len(token) >= 5 for token in tokens) or cleaned in {"work", "school", "health", "future", "project"}


def _prior_user_from_context(prior_context: list[str]) -> str:
    for note in reversed(prior_context):
        if note.lower().startswith("previous user message:"):
            return _user_facing_fragment(note.split(":", 1)[1].strip())
    return ""


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _normalize(left), _normalize(right)).ratio()
