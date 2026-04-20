"""Template-driven fallback generation with higher variety and profile-aware continuity."""

from __future__ import annotations

import hashlib
import random
import re
import secrets
from dataclasses import dataclass

from .medical_ontology import MedicalContext
from .schemas import UserProfile


def _seeded_rng(*parts: str, use_entropy: bool = True) -> random.Random:
    seed_material = "||".join(str(part) for part in parts)
    if use_entropy:
        seed_material += f"||{secrets.token_hex(8)}"
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


def _trigram_overlap(left: str, right: str) -> float:
    def trigrams(text: str) -> set[tuple[str, str, str]]:
        words = text.lower().split()
        return {tuple(words[index : index + 3]) for index in range(len(words) - 2)}

    left_grams = trigrams(left)
    right_grams = trigrams(right)
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / min(len(left_grams), len(right_grams))


def _choose(rng: random.Random, items: list[str]) -> str:
    return items[rng.randrange(len(items))]


def _sentences(text: str) -> list[str]:
    return [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text.strip()) if chunk.strip()]


@dataclass(frozen=True)
class TemplateRequest:
    latest_user: str
    mode: str
    request_id: str
    conversation_id: str | None = None
    profile: UserProfile | None = None
    medical_context: MedicalContext | None = None
    recent_assistant_messages: tuple[str, ...] = ()


class TemplateEngine:
    """Creates varied, mode-aware fallback text without repeating canned paragraphs."""

    def __init__(self, use_entropy: bool = True) -> None:
        self._use_entropy = use_entropy

    def render(self, request: TemplateRequest) -> str:
        last_candidate = self._general(request, _seeded_rng("fallback", use_entropy=self._use_entropy))
        for salt in tuple(str(index) for index in range(8)):
            rng = _seeded_rng(
                request.request_id,
                request.conversation_id or "no-conversation",
                request.mode,
                request.latest_user,
                salt,
                use_entropy=self._use_entropy,
            )
            candidate = self._candidate_for_mode(request, rng)
            candidate = self._apply_length_preference(candidate, request, rng)
            candidate = re.sub(r"\s+", " ", candidate).strip()
            last_candidate = candidate
            if not self._validate_template_output(candidate, request.mode):
                continue
            if any(_trigram_overlap(candidate, prior) > 0.4 for prior in request.recent_assistant_messages[-4:]):
                continue
            return candidate
        return last_candidate

    def _candidate_for_mode(self, request: TemplateRequest, rng: random.Random) -> str:
        if request.mode == "psych":
            return self._psych(request, rng)
        if request.mode == "health":
            return self._health(request, rng)
        if request.mode == "portfolio":
            return self._portfolio(request, rng)
        if request.mode == "crisis":
            return self._crisis(request, rng)
        return self._general(request, rng)

    def _validate_template_output(self, text: str, mode: str) -> bool:
        lowered = text.lower()
        min_words = {
            "crisis": 30,
            "psych": 40,
            "health": 40,
            "portfolio": 30,
            "general": 25,
        }
        if len(text.split()) < min_words.get(mode, 25):
            return False

        if mode == "crisis":
            return "988" in text or "emergency" in lowered
        if mode == "psych":
            return "?" in text and not any(term in lowered for term in ("you have", "you are diagnosed", "disorder"))
        if mode == "health":
            has_uncertainty = any(term in lowered for term in ("may", "can", "often", "generally", "possible", "sometimes"))
            has_unsafe_medical_claim = any(
                term in lowered
                for term in ("you definitely have", "you have", "take ", "dosage", "milligrams", "increase your dose")
            )
            return has_uncertainty and not has_unsafe_medical_claim
        if mode == "portfolio":
            return any(term in lowered for term in ("product", "mode-aware", "safety", "memory", "trust", "medbrief ai"))
        if mode == "general":
            if lowered.startswith("thank you for sharing"):
                return False
            return any(
                term in lowered
                for term in (
                    "next step",
                    "if you want",
                    "tell me",
                    "practical",
                    "tradeoff",
                    "checklist",
                    "decision",
                )
            )
        return True

    def _continuity_line(self, request: TemplateRequest, mode: str, rng: random.Random) -> str:
        profile = request.profile
        if profile is None:
            return ""

        lower = request.latest_user.lower()
        recurring_signals = [
            *profile.recurring_topics,
            *profile.themes,
            *profile.patterns,
            *profile.recurringStressors,
        ]
        match = next((item for item in recurring_signals if item and item.lower() in lower), None)
        if not match:
            return ""

        if mode == "psych":
            return _choose(
                rng,
                [
                    f"Because {match} has come up for you before, it may help to treat this as a recurring pattern instead of an isolated moment.",
                    f"Since {match} seems to be a repeating theme for you, continuity probably matters more here than a one-off tip.",
                    f"The fact that {match} has shown up before suggests there may be a loop here worth mapping gently.",
                ],
            )
        if mode == "health":
            return _choose(
                rng,
                [
                    f"Because {match} appears in your longer-running context too, it may help to notice whether the timing or triggers are repeating.",
                    f"Since {match} has been part of your broader context, the pattern over time may matter as much as the symptom itself.",
                    f"If {match} has been recurring for you, tracking when it appears can be more useful than focusing on one isolated moment.",
                ],
            )
        return _choose(
            rng,
            [
                f"Since {match} seems to be part of the bigger picture for you, it may help to anchor this answer in that ongoing context.",
                f"Because {match} appears to be a recurring thread for you, the useful answer is probably the one that fits the longer pattern.",
            ],
        )

    def _apply_length_preference(self, text: str, request: TemplateRequest, rng: random.Random) -> str:
        profile = request.profile
        preference = profile.preferences.response_length if profile else "balanced"
        if preference == "concise":
            chunks = _sentences(text)
            return " ".join(chunks[:2]) if len(chunks) > 2 else text
        if preference == "detailed":
            detail = self._detail_line(request.mode, request, rng)
            if detail:
                return f"{text} {detail}"
        return text

    def _detail_line(self, mode: str, request: TemplateRequest, rng: random.Random) -> str:
        continuity = self._continuity_line(request, mode, rng)
        detailers = {
            "psych": [
                "If it helps, we can slow this down and map what happens before, during, and after the hardest moment.",
                "A useful next layer is to separate the trigger, the interpretation, and the coping move that follows.",
            ],
            "health": [
                "If you want, tell me about the timing, severity, and what else shows up with it, and I can help frame the general possibilities more carefully.",
                "If you want, I can help you think through what details a clinician would usually want to hear first.",
            ],
            "portfolio": [
                "If you want, I can also walk through the product architecture, safety layer, or memory design decisions behind it.",
                "If you want, I can explain the system from a product, UX, or technical perspective.",
            ],
            "general": [
                "If you want, give me one concrete example and I can make the answer less abstract.",
                "If you want, I can turn this into a checklist, comparison, or decision tree.",
            ],
        }
        options = detailers.get(mode, [])
        if continuity:
            options = [continuity, *options]
        return _choose(rng, options) if options else continuity

    def _psych(self, request: TemplateRequest, rng: random.Random) -> str:
        lower = request.latest_user.lower()
        opener = _choose(
            rng,
            [
                "That sounds emotionally heavy, and it makes sense that it is taking up real space.",
                "What you're describing sounds draining, not trivial.",
                "That does not sound like a small passing feeling. It sounds like something your mind keeps having to carry.",
                "I can hear that there is real emotional weight in this.",
                "That sounds like the kind of thing that can quietly take over a whole day.",
                "It makes sense that this would stay mentally loud.",
                "That sounds exhausting in a way that is easy for other people to miss.",
                "There is a real emotional pattern underneath what you just described.",
                "That sounds like more than a bad moment. It sounds like something that keeps looping back.",
                "What you're carrying here sounds persistent, not incidental.",
                "That sounds tiring in the deep sense, not just in the obvious sense.",
                "I can hear that this is not just frustrating. It sounds personally costly.",
            ],
        )
        pattern = _choose(
            rng,
            [
                "A useful next step is to slow the loop down into trigger, feeling, interpretation, and reaction.",
                "These situations usually become clearer when you separate what happened from what your mind immediately concluded.",
                "It often helps to map the pattern instead of treating each moment like a new mystery.",
                "The leverage point is often hidden in the sequence: what happened first, what tightened next, and what you did to cope.",
                "What looks like one big feeling is often a chain of smaller moves happening very fast.",
                "When a feeling keeps repeating, the pattern usually matters more than the single moment.",
                "The turning point is often noticing the loop before it feels inevitable.",
                "You usually get more clarity by naming the cycle than by arguing with the feeling head-on.",
                "The mind often turns one painful moment into a broader story very quickly, and that story is worth slowing down.",
                "The part worth understanding is often the transition between the trigger and the coping move.",
                "A lot of these loops get less overwhelming once you can name the first internal shift instead of only the aftermath.",
                "When something keeps returning, it usually helps to ask what function the pattern is serving, even if it is costly.",
            ],
        )
        continuity = self._continuity_line(request, "psych", rng)
        question = _choose(
            rng,
            [
                "What tends to happen first for you: the thought, the body tension, or the urge to shut down?",
                "When this starts, what is the first thing your mind tells you?",
                "If you zoom in on the last time it happened, what was the trigger right before it escalated?",
                "What part feels most mentally loud right now: the feeling itself, what it might mean, or what you think it says about you?",
                "Does this usually hit harder when you are tired, pressed for time, or already feeling behind?",
                "What do you usually do next when this feeling spikes?",
                "If we mapped the cycle clearly, where do you think the first turn happens?",
                "What feels more accurate here: pressure, fear, shame, grief, or something else?",
                "When this hits, do you usually feel more trapped, more exhausted, or more self-critical?",
                "What does this feeling usually push you to do next, even if that move only helps for a moment?",
                "If this has a repeating pattern, what seems to reset the loop each time?",
                "What part of this feels hardest to say out loud?",
            ],
        )

        if "anx" in lower or "panic" in lower or "worry" in lower:
            angle = _choose(
                rng,
                [
                    "Anxiety tends to make the future feel urgent and slippery at the same time.",
                    "Anxiety often turns uncertainty into something that feels much closer than it really is.",
                    "Anxiety usually narrows attention until the threat feels like the only thing in the room.",
                    "Anxiety can make the body feel like the decision has already been made for you.",
                    "Anxiety often makes every possibility feel equally immediate, even when they are not.",
                    "Anxiety can turn a maybe into a felt certainty very quickly.",
                ],
            )
            parts = [opener, angle, continuity, pattern, question]
            return " ".join(part for part in parts if part)

        if any(token in lower for token in ("burnout", "drained", "exhausted", "numb")):
            angle = _choose(
                rng,
                [
                    "Burnout is often less about weakness and more about a nervous system that has been asked to stay activated for too long.",
                    "Burnout usually makes even reasonable tasks feel heavier because the system is already overdrawn.",
                    "That sounds closer to depletion than ordinary stress.",
                    "When you are this depleted, even simple choices can start feeling unfairly expensive.",
                    "Burnout often narrows life into obligation, then makes recovery feel like more work too.",
                    "This sounds like depletion, not a motivation problem.",
                ],
            )
            parts = [opener, angle, continuity, pattern, question]
            return " ".join(part for part in parts if part)

        parts = [opener, continuity, pattern, question]
        return " ".join(part for part in parts if part)

    def _health(self, request: TemplateRequest, rng: random.Random) -> str:
        terminology = request.profile.preferences.terminology if request.profile else "lay"
        med = request.medical_context or MedicalContext([], [], [], [], [])
        opening = _choose(
            rng,
            [
                "Generally speaking, that can have more than one explanation, so I do not want to collapse it into a diagnosis.",
                "A careful way to think about that is to separate possibilities from certainty.",
                "The safest way to frame that is as general medical information rather than a diagnosis.",
                "That question deserves a cautious answer because the right interpretation depends on details.",
                "A measured answer here starts with uncertainty, not overconfidence.",
                "The clinical reality is usually more than one possibility, so caution matters.",
                "A responsible answer here should stay educational rather than diagnostic.",
                "There are several possible explanations, and the details change how concerning it is.",
                "The useful answer here is usually probabilistic, not definitive.",
                "The careful approach is to treat this as a pattern-recognition problem, not a snap diagnosis.",
            ],
        )
        factors = _choose(
            rng,
            [
                "What matters most is timing, severity, duration, and what else is happening alongside it.",
                "The useful details are onset, intensity, triggers, and whether other symptoms travel with it.",
                "The key questions are how long it lasts, whether it is worsening, and what other changes show up with it.",
                "The safest next step is to look at pattern, severity, and whether daily functioning is being affected.",
                "A clinician would usually care about onset, progression, associated symptoms, and what makes it better or worse.",
                "It helps to notice whether this is new, escalating, recurrent, or linked to stress, sleep, food, or exertion.",
                "The right frame is not just what the symptom is, but when it appears, how strong it is, and whether it is changing.",
                "The most useful details are the timeline, the body systems involved, and whether the symptom is new for you.",
            ],
        )
        boundary = _choose(
            rng,
            [
                "If it is severe, persistent, worsening, or paired with alarming symptoms, a clinician should evaluate it.",
                "If it is clearly escalating, interfering with function, or just feels medically off, professional evaluation is the right next step.",
                "If this is intense, changing quickly, or clustering with other symptoms, clinical care is the safest move.",
                "If it is new, worsening, or hard to explain, a clinician can sort out what actually needs follow-up.",
                "If it is persistent or severe, a clinician should evaluate the full picture rather than guessing from one symptom.",
                "If this keeps happening or feels outside your normal pattern, a clinician can assess it more safely than a chatbot can.",
                "If the picture is getting more intense or more complex, in-person medical care is the right next step.",
                "If this is disrupting daily life or stacking with other symptoms, professional care makes sense.",
            ],
        )
        style_hint = (
            " I can use more clinical terminology if that is your preference."
            if terminology == "professional"
            else " I can also translate this into more patient-friendly language if you want."
        )
        continuity = self._continuity_line(request, "health", rng)
        med_line = ""
        if med.drugs:
            med_line = (
                f" Since medications like {', '.join(med.drugs[:2])} can matter, a clinician or pharmacist should be the one to interpret drug-specific risk."
            )
        elif med.diagnoses:
            med_line = (
                f" Terms like {', '.join(med.diagnoses[:2])} are relevant context, but they still should not be treated as a diagnosis from one chat message."
            )

        parts = [opening, continuity, factors, boundary]
        return (" ".join(part for part in parts if part) + style_hint + med_line).strip()

    def _portfolio(self, request: TemplateRequest, rng: random.Random) -> str:
        opener = _choose(
            rng,
            [
                "MedBrief AI is meant to feel like a product with a point of view, not a generic chatbot dropped into a portfolio.",
                "The differentiator is not just model access; it is the way the whole product is structured around trust, tone, and safety.",
                "What makes MedBrief AI interesting as a product is the combination of interaction design, safety policy, and runtime behavior.",
                "The product story is really about making a high-stakes assistant feel intentional rather than improvised.",
                "The most important difference is that MedBrief AI tries to behave like a coherent product surface, not just a language model endpoint.",
                "The value is in how the system behaves under different modes, not just that it can answer text prompts.",
            ],
        )
        differentiator = _choose(
            rng,
            [
                "mode-aware response design",
                "memory continuity",
                "medical boundary handling",
                "crisis interception",
                "OpenAI-compatible integration",
                "feedback-driven improvement loops",
                "template and model hybrid resilience",
                "clear runtime observability",
            ],
        )
        outcome = _choose(
            rng,
            [
                "That makes the experience feel calmer and more trustworthy under pressure.",
                "That helps the assistant stay coherent when the conversation moves between emotional support and medical explanation.",
                "That gives users something closer to product trust than novelty.",
                "That matters because people feel the difference between a polished system and a demo immediately.",
                "That is what turns the interface into something credible rather than merely functional.",
                "That makes the product easier to trust before the user even thinks about model architecture.",
            ],
        )
        return f"{opener} One clear differentiator is {differentiator}. {outcome}"

    def _crisis(self, request: TemplateRequest, rng: random.Random) -> str:
        del request, rng
        return (
            "I'm really glad you said something. Your safety matters most right now. "
            "Please call or text 988 right now if you're in the US, or contact local emergency services if you may act on these thoughts. "
            "If there is someone nearby, tell them clearly that you need help staying safe right now."
        )

    def _general(self, request: TemplateRequest, rng: random.Random) -> str:
        lower = request.latest_user.lower().strip()
        if "meaning of life" in lower or "purpose" in lower:
            return _choose(
                rng,
                [
                    "A grounded answer is that meaning is usually something we build rather than discover once and keep forever. It grows out of love, responsibility, curiosity, service, and the things we choose not to betray. If you want, say whether you want the philosophical, psychological, or practical version of that answer.",
                    "The most honest answer is that meaning is less like a fact you find and more like a pattern you create through commitment, relationships, and what you keep showing up for. If you want, I can make that more personal or more philosophical.",
                    "Meaning usually comes from the intersection of care, responsibility, and attention: who matters to you, what you work on, and what you are willing to carry on purpose. If you want, I can help apply that idea to your actual life instead of keeping it abstract.",
                    "A useful answer is that meaning is often built through attachment, effort, and direction rather than uncovered all at once. If you want, give me the angle you care about most and I can make this more concrete.",
                ],
            )
        if "death" in lower:
            return _choose(
                rng,
                [
                    "Death matters because it places a boundary around life. That boundary is part of what gives love, choice, regret, urgency, and repair their emotional weight. If you want, I can answer this more philosophically or more personally.",
                    "Death is frightening partly because it holds loss, uncertainty, and finality together. But that same finitude is part of why time and relationship feel precious at all. If you want, I can take this in a practical, emotional, or philosophical direction.",
                    "A philosophical answer is that death gives life urgency. A personal answer is that it confronts us with loss, uncertainty, and the limit of our control. If you want, say which layer matters more right now.",
                    "Death is difficult because it is both abstract and intimate at once: a concept until it becomes a person, a time, or a future you can no longer have. If you want, I can stay with the philosophy or move toward the emotional side of it.",
                ],
            )

        opener = _choose(
            rng,
            [
                "The clearest path through this is to break the question into its smaller working parts.",
                "The most useful starting point is clarifying what kind of answer would actually help right now.",
                "There's usually a simpler version of this question hiding inside the complex one.",
                "Separating facts from assumptions is usually where clarity starts.",
                "That question has a practical layer and a conceptual layer, and it helps to decide which one matters more first.",
                "The honest first step is admitting what we know versus what we're guessing at.",
                "There are usually two or three valid ways to frame this, and the framing changes the useful answer.",
                "The clearest move here is probably to zoom out before going deeper.",
                "Good questions like this usually get better when you tighten the actual goal first.",
                "Let me work backward from what a useful answer would actually look like.",
                "Complexity here usually lives in the transitions between parts, not in any single part.",
                "Most questions like this come apart cleanly once you separate what's known, assumed, and still needed.",
                "The short answer is possible, but the useful answer comes from naming the missing detail.",
                "The practical move here is to name the real decision you're trying to make.",
                "There's a tension in this question worth naming before trying to resolve it.",
                "The most grounded approach is to hold multiple possibilities before committing to one.",
                "Usually the loop breaks when you change one assumption, not when you simply push harder.",
                "Working backward from the outcome tends to make the path clearer than pushing forward from the confusion.",
                "The useful part of this question is what it's actually asking you to decide.",
                "Before solving this, it helps to know whether you want validation, options, or a concrete plan.",
            ],
        )
        continuity = self._continuity_line(request, "general", rng)
        closer = _choose(
            rng,
            [
                "A good next step is to tell me the exact situation, and I can turn it into a practical breakdown.",
                "If you want, I can make this more direct, more structured, or more reflective.",
                "If you want, I can turn this into a short decision framework instead of leaving it abstract.",
                "The fastest way forward is to name the constraint that matters most, and we can work from there.",
                "If you give me the concrete context, I can compare options and tradeoffs with you.",
                "A useful next step is to separate the part you can act on today from the part that simply needs more information.",
                "If you want, I can help you find the next action instead of trying to solve the whole problem at once.",
                "The practical next move is to define what a good outcome would look like in one sentence.",
                "If you want, I can help you pressure-test the assumptions before we choose a direction.",
                "A concrete next step is to tell me what feels most stuck, and I can help untangle that part first.",
                "If you want, I can answer this philosophically, practically, or as a step-by-step plan.",
                "The next useful move is to say whether the problem is uncertainty, overload, or conflicting priorities.",
                "If you want, I can turn this into a checklist you can actually use.",
                "A grounded next step is to tell me what decision or outcome is closest in time.",
                "If you want, I can help convert this into a clearer question before we answer it.",
                "The most actionable next step is to give me the specifics you're weighing.",
                "If you want, I can help identify the leverage point instead of treating everything as equally important.",
                "A good next step is to name what feels non-negotiable here, and we can build around that.",
                "If you want, I can walk through the options one by one and show where the real tradeoff lives.",
                "The clearest next step is to anchor this in one real example, then we can make the answer more useful.",
            ],
        )
        parts = [opener, continuity, closer]
        return " ".join(part for part in parts if part)
