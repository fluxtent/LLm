import unittest
from types import SimpleNamespace

from backend.app.inference import BaseInferenceEngine, InferenceResult, LocalResponderEngine
from backend.app.main import ModelUnavailableError, _generate_completion
from backend.app.personalization import (
    apply_memory_updates,
    build_personalization_context,
    build_response_plan,
    evaluate_response_quality,
)
from backend.app.schemas import ChatCompletionRequest, ChatMessage, UserProfile
from backend.app.safety import evaluate_request, is_low_quality_response
from backend.app.settings import Settings


def plan_for(
    messages: list[tuple[str, str]],
    profile: UserProfile | None = None,
    memory_summary: str | None = None,
):
    request = ChatCompletionRequest(
        messages=[ChatMessage(role=role, content=content) for role, content in messages],
        memory_summary=memory_summary,
    )
    context = build_personalization_context(request, profile)
    return build_response_plan(context, requested_mode=request.mode)


class FakeEngine(BaseInferenceEngine):
    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    async def complete(self, **kwargs) -> InferenceResult:
        del kwargs
        if self.responses:
            text = self.responses.pop(0)
        else:
            text = "The important part is repeated again. Tell me what outcome you want."
        return InferenceResult(text=text, upstream_model="fake-local-llm")


class PersonalizationPlanningTests(unittest.TestCase):
    def test_confusion_gets_next_step_understanding(self) -> None:
        plan = plan_for([("user", "i just dont know what to do")])

        self.assertEqual(plan.understanding.user_intent, "next_step")
        self.assertIn("practical next step", plan.understanding.implied_need)
        self.assertIn("actual words", " ".join(plan.avoid).lower())

    def test_escalating_self_hate_uses_current_chat_context(self) -> None:
        plan = plan_for(
            [
                ("user", "i just dont know what to do"),
                ("assistant", "The cleanest way to approach that is to separate what is factual."),
                ("user", "but i just hate my life"),
            ]
        )

        self.assertEqual(plan.understanding.emotional_state, "self_hate")
        self.assertTrue(plan.understanding.escalation_detected)
        self.assertIn("hate my life", plan.understanding.topic.lower())
        self.assertTrue(plan.prior_context)

    def test_frustration_at_assistant_triggers_repair_plan(self) -> None:
        plan = plan_for(
            [
                ("user", "i just dont know what to do"),
                ("assistant", "The cleanest way to approach that is to separate what is factual."),
                ("user", "do you not care either"),
            ]
        )

        self.assertTrue(plan.understanding.frustration_at_assistant)
        self.assertIn("repair trust", plan.understanding.implied_need)
        self.assertIn("assistant", " ".join(plan.understanding.continuity_notes).lower())

    def test_broad_emotional_categories_are_understood(self) -> None:
        examples = [
            ("i am so confused and lost", "confused"),
            ("i feel ashamed of myself", "self_hate"),
            ("i am angry that nobody listened", "anger"),
            ("i feel completely alone tonight", "lonely"),
            ("i am grieving and miss them", "grief"),
            ("i am under so much pressure", "overwhelmed"),
        ]

        for message, expected_emotion in examples:
            with self.subTest(message=message):
                plan = plan_for([("user", message)])
                self.assertEqual(plan.understanding.emotional_state, expected_emotion)
                self.assertGreaterEqual(plan.understanding.distress_level, 1)

    def test_followup_resolves_prior_context(self) -> None:
        plan = plan_for(
            [
                ("user", "I am choosing between computer science and premed"),
                ("assistant", "Let's compare the paths."),
                ("user", "what should I do about that"),
            ]
        )

        self.assertTrue(plan.understanding.references_prior_context)
        self.assertIn("computer science", " ".join(plan.prior_context).lower())

    def test_relevant_profile_memory_is_used_when_related(self) -> None:
        profile = UserProfile(
            user_id="user-1",
            recurring_topics=["burnout"],
            patterns=["shutdown"],
        )
        plan = plan_for([("user", "it is happening again at work")], profile=profile)

        self.assertTrue(any("burnout" in item.lower() for item in plan.relevant_memory))

    def test_generic_duplicate_response_is_rejected(self) -> None:
        plan = plan_for(
            [
                ("user", "i just dont know what to do"),
                (
                    "assistant",
                    "The cleanest way to approach that is to separate what is factual, what is interpretive, and what kind of answer would actually help right now.",
                ),
                ("user", "but i just hate my life"),
            ]
        )
        quality = evaluate_response_quality(
            "The cleanest way to approach that is to separate what is factual, what is interpretive, and what kind of answer would actually help right now.",
            plan,
        )

        self.assertTrue(quality.generic_response)
        self.assertTrue(quality.duplicate_response)
        self.assertTrue(quality.should_override)

    def test_meta_and_coded_renderer_phrases_are_rejected(self) -> None:
        plan = plan_for([("user", "fuck life")])
        quality = evaluate_response_quality(
            "The important part is [redacted] life. Tell me what outcome you want, and I will shape the answer around that instead of guessing.",
            plan,
        )

        self.assertTrue(quality.generic_response)
        self.assertTrue(quality.should_override)

    def test_old_practical_layer_template_is_rejected(self) -> None:
        plan = plan_for([("user", "hi")])
        quality = evaluate_response_quality(
            "That question has a practical layer and a conceptual layer, and it helps to decide which one matters more first.",
            plan,
        )

        self.assertTrue(quality.generic_response)
        self.assertTrue(quality.should_override)

    def test_purpose_answer_can_be_conceptual_without_copying_user_words(self) -> None:
        plan = plan_for([("user", "what is the purpose of anything")])
        quality = evaluate_response_quality(
            "Meaning is the reason something matters enough to shape a choice, a life, or a relationship.",
            plan,
        )

        self.assertFalse(quality.ignored_latest_user)
        self.assertFalse(quality.should_override)

    def test_incomplete_generation_is_rejected(self) -> None:
        self.assertTrue(
            is_low_quality_response(
                'This sounds unbearable right now, and the first priority is getting you through the next minute, that is how "'
            )
        )

    def test_explicit_crisis_still_routes_to_safety(self) -> None:
        plan = plan_for([("user", "I want to die")])
        decision = evaluate_request(plan.mode, plan.understanding.latest_user)

        self.assertFalse(decision.allow_model)
        self.assertEqual(decision.safety_flag, "crisis_intercept")

    def test_memory_updates_are_generalized(self) -> None:
        profile = UserProfile(user_id="user-1")
        plan = plan_for([("user", "my school project keeps making me spiral")], profile=profile)
        updated = apply_memory_updates(profile, plan.understanding)

        self.assertIn("school", updated.themes)
        self.assertIn("project", updated.recurringStressors)
        self.assertTrue(updated.patterns)


class PersonalizationRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_uses_local_fallback_instead_of_coded_generic_response(self) -> None:
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="fuck life")],
            max_tokens=120,
            stream=False,
        )
        engine = FakeEngine(
            [
                "The important part is [redacted] life. Tell me what outcome you want, and I will shape the answer around that instead of guessing.",
                "The important part is [redacted] life. Tell me what outcome you want, and I will shape the answer around that instead of guessing.",
            ]
        )

        _body, telemetry, text = await _generate_completion(
            SimpleNamespace(client=None),
            request,
            Settings(),
            engine,
            fallback_engine=LocalResponderEngine(),
        )

        self.assertIn("Fuck life", text)
        self.assertIn("safe", text.lower())
        self.assertNotIn("Tell me what outcome you want", text)
        self.assertNotIn("[redacted]", text)
        self.assertTrue(telemetry["fallback_flag"])

    async def test_local_fallback_answers_health_definition(self) -> None:
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="what is nephritis")],
            max_tokens=160,
            stream=False,
        )
        engine = FakeEngine([])

        _body, telemetry, text = await _generate_completion(
            SimpleNamespace(client=None),
            request,
            Settings(inference_engine="vllm"),
            engine,
            fallback_engine=LocalResponderEngine(),
        )

        self.assertIn("Nephritis", text)
        self.assertIn("kidneys", text)
        self.assertNotIn("model backend unavailable", text)
        self.assertTrue(telemetry["fallback_flag"])

    async def test_local_fallback_answers_medical_specialty_without_generic_shell(self) -> None:
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="what is nephrology")],
            max_tokens=160,
            stream=False,
        )
        engine = FakeEngine([])

        _body, telemetry, text = await _generate_completion(
            SimpleNamespace(client=None),
            request,
            Settings(inference_engine="vllm"),
            engine,
            fallback_engine=LocalResponderEngine(),
        )

        self.assertIn("kidney", text.lower())
        self.assertIn("medicine", text.lower())
        self.assertNotIn("thing you are trying to pin down", text)
        self.assertNotIn("A useful definition should say", text)
        self.assertTrue(telemetry["fallback_flag"])

    async def test_unknown_definition_refuses_shell_instead_of_word_swap(self) -> None:
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="what is flermology")],
            max_tokens=160,
            stream=False,
        )
        engine = FakeEngine([])

        _body, _telemetry, text = await _generate_completion(
            SimpleNamespace(client=None),
            request,
            Settings(inference_engine="vllm"),
            engine,
            fallback_engine=LocalResponderEngine(),
        )

        self.assertNotIn("means the thing you are trying to pin down", text)
        self.assertIn("not have enough", text.lower())

    async def test_runtime_can_still_raise_without_any_fallback(self) -> None:
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="what is nephritis")],
            max_tokens=120,
            stream=False,
        )

        with self.assertRaises(ModelUnavailableError):
            await _generate_completion(
                SimpleNamespace(client=None),
                request,
                Settings(inference_engine="vllm"),
                FakeEngine([]),
                fallback_engine=None,
            )

    async def test_mock_runtime_uses_engine_output_without_plan_renderer(self) -> None:
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            max_tokens=120,
            stream=False,
        )
        engine = FakeEngine(
            [
                "Hello, I can continue from your current chat and answer directly instead of using a canned wrapper.",
            ]
        )

        _body, telemetry, text = await _generate_completion(
            SimpleNamespace(client=None),
            request,
            Settings(inference_engine="mock"),
            engine,
            fallback_engine=None,
        )

        self.assertIn("Hello", text)
        self.assertNotIn("practical layer", text)
        self.assertEqual(telemetry["engine"], "mock")

    async def test_runtime_does_not_redact_user_language_in_model_answer(self) -> None:
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="fuck life")],
            max_tokens=120,
            stream=False,
        )
        engine = FakeEngine(
            [
                "Fuck life sounds less like a topic and more like you are overloaded right now. The first question is whether you are safe.",
            ]
        )

        _body, _telemetry, text = await _generate_completion(
            SimpleNamespace(client=None),
            request,
            Settings(),
            engine,
            fallback_engine=None,
        )

        self.assertIn("Fuck life", text)
        self.assertNotIn("[redacted]", text)


if __name__ == "__main__":
    unittest.main()
