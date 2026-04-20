import unittest

from backend.app.schemas import UserPreferences, UserProfile
from backend.app.template_engine import TemplateEngine, TemplateRequest


class TemplateEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = TemplateEngine(use_entropy=False)

    def test_general_template_meets_quality_bar(self) -> None:
        request = TemplateRequest(
            latest_user="How should I think about a hard decision at work?",
            mode="general",
            request_id="req-1",
            conversation_id="conv-1",
        )

        response = self.engine.render(request)

        self.assertTrue(self.engine._validate_template_output(response, "general"))

    def test_psych_template_uses_profile_continuity(self) -> None:
        profile = UserProfile(
            user_id="user-1",
            recurring_topics=["burnout"],
            themes=["work"],
            patterns=["self-criticism"],
            preferences=UserPreferences(response_length="detailed"),
        )
        request = TemplateRequest(
            latest_user="My burnout is getting worse and I keep shutting down",
            mode="psych",
            request_id="req-2",
            conversation_id="conv-1",
            profile=profile,
        )

        response = self.engine.render(request)

        self.assertIn("burnout", response.lower())
        self.assertTrue(self.engine._validate_template_output(response, "psych"))

    def test_recent_duplicate_is_avoided(self) -> None:
        first_request = TemplateRequest(
            latest_user="How do I structure this decision?",
            mode="general",
            request_id="req-3",
            conversation_id="conv-2",
        )
        first_response = self.engine.render(first_request)

        second_request = TemplateRequest(
            latest_user="How do I structure this decision?",
            mode="general",
            request_id="req-4",
            conversation_id="conv-2",
            recent_assistant_messages=(first_response,),
        )
        second_response = self.engine.render(second_request)

        self.assertNotEqual(first_response, second_response)
