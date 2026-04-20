import unittest

from backend.app.constants import CRISIS_KEYWORDS, HIGH_CONFIDENCE_CRISIS_KEYWORDS
from backend.app.safety import clean_response_text, evaluate_request, is_crisis, is_low_quality_response


class CrisisDetectionTests(unittest.TestCase):
    def test_every_high_confidence_keyword_is_detected_in_context(self) -> None:
        for keyword in HIGH_CONFIDENCE_CRISIS_KEYWORDS:
            with self.subTest(keyword=keyword):
                self.assertTrue(is_crisis(f"I am feeling like {keyword} right now"))

    def test_expanded_hopelessness_language_triggers_intercept(self) -> None:
        decision = evaluate_request("general", "i dont know how to go on with life")
        self.assertFalse(decision.allow_model)
        self.assertEqual(decision.safety_flag, "crisis_intercept")
        self.assertIn("988", decision.response_text)

    def test_keyword_plus_distress_signal_triggers_intercept(self) -> None:
        self.assertTrue(is_crisis("what's the point anymore when I feel so alone"))

    def test_ambiguous_phrase_without_distress_is_not_treated_as_crisis(self) -> None:
        self.assertFalse(is_crisis("what's the point of this feature flag"))

    def test_broader_keywords_need_distress_or_multiple_signals(self) -> None:
        softer_keywords = [keyword for keyword in CRISIS_KEYWORDS if keyword not in HIGH_CONFIDENCE_CRISIS_KEYWORDS]
        for keyword in softer_keywords:
            with self.subTest(keyword=keyword):
                probe = f"I keep thinking {keyword} and I feel so alone"
                self.assertTrue(is_crisis(probe))


class ResponseQualityTests(unittest.TestCase):
    def test_low_quality_detects_artifacts(self) -> None:
        text = "[INST] hello [/INST] assistant: please continue with tokens and tokens and tokens and tokens."
        self.assertTrue(is_low_quality_response(text))

    def test_low_quality_detects_prompt_echo(self) -> None:
        text = "You are MedBrief AI and you are MedBrief AI and you are MedBrief AI in this reply."
        self.assertTrue(is_low_quality_response(text))

    def test_clean_response_trims_incomplete_tail_after_last_full_sentence(self) -> None:
        text = (
            "Cardiology is a branch of medicine that focuses on the heart and blood vessels. "
            "Cardiologists often deal with conditions like coronary artery disease, arrhythmias, and"
        )
        self.assertEqual(
            clean_response_text(text),
            "Cardiology is a branch of medicine that focuses on the heart and blood vessels.",
        )
