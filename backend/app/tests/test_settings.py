import unittest

from backend.app.settings import Settings


class SettingsValidationTests(unittest.TestCase):
    def test_validate_for_production_rejects_mock_and_default_key(self) -> None:
        settings = Settings(
            environment="production",
            inference_engine="mock",
            vllm_base_url="",
            vllm_api_key="medbrief-local",
        )

        errors = settings.validate_for_production()

        self.assertTrue(any("active_engine is mock" in error for error in errors))

    def test_validate_for_production_accepts_vllm(self) -> None:
        settings = Settings(
            environment="production",
            inference_engine="vllm",
            vllm_base_url="http://127.0.0.1:8000",
            vllm_api_key="super-secret",
        )

        self.assertEqual(settings.validate_for_production(), [])

    def test_default_engine_prefers_ollama(self) -> None:
        settings = Settings(
            inference_engine="",
            vllm_base_url="",
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model="llama3.1:8b",
        )

        self.assertEqual(settings.active_engine, "ollama")
        self.assertEqual(settings.runtime_model_id, "llama3.1:8b")
