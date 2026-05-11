import unittest
from unittest.mock import patch

from backend.app.inference import VLLMChatEngine
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
            allow_local_responder_fallback=False,
        )

        self.assertEqual(settings.validate_for_production(), [])

    def test_validate_for_production_rejects_local_responder_fallback(self) -> None:
        settings = Settings(
            environment="production",
            inference_engine="vllm",
            vllm_base_url="http://127.0.0.1:8000",
            vllm_api_key="super-secret",
            allow_local_responder_fallback=True,
        )

        errors = settings.validate_for_production()

        self.assertTrue(any("scripted local fallback" in error for error in errors))

    def test_vercel_defaults_disable_local_responder_fallback(self) -> None:
        with patch.dict("os.environ", {"VERCEL": "1"}, clear=False):
            settings = Settings(
                inference_engine="vllm",
                vllm_base_url="http://127.0.0.1:8000",
                vllm_api_key="super-secret",
            )

        self.assertEqual(settings.environment, "production")
        self.assertFalse(settings.allow_local_responder_fallback)

    def test_legacy_provider_env_overrides_mock_engine(self) -> None:
        settings = Settings(
            environment="production",
            inference_engine="mock",
            vllm_base_url="https://provider.example",
            vllm_api_key="super-secret",
            vllm_model="microsoft/Phi-3-mini-4k-instruct",
            allow_local_responder_fallback=False,
        )

        self.assertEqual(settings.active_engine, "vllm")
        self.assertEqual(settings.runtime_model_id, "microsoft/Phi-3-mini-4k-instruct")
        self.assertEqual(settings.validate_for_production(), [])

    def test_placeholder_legacy_model_uses_gateway_default(self) -> None:
        settings = Settings(
            environment="production",
            inference_engine="mock",
            vllm_base_url="https://provider.example/v1",
            vllm_api_key="super-secret",
            vllm_model="meta/llama-3.3-70b",
        )

        self.assertEqual(settings.runtime_model_id, "meta/llama-3.3-70b")

    def test_vllm_adapter_accepts_base_url_that_already_includes_v1(self) -> None:
        settings = Settings(
            inference_engine="vllm",
            vllm_base_url="https://provider.example/v1",
            vllm_api_key="super-secret",
            vllm_model="meta/llama-3.3-70b",
        )
        engine = VLLMChatEngine(settings)

        self.assertEqual(
            engine._openai_path("chat/completions"),
            "https://provider.example/v1/chat/completions",
        )

    def test_vercel_ai_gateway_prefers_oidc_over_legacy_provider_key(self) -> None:
        settings = Settings(
            inference_engine="vllm",
            vllm_base_url="https://ai-gateway.vercel.sh/v1",
            vllm_api_key="legacy-provider-key",
            ai_gateway_api_key="",
            vercel_oidc_token="oidc-token",
        )
        engine = VLLMChatEngine(settings)

        self.assertEqual(engine._api_key, "oidc-token")

    def test_vercel_ai_gateway_accepts_oidc_without_legacy_provider_key(self) -> None:
        settings = Settings(
            environment="production",
            inference_engine="vllm",
            vllm_base_url="https://ai-gateway.vercel.sh/v1",
            vllm_api_key="medbrief-local",
            ai_gateway_api_key="",
            vercel_oidc_token="oidc-token",
            allow_local_responder_fallback=False,
        )

        self.assertEqual(settings.validate_for_production(), [])

    def test_default_engine_uses_local_ollama_when_available(self) -> None:
        with patch("backend.app.settings._ollama_model_installed", return_value=True):
            settings = Settings(
                inference_engine="",
                openai_api_key="",
                vllm_base_url="",
                ollama_base_url="http://127.0.0.1:11434",
                ollama_model="phi3:mini",
            )

            self.assertEqual(settings.active_engine, "ollama")
            self.assertEqual(settings.runtime_model_id, "phi3:mini")

    def test_default_engine_uses_custom_when_no_local_model_exists(self) -> None:
        with patch("backend.app.settings._ollama_model_installed", return_value=False):
            settings = Settings(
                inference_engine="",
                openai_api_key="",
                vllm_base_url="",
                ollama_base_url="http://127.0.0.1:11434",
                ollama_model="definitely-not-installed:latest",
            )

            self.assertEqual(settings.active_engine, "custom")
            self.assertEqual(settings.runtime_model_id, "medbrief-phi3-med")

    def test_openai_requires_explicit_engine(self) -> None:
        settings = Settings(
            inference_engine="",
            openai_api_key="sk-test",
            openai_model="gpt-5.4-mini",
            vllm_base_url="",
        )

        self.assertNotEqual(settings.active_engine, "openai")

    def test_openai_can_be_selected_explicitly(self) -> None:
        settings = Settings(
            inference_engine="openai",
            openai_api_key="sk-test",
            openai_model="gpt-5.4-mini",
            vllm_base_url="",
        )

        self.assertEqual(settings.active_engine, "openai")
        self.assertEqual(settings.runtime_model_id, "gpt-5.4-mini")
