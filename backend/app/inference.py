"""Inference engine adapters for Ollama, vLLM, and local fallback templates."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from dataclasses import dataclass

import httpx

from .medical_ontology import detect_medical_context
from .schemas import UserProfile
from .settings import Settings
from .template_engine import TemplateEngine, TemplateRequest


@dataclass(frozen=True)
class InferenceResult:
    text: str
    finish_reason: str = "stop"
    latency_ms: int = 0
    upstream_model: str | None = None


class BaseInferenceEngine:
    name = "base"

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        request_id: str,
        mode: str,
        conversation_id: str | None = None,
        profile: UserProfile | None = None,
    ) -> InferenceResult:
        raise NotImplementedError

    async def health(self) -> bool:
        return True

    async def warmup(self) -> None:
        return None


class MockInferenceEngine(BaseInferenceEngine):
    name = "mock"

    def __init__(self) -> None:
        self._templates = TemplateEngine()

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        request_id: str,
        mode: str,
        conversation_id: str | None = None,
        profile: UserProfile | None = None,
    ) -> InferenceResult:
        del max_tokens, temperature, top_p
        started = time.perf_counter()
        latest_user = next(message["content"] for message in reversed(messages) if message["role"] == "user")
        recent_assistant = tuple(
            message["content"] for message in messages if message["role"] == "assistant"
        )
        text = self._templates.render(
            TemplateRequest(
                latest_user=latest_user,
                mode=mode,
                request_id=request_id,
                conversation_id=conversation_id,
                profile=profile,
                medical_context=detect_medical_context(latest_user),
                recent_assistant_messages=recent_assistant,
            )
        )
        return InferenceResult(
            text=text,
            latency_ms=int((time.perf_counter() - started) * 1000),
            upstream_model=model,
        )


class OllamaChatEngine(BaseInferenceEngine):
    name = "ollama"

    def __init__(self, settings: Settings):
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._default_model = settings.ollama_model
        self._public_model_id = settings.public_model_id
        self._timeout = settings.ollama_timeout_seconds
        self._keep_alive = settings.ollama_keep_alive
        self._num_ctx = settings.ollama_num_ctx

    def _resolve_model(self, requested_model: str | None) -> str:
        if not requested_model or requested_model == self._public_model_id:
            return self._default_model
        return requested_model

    def _payload(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict[str, object]:
        return {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "num_ctx": self._num_ctx,
            },
        }

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        request_id: str,
        mode: str,
        conversation_id: str | None = None,
        profile: UserProfile | None = None,
    ) -> InferenceResult:
        del request_id, mode, conversation_id, profile
        started = time.perf_counter()
        requested_model = self._resolve_model(model)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json=self._payload(
                    model=requested_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                ),
            )
            response.raise_for_status()
            data = response.json()
        content = data["message"]["content"]
        return InferenceResult(
            text=content,
            finish_reason=data.get("done_reason", "stop"),
            latency_ms=int((time.perf_counter() - started) * 1000),
            upstream_model=data.get("model", requested_model),
        )

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
            models = {model.get("name") for model in data.get("models", []) if isinstance(model, dict)}
            return self._default_model in models
        except Exception:
            return False

    async def warmup(self) -> None:
        try:
            await self.complete(
                messages=[
                    {
                        "role": "system",
                        "content": "Reply with ok.",
                    },
                    {
                        "role": "user",
                        "content": "ok",
                    },
                ],
                model=self._default_model,
                max_tokens=8,
                temperature=0.0,
                top_p=1.0,
                request_id="warmup",
                mode="general",
            )
        except Exception:
            return None


class VLLMChatEngine(BaseInferenceEngine):
    name = "vllm"

    def __init__(self, settings: Settings):
        self._base_url = settings.vllm_base_url.rstrip("/")
        self._api_key = settings.vllm_api_key
        self._timeout = settings.request_timeout_seconds
        self._signing_secret = settings.vllm_signing_secret

    def _headers(self, request_id: str) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        if self._signing_secret:
            signature = hmac.new(
                self._signing_secret.encode("utf-8"),
                request_id.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            headers["X-MedBrief-Signature"] = signature
        return headers

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        request_id: str,
        mode: str,
        conversation_id: str | None = None,
        profile: UserProfile | None = None,
    ) -> InferenceResult:
        started = time.perf_counter()
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
            "extra_body": {
                "mode": mode,
                "request_id": request_id,
                "conversation_id": conversation_id,
                "preferences": profile.preferences.model_dump() if profile else None,
            },
        }

        delay = 0.5
        last_error: Exception | None = None
        for _attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        f"{self._base_url}/v1/chat/completions",
                        headers=self._headers(request_id),
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                content = data["choices"][0]["message"]["content"]
                return InferenceResult(
                    text=content,
                    finish_reason=data["choices"][0].get("finish_reason", "stop"),
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    upstream_model=data.get("model", model),
                )
            except Exception as exc:  # pragma: no cover - network path
                last_error = exc
                await asyncio.sleep(delay)
                delay *= 2
        raise RuntimeError("vLLM completion failed after retries") from last_error

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.5) as client:
                response = await client.get(
                    f"{self._base_url}/v1/models",
                    headers=self._headers("health-check"),
                )
            return response.is_success
        except httpx.HTTPError:
            return False


def create_inference_engine(settings: Settings) -> BaseInferenceEngine:
    if settings.active_engine == "ollama":
        return OllamaChatEngine(settings)
    if settings.active_engine == "vllm":
        return VLLMChatEngine(settings)
    return MockInferenceEngine()
