"""Canonical FastAPI gateway for MedBrief AI."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .constants import FRONTEND_FEATURE_FLAGS, PRIVACY_DISCLAIMER, SUPPORTED_MODES
from .inference import BaseInferenceEngine, MockInferenceEngine, create_inference_engine
from .profile_store import STORE
from .prompting import PromptBundle, build_prompt_bundle, is_definition_request
from .safety import (
    apply_profanity_filter,
    clean_response_text,
    degraded_mode_response,
    ensure_crisis_resources,
    evaluate_request,
    fallback_response,
    inject_privacy_disclaimer,
    is_low_quality_response,
    postprocess_health_response,
)
from .schemas import (
    BackendConfigResponse,
    ChatCompletionRequest,
    DeleteUserResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MemorySummarizeRequest,
    MemorySummarizeResponse,
    ModelsResponse,
    RuntimeConfigResponse,
    SessionInitRequest,
    SessionInitResponse,
    UserProfile,
    UserProfileResponse,
    UserProfileUpsertRequest,
)
from .settings import Settings, get_settings
from .telemetry import emit_event, telemetry_enabled


APP_STARTED_AT = time.time()
RATE_LIMIT_BUCKETS: defaultdict[str, list[float]] = defaultdict(list)
GENERIC_CACHE: dict[str, tuple[float, dict[str, object], str]] = {}
CACHE_TTL_SECONDS = 300
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "recall-app"


def _count_tokens_rough(text: str) -> int:
    return max(1, len(text.split()))


def _build_chat_response(
    *,
    response_text: str,
    request_id: str,
    model_id: str,
    prompt_tokens: int,
    finish_reason: str = "stop",
) -> dict[str, object]:
    completion_tokens = _count_tokens_rough(response_text)
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _make_cache_key(request: ChatCompletionRequest, prompt_bundle: PromptBundle) -> str | None:
    user_id = request.metadata.get("user_id")
    if request.memory_summary or user_id or request.conversation_id:
        return None
    if prompt_bundle.mode not in {"general", "health", "portfolio"}:
        return None
    return f"{prompt_bundle.mode}::{prompt_bundle.latest_user_text.strip().lower()}::{request.max_tokens}"


def _get_cached_completion(cache_key: str) -> tuple[dict[str, object], str] | None:
    cached = GENERIC_CACHE.get(cache_key)
    if not cached:
        return None
    stored_at, response_body, cleaned_text = cached
    if time.time() - stored_at > CACHE_TTL_SECONDS:
        GENERIC_CACHE.pop(cache_key, None)
        return None
    return response_body, cleaned_text


def _store_cached_completion(cache_key: str | None, response_body: dict[str, object], cleaned_text: str) -> None:
    if not cache_key:
        return
    GENERIC_CACHE[cache_key] = (time.time(), response_body, cleaned_text)


def _summarize_messages(messages: list[dict[str, str]]) -> str:
    users = [message["content"] for message in messages if message["role"] == "user"]
    assistants = [message["content"] for message in messages if message["role"] == "assistant"]
    first_issue = users[0] if users else "the user's concerns"
    latest_issue = users[-1] if users else "their main question"
    helpful = assistants[-1] if assistants else "general support and guidance"
    sentence_one = f"The user was primarily dealing with {first_issue[:180].rstrip('.!?')}."
    sentence_two = f"What seemed most helpful was {helpful[:180].rstrip('.!?')}."
    sentence_three = f"A reasonable follow-up is to revisit {latest_issue[:180].rstrip('.!?')} with any new details or changes."
    return " ".join([sentence_one, sentence_two, sentence_three])


async def _stream_sanitized_text(
    *,
    response_text: str,
    request_id: str,
    model_id: str,
) -> AsyncIterator[str]:
    words = response_text.split()
    for index, word in enumerate(words):
        rendered = f"{word} " if index < len(words) - 1 else word
        payload = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": rendered},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(payload)}\n\n"
        await asyncio.sleep(0)

    done_payload = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done_payload)}\n\n"
    yield "data: [DONE]\n\n"


def _rate_limit_or_raise(request: Request, bucket: str, limit: int, window_seconds: int) -> None:
    identifier = request.client.host if request.client else "unknown"
    key = f"{bucket}:{identifier}"
    now = time.time()
    RATE_LIMIT_BUCKETS[key] = [stamp for stamp in RATE_LIMIT_BUCKETS[key] if now - stamp < window_seconds]
    if len(RATE_LIMIT_BUCKETS[key]) >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    RATE_LIMIT_BUCKETS[key].append(now)


def _profile_from_request(request: ChatCompletionRequest) -> UserProfile | None:
    raw_profile = request.metadata.get("user_profile")
    if isinstance(raw_profile, dict) and raw_profile.get("user_id"):
        try:
            return UserProfile.model_validate(raw_profile)
        except Exception:
            pass
    user_id = request.metadata.get("user_id")
    if isinstance(user_id, str):
        return STORE.get_profile(user_id)
    return None


def _resolve_generation_settings(
    request: ChatCompletionRequest,
    prompt_bundle: PromptBundle,
) -> tuple[int, float, float]:
    max_tokens = request.max_tokens
    temperature = request.temperature
    top_p = request.top_p

    if is_definition_request(prompt_bundle.latest_user_text):
        return min(max_tokens, 64), min(temperature, 0.18), min(top_p, 0.78)
    if prompt_bundle.mode == "health":
        return min(max_tokens, 72), min(temperature, 0.18), min(top_p, 0.8)
    if prompt_bundle.mode == "psych":
        return min(max_tokens, 64), min(temperature, 0.22), min(top_p, 0.82)
    if prompt_bundle.mode == "general":
        return min(max_tokens, 72), min(temperature, 0.22), min(top_p, 0.8)
    if prompt_bundle.mode == "portfolio":
        return min(max_tokens, 88), min(temperature, 0.18), min(top_p, 0.8)
    return max_tokens, temperature, top_p


async def _generate_completion(
    http_request: Request,
    request: ChatCompletionRequest,
    settings: Settings,
    engine: BaseInferenceEngine,
    fallback_engine: BaseInferenceEngine | None = None,
) -> tuple[dict[str, object], dict[str, object], str]:
    request_id = request.request_id or str(uuid.uuid4())
    profile = _profile_from_request(request)
    prompt_bundle: PromptBundle = build_prompt_bundle(request)
    prompt_tokens = sum(_count_tokens_rough(message["content"]) for message in prompt_bundle.upstream_messages)
    max_tokens, temperature, top_p = _resolve_generation_settings(request, prompt_bundle)
    safety_decision = evaluate_request(prompt_bundle.mode, prompt_bundle.latest_user_text)
    cache_key = _make_cache_key(request, prompt_bundle)
    if cache_key:
        cached = _get_cached_completion(cache_key)
        if cached:
            response_body, cleaned = cached
            telemetry = {
                "request_id": request_id,
                "mode": prompt_bundle.mode,
                "latency_ms": 0,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": response_body["usage"]["completion_tokens"],
                "fallback_flag": response_body["model"] == settings.public_model_id,
                "safety_flag": "cache_hit",
                "model_version": settings.runtime_base_model_id,
                "adapter_version": settings.adapter_id or None,
                "engine": settings.active_engine,
            }
            return response_body, telemetry, cleaned

    fallback_used = settings.active_engine == "mock"
    started = time.perf_counter()

    if safety_decision.allow_model:
        try:
            inference = await engine.complete(
                messages=prompt_bundle.upstream_messages,
                model=request.model or settings.public_model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                request_id=request_id,
                mode=prompt_bundle.mode,
                conversation_id=request.conversation_id,
                profile=profile,
            )
            response_text = inference.text
            upstream_model = inference.upstream_model or settings.public_model_id
            finish_reason = inference.finish_reason
        except Exception:
            fallback_used = True
            if fallback_engine is not None:
                try:
                    inference = await fallback_engine.complete(
                        messages=prompt_bundle.upstream_messages,
                        model=request.model or settings.public_model_id,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        request_id=request_id,
                        mode=prompt_bundle.mode,
                        conversation_id=request.conversation_id,
                        profile=profile,
                    )
                    response_text = inference.text
                    upstream_model = settings.public_model_id
                    finish_reason = inference.finish_reason
                except Exception:
                    response_text = degraded_mode_response()
                    upstream_model = settings.public_model_id
                    finish_reason = "stop"
            else:
                response_text = degraded_mode_response()
                upstream_model = settings.public_model_id
                finish_reason = "stop"
    else:
        response_text = safety_decision.response_text or degraded_mode_response()
        upstream_model = settings.public_model_id
        finish_reason = "stop"

    cleaned = clean_response_text(response_text)
    cleaned = apply_profanity_filter(cleaned)
    if prompt_bundle.mode == "health":
        cleaned, medical_guard_hit = postprocess_health_response(cleaned)
        fallback_used = fallback_used or medical_guard_hit
    if prompt_bundle.mode == "crisis":
        cleaned = ensure_crisis_resources(cleaned)
    if is_low_quality_response(cleaned):
        cleaned = fallback_response()
        fallback_used = True

    response_body = _build_chat_response(
        response_text=cleaned,
        request_id=request_id,
        model_id=upstream_model,
        prompt_tokens=prompt_tokens,
        finish_reason=finish_reason,
    )
    telemetry = {
        "request_id": request_id,
        "mode": prompt_bundle.mode,
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": response_body["usage"]["completion_tokens"],
        "fallback_flag": fallback_used,
        "safety_flag": safety_decision.safety_flag,
        "model_version": settings.runtime_base_model_id,
        "adapter_version": settings.adapter_id or None,
        "engine": settings.active_engine,
        "user_id": request.metadata.get("user_id"),
        "conversation_id": request.conversation_id,
        "client": http_request.client.host if http_request.client else None,
    }
    _store_cached_completion(cache_key, response_body, cleaned)
    return response_body, telemetry, cleaned


def create_app() -> FastAPI:
    settings = get_settings()
    if settings.environment.lower() == "production":
        errors = settings.validate_for_production()
        if errors:
            raise RuntimeError("Production config errors: " + "; ".join(errors))
    engine = create_inference_engine(settings)
    fallback_engine: BaseInferenceEngine | None = None if settings.active_engine == "mock" else MockInferenceEngine()

    app = FastAPI(title=settings.api_title, version=settings.release_version)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def warm_runtime_model() -> None:
        if settings.active_engine == "ollama" and settings.ollama_warmup:
            await engine.warmup()

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        upstream_ok = await engine.health()
        status = "healthy" if upstream_ok or settings.active_engine == "mock" else "degraded"
        return HealthResponse(
            status=status,
            model_loaded=upstream_ok or settings.active_engine == "mock",
            adapter_loaded=settings.adapter_loaded,
            engine=settings.active_engine,
            gpu_type=settings.gpu_type,
            build_sha=settings.build_sha,
            model_version=settings.runtime_base_model_id,
            telemetry_enabled=telemetry_enabled(),
        )

    @app.get("/api/config", response_model=BackendConfigResponse)
    async def api_config() -> BackendConfigResponse:
        return BackendConfigResponse(
            model_id=settings.runtime_model_id,
            active_model=settings.runtime_model_id,
            adapter_id=settings.adapter_id or None,
            stream_default=settings.stream_default,
            max_tokens_default=settings.default_max_tokens,
            temperature_default=settings.default_temperature,
            supported_modes=list(SUPPORTED_MODES),
            frontend_features=dict(FRONTEND_FEATURE_FLAGS),
            default_generation={
                "max_new_tokens": settings.default_max_tokens,
                "temperature": settings.default_temperature,
                "top_p": settings.default_top_p,
            },
        )

    @app.get("/runtime-config.json", response_model=RuntimeConfigResponse)
    async def runtime_config() -> RuntimeConfigResponse:
        return RuntimeConfigResponse(
            apiBaseUrl=settings.runtime_config_api_base,
            defaultModel=settings.runtime_model_id,
            stream=settings.stream_default,
            enabledFeatures=dict(FRONTEND_FEATURE_FLAGS),
            maxTokensDefault=settings.default_max_tokens,
            temperatureDefault=settings.default_temperature,
        )

    @app.get("/v1/models", response_model=ModelsResponse)
    async def models() -> ModelsResponse:
        return ModelsResponse(
            data=[
                {
                    "id": settings.runtime_model_id,
                    "object": "model",
                    "owned_by": "medbrief",
                    "metadata": {
                        "base_model_id": settings.runtime_base_model_id,
                        "adapter_id": settings.adapter_id or None,
                        "release_version": settings.release_version,
                    },
                }
            ],
            active_model=settings.runtime_model_id,
            active_adapter=settings.adapter_id or None,
            release_version=settings.release_version,
        )

    @app.get("/", include_in_schema=False)
    async def root() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/index.html", include_in_schema=False)
    async def root_index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/api", include_in_schema=False)
    async def api_root() -> dict[str, object]:
        return {
            "message": "MedBrief AI API Server",
            "status": "running",
            "privacy_disclaimer": PRIVACY_DISCLAIMER,
            "endpoints": {
                "health": "/health",
                "config": "/api/config",
                "models": "/v1/models",
                "chat": "/v1/chat/completions",
                "profile": "/v1/profile",
                "feedback": "/v1/feedback",
                "memory_summarize": "/v1/memory/summarize",
                "session_init": "/v1/session/init",
            },
        }

    @app.post("/v1/profile", response_model=UserProfileResponse)
    async def upsert_profile(profile_request: UserProfileUpsertRequest, request: Request) -> UserProfileResponse:
        _rate_limit_or_raise(request, "profile", limit=60, window_seconds=60)
        stored = STORE.upsert_profile(profile_request.profile)
        emit_event("profile_upsert", user_id=stored.user_id)
        return UserProfileResponse(user_id=stored.user_id, profile=stored)

    @app.get("/v1/profile/{user_id}", response_model=UserProfileResponse)
    async def get_profile(user_id: str) -> UserProfileResponse:
        profile = STORE.get_profile(user_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Profile not found")
        return UserProfileResponse(user_id=user_id, profile=profile)

    @app.post("/v1/memory/summarize", response_model=MemorySummarizeResponse)
    async def summarize_memory(payload: MemorySummarizeRequest, request: Request) -> MemorySummarizeResponse:
        _rate_limit_or_raise(request, "summarize", limit=30, window_seconds=60)
        summary = _summarize_messages([message.model_dump() for message in payload.messages])
        stored = False
        if payload.user_id and payload.session_id:
            STORE.set_session_summary(payload.user_id, payload.session_id, summary)
            stored = True
        emit_event("memory_summarized", user_id=payload.user_id, session_id=payload.session_id, stored=stored)
        return MemorySummarizeResponse(summary=summary, stored=stored)

    @app.post("/v1/session/init", response_model=SessionInitResponse)
    async def session_init(payload: SessionInitRequest) -> SessionInitResponse:
        profile = STORE.get_profile(payload.user_id)
        summary = STORE.latest_summary(payload.user_id)
        return SessionInitResponse(session_id=payload.session_id, memory_summary=summary, profile=profile)

    @app.post("/v1/feedback", response_model=FeedbackResponse)
    async def feedback(payload: FeedbackRequest, request: Request) -> FeedbackResponse:
        _rate_limit_or_raise(request, "feedback", limit=120, window_seconds=60)
        count = STORE.add_feedback(payload)
        emit_event("feedback_received", user_id=payload.user_id, rating=payload.rating, mode=payload.mode)
        return FeedbackResponse(stored=True, feedback_count=count)

    @app.delete("/v1/user/{user_id}", response_model=DeleteUserResponse)
    async def delete_user(user_id: str) -> DeleteUserResponse:
        STORE.delete_user(user_id)
        emit_event("user_deleted", user_id=user_id)
        return DeleteUserResponse(user_id=user_id, deleted=True)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, http_request: Request) -> Response:
        _rate_limit_or_raise(http_request, "chat", limit=30, window_seconds=60)
        try:
            response_body, telemetry, sanitized_text = await _generate_completion(
                http_request,
                request,
                settings,
                engine,
                fallback_engine,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        emit_event("chat_completion", **telemetry)
        headers = {"X-Request-ID": telemetry["request_id"]}

        if request.stream:
            return StreamingResponse(
                _stream_sanitized_text(
                    response_text=sanitized_text,
                    request_id=telemetry["request_id"],
                    model_id=response_body["model"],
                ),
                media_type="text/event-stream",
                headers=headers,
            )

        return JSONResponse(content=response_body, headers=headers)

    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    return app


app = create_app()
