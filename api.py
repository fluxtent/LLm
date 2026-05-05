from __future__ import annotations

import concurrent.futures
import functools
import json
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from generate import MODE_PARAMETERS, SYSTEM_PROMPT, build_prompt, generate_response, load_runtime
from utils import (
    CRISIS_KEYWORDS,
    chunk_words,
    clean_response,
    detect_mode,
    is_low_quality_response,
    looks_like_medication_dosing,
)


APP_STARTED_AT = time.time()
LAST_INFERENCE_AT: float | None = None
RUNTIME = load_runtime()
GENERATION_LOCK = threading.Lock()
EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)

CRISIS_RESPONSE = (
    "I'm really glad you said something. Your safety matters most right now. "
    "Please call or text 988 right now if you're in the US, or contact local emergency services if you might act on these thoughts. "
    "If you can, reach out to someone nearby and tell them you need help staying safe."
)
MEDICATION_RESPONSE = (
    "I can't tell you how much medication to take. That decision needs a licensed clinician, pharmacist, or poison control professional "
    "who can consider your medication, health history, and safety risks."
)
DEGRADED_RESPONSE = "MedBrief AI is taking a brief break. Please try again in a moment."

FRONTEND_FEATURES = {
    "apiKeysEnabled": False,
    "moodCheckEnabled": True,
    "memoryInsightsEnabled": True,
}

ACTIVE_MODEL_ID = "medbrief-transformer"
ALLOWED_ORIGINS = [
    "https://medbriefai.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app = Flask(__name__)
CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=True)
limiter = Limiter(key_func=get_remote_address, app=app, default_limits=["30 per minute"])


def runtime_revision() -> str:
    model_path = Path("model.pth")
    return str(model_path.stat().st_mtime) if model_path.exists() else "fallback"


def build_messages(payload: dict) -> tuple[list[dict[str, str]], str, str | None]:
    messages = payload.get("messages", [])
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages are required")

    normalized: list[dict[str, str]] = []
    for item in messages:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content})
    if not normalized or not any(message["role"] == "user" for message in normalized):
        raise ValueError("at least one user message is required")

    mode = payload.get("mode") or detect_mode(next(m["content"] for m in reversed(normalized) if m["role"] == "user"))
    memory_summary = (payload.get("memory_summary") or "").strip() or None
    return normalized, mode, memory_summary


def build_response_payload(
    request_id: str,
    content: str,
    prompt_tokens: int,
    finish_reason: str = "stop",
) -> dict:
    completion_tokens = max(1, len(content.split()))
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": ACTIVE_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@functools.lru_cache(maxsize=128)
def cached_completion(
    revision: str,
    serialized_messages: str,
    mode: str,
    memory_summary: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> str:
    del revision
    messages = json.loads(serialized_messages)
    with GENERATION_LOCK:
        return generate_response(
            RUNTIME,
            messages,
            mode=mode,
            memory_summary=memory_summary or None,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )


def complete_with_timeout(
    messages: list[dict[str, str]],
    mode: str,
    memory_summary: str | None,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> str:
    future = EXECUTOR.submit(
        cached_completion,
        runtime_revision(),
        json.dumps(messages, sort_keys=True),
        mode,
        memory_summary or "",
        max_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
    )
    try:
        return future.result(timeout=30)
    except concurrent.futures.TimeoutError:
        return DEGRADED_RESPONSE


def stream_response(content: str, request_id: str) -> Response:
    def generate() -> str:
        chunks = chunk_words(content, words_per_chunk=12)
        for index, chunk in enumerate(chunks):
            rendered = chunk if index == len(chunks) - 1 else f"{chunk} "
            payload = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": ACTIVE_MODEL_ID,
                "choices": [{"index": 0, "delta": {"content": rendered}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(payload)}\n\n"
        done = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": ACTIVE_MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done)}\n\n"
        yield "data: [DONE]\n\n"

    response = Response(generate(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/health")
def health() -> Response:
    status = "healthy" if (RUNTIME["model_loaded"] or RUNTIME["fallback_available"]) else "degraded"
    payload = {
        "status": status,
        "model_loaded": RUNTIME["model_loaded"],
        "tokenizer_loaded": RUNTIME["tokenizer"] is not None,
        "backend_mode": RUNTIME.get("serve_strategy", "fallback"),
        "uptime_seconds": int(time.time() - APP_STARTED_AT),
        "last_inference_time": LAST_INFERENCE_AT,
    }
    return jsonify(payload)


@app.get("/api/config")
def api_config() -> Response:
    return jsonify(
        {
            "model_id": ACTIVE_MODEL_ID,
            "active_model": ACTIVE_MODEL_ID,
            "stream_default": True,
            "max_tokens_default": 180,
            "temperature_default": 0.7,
            "supported_modes": list(MODE_PARAMETERS.keys()),
            "frontend_features": FRONTEND_FEATURES,
            "default_generation": {"max_new_tokens": 180, "temperature": 0.7, "top_p": 0.95},
        }
    )


@app.get("/runtime-config.json")
def runtime_config() -> Response:
    return jsonify(
        {
            "apiBaseUrl": "http://127.0.0.1:5000",
            "defaultModel": ACTIVE_MODEL_ID,
            "stream": True,
            "enabledFeatures": FRONTEND_FEATURES,
            "maxTokensDefault": 180,
            "temperatureDefault": 0.7,
        }
    )


@app.get("/v1/models")
def models() -> Response:
    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": ACTIVE_MODEL_ID,
                    "object": "model",
                    "owned_by": "medbrief",
                    "metadata": {
                        "runtime_mode": RUNTIME.get("serve_strategy", "fallback"),
                        "custom_model": True,
                    },
                }
            ],
            "active_model": ACTIVE_MODEL_ID,
        }
    )


@app.post("/v1/chat/completions")
@limiter.limit("30 per minute")
def chat_completions() -> Response:
    global LAST_INFERENCE_AT

    try:
        payload = request.get_json(force=True, silent=False) or {}
        messages, mode, memory_summary = build_messages(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    request_id = payload.get("request_id") or str(uuid.uuid4())
    latest_user = next(message["content"] for message in reversed(messages) if message["role"] == "user")

    if any(keyword in latest_user.lower() for keyword in CRISIS_KEYWORDS):
        content = CRISIS_RESPONSE
    elif looks_like_medication_dosing(latest_user):
        content = MEDICATION_RESPONSE
    else:
        mode_params = {**MODE_PARAMETERS["general"], **MODE_PARAMETERS.get(mode, {})}
        max_tokens = int(payload.get("max_tokens") or 180)
        temperature = float(payload.get("temperature") if payload.get("temperature") is not None else mode_params["temperature"])
        top_k = int(payload.get("top_k") if payload.get("top_k") is not None else mode_params["top_k"])
        top_p = float(payload.get("top_p") if payload.get("top_p") is not None else mode_params["top_p"])
        repetition_penalty = float(
            payload.get("repetition_penalty")
            if payload.get("repetition_penalty") is not None
            else mode_params["repetition_penalty"]
        )
        content = complete_with_timeout(
            messages,
            mode,
            memory_summary,
            max_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        )

    content = clean_response(content)
    if is_low_quality_response(content):
        content = "I want to make sure I give you a thoughtful response. Could you tell me a bit more about what you mean?"
    if mode == "crisis" and "988" not in content:
        content += " Please call or text 988 right now if you are in the US."

    prompt, _ = build_prompt(messages, mode=mode, system_prompt=SYSTEM_PROMPT, memory_summary=memory_summary)
    response_payload = build_response_payload(request_id, content, prompt_tokens=max(1, len(prompt.split())))
    headers = {"X-Request-ID": request_id}
    LAST_INFERENCE_AT = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if payload.get("stream", True):
        stream = stream_response(content, request_id)
        stream.headers.extend(headers)
        return stream

    response = jsonify(response_payload)
    response.headers.extend(headers)
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
