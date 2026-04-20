"""Environment-backed settings for the MedBrief gateway."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from .constants import FRONTEND_FEATURE_FLAGS


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _list_env(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    environment: str = os.getenv("MEDBRIEF_ENV", "development")
    api_title: str = os.getenv("MEDBRIEF_API_TITLE", "MedBrief AI Gateway")
    release_version: str = os.getenv("MEDBRIEF_RELEASE_VERSION", "0.2.0")
    public_model_id: str = os.getenv("MEDBRIEF_MODEL_ID", "medbrief-phi3-med")
    base_model_id: str = os.getenv("MEDBRIEF_BASE_MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
    adapter_id: str = os.getenv("MEDBRIEF_ADAPTER_ID", "")
    build_sha: str = os.getenv("MEDBRIEF_BUILD_SHA", "dev")
    inference_engine: str = os.getenv("MEDBRIEF_INFERENCE_ENGINE", "")
    vllm_base_url: str = os.getenv("MEDBRIEF_VLLM_BASE_URL", "")
    vllm_api_key: str = os.getenv("MEDBRIEF_VLLM_API_KEY", "medbrief-local")
    vllm_signing_secret: str = os.getenv("MEDBRIEF_VLLM_SIGNING_SECRET", "")
    ollama_base_url: str = os.getenv("MEDBRIEF_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("MEDBRIEF_OLLAMA_MODEL", "phi3:mini")
    ollama_keep_alive: str = os.getenv("MEDBRIEF_OLLAMA_KEEP_ALIVE", "30m")
    ollama_num_ctx: int = int(os.getenv("MEDBRIEF_OLLAMA_NUM_CTX", "4096"))
    ollama_timeout_seconds: float = float(os.getenv("MEDBRIEF_OLLAMA_TIMEOUT_SECONDS", "180"))
    ollama_warmup: bool = _bool_env("MEDBRIEF_OLLAMA_WARMUP", True)
    request_timeout_seconds: float = float(os.getenv("MEDBRIEF_REQUEST_TIMEOUT_SECONDS", "45"))
    default_max_tokens: int = int(os.getenv("MEDBRIEF_DEFAULT_MAX_TOKENS", "80"))
    default_temperature: float = float(os.getenv("MEDBRIEF_DEFAULT_TEMPERATURE", "0.25"))
    default_top_p: float = float(os.getenv("MEDBRIEF_DEFAULT_TOP_P", "0.85"))
    stream_default: bool = _bool_env("MEDBRIEF_STREAM_DEFAULT", True)
    gpu_type: str = os.getenv("MEDBRIEF_GPU_TYPE", "L4")
    runtime_config_api_base: str = os.getenv("MEDBRIEF_RUNTIME_API_BASE", "")
    allowed_origins: tuple[str, ...] = _list_env(
        "MEDBRIEF_ALLOWED_ORIGINS",
        (
            "https://medbriefai.vercel.app,"
            "http://localhost:3000,http://127.0.0.1:3000,"
            "http://localhost:3001,http://127.0.0.1:3001,"
            "http://localhost:3003,http://127.0.0.1:3003,"
            "http://localhost:3004,http://127.0.0.1:3004,"
            "http://localhost:5173,http://127.0.0.1:5173"
        ),
    )

    @property
    def active_engine(self) -> str:
        if self.inference_engine.strip():
            return self.inference_engine.strip()
        if self.vllm_base_url:
            return "vllm"
        return "ollama"

    @property
    def runtime_model_id(self) -> str:
        if self.active_engine == "ollama":
            return self.ollama_model
        return self.public_model_id

    @property
    def runtime_base_model_id(self) -> str:
        if self.active_engine == "ollama":
            return self.ollama_model
        return self.base_model_id

    @property
    def adapter_loaded(self) -> bool:
        return bool(self.adapter_id)

    @property
    def frontend_features(self) -> dict[str, bool]:
        return dict(FRONTEND_FEATURE_FLAGS)

    def validate_for_production(self) -> list[str]:
        errors: list[str] = []
        if self.active_engine == "mock":
            errors.append("active_engine is mock - set MEDBRIEF_VLLM_BASE_URL or MEDBRIEF_INFERENCE_ENGINE")
        if self.active_engine == "vllm" and (not self.vllm_api_key or self.vllm_api_key == "medbrief-local"):
            errors.append("MEDBRIEF_VLLM_API_KEY is using the insecure default value")
        if self.active_engine == "ollama":
            if not self.ollama_base_url:
                errors.append("MEDBRIEF_OLLAMA_BASE_URL is required when using the ollama engine")
            if not self.ollama_model:
                errors.append("MEDBRIEF_OLLAMA_MODEL is required when using the ollama engine")
        return errors


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
