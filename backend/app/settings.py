"""Environment-backed settings for the MedBrief gateway."""

from __future__ import annotations

import os
import json
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .constants import FRONTEND_FEATURE_FLAGS


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _list_env(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _ollama_model_installed(model_name: str) -> bool:
    """Best-effort local check that avoids choosing Ollama on machines without models."""
    if not model_name:
        return False
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=0.35) as response:
            data = json.loads(response.read().decode("utf-8"))
        installed = {
            item.get("name") or item.get("model")
            for item in data.get("models", [])
            if isinstance(item, dict)
        }
        if model_name in installed:
            return True
    except Exception:
        pass
    name, _, tag = model_name.partition(":")
    tag = tag or "latest"
    candidates = [
        Path.home() / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library" / name / tag,
        Path.home() / ".ollama" / "models" / "manifests" / name / tag,
    ]
    return any(path.exists() for path in candidates)


@dataclass(frozen=True)
class Settings:
    environment: str = os.getenv("MEDBRIEF_ENV", "development")
    api_title: str = os.getenv("MEDBRIEF_API_TITLE", "MedBrief AI Gateway")
    release_version: str = os.getenv("MEDBRIEF_RELEASE_VERSION", "0.3.0")
    public_model_id: str = os.getenv("MEDBRIEF_MODEL_ID", "medbrief-phi3-med")
    base_model_id: str = os.getenv("MEDBRIEF_BASE_MODEL_ID", "medbrief-transformer-v1")
    adapter_id: str = os.getenv("MEDBRIEF_ADAPTER_ID", "")
    build_sha: str = os.getenv("MEDBRIEF_BUILD_SHA", "dev")
    inference_engine: str = os.getenv("MEDBRIEF_INFERENCE_ENGINE", "")
    openai_api_key: str = os.getenv("MEDBRIEF_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = os.getenv("MEDBRIEF_OPENAI_BASE_URL", "https://api.openai.com")
    openai_model: str = os.getenv("MEDBRIEF_OPENAI_MODEL", "gpt-5.4-mini")
    vllm_base_url: str = os.getenv("MEDBRIEF_VLLM_BASE_URL", "")
    vllm_api_key: str = os.getenv("MEDBRIEF_VLLM_API_KEY", "medbrief-local")
    vllm_signing_secret: str = os.getenv("MEDBRIEF_VLLM_SIGNING_SECRET", "")
    ollama_base_url: str = os.getenv("MEDBRIEF_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("MEDBRIEF_OLLAMA_MODEL", "phi3:mini")
    ollama_keep_alive: str = os.getenv("MEDBRIEF_OLLAMA_KEEP_ALIVE", "30m")
    ollama_num_ctx: int = int(os.getenv("MEDBRIEF_OLLAMA_NUM_CTX", "1024"))
    ollama_num_thread: int = int(os.getenv("MEDBRIEF_OLLAMA_NUM_THREAD", "8"))
    ollama_timeout_seconds: float = float(os.getenv("MEDBRIEF_OLLAMA_TIMEOUT_SECONDS", "180"))
    ollama_warmup: bool = _bool_env("MEDBRIEF_OLLAMA_WARMUP", True)
    request_timeout_seconds: float = float(os.getenv("MEDBRIEF_REQUEST_TIMEOUT_SECONDS", "45"))
    default_max_tokens: int = int(os.getenv("MEDBRIEF_DEFAULT_MAX_TOKENS", "180"))
    default_temperature: float = float(os.getenv("MEDBRIEF_DEFAULT_TEMPERATURE", "0.65"))
    default_top_p: float = float(os.getenv("MEDBRIEF_DEFAULT_TOP_P", "0.92"))
    stream_default: bool = _bool_env("MEDBRIEF_STREAM_DEFAULT", True)
    gpu_type: str = os.getenv("MEDBRIEF_GPU_TYPE", "L4")
    runtime_config_api_base: str = os.getenv("MEDBRIEF_RUNTIME_API_BASE", "")
    custom_model_path: str = os.getenv("MEDBRIEF_CUSTOM_MODEL_PATH", "model.pth")
    custom_vocab_path: str = os.getenv("MEDBRIEF_CUSTOM_VOCAB_PATH", "vocab.json")
    custom_merges_path: str = os.getenv("MEDBRIEF_CUSTOM_MERGES_PATH", "merges.pkl")
    custom_allow_cpu: bool = _bool_env("MEDBRIEF_CUSTOM_ALLOW_CPU", True)
    store_path: str = os.getenv(
        "MEDBRIEF_STORE_PATH",
        str(Path(__file__).resolve().parents[1] / ".data" / "medbrief_store.json"),
    )
    api_keys_enabled: bool = _bool_env("MEDBRIEF_API_KEYS_ENABLED", True)
    require_api_key: bool = _bool_env("MEDBRIEF_REQUIRE_API_KEY", False)
    admin_token: str = os.getenv("MEDBRIEF_ADMIN_TOKEN", "")
    allow_public_key_generation: bool = _bool_env(
        "MEDBRIEF_ALLOW_PUBLIC_KEY_GENERATION",
        os.getenv("MEDBRIEF_ENV", "development").lower() != "production",
    )
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
            return self.inference_engine.strip().lower()
        if self.vllm_base_url:
            return "vllm"
        if _ollama_model_installed(self.ollama_model):
            return "ollama"
        return "custom"

    @property
    def runtime_model_id(self) -> str:
        if self.active_engine == "openai":
            return self.openai_model
        if self.active_engine == "custom":
            return self.public_model_id
        if self.active_engine == "ollama":
            return self.ollama_model
        return self.public_model_id

    @property
    def runtime_base_model_id(self) -> str:
        if self.active_engine == "openai":
            return self.openai_model
        if self.active_engine == "custom":
            return self.base_model_id
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
        if self.active_engine not in {"custom", "mock", "openai", "ollama", "vllm"}:
            errors.append(f"unsupported MEDBRIEF_INFERENCE_ENGINE: {self.active_engine}")
        if self.active_engine == "mock":
            errors.append("active_engine is mock - set MEDBRIEF_VLLM_BASE_URL or MEDBRIEF_INFERENCE_ENGINE")
        if self.active_engine == "openai":
            if not self.openai_api_key:
                errors.append("MEDBRIEF_OPENAI_API_KEY or OPENAI_API_KEY is required when using the OpenAI engine")
            if not self.openai_model:
                errors.append("MEDBRIEF_OPENAI_MODEL is required when using the OpenAI engine")
        if self.active_engine == "custom":
            for label, path in (
                ("MEDBRIEF_CUSTOM_MODEL_PATH", self.custom_model_path),
                ("MEDBRIEF_CUSTOM_VOCAB_PATH", self.custom_vocab_path),
                ("MEDBRIEF_CUSTOM_MERGES_PATH", self.custom_merges_path),
            ):
                if not Path(path).exists():
                    errors.append(f"{label} does not exist: {path}")
        if self.active_engine == "vllm" and (not self.vllm_api_key or self.vllm_api_key == "medbrief-local"):
            errors.append("MEDBRIEF_VLLM_API_KEY is using the insecure default value")
        if self.active_engine == "ollama":
            if not self.ollama_base_url:
                errors.append("MEDBRIEF_OLLAMA_BASE_URL is required when using the ollama engine")
            if not self.ollama_model:
                errors.append("MEDBRIEF_OLLAMA_MODEL is required when using the ollama engine")
        if self.require_api_key and not self.api_keys_enabled:
            errors.append("MEDBRIEF_REQUIRE_API_KEY cannot be true while API keys are disabled")
        if (
            self.environment.lower() == "production"
            and self.api_keys_enabled
            and self.allow_public_key_generation
            and not self.admin_token
            and os.getenv("MEDBRIEF_ALLOW_PUBLIC_KEY_GENERATION") is not None
        ):
            errors.append("public API key generation is enabled in production without MEDBRIEF_ADMIN_TOKEN")
        return errors


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
