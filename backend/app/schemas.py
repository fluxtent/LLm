"""Pydantic request and response models for the MedBrief gateway."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ModeLiteral = Literal["psych", "health", "crisis", "portfolio", "general"]
RoleLiteral = Literal["system", "user", "assistant"]
TerminologyLiteral = Literal["lay", "professional"]
LengthLiteral = Literal["concise", "balanced", "detailed"]
ToneLiteral = Literal["supportive", "clinical", "direct"]


class ChatMessage(BaseModel):
    role: RoleLiteral
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message content cannot be empty")
        return cleaned


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=350, ge=32, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    stream: bool = True
    mode: ModeLiteral | None = None
    conversation_id: str | None = None
    memory_summary: str | None = None
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        if not value:
            raise ValueError("messages are required")
        if not any(message.role == "user" for message in value):
            raise ValueError("at least one user message is required")
        for message in value:
            limit = 12000 if message.role == "system" else 4000
            if len(message.content) > limit:
                raise ValueError(f"{message.role} messages must be {limit} characters or fewer")
        return value


class UserPreferences(BaseModel):
    terminology: TerminologyLiteral = "lay"
    response_length: LengthLiteral = "balanced"
    tone: ToneLiteral = "supportive"
    memory_enabled: bool = True


class UserProfile(BaseModel):
    user_id: str
    display_name: str | None = None
    communication_style: str | None = None
    medical_context: list[str] = Field(default_factory=list)
    recurring_topics: list[str] = Field(default_factory=list)
    session_history: list[str] = Field(default_factory=list)
    mood_history: list[dict[str, Any]] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    copingPrefs: list[str] = Field(default_factory=list)
    recurringStressors: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    createdAt: int | None = None
    preferences: UserPreferences = Field(default_factory=UserPreferences)


class UserProfileUpsertRequest(BaseModel):
    user_id: str
    profile: UserProfile


class UserProfileResponse(BaseModel):
    user_id: str
    profile: UserProfile


class MemorySummarizeRequest(BaseModel):
    user_id: str | None = None
    session_id: str | None = None
    messages: list[ChatMessage]


class MemorySummarizeResponse(BaseModel):
    summary: str
    stored: bool = False


class SessionInitRequest(BaseModel):
    user_id: str
    session_id: str


class SessionInitResponse(BaseModel):
    session_id: str
    memory_summary: str | None = None
    profile: UserProfile | None = None


class FeedbackRequest(BaseModel):
    user_id: str | None = None
    conversation_id: str | None = None
    request_id: str | None = None
    rating: Literal["up", "down"]
    prompt: str
    response: str
    mode: ModeLiteral


class FeedbackResponse(BaseModel):
    stored: bool = True
    feedback_count: int


class DeleteUserResponse(BaseModel):
    user_id: str
    deleted: bool = True


class RuntimeConfigResponse(BaseModel):
    apiBaseUrl: str
    defaultModel: str
    stream: bool
    enabledFeatures: dict[str, bool]
    maxTokensDefault: int
    temperatureDefault: float


class BackendConfigResponse(BaseModel):
    model_id: str
    active_model: str
    adapter_id: str | None
    stream_default: bool
    max_tokens_default: int
    temperature_default: float
    supported_modes: list[ModeLiteral]
    frontend_features: dict[str, bool]
    default_generation: dict[str, float | int]


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    model_loaded: bool
    adapter_loaded: bool
    engine: str
    gpu_type: str
    build_sha: str
    model_version: str
    telemetry_enabled: bool = False


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[dict[str, Any]]
    active_model: str
    active_adapter: str | None
    release_version: str
