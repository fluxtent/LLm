"""Schemas for MedBrief training assets."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ModeLiteral = Literal["psych", "health", "crisis", "portfolio", "general"]
RoleLiteral = Literal["user", "assistant", "system"]
SafetyLiteral = Literal["standard", "caution", "crisis", "restricted"]


class ConversationMessage(BaseModel):
    role: RoleLiteral
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("content cannot be empty")
        return cleaned


class SFTConversation(BaseModel):
    messages: list[ConversationMessage]
    mode: ModeLiteral
    source: str
    source_type: str
    tags: list[str] = Field(default_factory=list)
    safety_level: SafetyLiteral


class EvalPrompt(BaseModel):
    mode: ModeLiteral
    prompt: str
    expected_traits: list[str]
    red_lines: list[str]


class PreferencePair(BaseModel):
    mode: ModeLiteral
    prompt: str
    chosen: str
    rejected: str
    source: str
