"""In-memory profile, session summary, and feedback storage for local MedBrief use."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from .schemas import FeedbackRequest, UserProfile


@dataclass
class MemoryStore:
    profiles: dict[str, UserProfile] = field(default_factory=dict)
    session_summaries: dict[tuple[str, str], str] = field(default_factory=dict)
    feedback: list[dict] = field(default_factory=list)

    def upsert_profile(self, profile: UserProfile) -> UserProfile:
        self.profiles[profile.user_id] = deepcopy(profile)
        return deepcopy(self.profiles[profile.user_id])

    def get_profile(self, user_id: str) -> UserProfile | None:
        profile = self.profiles.get(user_id)
        return deepcopy(profile) if profile else None

    def set_session_summary(self, user_id: str, session_id: str, summary: str) -> None:
        self.session_summaries[(user_id, session_id)] = summary.strip()
        profile = self.profiles.get(user_id)
        if profile:
            profile.session_history = [summary.strip(), *profile.session_history][:10]

    def latest_summary(self, user_id: str) -> str | None:
        for (stored_user_id, _session_id), summary in reversed(list(self.session_summaries.items())):
            if stored_user_id == user_id:
                return summary
        profile = self.profiles.get(user_id)
        if profile and profile.session_history:
            return profile.session_history[0]
        return None

    def add_feedback(self, feedback: FeedbackRequest) -> int:
        self.feedback.append(feedback.model_dump())
        return len(self.feedback)

    def delete_user(self, user_id: str) -> None:
        self.profiles.pop(user_id, None)
        self.feedback = [item for item in self.feedback if item.get("user_id") != user_id]
        self.session_summaries = {
            key: value for key, value in self.session_summaries.items() if key[0] != user_id
        }


STORE = MemoryStore()
