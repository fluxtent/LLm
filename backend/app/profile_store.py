"""Persistent profile, session summary, feedback, and API-key storage."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from .schemas import FeedbackRequest, UserProfile
from .settings import get_settings


def _public_api_key_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "label": record["label"],
        "prefix": record["prefix"],
        "created_at": record["created_at"],
        "last_used_at": record.get("last_used_at"),
        "revoked_at": record.get("revoked_at"),
        "usage_count": record.get("usage_count", 0),
    }


@dataclass
class MemoryStore:
    storage_path: Path | None = None
    profiles: dict[str, UserProfile] = field(default_factory=dict)
    session_summaries: dict[tuple[str, str], str] = field(default_factory=dict)
    feedback: list[dict] = field(default_factory=list)
    api_keys: dict[str, dict[str, Any]] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.storage_path is not None:
            self.storage_path = Path(self.storage_path)
            self.load()

    @staticmethod
    def _hash_api_key(api_key: str) -> str:
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()

    def load(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        with self._lock:
            profiles = {}
            for user_id, raw_profile in data.get("profiles", {}).items():
                try:
                    profiles[user_id] = UserProfile.model_validate(raw_profile)
                except Exception:
                    continue
            self.profiles = profiles

            summaries: dict[tuple[str, str], str] = {}
            for item in data.get("session_summaries", []):
                user_id = item.get("user_id")
                session_id = item.get("session_id")
                summary = item.get("summary")
                if user_id and session_id and summary:
                    summaries[(user_id, session_id)] = str(summary)
            self.session_summaries = summaries

            self.feedback = [item for item in data.get("feedback", []) if isinstance(item, dict)]
            self.api_keys = {
                item["id"]: item
                for item in data.get("api_keys", [])
                if isinstance(item, dict) and item.get("id") and item.get("key_hash")
            }

    def save(self) -> None:
        if self.storage_path is None:
            return
        with self._lock:
            payload = {
                "profiles": {
                    user_id: profile.model_dump()
                    for user_id, profile in self.profiles.items()
                },
                "session_summaries": [
                    {
                        "user_id": user_id,
                        "session_id": session_id,
                        "summary": summary,
                    }
                    for (user_id, session_id), summary in self.session_summaries.items()
                ],
                "feedback": self.feedback,
                "api_keys": list(self.api_keys.values()),
            }
            try:
                self.storage_path.parent.mkdir(parents=True, exist_ok=True)
                self.storage_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            except OSError:
                # Serverless platforms such as Vercel may expose a read-only
                # application directory. Keep the in-memory state for the warm
                # invocation instead of taking chat/profile requests down.
                self.storage_path = None

    def upsert_profile(self, profile: UserProfile) -> UserProfile:
        with self._lock:
            self.profiles[profile.user_id] = deepcopy(profile)
            self.save()
            return deepcopy(self.profiles[profile.user_id])

    def get_profile(self, user_id: str) -> UserProfile | None:
        with self._lock:
            profile = self.profiles.get(user_id)
            return deepcopy(profile) if profile else None

    def set_session_summary(self, user_id: str, session_id: str, summary: str) -> None:
        with self._lock:
            cleaned = summary.strip()
            self.session_summaries[(user_id, session_id)] = cleaned
            profile = self.profiles.get(user_id)
            if profile:
                profile.session_history = [cleaned, *profile.session_history][:10]
            self.save()

    def latest_summary(self, user_id: str) -> str | None:
        with self._lock:
            for (stored_user_id, _session_id), summary in reversed(list(self.session_summaries.items())):
                if stored_user_id == user_id:
                    return summary
            profile = self.profiles.get(user_id)
            if profile and profile.session_history:
                return profile.session_history[0]
            return None

    def add_feedback(self, feedback: FeedbackRequest) -> int:
        with self._lock:
            self.feedback.append(feedback.model_dump())
            self.save()
            return len(self.feedback)

    def delete_user(self, user_id: str) -> None:
        with self._lock:
            self.profiles.pop(user_id, None)
            self.feedback = [item for item in self.feedback if item.get("user_id") != user_id]
            self.session_summaries = {
                key: value for key, value in self.session_summaries.items() if key[0] != user_id
            }
            self.save()

    def create_api_key(self, label: str) -> tuple[str, dict[str, Any]]:
        raw_key = f"mbk_{secrets.token_urlsafe(32)}"
        now = int(time.time())
        record = {
            "id": secrets.token_hex(8),
            "label": (label or "MedBrief API key").strip()[:80] or "MedBrief API key",
            "prefix": raw_key[:12],
            "key_hash": self._hash_api_key(raw_key),
            "created_at": now,
            "last_used_at": None,
            "revoked_at": None,
            "usage_count": 0,
        }
        with self._lock:
            self.api_keys[record["id"]] = record
            self.save()
        return raw_key, _public_api_key_record(record)

    def list_api_keys(self, include_revoked: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            records = [
                _public_api_key_record(record)
                for record in self.api_keys.values()
                if include_revoked or not record.get("revoked_at")
            ]
        return sorted(records, key=lambda item: item["created_at"], reverse=True)

    def authenticate_api_key(self, api_key: str) -> dict[str, Any] | None:
        key_hash = self._hash_api_key(api_key.strip())
        now = int(time.time())
        with self._lock:
            for record in self.api_keys.values():
                if record.get("revoked_at"):
                    continue
                if hmac.compare_digest(record.get("key_hash", ""), key_hash):
                    record["last_used_at"] = now
                    record["usage_count"] = int(record.get("usage_count", 0)) + 1
                    self.save()
                    return _public_api_key_record(record)
        return None

    def revoke_api_key(self, key_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self.api_keys.get(key_id)
            if record is None:
                return None
            record["revoked_at"] = int(time.time())
            self.save()
            return _public_api_key_record(record)


STORE = MemoryStore(storage_path=Path(get_settings().store_path))
