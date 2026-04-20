"""Structured telemetry with PII scrubbing and optional OpenTelemetry hooks."""

from __future__ import annotations

import json
import logging
import re
from typing import Any


LOGGER = logging.getLogger("medbrief.telemetry")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

try:
    from opentelemetry import trace  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    trace = None


def scrub_pii(value: Any) -> Any:
    if isinstance(value, str):
        value = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", value)
        value = PHONE_PATTERN.sub("[REDACTED_PHONE]", value)
        value = SSN_PATTERN.sub("[REDACTED_SSN]", value)
        return value
    if isinstance(value, dict):
        return {key: scrub_pii(item) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub_pii(item) for item in value]
    return value


def telemetry_enabled() -> bool:
    return trace is not None


def emit_event(event_name: str, **fields: Any) -> None:
    payload = {"event": event_name, **scrub_pii(fields)}
    LOGGER.info(json.dumps(payload, sort_keys=True))
    if trace is not None:  # pragma: no cover - depends on optional package
        tracer = trace.get_tracer("medbrief")
        with tracer.start_as_current_span(event_name) as span:
            for key, value in payload.items():
                span.set_attribute(key, json.dumps(value) if isinstance(value, (dict, list)) else value)
