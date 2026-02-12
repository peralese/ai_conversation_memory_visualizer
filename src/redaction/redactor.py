from __future__ import annotations

import re
from dataclasses import dataclass


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# Lightweight heuristic for person-like names (capitalized pairs).
NAME_RE = re.compile(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b")


@dataclass
class RedactionConfig:
    redact_emails: bool = True
    redact_phones: bool = True
    redact_ips: bool = True
    redact_names: bool = False


def redact_text(text: str, config: RedactionConfig) -> str:
    redacted = text
    if config.redact_emails:
        redacted = EMAIL_RE.sub("[REDACTED_EMAIL]", redacted)
    if config.redact_phones:
        redacted = PHONE_RE.sub("[REDACTED_PHONE]", redacted)
    if config.redact_ips:
        redacted = IP_RE.sub("[REDACTED_IP]", redacted)
    if config.redact_names:
        redacted = NAME_RE.sub("[REDACTED_NAME]", redacted)
    return redacted
