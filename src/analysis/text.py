from __future__ import annotations

import re

_BOILERPLATE_STOPWORDS = {
    "please",
    "help",
    "generate",
    "create",
    "rewrite",
    "draft",
    "write",
    "make",
    "build",
    "provide",
    "show",
    "give",
    "prompted",
}

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def build_analysis_text(text: str) -> str:
    if not text:
        return ""

    cleaned = text.strip()
    cleaned = re.sub(r"^\s*prompted\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    tokens = _TOKEN_PATTERN.findall(cleaned)
    kept = [tok for tok in tokens if tok not in _BOILERPLATE_STOPWORDS]
    return " ".join(kept)
