from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_DEFAULT_DOMAIN_STOPWORDS = {
    "id",
    "data",
    "file",
    "path",
    "uploaded",
    "user",
    "turn",
    "messages",
    "prompted",
    "imagedisplayed",
    "json",
    "csv",
    "xlsx",
    "pdf",
    "assistant",
    "system",
    "role",
    "timestamp",
    "conversation",
    "model",
    "source",
    "import",
    "def",
    "return",
    "http",
    "https",
    "aws",
    "migration",
    "cloud",
    "service",
    "services",
    "account",
    "region",
    "vpc",
    "ec2",
    "s3",
    "iam",
    "application",
    "applications",
}

_STOPWORDS_PATH = Path(__file__).resolve().parents[2] / "config" / "domain_stopwords.txt"


@lru_cache(maxsize=1)
def get_domain_stopwords() -> set[str]:
    if not _STOPWORDS_PATH.exists():
        return set(_DEFAULT_DOMAIN_STOPWORDS)

    loaded: set[str] = set()
    for line in _STOPWORDS_PATH.read_text(encoding="utf-8").splitlines():
        token = line.strip().lower()
        if not token or token.startswith("#"):
            continue
        loaded.add(token.lstrip("."))
    return loaded or set(_DEFAULT_DOMAIN_STOPWORDS)


DOMAIN_STOPWORDS = sorted(get_domain_stopwords())

DOMAIN_SIGNAL_TOKENS = {"aws", "migration", "cloud"}
