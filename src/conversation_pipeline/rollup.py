from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from typing import Any

from src.clustering.semantic_labels import normalize_for_label
from src.storage.repository import SQLiteRepository


@dataclass
class RollupConfig:
    max_snippet_chars: int = 240
    max_representative_snippets: int = 6
    max_top_terms: int = 12
    config_version: str = "conv_rollup_v1"
    exclude_domain_stopwords: bool = True


def build_conversation_rollups(repo: SQLiteRepository, config: RollupConfig | None = None) -> list[str]:
    cfg = config or RollupConfig()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in repo.all_conversation_messages():
        grouped[str(row["conversation_id"])].append(row)

    updated_ids: list[str] = []
    now = datetime.now(timezone.utc).isoformat()

    for conversation_id, rows in sorted(grouped.items(), key=lambda kv: kv[0]):
        rows_sorted = sorted(rows, key=lambda r: str(r.get("timestamp") or ""))
        if not rows_sorted:
            continue

        source = str(rows_sorted[0].get("source") or "")
        started_at = str(rows_sorted[0].get("timestamp") or "")
        ended_at = str(rows_sorted[-1].get("timestamp") or "")
        message_count = len(rows_sorted)
        user_message_count = sum(1 for r in rows_sorted if str(r.get("speaker_role") or "").lower() == "user")
        assistant_message_count = sum(1 for r in rows_sorted if str(r.get("speaker_role") or "").lower() == "assistant")
        avg_message_length = sum(len(str(r.get("original_text") or "")) for r in rows_sorted) / max(1, message_count)

        snippets = _representative_snippets(rows_sorted, max_chars=cfg.max_snippet_chars, max_count=cfg.max_representative_snippets)
        conv_title = str(rows_sorted[0].get("conversation_title") or "").strip()
        normalized_parts = []
        if conv_title:
            normalized_title = normalize_for_label(conv_title, exclude_domain_stopwords=cfg.exclude_domain_stopwords)
            if normalized_title:
                normalized_parts.append(normalized_title)

        top_terms = _top_terms(
            [str(r.get("original_text") or "") for r in rows_sorted],
            top_n=cfg.max_top_terms,
            exclude_domain_stopwords=cfg.exclude_domain_stopwords,
        )
        if top_terms:
            normalized_parts.append("top terms: " + ", ".join(top_terms))

        for snippet in snippets:
            normalized = normalize_for_label(snippet, exclude_domain_stopwords=cfg.exclude_domain_stopwords)
            if normalized:
                normalized_parts.append(normalized)

        rollup_text = "\n".join(normalized_parts).strip()
        rollup_hash = hashlib.sha256(f"{cfg.config_version}\n{rollup_text}".encode("utf-8")).hexdigest()

        repo.upsert_conversation_rollup(
            {
                "conversation_id": conversation_id,
                "source": source,
                "started_at": started_at,
                "ended_at": ended_at,
                "message_count": message_count,
                "user_message_count": user_message_count,
                "assistant_message_count": assistant_message_count,
                "avg_message_length": avg_message_length,
                "top_terms": top_terms,
                "representative_snippets": snippets,
                "rollup_text": rollup_text,
                "rollup_hash": rollup_hash,
                "updated_at": now,
            }
        )
        updated_ids.append(conversation_id)

    return updated_ids


def _representative_snippets(rows: list[dict[str, Any]], *, max_chars: int, max_count: int) -> list[str]:
    user_rows = [r for r in rows if str(r.get("speaker_role") or "").lower() == "user"]
    if not user_rows:
        user_rows = rows

    selected: list[dict[str, Any]] = []
    if user_rows:
        selected.append(user_rows[0])
        if len(user_rows) > 1:
            selected.append(user_rows[-1])

    mid_candidates = user_rows[1:-1] if len(user_rows) > 2 else []
    if mid_candidates:
        picks = min(4, len(mid_candidates))
        step = max(1, len(mid_candidates) // picks)
        for idx in range(0, len(mid_candidates), step):
            selected.append(mid_candidates[idx])
            if len(selected) >= max_count:
                break

    snippets: list[str] = []
    seen: set[str] = set()
    for row in selected:
        text = " ".join(str(row.get("original_text") or "").split()).strip()
        if not text:
            continue
        clipped = text[:max_chars]
        key = clipped.lower()
        if key in seen:
            continue
        seen.add(key)
        snippets.append(clipped)
        if len(snippets) >= max_count:
            break

    return snippets


def _top_terms(texts: list[str], *, top_n: int, exclude_domain_stopwords: bool) -> list[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        cleaned = normalize_for_label(text, exclude_domain_stopwords=exclude_domain_stopwords)
        counter.update([tok for tok in cleaned.split() if tok])
    return [tok for tok, _count in counter.most_common(top_n)]
