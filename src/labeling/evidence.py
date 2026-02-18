from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
from typing import Any

from src.clustering.keywords import build_distinctive_labels
from src.storage.repository import SQLiteRepository

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,2}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")
_SPACE_RE = re.compile(r"\s+")


def scrub_pii(text: str) -> str:
    cleaned = _EMAIL_RE.sub("[redacted-email]", str(text or ""))
    cleaned = _PHONE_RE.sub("[redacted-phone]", cleaned)
    return _SPACE_RE.sub(" ", cleaned).strip()


def truncate_text(text: str, max_chars: int) -> str:
    value = scrub_pii(text)
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip()


def compute_evidence_hash(packet: dict[str, Any]) -> str:
    canonical = json.dumps(packet, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_cluster_evidence_packet(
    repo: SQLiteRepository,
    cluster_id: int,
    *,
    max_items: int = 12,
    max_chars: int = 400,
) -> dict[str, Any]:
    rows = repo.cluster_member_records(cluster_id)
    top_terms = _cluster_top_terms(rows, cluster_id)
    source_counts = Counter(str(r.get("source") or "UNKNOWN").upper().strip() for r in rows)
    timestamps = sorted(str(r.get("timestamp") or "") for r in rows if r.get("timestamp"))

    representatives = _representative_cluster_messages(rows, max_items=max_items)
    messages = [
        {
            "message_id": str(r.get("id") or ""),
            "timestamp": str(r.get("timestamp") or ""),
            "source": str(r.get("source") or ""),
            "conversation_id": str(r.get("conversation_id") or ""),
            "role": str(r.get("speaker_role") or ""),
            "content": truncate_text(str(r.get("original_text") or ""), max_chars),
        }
        for r in representatives
    ]

    packet = {
        "kind": "cluster",
        "cluster_id": int(cluster_id),
        "top_terms": top_terms,
        "message_count": int(len(rows)),
        "date_range": {
            "first_seen": timestamps[0] if timestamps else None,
            "last_seen": timestamps[-1] if timestamps else None,
        },
        "source_distribution": dict(sorted(source_counts.items(), key=lambda kv: kv[0])),
        "representative_messages": messages,
    }
    packet["evidence_hash"] = compute_evidence_hash(packet)
    return packet


def build_conv_cluster_evidence_packet(
    repo: SQLiteRepository,
    conv_cluster_id: int,
    *,
    max_items: int = 12,
    max_chars: int = 400,
) -> dict[str, Any]:
    members = repo.get_conv_cluster_members(conv_cluster_id)
    source_counts = Counter(str(m.get("source") or "UNKNOWN").upper().strip() for m in members)
    started = sorted(str(m.get("started_at") or "") for m in members if m.get("started_at"))
    ended = sorted(str(m.get("ended_at") or "") for m in members if m.get("ended_at"))
    total_messages = sum(int(m.get("message_count") or 0) for m in members)

    reps = sorted(
        members,
        key=lambda m: (
            -int(m.get("is_representative") or 0),
            float(m.get("distance") or 0.0),
            str(m.get("conversation_id") or ""),
        ),
    )[: max(1, int(max_items))]
    representative_conversations = []
    term_counter: Counter[str] = Counter()
    for member in reps:
        rollup_text = scrub_pii(str(member.get("rollup_text") or ""))
        term_counter.update([t for t in rollup_text.split() if len(t) >= 3][:40])
        snippets = [truncate_text(str(s), max_chars) for s in (member.get("representative_snippets") or [])[:2]]
        representative_conversations.append(
            {
                "conversation_id": str(member.get("conversation_id") or ""),
                "source": str(member.get("source") or ""),
                "started_at": member.get("started_at"),
                "ended_at": member.get("ended_at"),
                "preview": " | ".join(snippets),
            }
        )

    packet = {
        "kind": "conv_cluster",
        "conv_cluster_id": int(conv_cluster_id),
        "top_terms": [term for term, _ in term_counter.most_common(14)],
        "conversation_count": int(len(members)),
        "message_count": int(total_messages),
        "date_range": {
            "first_seen": started[0] if started else None,
            "last_seen": ended[-1] if ended else None,
        },
        "source_distribution": dict(sorted(source_counts.items(), key=lambda kv: kv[0])),
        "representative_conversations": representative_conversations,
    }
    packet["evidence_hash"] = compute_evidence_hash(packet)
    return packet


def _cluster_top_terms(rows: list[dict[str, Any]], cluster_id: int) -> list[str]:
    if not rows:
        return []
    texts_by_cluster = {int(cluster_id): [str(r.get("analysis_text") or "") for r in rows]}
    info = build_distinctive_labels(texts_by_cluster, top_n=8, exclude_domain_stopwords=True).get(int(cluster_id))
    if info and info.terms:
        return [str(t) for t in info.terms]
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update([tok for tok in str(row.get("analysis_text") or "").split() if len(tok) >= 3])
    return [term for term, _ in counter.most_common(8)]


def _representative_cluster_messages(rows: list[dict[str, Any]], *, max_items: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    with_vectors = [r for r in rows if isinstance(r.get("vector"), list) and r.get("vector")]
    if with_vectors:
        ranked = sorted(
            with_vectors,
            key=lambda r: (
                _vector_magnitude([float(x) for x in r.get("vector") or []]),
                str(r.get("timestamp") or ""),
                str(r.get("id") or ""),
            ),
            reverse=True,
        )
        return ranked[: max(1, int(max_items))]
    by_time = sorted(rows, key=lambda r: (str(r.get("timestamp") or ""), str(r.get("id") or "")))
    return by_time[: max(1, int(max_items))]


def _vector_magnitude(vector: list[float]) -> float:
    if not vector:
        return 0.0
    return sum(v * v for v in vector) ** 0.5
