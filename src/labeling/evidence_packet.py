from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from typing import Any

from src.storage.repository import SQLiteRepository


@dataclass
class EvidenceConfig:
    prompt_version: str = "conv_label_v1"
    model: str = "gpt-4o-mini"
    max_top_terms: int = 14
    max_representatives: int = 8


def build_cluster_evidence_packet(
    repo: SQLiteRepository,
    conv_cluster_id: int,
    config: EvidenceConfig | None = None,
) -> dict[str, Any]:
    cfg = config or EvidenceConfig()
    members = repo.get_conv_cluster_members(conv_cluster_id)
    if not members:
        packet = {
            "cluster_id": int(conv_cluster_id),
            "size": {"conversations": 0, "messages": 0},
            "date_range": {"started_at": None, "ended_at": None},
            "source_breakdown": {},
            "top_terms": [],
            "representative_conversations": [],
        }
        packet["evidence_hash"] = _evidence_hash(packet, cfg.prompt_version, cfg.model)
        return packet

    started_values = [str(m.get("started_at") or "") for m in members if m.get("started_at")]
    ended_values = [str(m.get("ended_at") or "") for m in members if m.get("ended_at")]
    total_messages = sum(int(m.get("message_count") or 0) for m in members)

    source_counter: Counter[str] = Counter()
    term_counter: Counter[str] = Counter()
    reps: list[dict[str, Any]] = []

    for member in members:
        source = str(member.get("source") or "UNKNOWN")
        source_counter[source] += 1

        rollup_text = str(member.get("rollup_text") or "")
        term_counter.update([tok for tok in rollup_text.split() if tok])

        if int(member.get("is_representative") or 0) == 1 and len(reps) < cfg.max_representatives:
            snippets = member.get("representative_snippets") or []
            preview = " | ".join([str(s) for s in snippets[:2]])
            reps.append(
                {
                    "conversation_id": str(member.get("conversation_id") or ""),
                    "started_at": member.get("started_at"),
                    "ended_at": member.get("ended_at"),
                    "rollup_preview": preview[:360],
                }
            )

    top_terms = [term for term, _ in term_counter.most_common(cfg.max_top_terms)]

    packet = {
        "cluster_id": int(conv_cluster_id),
        "size": {
            "conversations": len(members),
            "messages": total_messages,
        },
        "date_range": {
            "started_at": min(started_values) if started_values else None,
            "ended_at": max(ended_values) if ended_values else None,
        },
        "source_breakdown": dict(sorted(source_counter.items(), key=lambda kv: kv[0])),
        "top_terms": top_terms,
        "representative_conversations": sorted(reps, key=lambda r: (str(r.get("started_at") or ""), str(r.get("conversation_id") or ""))),
    }
    packet["evidence_hash"] = _evidence_hash(packet, cfg.prompt_version, cfg.model)
    return packet


def _evidence_hash(packet: dict[str, Any], prompt_version: str, model: str) -> str:
    payload = {
        "packet": packet,
        "prompt_version": prompt_version,
        "model": model,
        "generated_at": None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
