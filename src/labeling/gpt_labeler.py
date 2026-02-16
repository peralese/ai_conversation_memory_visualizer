from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import re
import time
from typing import Any

from src.labeling.evidence_packet import EvidenceConfig
from src.storage.repository import SQLiteRepository

PROMPT_VERSION = "conv_label_v1"


@dataclass
class GPTLabelerConfig:
    model: str = "gpt-4o-mini"
    min_seconds_between_requests: float = 1.5
    max_requests_per_run: int = 100
    max_retries: int = 4
    dry_run: bool = False


class GPTClusterLabeler:
    def __init__(self, repo: SQLiteRepository, config: GPTLabelerConfig | None = None):
        self.repo = repo
        self.config = config or GPTLabelerConfig()
        self._last_request_at = 0.0
        self._requests_made = 0
        self._client = None

    def generate_label(
        self,
        conv_cluster_id: int,
        packet: dict[str, Any],
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        evidence_hash = str(packet.get("evidence_hash") or "")
        cached = self.repo.get_conv_cluster_label(conv_cluster_id)
        if (
            not force_refresh
            and cached is not None
            and str(cached.get("evidence_hash") or "") == evidence_hash
            and str(cached.get("prompt_version") or "") == PROMPT_VERSION
            and str(cached.get("model") or "") == self.config.model
        ):
            return {
                "conv_cluster_id": conv_cluster_id,
                "cached": True,
                "label_source": str(cached.get("label_source") or "heuristic"),
                "title": str(cached.get("title") or ""),
                "summary": str(cached.get("summary") or ""),
                "tags": list(cached.get("tags") or []),
            }

        if self.config.dry_run:
            label = _heuristic_label(packet)
            self._persist(conv_cluster_id, label, evidence_hash=evidence_hash, label_source="heuristic", tokens_in=None, tokens_out=None)
            return {"conv_cluster_id": conv_cluster_id, "cached": False, **label, "label_source": "heuristic"}

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or self._requests_made >= self.config.max_requests_per_run:
            label = _heuristic_label(packet)
            self._persist(conv_cluster_id, label, evidence_hash=evidence_hash, label_source="heuristic", tokens_in=None, tokens_out=None)
            return {"conv_cluster_id": conv_cluster_id, "cached": False, **label, "label_source": "heuristic"}

        payload, tokens_in, tokens_out = self._call_with_retry(packet)
        label = _sanitize_label_payload(payload)
        self._persist(conv_cluster_id, label, evidence_hash=evidence_hash, label_source="gpt", tokens_in=tokens_in, tokens_out=tokens_out)
        return {"conv_cluster_id": conv_cluster_id, "cached": False, **label, "label_source": "gpt"}

    def _persist(
        self,
        conv_cluster_id: int,
        label: dict[str, Any],
        *,
        evidence_hash: str,
        label_source: str,
        tokens_in: int | None,
        tokens_out: int | None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.repo.upsert_conv_cluster_label(
            {
                "conv_cluster_id": conv_cluster_id,
                "label_source": label_source,
                "title": label["title"],
                "summary": label["summary"],
                "tags": label["tags"],
                "evidence_hash": evidence_hash,
                "prompt_version": PROMPT_VERSION,
                "model": self.config.model,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "created_at": now,
                "updated_at": now,
            }
        )

    def _call_with_retry(self, packet: dict[str, Any]) -> tuple[dict[str, Any], int | None, int | None]:
        backoff = 1.0
        for attempt in range(self.config.max_retries):
            try:
                self._throttle()
                payload, tokens_in, tokens_out = self._call_openai(packet)
                return payload, tokens_in, tokens_out
            except Exception:
                if attempt >= self.config.max_retries - 1:
                    raise
                time.sleep(backoff)
                backoff *= 2.0
        raise RuntimeError("Failed to get GPT label")

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_at
        wait_for = self.config.min_seconds_between_requests - elapsed
        if wait_for > 0:
            time.sleep(wait_for)

    def _call_openai(self, packet: dict[str, Any]) -> tuple[dict[str, Any], int | None, int | None]:
        if self._client is None:
            from openai import OpenAI  # type: ignore

            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        system_prompt = (
            "You are an analyst. Produce a concise topic label and summary for a cluster of conversations. "
            "Avoid generic tokens like id/file/str/click/run. Prefer concrete domains. "
            "Return strict JSON with keys: title, summary, tags."
        )
        user_prompt = (
            "Use this evidence packet JSON:\n"
            f"{json.dumps(packet, sort_keys=True)}\n\n"
            "Return JSON with:\n"
            "title: 3-7 words\n"
            "summary: 1-2 sentences\n"
            "tags: 5-12 short tags"
        )

        response = self._client.chat.completions.create(
            model=self.config.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        self._requests_made += 1
        self._last_request_at = time.time()

        content = response.choices[0].message.content if response.choices else "{}"
        parsed = json.loads(content or "{}")
        usage = response.usage
        tokens_in = int(usage.prompt_tokens) if usage and usage.prompt_tokens is not None else None
        tokens_out = int(usage.completion_tokens) if usage and usage.completion_tokens is not None else None
        return parsed, tokens_in, tokens_out


def _heuristic_label(packet: dict[str, Any]) -> dict[str, Any]:
    terms = [str(t) for t in (packet.get("top_terms") or []) if str(t).strip()]
    title_terms = terms[:4] if terms else ["Conversation", "Cluster"]
    title = " ".join(title_terms[:4]).title()
    subtitle_terms = terms[4:10]
    summary = (
        f"This conversation cluster is mostly about {title.lower()}. "
        f"Common threads include: {', '.join(subtitle_terms) if subtitle_terms else 'related workflows and planning'}."
    )
    tags = terms[:10] if terms else ["conversation", "cluster"]
    return {
        "title": _truncate(re.sub(r"\s+", " ", title).strip(), 60),
        "summary": _truncate(re.sub(r"\s+", " ", summary).strip(), 280),
        "tags": tags,
    }


def _sanitize_label_payload(payload: dict[str, Any]) -> dict[str, Any]:
    title = _truncate(str(payload.get("title") or "Conversation Cluster").strip(), 60)
    summary = _truncate(str(payload.get("summary") or "").strip(), 280)
    tags_raw = payload.get("tags")
    if isinstance(tags_raw, list):
        tags = [str(t).strip() for t in tags_raw if str(t).strip()]
    elif isinstance(tags_raw, str):
        tags = [p.strip() for p in tags_raw.split(",") if p.strip()]
    else:
        tags = []
    if not tags:
        tags = [tok for tok in re.findall(r"[a-z0-9_-]+", title.lower()) if len(tok) > 2]
    return {"title": title, "summary": summary, "tags": tags[:12]}


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip(" ,.;:-")
