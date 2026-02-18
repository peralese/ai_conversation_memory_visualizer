from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
import re
import time
from typing import Any

from src.storage.repository import SQLiteRepository

PROMPT_VERSION = "semantic_label_v1"
logger = logging.getLogger(__name__)


@dataclass
class SemanticLabelerConfig:
    model: str = "gpt-4o-mini"
    rpm: int = 60
    concurrency: int = 2
    max_retries: int = 4
    dry_run: bool = False


class SemanticLabeler:
    def __init__(self, config: SemanticLabelerConfig | None = None):
        self.config = config or SemanticLabelerConfig()
        self.provider = "openai" if os.getenv("OPENAI_API_KEY") else "heuristic"
        if self.provider != "openai":
            logger.info("OPENAI_API_KEY missing; semantic labels will use heuristic provider.")
        self._sem = asyncio.Semaphore(max(1, int(self.config.concurrency)))
        self._request_times: deque[float] = deque()
        self._rate_lock = asyncio.Lock()
        self._client = None

    async def label_cluster(self, packet: dict[str, Any]) -> dict[str, Any]:
        return await self._label_packet(packet, kind="cluster")

    async def label_conv_cluster(self, packet: dict[str, Any]) -> dict[str, Any]:
        return await self._label_packet(packet, kind="conv_cluster")

    async def _label_packet(self, packet: dict[str, Any], *, kind: str) -> dict[str, Any]:
        if self.config.dry_run or self.provider != "openai":
            label = _heuristic_label(packet, kind=kind)
            return {**label, "provider": "heuristic"}

        async with self._sem:
            await self._wait_for_rate_slot()
            payload = await self._call_openai_with_retry(packet, kind=kind)
            label = _sanitize_label_payload(payload)
            return {**label, "provider": "openai"}

    async def _wait_for_rate_slot(self) -> None:
        rpm = max(1, int(self.config.rpm))
        window = 60.0
        while True:
            async with self._rate_lock:
                now = time.monotonic()
                while self._request_times and now - self._request_times[0] >= window:
                    self._request_times.popleft()
                if len(self._request_times) < rpm:
                    self._request_times.append(now)
                    return
                sleep_for = max(0.05, window - (now - self._request_times[0]))
            await asyncio.sleep(sleep_for)

    async def _call_openai_with_retry(self, packet: dict[str, Any], *, kind: str) -> dict[str, Any]:
        backoff = 1.0
        for attempt in range(max(1, int(self.config.max_retries))):
            try:
                return await asyncio.to_thread(self._call_openai, packet, kind)
            except Exception:
                if attempt >= max(1, int(self.config.max_retries)) - 1:
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2.0
        raise RuntimeError("OpenAI labeling failed")

    def _call_openai(self, packet: dict[str, Any], kind: str) -> dict[str, Any]:
        if self._client is None:
            from openai import OpenAI  # type: ignore

            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        system_prompt = (
            "You are an analyst creating concise semantic topic labels for clustered conversations. "
            "Return strict JSON with keys: label, summary, tags. "
            "Constraints: label <= 5 words; summary 1-2 sentences; tags 3-8 short strings. "
            "Avoid boilerplate, generic code tokens, and stopwords."
        )
        user_prompt = (
            f"Cluster type: {kind}\n"
            "Evidence packet JSON follows:\n"
            f"{json.dumps(packet, sort_keys=True)}"
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
        content = response.choices[0].message.content if response.choices else "{}"
        return json.loads(content or "{}")


@dataclass
class GPTLabelerConfig:
    model: str = "gpt-4o-mini"
    min_seconds_between_requests: float = 1.5
    max_requests_per_run: int = 100
    max_retries: int = 4
    dry_run: bool = False


class GPTClusterLabeler:
    """Backward-compatible wrapper used by conversation pipeline."""

    def __init__(self, repo: SQLiteRepository, config: GPTLabelerConfig | None = None):
        self.repo = repo
        self.config = config or GPTLabelerConfig()
        rpm = max(1, int(60.0 / max(0.05, float(self.config.min_seconds_between_requests))))
        self._labeler = SemanticLabeler(
            SemanticLabelerConfig(
                model=self.config.model,
                rpm=rpm,
                concurrency=2,
                max_retries=self.config.max_retries,
                dry_run=self.config.dry_run,
            )
        )
        self._requests_made = 0

    def generate_label(
        self,
        conv_cluster_id: int,
        packet: dict[str, Any],
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        evidence_hash = str(packet.get("evidence_hash") or "")
        cached = self.repo.get_conv_cluster_semantic_label(conv_cluster_id)
        if (
            not force_refresh
            and cached is not None
            and str(cached.get("evidence_hash") or "") == evidence_hash
        ):
            return {
                "conv_cluster_id": conv_cluster_id,
                "cached": True,
                "label_source": str(cached.get("provider") or "heuristic"),
                "title": str(cached.get("label") or ""),
                "summary": str(cached.get("summary") or ""),
                "tags": list(cached.get("tags") or []),
            }

        if self._requests_made >= int(self.config.max_requests_per_run):
            payload = _heuristic_label(packet, kind="conv_cluster")
            provider = "heuristic"
        else:
            payload, _tokens_in, _tokens_out = self._call_openai(packet)
            provider = str(payload.get("provider") or "openai")
            self._requests_made += 1

        label = str(payload.get("label") or "").strip() or f"Conversation Cluster {conv_cluster_id}"
        summary = str(payload.get("summary") or "").strip()
        tags = [str(t).strip() for t in (payload.get("tags") or []) if str(t).strip()]

        self.repo.upsert_conv_cluster_semantic_label(
            conv_cluster_id=int(conv_cluster_id),
            label=label,
            summary=summary,
            tags=tags,
            provider=provider,
            evidence_hash=evidence_hash,
        )

        # Keep legacy table in sync for backward compatibility.
        now = datetime.now(timezone.utc).isoformat()
        self.repo.upsert_conv_cluster_label(
            {
                "conv_cluster_id": int(conv_cluster_id),
                "label_source": provider,
                "title": label,
                "summary": summary,
                "tags": tags,
                "evidence_hash": evidence_hash,
                "prompt_version": PROMPT_VERSION,
                "model": self.config.model,
                "tokens_in": None,
                "tokens_out": None,
                "created_at": now,
                "updated_at": now,
            }
        )
        return {
            "conv_cluster_id": conv_cluster_id,
            "cached": False,
            "label_source": provider,
            "title": label,
            "summary": summary,
            "tags": tags,
        }

    def _call_openai(self, packet: dict[str, Any]) -> tuple[dict[str, Any], int | None, int | None]:
        payload = asyncio.run(self._labeler.label_conv_cluster(packet))
        return payload, None, None


def _heuristic_label(packet: dict[str, Any], *, kind: str) -> dict[str, Any]:
    terms = [str(t) for t in (packet.get("top_terms") or []) if str(t).strip()]
    if not terms:
        label = "Conversation Cluster" if kind == "conv_cluster" else "Topic Cluster"
        return {
            "label": label,
            "summary": f"This {kind.replace('_', ' ')} groups related discussions with overlapping intents.",
            "tags": ["cluster", "topic", "conversation"],
        }
    label = " ".join(terms[:4]).title()
    summary = (
        f"This {kind.replace('_', ' ')} is mostly about {label.lower()}. "
        f"Common threads include: {', '.join(terms[4:9]) if len(terms) > 4 else ', '.join(terms[:3])}."
    )
    tags = terms[:8]
    return {
        "label": _truncate(label, 60),
        "summary": _truncate(summary, 280),
        "tags": tags,
    }


def _sanitize_label_payload(payload: dict[str, Any]) -> dict[str, Any]:
    label = _truncate(str(payload.get("label") or "Topic Cluster").strip(), 60)
    label = re.sub(r"\s+", " ", label)
    summary = _truncate(str(payload.get("summary") or "").strip(), 280)

    tags_raw = payload.get("tags")
    if isinstance(tags_raw, list):
        tags = [str(t).strip() for t in tags_raw if str(t).strip()]
    elif isinstance(tags_raw, str):
        tags = [p.strip() for p in tags_raw.split(",") if p.strip()]
    else:
        tags = []
    if not tags:
        tags = [tok for tok in re.findall(r"[a-z0-9_-]+", label.lower()) if len(tok) >= 3]
    return {"label": label, "summary": summary, "tags": tags[:8]}


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip(" ,.;:-")
