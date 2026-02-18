from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from src.labeling.evidence import (
    build_cluster_evidence_packet,
    build_conv_cluster_evidence_packet,
)
from src.labeling.gpt_labeler import SemanticLabeler, SemanticLabelerConfig
from src.storage.repository import SQLiteRepository

logger = logging.getLogger(__name__)


class SemanticLabelService:
    def __init__(
        self,
        repo: SQLiteRepository,
        labeler: SemanticLabeler | None = None,
        config: SemanticLabelerConfig | None = None,
    ):
        self.repo = repo
        self.labeler = labeler or SemanticLabeler(config)

    def label_one_cluster(self, cluster_id: int, force: bool = False) -> dict[str, Any]:
        packet = build_cluster_evidence_packet(self.repo, int(cluster_id))
        evidence_hash = str(packet.get("evidence_hash") or "")
        existing = self.repo.get_cluster_semantic_label(int(cluster_id))
        if (
            not force
            and existing is not None
            and str(existing.get("evidence_hash") or "") == evidence_hash
        ):
            return {"cluster_id": int(cluster_id), "cached": True, "provider": str(existing.get("provider") or "heuristic")}

        result = asyncio.run(self.labeler.label_cluster(packet))
        self.repo.upsert_cluster_semantic_label(
            cluster_id=int(cluster_id),
            label=str(result.get("label") or f"Cluster {cluster_id}"),
            summary=str(result.get("summary") or ""),
            tags=[str(t) for t in (result.get("tags") or [])],
            provider=str(result.get("provider") or "heuristic"),
            evidence_hash=evidence_hash,
        )
        return {"cluster_id": int(cluster_id), "cached": False, "provider": str(result.get("provider") or "heuristic")}

    def label_one_conv_cluster(self, conv_cluster_id: int, force: bool = False) -> dict[str, Any]:
        packet = build_conv_cluster_evidence_packet(self.repo, int(conv_cluster_id))
        evidence_hash = str(packet.get("evidence_hash") or "")
        existing = self.repo.get_conv_cluster_semantic_label(int(conv_cluster_id))
        if (
            not force
            and existing is not None
            and str(existing.get("evidence_hash") or "") == evidence_hash
        ):
            return {"conv_cluster_id": int(conv_cluster_id), "cached": True, "provider": str(existing.get("provider") or "heuristic")}

        result = asyncio.run(self.labeler.label_conv_cluster(packet))
        self.repo.upsert_conv_cluster_semantic_label(
            conv_cluster_id=int(conv_cluster_id),
            label=str(result.get("label") or f"Conversation Cluster {conv_cluster_id}"),
            summary=str(result.get("summary") or ""),
            tags=[str(t) for t in (result.get("tags") or [])],
            provider=str(result.get("provider") or "heuristic"),
            evidence_hash=evidence_hash,
        )
        return {"conv_cluster_id": int(conv_cluster_id), "cached": False, "provider": str(result.get("provider") or "heuristic")}

    def label_all_clusters(self, force: bool = False, limit: int | None = None) -> dict[str, Any]:
        cluster_ids = self.repo.list_clusters_missing_semantic_labels(force=force)
        if force:
            cluster_ids = sorted(int(c["cluster_id"]) for c in self.repo.list_clusters())
        if limit is not None:
            cluster_ids = cluster_ids[: max(0, int(limit))]
        started = time.perf_counter()
        result = self._run_batch_cluster_labels(cluster_ids, force=force)
        result["duration_seconds"] = round(time.perf_counter() - started, 3)
        logger.info("Cluster semantic labeling complete: %s", result)
        return result

    def label_all_conv_clusters(self, force: bool = False, limit: int | None = None) -> dict[str, Any]:
        cluster_ids = self.repo.list_conv_clusters_missing_semantic_labels(force=force)
        if force:
            cluster_ids = sorted(int(r["conv_cluster_id"]) for r in self.repo.list_conv_clusters())
        if limit is not None:
            cluster_ids = cluster_ids[: max(0, int(limit))]
        started = time.perf_counter()
        result = self._run_batch_conv_cluster_labels(cluster_ids, force=force)
        result["duration_seconds"] = round(time.perf_counter() - started, 3)
        logger.info("Conversation cluster semantic labeling complete: %s", result)
        return result

    def _run_batch_cluster_labels(self, cluster_ids: list[int], *, force: bool) -> dict[str, Any]:
        async def _run() -> dict[str, int]:
            generated = 0
            cached = 0
            errors = 0
            tasks = [self._label_cluster_async(cid, force=force) for cid in cluster_ids]
            for result in await asyncio.gather(*tasks, return_exceptions=True):
                if isinstance(result, Exception):
                    errors += 1
                    continue
                if result.get("cached"):
                    cached += 1
                else:
                    generated += 1
            return {"generated": generated, "cached": cached, "errors": errors}

        counts = asyncio.run(_run()) if cluster_ids else {"generated": 0, "cached": 0, "errors": 0}
        return {"clusters_considered": len(cluster_ids), **counts}

    def _run_batch_conv_cluster_labels(self, cluster_ids: list[int], *, force: bool) -> dict[str, Any]:
        async def _run() -> dict[str, int]:
            generated = 0
            cached = 0
            errors = 0
            tasks = [self._label_conv_cluster_async(cid, force=force) for cid in cluster_ids]
            for result in await asyncio.gather(*tasks, return_exceptions=True):
                if isinstance(result, Exception):
                    errors += 1
                    continue
                if result.get("cached"):
                    cached += 1
                else:
                    generated += 1
            return {"generated": generated, "cached": cached, "errors": errors}

        counts = asyncio.run(_run()) if cluster_ids else {"generated": 0, "cached": 0, "errors": 0}
        return {"conv_clusters_considered": len(cluster_ids), **counts}

    async def _label_cluster_async(self, cluster_id: int, *, force: bool) -> dict[str, Any]:
        packet = build_cluster_evidence_packet(self.repo, int(cluster_id))
        evidence_hash = str(packet.get("evidence_hash") or "")
        existing = self.repo.get_cluster_semantic_label(int(cluster_id))
        if (
            not force
            and existing is not None
            and str(existing.get("evidence_hash") or "") == evidence_hash
        ):
            return {"cluster_id": int(cluster_id), "cached": True}
        result = await self.labeler.label_cluster(packet)
        self.repo.upsert_cluster_semantic_label(
            cluster_id=int(cluster_id),
            label=str(result.get("label") or f"Cluster {cluster_id}"),
            summary=str(result.get("summary") or ""),
            tags=[str(t) for t in (result.get("tags") or [])],
            provider=str(result.get("provider") or "heuristic"),
            evidence_hash=evidence_hash,
        )
        return {"cluster_id": int(cluster_id), "cached": False}

    async def _label_conv_cluster_async(self, conv_cluster_id: int, *, force: bool) -> dict[str, Any]:
        packet = build_conv_cluster_evidence_packet(self.repo, int(conv_cluster_id))
        evidence_hash = str(packet.get("evidence_hash") or "")
        existing = self.repo.get_conv_cluster_semantic_label(int(conv_cluster_id))
        if (
            not force
            and existing is not None
            and str(existing.get("evidence_hash") or "") == evidence_hash
        ):
            return {"conv_cluster_id": int(conv_cluster_id), "cached": True}
        result = await self.labeler.label_conv_cluster(packet)
        self.repo.upsert_conv_cluster_semantic_label(
            conv_cluster_id=int(conv_cluster_id),
            label=str(result.get("label") or f"Conversation Cluster {conv_cluster_id}"),
            summary=str(result.get("summary") or ""),
            tags=[str(t) for t in (result.get("tags") or [])],
            provider=str(result.get("provider") or "heuristic"),
            evidence_hash=evidence_hash,
        )
        return {"conv_cluster_id": int(conv_cluster_id), "cached": False}
