from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import sqrt
from typing import Any

from src.storage.repository import SQLiteRepository

MIN_POINTS_PER_BUCKET = 3
VOLATILITY_STABLE_THRESHOLD = 0.05


class DriftService:
    def __init__(self, repo: SQLiteRepository):
        self.repo = repo

    def compute_and_persist(self, level: str = "cluster") -> dict[str, Any]:
        level = (level or "cluster").lower()
        if level not in {"cluster", "subcluster"}:
            raise ValueError("level must be cluster or subcluster")

        points = self.repo.cluster_embedding_points() if level == "cluster" else self.repo.subcluster_embedding_points()
        grouped: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
        for point in points:
            ts = datetime.fromisoformat(str(point["timestamp"]))
            bucket_start = (ts - timedelta(days=ts.weekday())).date().isoformat()
            grouped[(str(point["entity_id"]), bucket_start)].append(list(map(float, point["vector"])))

        rows = []
        now = datetime.now(timezone.utc).isoformat()
        for (entity_id, bucket), vectors in grouped.items():
            if len(vectors) < MIN_POINTS_PER_BUCKET:
                continue
            rows.append(
                {
                    "entity_id": entity_id,
                    "bucket_start_date": bucket,
                    "centroid_vector": _centroid(vectors),
                    "message_count": len(vectors),
                    "created_at": now,
                }
            )

        self.repo.replace_drift_rows(level, rows)
        return {"level": level, "rows": len(rows)}

    def ensure_fresh(self, level: str = "cluster") -> None:
        level = (level or "cluster").lower()
        if self.repo.drift_stale(level):
            self.compute_and_persist(level)

    def summary(self, level: str = "cluster") -> list[dict[str, Any]]:
        level = (level or "cluster").lower()
        self.ensure_fresh(level)
        rows = self.repo.drift_rows(level)
        by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_entity[str(row["entity_id"])].append(row)

        labels = _labels_for_level(self.repo, level)
        out = []
        for entity_id, items in by_entity.items():
            items_sorted = sorted(items, key=lambda r: r["bucket_start_date"])
            if len(items_sorted) < 2:
                continue
            week_to_week = []
            for idx in range(1, len(items_sorted)):
                prev = items_sorted[idx - 1]["centroid_vector"]
                cur = items_sorted[idx]["centroid_vector"]
                week_to_week.append(_cosine_distance(prev, cur))
            cumulative = _cosine_distance(items_sorted[0]["centroid_vector"], items_sorted[-1]["centroid_vector"])
            volatility = sum(week_to_week) / len(week_to_week) if week_to_week else 0.0
            out.append(
                {
                    "cluster_id": entity_id,
                    "label": labels.get(entity_id, entity_id),
                    "first_bucket": items_sorted[0]["bucket_start_date"],
                    "last_bucket": items_sorted[-1]["bucket_start_date"],
                    "cumulative_drift": round(cumulative, 6),
                    "volatility": round(volatility, 6),
                    "active_buckets_count": len(items_sorted),
                    "stability_tag": "stable" if volatility < VOLATILITY_STABLE_THRESHOLD else "evolving",
                }
            )
        out.sort(key=lambda r: r["cumulative_drift"], reverse=True)
        return out

    def detail(self, level: str = "cluster", cluster_id: str | None = None) -> dict[str, Any]:
        level = (level or "cluster").lower()
        self.ensure_fresh(level)
        if not cluster_id:
            return {"level": level, "items": self.summary(level)}

        rows = self.repo.drift_rows(level, entity_id=str(cluster_id))
        rows = sorted(rows, key=lambda r: r["bucket_start_date"])
        series = []
        for idx, row in enumerate(rows):
            drift = None
            if idx > 0:
                drift = _cosine_distance(rows[idx - 1]["centroid_vector"], row["centroid_vector"])
            series.append(
                {
                    "bucket_start_date": row["bucket_start_date"],
                    "message_count": row["message_count"],
                    "week_to_week_drift": None if drift is None else round(drift, 6),
                }
            )
        labels = _labels_for_level(self.repo, level)
        return {
            "level": level,
            "cluster_id": str(cluster_id),
            "label": labels.get(str(cluster_id), str(cluster_id)),
            "series": series,
        }


def _centroid(vectors: list[list[float]]) -> list[float]:
    dim = len(vectors[0])
    sums = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            sums[i] += float(v[i])
    return [s / len(vectors) for s in sums]


def _cosine_distance(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(float(a[i]) * float(b[i]) for i in range(n))
    na = sqrt(sum(float(a[i]) * float(a[i]) for i in range(n)))
    nb = sqrt(sum(float(b[i]) * float(b[i]) for i in range(n)))
    if na == 0 or nb == 0:
        return 0.0
    cosine_sim = max(-1.0, min(1.0, dot / (na * nb)))
    return 1.0 - cosine_sim


def _labels_for_level(repo: SQLiteRepository, level: str) -> dict[str, str]:
    if level == "subcluster":
        return {str(r["id"]): str(r["label"]) for r in repo.all_subclusters()}
    return {str(r["cluster_id"]): str(r["label"]) for r in repo.list_clusters()}
