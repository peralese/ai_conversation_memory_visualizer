from __future__ import annotations

from typing import Any

from src.storage.repository import SQLiteRepository

SOURCES = ("CHATGPT", "CLAUDE", "GEMINI")


class ModelSpecializationService:
    def __init__(self, repo: SQLiteRepository):
        self.repo = repo

    def compute(self, level: str = "cluster") -> dict[str, Any]:
        level = (level or "cluster").lower()
        baseline_counts = self.repo.dataset_source_counts()
        total_messages = sum(baseline_counts.values())
        baseline_available = {s: int(baseline_counts.get(s, 0)) > 0 for s in SOURCES}

        baseline = {
            s: {
                "count": int(baseline_counts.get(s, 0)),
                "pct": round((int(baseline_counts.get(s, 0)) / max(1, total_messages)) * 100.0, 2),
                "p": float(int(baseline_counts.get(s, 0)) / max(1, total_messages)),
            }
            for s in SOURCES
        }

        entities = []
        if level == "subcluster":
            for row in self.repo.all_subclusters():
                sid = int(row["id"])
                message_count = int(row["message_count"] or 0)
                source_counts = self.repo.subcluster_source_counts(sid)
                entities.append(
                    self._entity_payload(
                        entity_id=str(sid),
                        label=str(row["label"]),
                        message_count=message_count,
                        source_counts=source_counts,
                        baseline=baseline,
                        baseline_available=baseline_available,
                        dataset_total=total_messages,
                    )
                )
        else:
            for row in self.repo.list_clusters():
                cid = int(row["cluster_id"])
                message_count = int(row["message_count"] or 0)
                source_counts = self.repo.cluster_source_counts(cid)
                payload = self._entity_payload(
                    entity_id=str(cid),
                    label=str(row["label"]),
                    message_count=message_count,
                    source_counts=source_counts,
                    baseline=baseline,
                    baseline_available=baseline_available,
                    dataset_total=total_messages,
                )
                payload["cluster_id"] = cid
                entities.append(payload)

        entities.sort(key=lambda r: (float(r["dominant_lift"]) if r["dominant_lift"] is not None else -1.0), reverse=True)
        return {
            "level": level,
            "baseline": baseline,
            "baseline_available": baseline_available,
            "items": entities,
        }

    def _entity_payload(
        self,
        *,
        entity_id: str,
        label: str,
        message_count: int,
        source_counts: dict[str, int],
        baseline: dict[str, dict[str, float | int]],
        baseline_available: dict[str, bool],
        dataset_total: int,
    ) -> dict[str, Any]:
        percents = {
            s: round((int(source_counts.get(s, 0)) / max(1, message_count)) * 100.0, 2)
            for s in SOURCES
        }
        p_cluster = {s: float(int(source_counts.get(s, 0)) / max(1, message_count)) for s in SOURCES}
        lift: dict[str, float | None] = {}
        for s in SOURCES:
            base_p = float(baseline[s]["p"])
            if base_p <= 0 or not baseline_available.get(s, False):
                lift[s] = None
            else:
                lift[s] = round(p_cluster[s] / base_p, 4)

        valid_sources = [s for s in SOURCES if lift[s] is not None]
        dominant_source: str | None = None
        dominant_lift: float | None = None
        if valid_sources:
            dominant_source = max(valid_sources, key=lambda s: float(lift[s] or -1.0))
            dominant_lift = float(lift[dominant_source]) if dominant_source is not None else None

        return {
            "id": entity_id,
            "label": label,
            "message_count": message_count,
            "dataset_percentage": round((message_count / max(1, dataset_total)) * 100.0, 2),
            "source_breakdown": {
                "counts": {s: int(source_counts.get(s, 0)) for s in SOURCES},
                "percents": percents,
            },
            "lift_by_source": {s: float(lift[s]) for s in SOURCES if lift[s] is not None},
            "dominant_source": dominant_source,
            "dominant_lift": dominant_lift,
        }
