from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from src.modes.scoring import score_mode_weights
from src.modes.taxonomy import MODE_TAXONOMY
from src.storage.repository import SQLiteRepository


class ModesService:
    def __init__(self, repo: SQLiteRepository):
        self.repo = repo

    def compute_and_persist(self, level: str = "cluster") -> dict[str, Any]:
        level = (level or "cluster").lower()
        if level not in {"cluster", "subcluster"}:
            raise ValueError("level must be cluster or subcluster")

        entities = self._entity_rows(level)
        rows = []
        now = datetime.now(timezone.utc).isoformat()
        for entity in entities:
            eid = str(entity["entity_id"])
            snippets = [str(s.get("snippet") or "") for s in entity.get("sample_messages", [])[:5]]
            scored = score_mode_weights(
                top_keywords=[str(k) for k in entity.get("top_keywords", [])],
                label=str(entity.get("label") or ""),
                sample_snippets=snippets,
            )
            rows.append(
                {
                    "entity_id": eid,
                    "mode_weights": scored["mode_weights"],
                    "dominant_mode": scored["dominant_mode"],
                    "dominant_weight": scored["dominant_weight"],
                    "created_at": now,
                }
            )
        self.repo.replace_mode_scores(level, rows)
        return {"level": level, "rows": len(rows)}

    def ensure_fresh(self, level: str = "cluster") -> None:
        level = (level or "cluster").lower()
        if self.repo.mode_scores_stale(level):
            self.compute_and_persist(level)

    def metrics(self, level: str = "cluster") -> dict[str, Any]:
        level = (level or "cluster").lower()
        self.ensure_fresh(level)

        scores = {r["entity_id"]: r for r in self.repo.mode_scores(level)}
        entities = self._entity_rows(level)
        total_messages = max(1, sum(int(e["message_count"]) for e in entities))

        overall = {m.id: 0.0 for m in MODE_TAXONOMY}
        per_entity = []
        for e in entities:
            eid = str(e["entity_id"])
            score = scores.get(eid)
            if not score:
                continue
            message_count = int(e["message_count"])
            dataset_pct = round((message_count / total_messages) * 100.0, 2)
            mode_weights = {k: float(v) for k, v in score["mode_weights"].items()}
            for mid, w in mode_weights.items():
                overall[mid] += (message_count / total_messages) * float(w)
            per_entity.append(
                {
                    "entity_id": eid,
                    "label": e["label"],
                    "message_count": message_count,
                    "dataset_percentage": dataset_pct,
                    "dominant_mode": score["dominant_mode"],
                    "dominant_weight": score["dominant_weight"],
                    "mode_weights": mode_weights,
                }
            )

        overall = {k: round(v, 4) for k, v in overall.items()}
        return {
            "level": level,
            "taxonomy": [{"id": m.id, "name": m.name, "description": m.description} for m in MODE_TAXONOMY],
            "overall_mode_distribution": overall,
            "per_entity_mode_weights": per_entity,
        }

    def by_source(self, level: str = "cluster") -> dict[str, Any]:
        level = (level or "cluster").lower()
        self.ensure_fresh(level)
        scores = {r["entity_id"]: r for r in self.repo.mode_scores(level)}
        entities = self._entity_rows(level)

        sources = ("CHATGPT", "CLAUDE", "GEMINI")
        totals_by_source = {s: 0 for s in sources}
        by_source = {s: {m.id: 0.0 for m in MODE_TAXONOMY} for s in sources}

        for e in entities:
            eid = str(e["entity_id"])
            score = scores.get(eid)
            if not score:
                continue
            source_counts = e.get("source_counts", {})
            for s in sources:
                totals_by_source[s] += int(source_counts.get(s, 0))

        for e in entities:
            eid = str(e["entity_id"])
            score = scores.get(eid)
            if not score:
                continue
            source_counts = e.get("source_counts", {})
            weights = score["mode_weights"]
            for s in sources:
                n_s = totals_by_source[s]
                if n_s <= 0:
                    continue
                contribution = int(source_counts.get(s, 0)) / n_s
                for mode_id, w in weights.items():
                    by_source[s][mode_id] += contribution * float(w)

        baseline_available = {s: totals_by_source[s] > 0 for s in sources}
        out = {}
        for s in sources:
            if not baseline_available[s]:
                out[s] = None
            else:
                out[s] = {k: round(v, 4) for k, v in by_source[s].items()}
        return {
            "level": level,
            "baseline_available": baseline_available,
            "mode_distribution_by_source": out,
        }

    def timeline(self, level: str = "cluster", bucket: str = "week") -> dict[str, Any]:
        level = (level or "cluster").lower()
        bucket = (bucket or "week").lower()
        self.ensure_fresh(level)
        scores = {r["entity_id"]: r for r in self.repo.mode_scores(level)}
        points = self.repo.mode_timeline_points(level)

        grouped_counts: dict[tuple[str, str], int] = defaultdict(int)
        for point in points:
            ts = datetime.fromisoformat(point["timestamp"])
            if bucket == "month":
                b = f"{ts.year}-{ts.month:02d}-01"
            else:
                b = (ts - timedelta(days=ts.weekday())).date().isoformat()
            grouped_counts[(b, str(point["entity_id"]))] += 1

        per_bucket: dict[str, dict[str, float]] = defaultdict(lambda: {m.id: 0.0 for m in MODE_TAXONOMY})
        bucket_total_messages: dict[str, int] = defaultdict(int)
        for (b, entity_id), count in grouped_counts.items():
            score = scores.get(entity_id)
            if not score:
                continue
            weights = score["mode_weights"]
            bucket_total_messages[b] += count
            for mode_id, w in weights.items():
                per_bucket[b][mode_id] += count * float(w)

        rows = []
        for b in sorted(per_bucket.keys()):
            total = max(1, bucket_total_messages[b])
            distribution = {mid: round(val / total, 4) for mid, val in per_bucket[b].items()}
            rows.append({"bucket_start": b, "message_count": bucket_total_messages[b], "mode_distribution": distribution})
        return {"level": level, "bucket": bucket, "rows": rows}

    def _entity_rows(self, level: str) -> list[dict[str, Any]]:
        if level == "subcluster":
            out = []
            for row in self.repo.all_subclusters():
                sid = int(row["id"])
                members = self.repo.subcluster_member_records(sid, limit=5)
                out.append(
                    {
                        "entity_id": str(sid),
                        "label": str(row["label"]),
                        "message_count": int(row["message_count"] or 0),
                        "top_keywords": self._top_keywords([str(m.get("original_text") or "") for m in members]),
                        "sample_messages": [{"snippet": str(m.get("original_text") or "")[:200]} for m in members],
                        "source_counts": self.repo.subcluster_source_counts(sid),
                    }
                )
            return out

        out = []
        for row in self.repo.list_clusters():
            cid = int(row["cluster_id"])
            members = self.repo.cluster_member_records(cid, limit=5)
            out.append(
                {
                    "entity_id": str(cid),
                    "label": str(row["label"]),
                    "message_count": int(row["message_count"] or 0),
                    "top_keywords": self._top_keywords([str(m.get("analysis_text") or "") for m in members]),
                    "sample_messages": [{"snippet": str(m.get("original_text") or "")[:200]} for m in members],
                    "source_counts": self.repo.cluster_source_counts(cid),
                }
            )
        return out

    def _top_keywords(self, texts: list[str], top_n: int = 6) -> list[str]:
        if not any(t.strip() for t in texts):
            return []
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            import numpy as np  # type: ignore

            vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=1000)
            mat = vec.fit_transform(texts)
            scores = np.asarray(mat.mean(axis=0)).ravel()
            if scores.size == 0:
                return []
            terms = np.array(vec.get_feature_names_out())
            idx = np.argsort(scores)[-top_n:][::-1]
            return [str(terms[i]) for i in idx if scores[i] > 0]
        except Exception:
            return []
