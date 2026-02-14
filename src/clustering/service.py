from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import sqrt
import random
from typing import Any

from src.analysis.domain import DOMAIN_STOPWORDS
from src.analysis.text import build_analysis_text
from src.models import Cluster
from src.storage.repository import SQLiteRepository

LARGE_CLUSTER_THRESHOLD_PERCENT = 10.0
LARGE_CLUSTER_THRESHOLD_MESSAGES = 100
SUBCLUSTER_MIN_SIZE = 3


class ClusteringService:
    def __init__(self, repo: SQLiteRepository):
        self.repo = repo

    def cluster_embeddings(self, k: int | None = None) -> dict[str, int]:
        rows = self.repo.load_embeddings()
        if len(rows) < 2:
            return {"clusters": 0, "members": 0}

        vectors = [list(map(float, r["vector"])) for r in rows]
        n = len(rows)
        if k is None:
            k = max(2, min(8, int(n**0.5)))

        try:
            from sklearn.cluster import KMeans  # type: ignore
            import numpy as np  # type: ignore

            matrix = np.array(vectors, dtype=float)
            model = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = model.fit_predict(matrix)
            centroids = [center.tolist() for center in model.cluster_centers_]
        except Exception:
            labels, centroids = _fallback_cluster(vectors, k)

        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row, label in zip(rows, labels):
            grouped[int(label)].append(row)

        clusters: list[Cluster] = []
        memberships: list[tuple[int, str]] = []
        topic_events: list[tuple[int, str, str, str]] = []
        label_terms_by_cluster: dict[int, list[str]] = {}

        for label, members in grouped.items():
            texts = [str(m.get("analysis_text") or build_analysis_text(str(m["original_text"]))) for m in members]
            label_terms_by_cluster[label] = _top_keywords_terms(texts, exclude_domain_stopwords=True)
            member_ids = [m["item_id"] for m in members]
            centroid = centroids[label] if label < len(centroids) else None

            clusters.append(
                Cluster(
                    cluster_id=label,
                    label="",
                    member_ids=member_ids,
                    centroid=centroid,
                    created_at=datetime.now(timezone.utc),
                )
            )

            for member in members:
                memberships.append((label, member["item_id"]))
                topic_events.append((label, member["timestamp"], member["conversation_id"], member["item_id"]))

        _apply_unique_cluster_labels(clusters, label_terms_by_cluster)
        self.repo.replace_clusters(clusters, memberships, topic_events)
        self.repo.clear_all_subclusters()
        return {"clusters": len(clusters), "members": len(memberships)}

    def list_clusters(self, *, exclude_domain_stopwords: bool = True, include_subclusters: bool = False) -> list[dict[str, Any]]:
        clusters = self.repo.list_clusters()
        records_by_cluster = self._records_by_cluster()

        for cluster in clusters:
            cid = int(cluster["cluster_id"])
            records = records_by_cluster.get(cid, [])
            terms = _top_keywords_terms(
                [r["analysis_text"] for r in records],
                exclude_domain_stopwords=exclude_domain_stopwords,
            )
            if terms:
                cluster["label"] = ", ".join(terms[:3])
            cluster["source_breakdown"] = _source_breakdown(records)
            stats = self.repo.cluster_context_stats(cid)
            cluster["dataset_percentage"] = stats["dataset_percentage"]
            cluster["average_message_length"] = stats["average_message_length"]
            cluster["conversations_count"] = stats["conversations_count"]

        _ensure_unique_labels_dicts(clusters)

        if include_subclusters:
            for cluster in clusters:
                cid = int(cluster["cluster_id"])
                cluster["subclusters"] = self._get_subclusters_for_parent(
                    cid, exclude_domain_stopwords=exclude_domain_stopwords
                )
        return clusters

    def cluster_detail(
        self,
        cluster_id: int,
        *,
        exclude_domain_stopwords: bool = True,
        include_subclusters: bool = True,
    ) -> dict[str, Any]:
        cluster = self.repo.get_cluster_row(cluster_id)

        records = self.repo.cluster_member_records(cluster_id)
        top_keywords = _top_keywords_terms(
            [r["analysis_text"] for r in records], top_n=8, exclude_domain_stopwords=exclude_domain_stopwords
        )
        if top_keywords:
            cluster["label"] = ", ".join(top_keywords[:3])
        samples = _select_representative_samples(records, cluster.get("centroid"), cluster_id, limit=10)
        source_breakdown = _source_breakdown(records)
        stats = self.repo.cluster_context_stats(cluster_id)

        payload = {
            "cluster_id": int(cluster["cluster_id"]),
            "label": cluster["label"],
            "message_count": stats["message_count"],
            "conversations_count": stats["conversations_count"],
            "dataset_percentage": stats["dataset_percentage"],
            "average_message_length": stats["average_message_length"],
            "top_keywords": top_keywords,
            "first_seen": cluster["first_seen"],
            "last_seen": cluster["last_seen"],
            "sample_messages": samples,
            "source_breakdown": source_breakdown,
        }

        if include_subclusters:
            payload["subclusters"] = self._get_subclusters_for_parent(
                int(cluster_id), exclude_domain_stopwords=exclude_domain_stopwords
            )
        return payload

    def subclusters_for_cluster(
        self,
        cluster_id: int,
        *,
        exclude_domain_stopwords: bool = True,
    ) -> dict[str, Any]:
        cluster = self.repo.get_cluster_row(cluster_id)
        return {
            "cluster_id": int(cluster_id),
            "label": str(cluster["label"]),
            "message_count": int(cluster["message_count"]),
            "subclusters": self._get_subclusters_for_parent(
                int(cluster_id), exclude_domain_stopwords=exclude_domain_stopwords
            ),
        }

    def subcluster_topic_evolution(
        self,
        *,
        source: str | None = None,
        min_messages: int = 1,
        top_n: int = 15,
        exclude_domain_stopwords: bool = True,
    ) -> list[dict[str, Any]]:
        subclusters = {
            int(c["cluster_id"]): self._get_subclusters_for_parent(
                int(c["cluster_id"]),
                exclude_domain_stopwords=exclude_domain_stopwords,
            )
            for c in self.repo.list_clusters()
        }
        message_to_entity: dict[str, tuple[str, str, int]] = {}
        for parent_id, subs in subclusters.items():
            for sub in subs:
                entity_id = str(sub["id"])
                label = str(sub["label"])
                for msg_id in sub.get("message_ids", []):
                    message_to_entity[str(msg_id)] = (entity_id, label, parent_id)

        cluster_labels = {
            int(c["cluster_id"]): str(c["label"]) for c in self.list_clusters(exclude_domain_stopwords=exclude_domain_stopwords)
        }

        rollup: dict[tuple[str, str], int] = defaultdict(int)
        totals: dict[str, int] = defaultdict(int)
        labels: dict[str, str] = {}

        for event in self.repo.topic_events(source=source):
            cluster_id = int(event["cluster_id"])
            msg_id = str(event["message_id"])
            mapped = message_to_entity.get(msg_id)
            if mapped is not None:
                entity_id, label, _parent = mapped
            else:
                entity_id = str(cluster_id)
                label = cluster_labels.get(cluster_id, f"#{cluster_id}")

            ts = datetime.fromisoformat(event["timestamp"])
            week_start = (ts - timedelta(days=ts.weekday())).date().isoformat()
            rollup[(entity_id, week_start)] += 1
            totals[entity_id] += 1
            labels[entity_id] = label

        eligible = [eid for eid, total in totals.items() if total >= max(1, min_messages)]
        eligible.sort(key=lambda eid: totals[eid], reverse=True)
        top_entities = set(eligible[: max(1, top_n)])

        out: list[dict[str, Any]] = []
        for (entity_id, week_start), count in sorted(rollup.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if entity_id not in top_entities:
                continue
            dt = datetime.fromisoformat(week_start)
            wy, wn, _wd = dt.isocalendar()
            out.append(
                {
                    "cluster_id": entity_id,
                    "label": labels.get(entity_id, entity_id),
                    "bucket": f"{wy}-W{wn:02d}",
                    "week_start": week_start,
                    "count": count,
                    "total_cluster_messages": totals.get(entity_id, 0),
                }
            )
        return out

    def _records_by_cluster(self) -> dict[int, list[dict[str, Any]]]:
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in self.repo.all_cluster_member_records():
            grouped[int(row["cluster_id"])].append(row)
        return grouped

    def _get_subclusters_for_parent(
        self,
        parent_cluster_id: int,
        *,
        exclude_domain_stopwords: bool = True,
    ) -> list[dict[str, Any]]:
        self._ensure_subclusters(parent_cluster_id, exclude_domain_stopwords=exclude_domain_stopwords)
        rows = self.repo.list_subclusters(parent_cluster_id)
        parent_stats = self.repo.cluster_context_stats(parent_cluster_id)
        parent_count = max(1, int(parent_stats["message_count"]))

        out = []
        for row in rows:
            sid = int(row["id"])
            samples = self.repo.subcluster_member_records(sid, limit=3)
            out.append(
                {
                    "id": sid,
                    "parent_cluster_id": int(row["parent_cluster_id"]),
                    "label": str(row["label"]),
                    "message_count": int(row["message_count"] or 0),
                    "conversations_count": int(row["conversations_count"] or 0),
                    "dataset_percentage": round((int(row["message_count"] or 0) / parent_count) * 100.0, 2),
                    "first_seen": row["first_seen"],
                    "last_seen": row["last_seen"],
                    "sample_messages": [
                        {
                            "message_id": str(m["id"]),
                            "timestamp": str(m["timestamp"]),
                            "role": str(m["speaker_role"]),
                            "source": str(m["source"]),
                            "conversation_id": str(m["conversation_id"]),
                            "conversation_title": str(m["conversation_title"] or ""),
                            "snippet": str(m["original_text"] or "")[:200],
                        }
                        for m in samples
                    ],
                }
            )
        return out

    def _ensure_subclusters(self, parent_cluster_id: int, *, exclude_domain_stopwords: bool = True) -> None:
        if not self._should_subcluster(parent_cluster_id):
            self.repo.clear_subclusters_for_parent(parent_cluster_id)
            return

        if not self.repo.subclusters_stale(parent_cluster_id):
            return

        records = self.repo.cluster_member_records(parent_cluster_id)
        vector_records = [r for r in records if isinstance(r.get("vector"), list) and r.get("vector")]
        if len(vector_records) < max(2 * SUBCLUSTER_MIN_SIZE, 6):
            self.repo.clear_subclusters_for_parent(parent_cluster_id)
            return

        vectors = [[float(x) for x in r["vector"]] for r in vector_records]
        unique_vectors = {tuple(v) for v in vectors}
        k = min(4, max(2, int(sqrt(len(vector_records)))))
        k = min(k, len(unique_vectors))
        if len(vector_records) <= k:
            self.repo.clear_subclusters_for_parent(parent_cluster_id)
            return

        try:
            from sklearn.cluster import KMeans  # type: ignore
            import numpy as np  # type: ignore

            model = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = model.fit_predict(np.array(vectors, dtype=float))
        except Exception:
            labels, _ = _fallback_cluster(vectors, k)

        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record, label in zip(vector_records, labels):
            grouped[int(label)].append(record)

        insert_rows: list[dict[str, Any]] = []
        for sub_label, members in sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True):
            if len(members) < SUBCLUSTER_MIN_SIZE:
                continue
            terms = _top_keywords_terms(
                [str(m["analysis_text"]) for m in members],
                top_n=6,
                exclude_domain_stopwords=exclude_domain_stopwords,
            )
            label = ", ".join(terms[:3]) if terms else f"Subtopic {sub_label}"
            insert_rows.append(
                {
                    "label": label,
                    "message_ids": [str(m["id"]) for m in members],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        # Ensure uniqueness within parent cluster.
        used: set[str] = set()
        for idx, row in enumerate(insert_rows):
            if row["label"] in used:
                row["label"] = f"{row['label']} ({parent_cluster_id}.{idx})"
            used.add(row["label"])

        if not insert_rows:
            self.repo.clear_subclusters_for_parent(parent_cluster_id)
            return
        self.repo.replace_subclusters_for_parent(parent_cluster_id, insert_rows)

    def _should_subcluster(self, parent_cluster_id: int) -> bool:
        stats = self.repo.cluster_context_stats(parent_cluster_id)
        return (
            int(stats["message_count"]) > LARGE_CLUSTER_THRESHOLD_MESSAGES
            or float(stats["dataset_percentage"]) > LARGE_CLUSTER_THRESHOLD_PERCENT
        )


def _top_keywords_terms(
    texts: list[str], top_n: int = 6, *, exclude_domain_stopwords: bool = True
) -> list[str]:
    if not any(t.strip() for t in texts):
        return []

    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer  # type: ignore
        import numpy as np  # type: ignore

        stopwords = set(ENGLISH_STOP_WORDS).union(
            {"prompted", "please", "help", "generate", "create", "rewrite", "draft"}
        )
        if exclude_domain_stopwords:
            stopwords = stopwords.union(set(DOMAIN_STOPWORDS))
        vec = TfidfVectorizer(stop_words=list(stopwords), max_features=2000, ngram_range=(1, 2))
        matrix = vec.fit_transform(texts)
        scores = np.asarray(matrix.mean(axis=0)).ravel()
        if scores.size == 0:
            return []

        terms = np.array(vec.get_feature_names_out())
        idx = np.argsort(scores)[-top_n:][::-1]
        return [str(terms[i]) for i in idx if scores[i] > 0]
    except Exception:
        words = []
        domain_set = set(DOMAIN_STOPWORDS) if exclude_domain_stopwords else set()
        for text in texts:
            for token in build_analysis_text(text).lower().split():
                token = "".join(ch for ch in token if ch.isalnum())
                if len(token) < 3 or token in domain_set:
                    continue
                words.append(token)
        if not words:
            return []
        counts: dict[str, int] = defaultdict(int)
        for w in words:
            counts[w] += 1
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return [w for w, _ in top]


def _apply_unique_cluster_labels(clusters: list[Cluster], terms_by_cluster: dict[int, list[str]]) -> None:
    used: set[str] = set()
    for cluster in sorted(clusters, key=lambda c: c.cluster_id):
        terms = terms_by_cluster.get(cluster.cluster_id, [])
        base_label = ", ".join(terms[:3]) if terms else "Unlabeled topic"

        label = base_label
        if label in used and terms:
            for extra_idx in range(3, len(terms)):
                candidate = ", ".join(terms[: extra_idx + 1])
                if candidate not in used:
                    label = candidate
                    break
        if label in used:
            label = f"{base_label} (#{cluster.cluster_id})"

        cluster.label = label
        used.add(label)


def _ensure_unique_labels_dicts(clusters: list[dict[str, Any]]) -> None:
    used: set[str] = set()
    for cluster in clusters:
        label = str(cluster.get("label") or "Unlabeled topic")
        if label in used:
            label = f"{label} (#{cluster['cluster_id']})"
            cluster["label"] = label
        used.add(label)


def _fallback_cluster(vectors: list[list[float]], k: int) -> tuple[list[int], list[list[float]]]:
    labels = []
    centroids = []
    step = max(1, (len(vectors) + max(1, k) - 1) // max(1, k))
    for i, _vector in enumerate(vectors):
        label = min(k - 1, i // step)
        labels.append(label)
    for label in range(k):
        members = [v for idx, v in enumerate(vectors) if labels[idx] == label]
        if len(members) == 0:
            centroids.append(vectors[0])
        else:
            dim = len(members[0])
            sums = [0.0] * dim
            for member in members:
                for i in range(dim):
                    sums[i] += member[i]
            centroids.append([s / len(members) for s in sums])
    return labels, centroids


def _source_breakdown(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    sources = ["CHATGPT", "CLAUDE", "GEMINI"]
    counts = {s: 0 for s in sources}
    for r in records:
        source = str(r.get("source") or "")
        if source in counts:
            counts[source] += 1
    total = max(1, len(records))
    percents = {s: round((counts[s] / total) * 100.0, 2) for s in sources}
    return {"counts": counts, "percents": percents}


def _select_representative_samples(
    records: list[dict[str, Any]],
    centroid: list[float] | None,
    cluster_id: int,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if not records:
        return []

    # Keep chronological helpers for fallback.
    by_time = sorted(records, key=lambda r: str(r.get("timestamp") or ""))
    selected_ids: set[str] = set()
    selected: list[dict[str, Any]] = []

    def add_record(rec: dict[str, Any]) -> None:
        rid = str(rec["id"])
        if rid in selected_ids or len(selected) >= limit:
            return
        selected_ids.add(rid)
        selected.append(
            {
                "message_id": rid,
                "timestamp": rec["timestamp"],
                "role": rec["speaker_role"],
                "source": rec["source"],
                "conversation_id": rec["conversation_id"],
                "conversation_title": rec.get("conversation_title") or "",
                "snippet": str(rec.get("original_text") or "")[:220],
            }
        )

    role_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        role_groups[str(r.get("speaker_role") or "")].append(r)

    if centroid and isinstance(centroid, list) and centroid:
        candidates = [r for r in records if isinstance(r.get("vector"), list) and r.get("vector")]
        by_distance = sorted(candidates, key=lambda r: _vector_distance(r["vector"], centroid))
        for role in ("user", "assistant"):
            role_distance = [r for r in by_distance if str(r.get("speaker_role")) == role]
            if role_distance:
                add_record(role_distance[0])
        for r in by_distance:
            if len(selected) >= limit:
                break
            add_record(r)

    # Fallback/coverage: earliest + latest and random fill.
    if by_time:
        add_record(by_time[0])
        add_record(by_time[-1])
    for role in ("user", "assistant"):
        group = sorted(role_groups.get(role, []), key=lambda r: str(r.get("timestamp") or ""))
        if group:
            add_record(group[0])
            add_record(group[-1])

    remaining = [r for r in records if str(r["id"]) not in selected_ids]
    rng = random.Random(cluster_id)
    rng.shuffle(remaining)
    for r in remaining:
        if len(selected) >= limit:
            break
        add_record(r)

    return sorted(selected, key=lambda r: r["timestamp"], reverse=True)[:limit]


def _vector_distance(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return float("inf")
    n = min(len(a), len(b))
    return sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(n)))
