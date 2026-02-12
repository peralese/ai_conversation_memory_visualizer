from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from src.models import Cluster
from src.storage.repository import SQLiteRepository


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

            # KMeans chosen for MVP simplicity and deterministic operation.
            matrix = np.array(vectors, dtype=float)
            model = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = model.fit_predict(matrix)
            centroids = [center.tolist() for center in model.cluster_centers_]
        except Exception:
            labels, centroids = _fallback_cluster(vectors, k)

        grouped: dict[int, list[dict]] = defaultdict(list)
        for row, label in zip(rows, labels):
            grouped[int(label)].append(row)

        clusters: list[Cluster] = []
        memberships: list[tuple[int, str]] = []
        topic_events: list[tuple[int, str, str, str]] = []

        for label, members in grouped.items():
            texts = [str(m["original_text"]) for m in members]
            cluster_label = _top_keywords_label(texts)
            member_ids = [m["item_id"] for m in members]
            centroid = centroids[label] if label < len(centroids) else None

            clusters.append(
                Cluster(
                    cluster_id=label,
                    label=cluster_label,
                    member_ids=member_ids,
                    centroid=centroid,
                    created_at=datetime.now(timezone.utc),
                )
            )

            for member in members:
                memberships.append((label, member["item_id"]))
                topic_events.append(
                    (
                        label,
                        member["timestamp"],
                        member["conversation_id"],
                        member["item_id"],
                    )
                )

        self.repo.replace_clusters(clusters, memberships, topic_events)
        return {"clusters": len(clusters), "members": len(memberships)}


def _top_keywords_label(texts: list[str], top_n: int = 3) -> str:
    if not any(t.strip() for t in texts):
        return "Unlabeled topic"

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        import numpy as np  # type: ignore

        vec = TfidfVectorizer(stop_words="english", max_features=1000)
        matrix = vec.fit_transform(texts)
        scores = np.asarray(matrix.mean(axis=0)).ravel()
        if scores.size == 0:
            return "Unlabeled topic"

        terms = np.array(vec.get_feature_names_out())
        idx = np.argsort(scores)[-top_n:][::-1]
        words = [terms[i] for i in idx if scores[i] > 0]
        return ", ".join(words) if words else "Unlabeled topic"
    except Exception:
        words = []
        for text in texts:
            for token in text.lower().split():
                token = "".join(ch for ch in token if ch.isalnum())
                if len(token) < 4:
                    continue
                words.append(token)
        if not words:
            return "Unlabeled topic"
        counts: dict[str, int] = defaultdict(int)
        for w in words:
            counts[w] += 1
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return ", ".join([w for w, _ in top]) if top else "Unlabeled topic"


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
