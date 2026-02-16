from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt
from typing import Any


from src.storage.repository import SQLiteRepository


@dataclass
class ConversationClusteringConfig:
    k: int | None = None
    algo: str = "kmeans"
    force_recluster: bool = False
    representatives_per_cluster: int = 6


def cluster_conversations(repo: SQLiteRepository, config: ConversationClusteringConfig | None = None) -> dict[str, Any]:
    cfg = config or ConversationClusteringConfig()
    rows = repo.list_conversation_embeddings()
    if len(rows) < 2:
        return {"clusters": 0, "members": 0, "cluster_ids": []}

    if cfg.force_recluster:
        repo.clear_conv_clusters()

    vectors = [list(map(float, r["embedding"])) for r in rows]
    n = len(rows)
    k = cfg.k if cfg.k is not None else max(2, min(10, int(sqrt(n))))

    labels, centroids = _fit_cluster(vectors, k)

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row, label in zip(rows, labels):
        grouped[int(label)].append(row)

    created_cluster_ids: list[int] = []
    total_members = 0
    for label, members in sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True):
        centroid = centroids[label] if label < len(centroids) else None
        params = {
            "k": k,
            "cluster_label": int(label),
            "member_count": len(members),
            "centroid_dim": len(centroid or []),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        conv_cluster_id = repo.create_conv_cluster_run(cfg.algo, params_json=_json_dumps(params))
        created_cluster_ids.append(conv_cluster_id)

        ranked = []
        for member in members:
            vec = list(map(float, member["embedding"]))
            dist = _l2_distance(vec, centroid or vec)
            ranked.append((member, dist))
        ranked.sort(key=lambda item: (item[1], str(item[0].get("conversation_id") or "")))

        representatives = {
            str(item[0]["conversation_id"])
            for item in ranked[: max(1, min(cfg.representatives_per_cluster, len(ranked)))]
        }

        repo.upsert_conv_cluster_members(
            conv_cluster_id,
            [
                {
                    "conversation_id": str(member["conversation_id"]),
                    "distance": dist,
                    "is_representative": str(member["conversation_id"]) in representatives,
                }
                for member, dist in ranked
            ],
        )
        total_members += len(ranked)

    return {"clusters": len(created_cluster_ids), "members": total_members, "cluster_ids": created_cluster_ids}


def _fit_cluster(vectors: list[list[float]], k: int) -> tuple[list[int], list[list[float]]]:
    try:
        from sklearn.cluster import KMeans  # type: ignore
        import numpy as np  # type: ignore

        matrix = np.array(vectors, dtype=float)
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = model.fit_predict(matrix)
        centroids = [center.tolist() for center in model.cluster_centers_]
        return [int(x) for x in labels], centroids
    except Exception:
        return _fallback_cluster(vectors, k)


def _fallback_cluster(vectors: list[list[float]], k: int) -> tuple[list[int], list[list[float]]]:
    labels: list[int] = []
    centroids: list[list[float]] = []
    step = max(1, (len(vectors) + max(1, k) - 1) // max(1, k))
    for idx, _vector in enumerate(vectors):
        labels.append(min(k - 1, idx // step))

    for label in range(k):
        members = [v for idx, v in enumerate(vectors) if labels[idx] == label]
        if not members:
            centroids.append(vectors[0])
            continue
        dim = len(members[0])
        sums = [0.0] * dim
        for member in members:
            for j in range(dim):
                sums[j] += member[j]
        centroids.append([s / len(members) for s in sums])
    return labels, centroids


def _l2_distance(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum((float(a[i]) - float(b[i])) ** 2 for i in range(n)) ** 0.5


def _json_dumps(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, sort_keys=True)
