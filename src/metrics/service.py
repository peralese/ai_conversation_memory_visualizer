from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from math import ceil

from src.storage.repository import SQLiteRepository


class MetricsService:
    def __init__(self, repo: SQLiteRepository):
        self.repo = repo

    def recurring_topics(self) -> list[dict]:
        events = self.repo.topic_events()
        buckets: dict[int, set[str]] = defaultdict(set)
        convs: dict[int, set[str]] = defaultdict(set)

        for e in events:
            ts = datetime.fromisoformat(e["timestamp"])
            bucket = f"{ts.isocalendar().year}-W{ts.isocalendar().week:02d}"
            cid = int(e["cluster_id"])
            buckets[cid].add(bucket)
            convs[cid].add(e["conversation_id"])

        out = []
        for cluster in self.repo.list_clusters():
            cid = int(cluster["cluster_id"])
            out.append(
                {
                    "cluster_id": cid,
                    "label": cluster["label"],
                    "time_buckets": len(buckets[cid]),
                    "conversation_count": len(convs[cid]),
                    "is_recurring": len(buckets[cid]) > 1 and len(convs[cid]) > 1,
                }
            )
        return out

    def topic_evolution(self, granularity: str = "week") -> list[dict]:
        events = self.repo.topic_events()
        rollup: dict[tuple[int, str], int] = defaultdict(int)

        for e in events:
            ts = datetime.fromisoformat(e["timestamp"])
            if granularity == "month":
                bucket = f"{ts.year}-{ts.month:02d}"
            else:
                bucket = f"{ts.isocalendar().year}-W{ts.isocalendar().week:02d}"
            rollup[(int(e["cluster_id"]), bucket)] += 1

        return [
            {"cluster_id": cid, "bucket": bucket, "count": count}
            for (cid, bucket), count in sorted(rollup.items(), key=lambda x: x[0][1])
        ]

    def idea_half_life(self) -> list[dict]:
        # Assumption: half-life is elapsed weeks until weekly activity falls below
        # 50% of the cluster's peak weekly volume after first mention.
        events = self.repo.topic_events()
        by_cluster: dict[int, list[datetime]] = defaultdict(list)
        for e in events:
            by_cluster[int(e["cluster_id"])].append(datetime.fromisoformat(e["timestamp"]))

        out: list[dict] = []
        for cluster in self.repo.list_clusters():
            cid = int(cluster["cluster_id"])
            times = sorted(by_cluster[cid])
            if not times:
                continue

            first = times[0]
            weekly_counts: dict[int, int] = defaultdict(int)
            for ts in times:
                week_offset = int((ts - first).days // 7)
                weekly_counts[week_offset] += 1

            peak = max(weekly_counts.values())
            threshold = peak * 0.5
            half_life_weeks = None
            for week in range(0, max(weekly_counts.keys()) + 1):
                if weekly_counts.get(week, 0) < threshold:
                    half_life_weeks = week
                    break

            out.append(
                {
                    "cluster_id": cid,
                    "label": cluster["label"],
                    "first_mention": first.isoformat(),
                    "peak_weekly_volume": peak,
                    "half_life_weeks": half_life_weeks,
                }
            )

        return out
