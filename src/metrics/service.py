from __future__ import annotations

from collections import Counter
from collections import defaultdict
from datetime import date, datetime, timedelta

from src.analysis.domain import DOMAIN_SIGNAL_TOKENS
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

    def topic_evolution(
        self,
        granularity: str = "week",
        source: str | None = None,
        min_messages: int = 1,
        top_n: int = 15,
        label_by_cluster: dict[int, str] | None = None,
    ) -> list[dict]:
        events = self.repo.topic_events(source=source)
        rollup: dict[tuple[int, str], int] = defaultdict(int)
        totals: dict[int, int] = defaultdict(int)
        labels = label_by_cluster or {int(c["cluster_id"]): str(c["label"]) for c in self.repo.list_clusters()}

        for e in events:
            ts = datetime.fromisoformat(e["timestamp"])
            if granularity == "month":
                bucket = f"{ts.year}-{ts.month:02d}"
            else:
                week_start_date = _iso_week_start(ts)
                bucket = week_start_date.isoformat()
            rollup[(int(e["cluster_id"]), bucket)] += 1
            totals[int(e["cluster_id"])] += 1

        eligible = [cid for cid, total in totals.items() if total >= max(1, min_messages)]
        eligible.sort(key=lambda cid: totals[cid], reverse=True)
        top_clusters = set(eligible[: max(1, top_n)])

        out = []
        for (cid, bucket), count in sorted(rollup.items(), key=lambda x: (x[0][1], x[0][0])):
            if cid not in top_clusters:
                continue
            if granularity == "month":
                bucket_label = bucket
            else:
                dt = datetime.fromisoformat(bucket)
                week_year, week_num, _weekday = dt.isocalendar()
                bucket_label = f"{week_year}-W{week_num:02d}"
            out.append(
                {
                    "cluster_id": cid,
                    "label": labels.get(cid, f"Cluster {cid}"),
                    "bucket": bucket_label,
                    "week_start": bucket,
                    "count": count,
                    "total_cluster_messages": totals.get(cid, 0),
                }
            )
        return out

    def idea_half_life(self, label_by_cluster: dict[int, str] | None = None) -> list[dict]:
        events = self.repo.topic_events()
        by_cluster: dict[int, dict[date, int]] = defaultdict(lambda: defaultdict(int))
        for e in events:
            ts = datetime.fromisoformat(e["timestamp"])
            by_cluster[int(e["cluster_id"])][_iso_week_start(ts)] += 1

        out: list[dict] = []
        labels = label_by_cluster or {int(c["cluster_id"]): str(c["label"]) for c in self.repo.list_clusters()}
        for cluster in self.repo.list_clusters():
            cid = int(cluster["cluster_id"])
            weekly = by_cluster[cid]
            if not weekly:
                continue

            sorted_weeks = sorted(weekly.items(), key=lambda kv: kv[0])
            counts = [count for _week, count in sorted_weeks]
            first = sorted_weeks[0][0]
            peak = max(counts)

            half_life_weeks = None
            if len(sorted_weeks) > 1:
                peak_idx = counts.index(peak)
                threshold = peak * 0.5
                peak_week = sorted_weeks[peak_idx][0]
                for idx in range(peak_idx + 1, len(sorted_weeks)):
                    week_start, count = sorted_weeks[idx]
                    if count <= threshold:
                        half_life_weeks = int((week_start - peak_week).days // 7)
                        break

            out.append(
                {
                    "cluster_id": cid,
                    "label": labels.get(cid, str(cluster["label"])),
                    "first_mention": datetime.combine(first, datetime.min.time()).isoformat(),
                    "peak_weekly_volume": peak,
                    "half_life_weeks": half_life_weeks,
                }
            )

        return out

    def dataset_profile(self, top_n: int = 30) -> dict:
        rows = self.repo.profile_message_rows()
        counts = self.repo.dataset_counts()

        global_counter: Counter[str] = Counter()
        by_source_counter: dict[str, Counter[str]] = defaultdict(Counter)
        signal_hits_global = {token: 0 for token in DOMAIN_SIGNAL_TOKENS}
        signal_hits_by_source: dict[str, dict[str, int]] = defaultdict(lambda: {token: 0 for token in DOMAIN_SIGNAL_TOKENS})
        messages_by_source: dict[str, int] = defaultdict(int)

        for row in rows:
            source = str(row["source"])
            text = str(row["analysis_text"] or "")
            tokens = [tok for tok in text.split() if tok]
            token_set = set(tokens)

            global_counter.update(tokens)
            by_source_counter[source].update(tokens)
            messages_by_source[source] += 1

            for token in DOMAIN_SIGNAL_TOKENS:
                if token in token_set:
                    signal_hits_global[token] += 1
                    signal_hits_by_source[source][token] += 1

        total_messages = max(1, counts["messages"])
        by_source = {}
        for source in sorted(messages_by_source.keys()):
            source_total = max(1, messages_by_source[source])
            by_source[source] = {
                "messages": messages_by_source[source],
                "top_tokens": [{"token": t, "count": c} for t, c in by_source_counter[source].most_common(top_n)],
                "domain_token_pct": {
                    token: round((signal_hits_by_source[source][token] / source_total) * 100.0, 2)
                    for token in sorted(DOMAIN_SIGNAL_TOKENS)
                },
            }

        return {
            "total_messages": counts["messages"],
            "total_conversations": counts["conversations"],
            "top_tokens": [{"token": t, "count": c} for t, c in global_counter.most_common(top_n)],
            "domain_token_pct": {
                token: round((signal_hits_global[token] / total_messages) * 100.0, 2)
                for token in sorted(DOMAIN_SIGNAL_TOKENS)
            },
            "by_source": by_source,
        }


def _iso_week_start(ts: datetime) -> date:
    return (ts - timedelta(days=ts.weekday())).date()
