from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from src.clustering.service import ClusteringService
from src.metrics.drift_service import DriftService
from src.metrics.modes_service import ModesService
from src.metrics.service import MetricsService
from src.metrics.specialization_service import ModelSpecializationService
from src.storage.repository import SQLiteRepository


class CognitiveSummaryReportGenerator:
    def __init__(self, repo: SQLiteRepository):
        self.repo = repo
        self.clustering = ClusteringService(repo)
        self.metrics = MetricsService(repo)
        self.specialization = ModelSpecializationService(repo)
        self.drift = DriftService(repo)
        self.modes = ModesService(repo)

    def generate_json_report(self) -> dict[str, Any]:
        counts = self.repo.dataset_counts()
        time_range = self.repo.dataset_time_range()
        source_counts = self.repo.dataset_source_counts()
        avg_len = self.repo.average_message_length()

        total_messages = max(1, counts["messages"])
        source_distribution = {
            src: {
                "count": int(source_counts.get(src, 0)),
                "pct": round((int(source_counts.get(src, 0)) / total_messages) * 100.0, 2),
            }
            for src in ("CHATGPT", "CLAUDE", "GEMINI")
        }

        clusters = self.clustering.list_clusters(
            exclude_domain_stopwords=True,
            include_subclusters=False,
            use_semantic_labels=True,
            show_legacy_labels=False,
        )
        report_label_by_cluster = {
            int(c["cluster_id"]): _report_label(str(c.get("label") or ""), str(c.get("legacy_label") or ""))
            for c in clusters
        }
        top_clusters = sorted(clusters, key=lambda c: float(c.get("dataset_percentage", 0)), reverse=True)[:10]

        specialization = self.specialization.compute(level="cluster")
        spec_by_cluster = {int(item.get("cluster_id") or item.get("id")): item for item in specialization["items"]}

        drift_summary = self.drift.summary(
            level="cluster",
            label_by_entity={str(cid): label for cid, label in report_label_by_cluster.items()},
        )
        drift_by_cluster = {int(item["cluster_id"]): item for item in drift_summary if str(item.get("cluster_id", "")).isdigit()}

        half_life = self.metrics.idea_half_life(label_by_cluster=report_label_by_cluster)
        half_life_by_cluster = {int(item["cluster_id"]): item for item in half_life}

        cluster_modes = []
        for c in top_clusters:
            cid = int(c["cluster_id"])
            detail = self.clustering.cluster_detail(cid, include_subclusters=False)
            spec = spec_by_cluster.get(cid, {})
            drift = drift_by_cluster.get(cid, {})
            hl = half_life_by_cluster.get(cid, {})
            cluster_modes.append(
                {
                    "cluster_id": cid,
                    "label": str(c["label"]),
                    "dataset_percentage": float(c.get("dataset_percentage", 0.0)),
                    "message_count": int(c.get("message_count", 0)),
                    "conversations_count": int(c.get("conversations_count", 0)),
                    "average_message_length": int(detail.get("average_message_length", 0)),
                    "half_life_weeks": hl.get("half_life_weeks"),
                    "dominant_source": spec.get("dominant_source"),
                    "dominant_lift": spec.get("dominant_lift"),
                    "drift_cumulative": drift.get("cumulative_drift"),
                    "drift_volatility": drift.get("volatility"),
                    "description": _describe_cluster(
                        str(c["label"]),
                        list(detail.get("top_keywords") or []),
                        [str(s.get("snippet") or "") for s in (detail.get("sample_messages") or [])[:2]],
                    ),
                }
            )

        specialization_highlights = _specialization_highlights(
            specialization["items"],
            specialization.get("baseline_available", {}),
        )
        specialization_notes = [
            f"Source {src} has no data and is excluded from specialization calculations."
            for src, available in specialization.get("baseline_available", {}).items()
            if not available
        ]
        drift_insights = _drift_insights(drift_summary)
        half_life_insights = _half_life_insights(half_life)
        timeline_summary = _timeline_summary(self.repo.topic_events())
        modes_metrics = self.modes.metrics(level="cluster")
        modes_by_source = self.modes.by_source(level="cluster")
        top_by_mode = _top_clusters_by_mode(modes_metrics.get("per_entity_mode_weights", []))

        return {
            "report_version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "overview": {
                "total_conversations": counts["conversations"],
                "total_messages": counts["messages"],
                "date_range": time_range,
                "source_distribution": source_distribution,
                "average_message_length": avg_len,
            },
            "top_cognitive_modes": cluster_modes,
            "model_specialization_highlights": specialization_highlights,
            "model_specialization_notes": specialization_notes,
            "overall_mode_distribution": modes_metrics.get("overall_mode_distribution", {}),
            "top_clusters_by_dominant_mode": top_by_mode,
            "mode_distribution_by_source": modes_by_source,
            "drift_insights": drift_insights,
            "half_life_insights": half_life_insights,
            "timeline_summary": timeline_summary,
        }

    def generate_markdown_report(self, report: dict[str, Any] | None = None) -> str:
        report = report or self.generate_json_report()
        lines: list[str] = []
        lines.append("# Cognitive Summary Report")
        lines.append("")
        lines.append(f"- Version: `{report['report_version']}`")
        lines.append(f"- Generated: `{report['generated_at']}`")
        lines.append("")

        ov = report["overview"]
        lines.append("## 1) Dataset Overview")
        lines.append(f"- Total conversations: **{ov['total_conversations']}**")
        lines.append(f"- Total messages: **{ov['total_messages']}**")
        lines.append(
            f"- Date range: **{ov['date_range']['first_message_at'] or 'N/A'}** -> **{ov['date_range']['last_message_at'] or 'N/A'}**"
        )
        lines.append(f"- Avg message length: **{ov['average_message_length']}** chars")
        lines.append("- Source distribution:")
        for src, val in ov["source_distribution"].items():
            lines.append(f"  - {src}: {val['count']} ({val['pct']}%)")
        lines.append("")

        lines.append("## 2) Top Cognitive Modes")
        for row in report["top_cognitive_modes"]:
            lines.append(
                f"- **#{row['cluster_id']} {row['label']}**: {row['dataset_percentage']}% dataset, {row['message_count']} msgs, {row['conversations_count']} convs, avg len {row['average_message_length']}, half-life {row['half_life_weeks'] if row['half_life_weeks'] is not None else 'insufficient data'}, dominant {row['dominant_source']} x{row['dominant_lift'] if row['dominant_lift'] is not None else 0}, drift(cum={row['drift_cumulative']}, vol={row['drift_volatility']})."
            )
            lines.append(f"  - {row['description']}")
        lines.append("")

        lines.append("## 3) Model Specialization Highlights")
        for note in report.get("model_specialization_notes", []):
            lines.append(f"- Note: {note}")
        for source, payload in report["model_specialization_highlights"].items():
            lines.append(f"- **{source}**: {payload['interpretation']}")
            for row in payload["top_clusters"]:
                lines.append(
                    f"  - #{row['cluster_id']} {row['label']} (lift {row['lift']}, share {row['share_pct']}%)"
                )
        lines.append("")

        lines.append("## Cognitive Mode Fingerprint")
        for mode_id, weight in report.get("overall_mode_distribution", {}).items():
            lines.append(f"- {mode_id}: {round(float(weight) * 100.0, 2)}%")
        lines.append("")

        lines.append("## Top Clusters by Dominant Mode")
        for mode_id, rows in report.get("top_clusters_by_dominant_mode", {}).items():
            lines.append(f"- **{mode_id}**")
            for row in rows:
                lines.append(f"  - #{row['entity_id']} {row['label']} (weight {row['dominant_weight']})")
        lines.append("")

        lines.append("## Mode Distribution by Source")
        by_source = report.get("mode_distribution_by_source", {})
        for source, available in by_source.get("baseline_available", {}).items():
            if not available:
                lines.append(f"- {source}: N/A (no baseline data)")
                continue
            dist = (by_source.get("mode_distribution_by_source") or {}).get(source) or {}
            lines.append(f"- {source}:")
            for mode_id, val in dist.items():
                lines.append(f"  - {mode_id}: {round(float(val) * 100.0, 2)}%")
        lines.append("")

        lines.append("## 4) Evolving vs Stable Topics (Drift)")
        lines.append("- Most evolving:")
        for row in report["drift_insights"]["most_evolving"]:
            lines.append(
                f"  - #{row['cluster_id']} {row['label']} (cumulative {row['cumulative_drift']}, volatility {row['volatility']}, buckets {row['active_buckets_count']})"
            )
        lines.append("- Most stable:")
        for row in report["drift_insights"]["most_stable"]:
            lines.append(
                f"  - #{row['cluster_id']} {row['label']} (cumulative {row['cumulative_drift']}, volatility {row['volatility']}, buckets {row['active_buckets_count']})"
            )
        if report["drift_insights"]["low_volume_warning"]:
            lines.append(f"- Warning: {report['drift_insights']['low_volume_warning']}")
        lines.append("")

        lines.append("## 5) Idea Half-Life Insights")
        lines.append("- Shortest half-life:")
        for row in report["half_life_insights"]["shortest"]:
            lines.append(f"  - #{row['cluster_id']} {row['label']} ({row['half_life_weeks']} weeks)")
        lines.append("- Longest half-life:")
        for row in report["half_life_insights"]["longest"]:
            lines.append(f"  - #{row['cluster_id']} {row['label']} ({row['half_life_weeks']} weeks)")
        if report["half_life_insights"]["insufficient_data_count"] > 0:
            lines.append(
                f"- Caveat: {report['half_life_insights']['insufficient_data_count']} clusters have insufficient data for half-life."
            )
        lines.append("")

        lines.append("## 6) Timeline Summary")
        ts = report["timeline_summary"]
        lines.append("- Most active weeks:")
        for row in ts["most_active_weeks"]:
            lines.append(f"  - {row['bucket']}: {row['count']} messages")
        lines.append("- Most active months:")
        for row in ts["most_active_months"]:
            lines.append(f"  - {row['bucket']}: {row['count']} messages")
        lines.append(f"- Seasonality note: {ts['seasonality_note']}")
        lines.append("")
        return "\n".join(lines)


def _describe_cluster(label: str, keywords: list[str], snippets: list[str]) -> str:
    keyword_text = ", ".join(keywords[:3]) if keywords else label
    snippets = [s[:120] for s in snippets if s]
    if len(snippets) >= 2:
        return (
            f"This cluster centers on {label}. Frequent themes include {keyword_text}. "
            f"Representative snippets include \"{snippets[0]}\" and \"{snippets[1]}\"."
        )
    if len(snippets) == 1:
        return (
            f"This cluster centers on {label}, with recurring themes around {keyword_text}. "
            f"Representative snippet: \"{snippets[0]}\"."
        )
    return f"This cluster centers on {label}, with recurring themes around {keyword_text}."


def _specialization_highlights(items: list[dict[str, Any]], baseline_available: dict[str, bool]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for source in ("CHATGPT", "CLAUDE", "GEMINI"):
        if not baseline_available.get(source, False):
            continue
        rows = sorted(
            items,
            key=lambda r: float(r.get("lift_by_source", {}).get(source) or 0.0),
            reverse=True,
        )[:5]
        top_rows = [
            {
                "cluster_id": int(r.get("cluster_id") or r.get("id")),
                "label": r["label"],
                "lift": float(r.get("lift_by_source", {}).get(source) or 0.0),
                "share_pct": float(r.get("source_breakdown", {}).get("percents", {}).get(source) or 0.0),
            }
            for r in rows
        ]
        if top_rows:
            first = top_rows[0]
            interpretation = (
                f"{source} is overrepresented in #{first['cluster_id']} ({first['label']}) with lift {first['lift']:.2f}, "
                f"suggesting stronger usage for that cognitive mode."
            )
        else:
            interpretation = f"{source} has no strong overrepresentation signals."
        out[source] = {"top_clusters": top_rows, "interpretation": interpretation}
    return out


def _drift_insights(rows: list[dict[str, Any]]) -> dict[str, Any]:
    most_evolving = rows[:5]
    stable_candidates = sorted(rows, key=lambda r: (float(r.get("volatility", 0.0)), -int(r.get("active_buckets_count", 0))))
    most_stable = stable_candidates[:5]
    low_volume = [r for r in rows if int(r.get("active_buckets_count", 0)) < 3]
    warning = None
    if low_volume:
        warning = f"{len(low_volume)} clusters have fewer than 3 active buckets; drift reliability may be low."
    return {
        "most_evolving": most_evolving,
        "most_stable": most_stable,
        "low_volume_warning": warning,
    }


def _half_life_insights(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r.get("half_life_weeks") is not None]
    valid_sorted = sorted(valid, key=lambda r: int(r["half_life_weeks"]))
    return {
        "shortest": valid_sorted[:5],
        "longest": list(reversed(valid_sorted[-5:])),
        "insufficient_data_count": len(rows) - len(valid),
    }


def _timeline_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    weekly: dict[str, int] = defaultdict(int)
    monthly: dict[str, int] = defaultdict(int)
    for e in events:
        ts = datetime.fromisoformat(str(e["timestamp"]))
        week_start = (ts - timedelta(days=ts.weekday())).date().isoformat()
        month = f"{ts.year}-{ts.month:02d}"
        weekly[week_start] += 1
        monthly[month] += 1

    most_active_weeks = [
        {"bucket": k, "count": v}
        for k, v in sorted(weekly.items(), key=lambda kv: kv[1], reverse=True)[:5]
    ]
    most_active_months = [
        {"bucket": k, "count": v}
        for k, v in sorted(monthly.items(), key=lambda kv: kv[1], reverse=True)[:5]
    ]
    note = "Activity is fairly steady over time."
    if weekly:
        avg = sum(weekly.values()) / len(weekly)
        peak = max(weekly.values())
        if peak >= avg * 1.5:
            note = "Activity shows clear spikes in specific weeks."
    return {
        "most_active_weeks": most_active_weeks,
        "most_active_months": most_active_months,
        "seasonality_note": note,
    }


def _top_clusters_by_mode(rows: list[dict[str, Any]], top_n: int = 5) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        mode = row.get("dominant_mode")
        if mode:
            grouped[str(mode)].append(row)
    out = {}
    for mode, items in grouped.items():
        out[mode] = sorted(items, key=lambda r: float(r.get("dominant_weight") or 0.0), reverse=True)[:top_n]
    return out


def _report_label(semantic_label: str, legacy_label: str) -> str:
    semantic_label = str(semantic_label or "").strip()
    legacy_label = str(legacy_label or "").strip()
    if semantic_label and legacy_label and semantic_label.lower() != legacy_label.lower():
        return f"{semantic_label} (legacy: {legacy_label})"
    return semantic_label or legacy_label or "Unlabeled topic"
