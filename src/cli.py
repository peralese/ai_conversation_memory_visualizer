from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.clustering.service import ClusteringService
from src.embeddings.service import EmbeddingService
from src.metrics.drift_service import DriftService
from src.metrics.service import MetricsService
from src.pipeline import import_file
from src.reports.generator import CognitiveSummaryReportGenerator
from src.storage.repository import SQLiteRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Conversation Memory Visualizer CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import", help="Import conversation export")
    p_import.add_argument("file_path")

    p_embed = sub.add_parser("embed", help="Generate embeddings")
    p_embed.add_argument("--since", default=None)
    p_embed.add_argument("--redact", action="store_true")

    p_cluster = sub.add_parser("cluster", help="Cluster embedded messages")
    p_cluster.add_argument("--k", type=int, default=None)

    p_profile = sub.add_parser("profile", help="Profile dataset token/domain skew")
    p_profile.add_argument("--top-n", type=int, default=30)

    p_drift = sub.add_parser("drift", help="Compute and persist drift metrics")
    p_drift.add_argument("--level", choices=["cluster", "subcluster"], default="cluster")

    p_report = sub.add_parser("report", help="Generate cognitive summary report")
    p_report.add_argument("--format", choices=["json", "md"], default="md")
    p_report.add_argument("--out", default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo = SQLiteRepository("data/memory_viz.db")

    if args.command == "import":
        result = import_file(args.file_path, repo, output_dir="data/normalized")
        print(result)
        return

    if args.command == "embed":
        service = EmbeddingService(repo)
        count = service.embed_since(since=args.since, redact=args.redact)
        print({"embedded": count})
        return

    if args.command == "cluster":
        service = ClusteringService(repo)
        result = service.cluster_embeddings(k=args.k)
        print(result)
        return

    if args.command == "profile":
        service = MetricsService(repo)
        print(service.dataset_profile(top_n=args.top_n))
        return

    if args.command == "drift":
        service = DriftService(repo)
        print(service.compute_and_persist(level=args.level))
        return

    if args.command == "report":
        service = CognitiveSummaryReportGenerator(repo)
        payload = service.generate_json_report()
        if args.format == "md":
            content = service.generate_markdown_report(payload)
        else:
            content = json.dumps(payload, indent=2)

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            print({"written": str(out_path), "format": args.format})
        else:
            print(content)
        return


if __name__ == "__main__":
    main()
