from __future__ import annotations

import argparse

from src.clustering.service import ClusteringService
from src.embeddings.service import EmbeddingService
from src.pipeline import import_file
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


if __name__ == "__main__":
    main()
