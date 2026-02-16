#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from src.conversation_pipeline.clustering import ConversationClusteringConfig
from src.conversation_pipeline.embeddings import ConversationEmbeddingConfig
from src.conversation_pipeline.rollup import RollupConfig
from src.conversation_pipeline.service import ConversationPipelineConfig, run_conversation_pipeline
from src.env_loader import load_dotenv
from src.labeling.evidence_packet import EvidenceConfig
from src.labeling.gpt_labeler import GPTLabelerConfig
from src.storage.repository import SQLiteRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run conversation-level clustering and semantic labeling")
    parser.add_argument("--db", default="data/memory_viz.db")
    parser.add_argument("--force-reembed", action="store_true")
    parser.add_argument("--force-recluster", action="store_true")
    parser.add_argument("--force-relabel", action="store_true")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--max-gpt", type=int, default=100)
    parser.add_argument("--min-seconds-between-gpt", type=float, default=1.5)
    parser.add_argument("--dry-run", action="store_true", help="Build packets and heuristic labels only; do not call GPT")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    repo = SQLiteRepository(args.db)

    config = ConversationPipelineConfig(
        rollup=RollupConfig(),
        embeddings=ConversationEmbeddingConfig(force_reembed=args.force_reembed),
        clustering=ConversationClusteringConfig(
            k=args.k,
            force_recluster=args.force_recluster,
            algo="kmeans",
        ),
        evidence=EvidenceConfig(),
        labeling=GPTLabelerConfig(
            min_seconds_between_requests=args.min_seconds_between_gpt,
            max_requests_per_run=args.max_gpt,
            dry_run=args.dry_run,
        ),
        force_relabel=args.force_relabel,
    )

    result = run_conversation_pipeline(repo, config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
