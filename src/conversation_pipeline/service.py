from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

from src.conversation_pipeline.clustering import ConversationClusteringConfig, cluster_conversations
from src.conversation_pipeline.embeddings import ConversationEmbeddingConfig, embed_conversations
from src.conversation_pipeline.rollup import RollupConfig, build_conversation_rollups
from src.labeling.evidence_packet import EvidenceConfig, build_cluster_evidence_packet
from src.labeling.gpt_labeler import GPTClusterLabeler, GPTLabelerConfig
from src.storage.repository import SQLiteRepository

logger = logging.getLogger(__name__)


@dataclass
class ConversationPipelineConfig:
    rollup: RollupConfig = field(default_factory=RollupConfig)
    embeddings: ConversationEmbeddingConfig = field(default_factory=ConversationEmbeddingConfig)
    clustering: ConversationClusteringConfig = field(default_factory=ConversationClusteringConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    labeling: GPTLabelerConfig = field(default_factory=GPTLabelerConfig)
    force_relabel: bool = False


def run_conversation_pipeline(repo: SQLiteRepository, config: ConversationPipelineConfig | None = None) -> dict[str, Any]:
    cfg = config or ConversationPipelineConfig()

    rollup_ids = build_conversation_rollups(repo, cfg.rollup)
    embed_result = embed_conversations(repo, config=cfg.embeddings)
    cluster_result = cluster_conversations(repo, cfg.clustering)

    labeler = GPTClusterLabeler(repo, cfg.labeling)
    generated = 0
    cached = 0
    label_errors = 0
    labeled_cluster_ids: list[int] = []
    for conv_cluster_id in cluster_result.get("cluster_ids", []):
        try:
            packet = build_cluster_evidence_packet(repo, int(conv_cluster_id), cfg.evidence)
            result = labeler.generate_label(int(conv_cluster_id), packet, force_refresh=cfg.force_relabel)
            labeled_cluster_ids.append(int(conv_cluster_id))
            if result.get("cached"):
                cached += 1
            else:
                generated += 1
        except Exception:
            label_errors += 1
            logger.exception("Failed to label conversation cluster %s", conv_cluster_id)

    return {
        "rollups": len(rollup_ids),
        "embedded": int(embed_result.get("embedded") or 0),
        "embed_skipped": int(embed_result.get("skipped") or 0),
        "clusters": int(cluster_result.get("clusters") or 0),
        "cluster_members": int(cluster_result.get("members") or 0),
        "labels_generated": generated,
        "labels_cached": cached,
        "label_errors": label_errors,
        "cluster_ids": labeled_cluster_ids,
        "dry_run": bool(cfg.labeling.dry_run),
    }
