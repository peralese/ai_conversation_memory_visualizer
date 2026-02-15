from __future__ import annotations

import re
from typing import Any

from src.modes.taxonomy import MODE_TAXONOMY

STRONG_WEIGHT = 3
MEDIUM_WEIGHT = 2
WEAK_WEIGHT = 1
NEGATION_GUARD_PHRASES = ("not a bug", "not bug", "no bug")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def score_mode_weights(
    *,
    top_keywords: list[str],
    label: str,
    sample_snippets: list[str],
) -> dict[str, Any]:
    evidence_parts = []
    evidence_parts.extend([k for k in top_keywords if k])
    evidence_parts.append(label or "")
    evidence_parts.extend([s for s in sample_snippets if s])
    evidence = " ".join(evidence_parts).lower()
    tokens = TOKEN_PATTERN.findall(evidence)
    token_set = set(tokens)

    scores: dict[str, float] = {mode.id: 0.0 for mode in MODE_TAXONOMY}
    for mode in MODE_TAXONOMY:
        score = 0.0
        score += _match_keywords(mode.strong_keywords, evidence, token_set, STRONG_WEIGHT)
        score += _match_keywords(mode.medium_keywords, evidence, token_set, MEDIUM_WEIGHT)
        score += _match_keywords(mode.weak_keywords, evidence, token_set, WEAK_WEIGHT)
        if mode.id == "troubleshooting_debugging" and any(p in evidence for p in NEGATION_GUARD_PHRASES):
            score = max(0.0, score - STRONG_WEIGHT)
        scores[mode.id] = score

    return _normalize_scores(scores)


def _match_keywords(keywords: tuple[str, ...], evidence: str, token_set: set[str], weight: int) -> float:
    score = 0.0
    for kw in keywords:
        k = kw.strip().lower()
        if not k:
            continue
        if " " in k:
            if k in evidence:
                score += weight
        elif k in token_set:
            score += weight
    return score


def _normalize_scores(scores: dict[str, float]) -> dict[str, Any]:
    total = sum(scores.values())
    weights: dict[str, float] = {}
    if total <= 0:
        # Fallback: place full mass on design_synthesis for V1 deterministic behavior.
        for mode_id in scores.keys():
            weights[mode_id] = 1.0 if mode_id == "design_synthesis" else 0.0
        dominant_mode = "design_synthesis"
        dominant_weight = 1.0
    else:
        for mode_id, s in scores.items():
            weights[mode_id] = round(float(s / total), 3)
        # rounding correction to keep sum near 1.0
        mode_ids = list(weights.keys())
        current = sum(weights.values())
        if mode_ids and abs(current - 1.0) > 1e-9:
            dominant_tmp = max(mode_ids, key=lambda mid: weights[mid])
            weights[dominant_tmp] = round(weights[dominant_tmp] + (1.0 - current), 3)
        dominant_mode = max(mode_ids, key=lambda mid: weights[mid]) if mode_ids else None
        dominant_weight = weights[dominant_mode] if dominant_mode is not None else None

    return {
        "mode_weights": weights,
        "dominant_mode": dominant_mode,
        "dominant_weight": dominant_weight,
        "raw_scores": scores,
    }
