from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log
import re
from typing import Any

from src.analysis.domain import get_domain_stopwords

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer  # type: ignore
except Exception:
    ENGLISH_STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "with",
    }


_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_STACKTRACE_RE = re.compile(
    r"(?im)^\s*(?:traceback \(most recent call last\):|file \"[^\"]+\", line \d+.*|[a-z_][a-z0-9_]*(?:error|exception):.*|at\s+[^\n]+)\s*$"
)
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s]+", re.IGNORECASE)
_PATH_RE = re.compile(r"(?:^|\s)(?:[a-zA-Z]:\\|/|\./|\../)[^\s]+")
_JSON_BLOB_RE = re.compile(r"\{[^{}]{80,}\}", re.DOTALL)
_UI_BOILERPLATE_RE = re.compile(r"(?im)^\s*the\s+user\s+has\s+uploaded\s+a\s+file\s+to\s*:\s*.*$")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)

_EXTRA_CODE_STOPWORDS = {
    "id",
    "data",
    "file",
    "files",
    "path",
    "str",
    "int",
    "float",
    "dict",
    "list",
    "json",
    "yaml",
    "csv",
    "tsv",
    "run",
    "args",
    "kwargs",
    "self",
    "return",
    "true",
    "false",
    "none",
    "null",
    "user",
    "assistant",
    "prompt",
    "system",
    "tool",
    "image",
    "uploaded",
    "download",
    "http",
    "https",
    "localhost",
    "api",
    "token",
    "key",
    "env",
    "config",
}

_BAD_PHRASE_TOKENS = {
    "str",
    "int",
    "float",
    "dict",
    "list",
    "json",
    "yaml",
    "csv",
    "tsv",
    "kwargs",
    "args",
    "self",
    "null",
    "none",
    "true",
    "false",
}


@dataclass
class SemanticLabel:
    title: str
    subtitle: str
    summary: str


@dataclass
class _RepSample:
    id: str
    timestamp: str
    text: str
    normalized: str


def normalize_for_label(text: str, *, exclude_domain_stopwords: bool = True) -> str:
    if not text:
        return ""

    cleaned = str(text)
    cleaned = _UI_BOILERPLATE_RE.sub(" ", cleaned)
    cleaned = _CODE_BLOCK_RE.sub(" ", cleaned)
    cleaned = _INLINE_CODE_RE.sub(" ", cleaned)
    cleaned = _STACKTRACE_RE.sub(" ", cleaned)
    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _PATH_RE.sub(" ", cleaned)
    cleaned = _JSON_BLOB_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()

    stopwords = set(t.lower() for t in ENGLISH_STOP_WORDS).union(_EXTRA_CODE_STOPWORDS)
    if exclude_domain_stopwords:
        stopwords.update(get_domain_stopwords())

    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(cleaned):
        token = raw.strip("_-").lower()
        if not token:
            continue
        if len(token) < 3:
            continue
        if token.isdigit():
            continue
        if token in stopwords:
            continue
        tokens.append(token)

    return " ".join(tokens)


def build_semantic_labels(
    records_by_cluster: dict[int, list[dict[str, Any]]],
    *,
    exclude_domain_stopwords: bool = True,
) -> dict[int, SemanticLabel]:
    representative_by_cluster: dict[int, list[_RepSample]] = {}
    titles_by_cluster: dict[int, list[str]] = {}

    for cid, records in records_by_cluster.items():
        representative_by_cluster[cid] = _representative_samples(records, exclude_domain_stopwords=exclude_domain_stopwords)
        titles_by_cluster[cid] = _top_titles(records, k=10, exclude_domain_stopwords=exclude_domain_stopwords)

    doc_tokens_by_cluster: dict[int, list[str]] = {}
    for cid in records_by_cluster.keys():
        parts = [s.normalized for s in representative_by_cluster.get(cid, []) if s.normalized]
        parts.extend(titles_by_cluster.get(cid, []))
        tokens: list[str] = []
        for p in parts:
            tokens.extend(_TOKEN_RE.findall(p.lower()))
        doc_tokens_by_cluster[cid] = [t for t in tokens if t]

    phrase_scores = _score_phrases(doc_tokens_by_cluster)

    out: dict[int, SemanticLabel] = {}
    for cid in records_by_cluster.keys():
        ranked = phrase_scores.get(cid, [])
        title_phrase = ranked[0] if ranked else "Unlabeled Topic"
        subtitle_phrases = ranked[1:4]
        title = _safe_title(title_phrase)
        subtitle = ", ".join(subtitle_phrases[:3])
        summary = _summary_for_cluster(title, subtitle, representative_by_cluster.get(cid, []))
        out[cid] = SemanticLabel(title=title, subtitle=subtitle, summary=summary)
    return out


def _representative_samples(
    records: list[dict[str, Any]],
    *,
    exclude_domain_stopwords: bool,
) -> list[_RepSample]:
    if not records:
        return []

    samples: list[_RepSample] = []
    for r in records:
        normalized = normalize_for_label(str(r.get("original_text") or ""), exclude_domain_stopwords=exclude_domain_stopwords)
        samples.append(
            _RepSample(
                id=str(r.get("id") or ""),
                timestamp=str(r.get("timestamp") or ""),
                text=str(r.get("original_text") or ""),
                normalized=normalized,
            )
        )

    by_time = sorted(samples, key=lambda s: s.timestamp)
    earliest = by_time[:3]
    recent = by_time[-3:]
    longest = sorted(samples, key=lambda s: len(s.text), reverse=True)[:3]

    unique_candidates = [s for s in samples if s.normalized]
    tfidf_ranked = _rank_by_message_uniqueness(unique_candidates)[:20]

    selected: dict[str, _RepSample] = {}
    for group in (tfidf_ranked, recent, earliest, longest):
        for sample in group:
            selected[sample.id] = sample

    return sorted(selected.values(), key=lambda s: (s.timestamp, s.id))


def _rank_by_message_uniqueness(samples: list[_RepSample]) -> list[_RepSample]:
    if not samples:
        return []

    docs = [s.normalized for s in samples]
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        matrix = vec.fit_transform(docs)
        scores = matrix.sum(axis=1)
        scored = []
        for idx, sample in enumerate(samples):
            value = float(scores[idx, 0])
            scored.append((sample, value))
        scored.sort(key=lambda row: (row[1], row[0].timestamp, row[0].id), reverse=True)
        return [s for s, _ in scored]
    except Exception:
        scored = []
        for sample in samples:
            scored.append((sample, len(set(sample.normalized.split()))))
        scored.sort(key=lambda row: (row[1], row[0].timestamp, row[0].id), reverse=True)
        return [s for s, _ in scored]


def _top_titles(records: list[dict[str, Any]], *, k: int, exclude_domain_stopwords: bool) -> list[str]:
    counts: Counter[str] = Counter()
    for r in records:
        title = str(r.get("conversation_title") or "").strip()
        if title:
            counts[title] += 1

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    out: list[str] = []
    for title, _ in ranked:
        normalized = normalize_for_label(title, exclude_domain_stopwords=exclude_domain_stopwords)
        if normalized:
            out.append(normalized)
    return out


def _score_phrases(doc_tokens_by_cluster: dict[int, list[str]]) -> dict[int, list[str]]:
    n_clusters = max(1, len(doc_tokens_by_cluster))
    tf_by_cluster: dict[int, Counter[str]] = {}
    df: Counter[str] = Counter()

    for cid, tokens in doc_tokens_by_cluster.items():
        phrases = _extract_phrases(tokens)
        tf_by_cluster[cid] = Counter(phrases)
        df.update(set(phrases))

    out: dict[int, list[str]] = {}
    for cid, tf in tf_by_cluster.items():
        ranked: list[tuple[str, float]] = []
        for phrase, count in tf.items():
            if not phrase:
                continue
            idf = log((1 + n_clusters) / (1 + df[phrase])) + 1.0
            score = float(count) * idf
            score += _phrase_quality_bonus(phrase)
            score -= _phrase_penalty(phrase)
            ranked.append((phrase, score))

        ranked.sort(key=lambda row: (-row[1], row[0]))
        filtered = [phrase for phrase, _score in ranked if _is_acceptable_phrase(phrase)]
        out[cid] = filtered[:6]
    return out


def _extract_phrases(tokens: list[str]) -> list[str]:
    phrases: list[str] = []
    n = len(tokens)
    for i in range(n):
        for size in (1, 2, 3):
            j = i + size
            if j > n:
                break
            phrase = " ".join(tokens[i:j]).strip()
            if phrase:
                phrases.append(phrase)
    return phrases


def _phrase_quality_bonus(phrase: str) -> float:
    words = phrase.split()
    alpha_words = sum(1 for w in words if any(ch.isalpha() for ch in w))
    bonus = 0.0
    if 2 <= len(words) <= 5:
        bonus += 0.35
    if alpha_words == len(words):
        bonus += 0.15
    return bonus


def _phrase_penalty(phrase: str) -> float:
    words = phrase.split()
    if not words:
        return 10.0

    penalty = 0.0
    bad_hits = sum(1 for w in words if w in _BAD_PHRASE_TOKENS)
    penalty += bad_hits * 0.8

    numeric_heavy = sum(1 for w in words if any(ch.isdigit() for ch in w))
    if numeric_heavy > 0:
        penalty += (numeric_heavy / len(words)) * 0.5

    return penalty


def _is_acceptable_phrase(phrase: str) -> bool:
    words = phrase.split()
    if not words:
        return False
    if len(words) > 5:
        return False
    if not any(any(ch.isalpha() for ch in w) for w in words):
        return False
    if all(w in _BAD_PHRASE_TOKENS for w in words):
        return False
    return True


def _safe_title(phrase: str) -> str:
    text = re.sub(r"\s+", " ", phrase.strip())
    if not text:
        return "Unlabeled Topic"
    text = text.title()
    if len(text) > 60:
        text = text[:60].rstrip(" ,.;:-")
    return text or "Unlabeled Topic"


def _summary_for_cluster(title: str, subtitle: str, samples: list[_RepSample]) -> str:
    snippets = [s.normalized for s in samples if s.normalized][:4]
    terms: Counter[str] = Counter()
    for snippet in snippets:
        terms.update(snippet.split())
    top_terms = [term for term, _ in terms.most_common(4)]
    work_bits = ", ".join(top_terms) if top_terms else "implementation details"
    subtitle_text = subtitle if subtitle else "related implementation details"
    return (
        f"This cluster is mostly about {title.lower()}. "
        f"Common threads: {subtitle_text}. Typical work includes: {work_bits}."
    )
