from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log
import re
from typing import Any

from src.analysis.domain import get_domain_stopwords

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # type: ignore
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
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s]+", re.IGNORECASE)
_UNIX_PATH_RE = re.compile(r"(?:^|\s)(?:/|\./|\../)[^\s]+")
_WINDOWS_PATH_RE = re.compile(r"(?:^|\s)[a-zA-Z]:\\[^\s]+")

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)
_UUID_RE = re.compile(r"^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$", re.IGNORECASE)
_LONG_HEX_RE = re.compile(r"^[0-9a-f]{16,}$", re.IGNORECASE)
_LONG_ID_RE = re.compile(r"^[a-z0-9_-]{24,}$", re.IGNORECASE)

_GENERIC_LABEL_TERMS = {
    "topic",
    "chat",
    "conversation",
    "assistant",
    "user",
    "message",
    "messages",
    "data",
    "file",
    "path",
    "model",
    "request",
    "response",
}


@dataclass
class LabelArtifacts:
    terms: list[str]
    label: str
    low_signal: bool
    warning: str | None
    debug: dict[str, Any] | None = None


@dataclass
class _TokenizationResult:
    raw_tokens: list[str]
    final_tokens: list[str]
    removed_by_rule: dict[str, list[str]]


def build_distinctive_labels(
    texts_by_cluster: dict[int, list[str]],
    *,
    top_n: int = 8,
    exclude_domain_stopwords: bool = True,
    include_debug_for: set[int] | None = None,
) -> dict[int, LabelArtifacts]:
    include_debug_for = include_debug_for or set()
    strict_tokens_by_cluster: dict[int, list[str]] = {}
    relaxed_tokens_by_cluster: dict[int, list[str]] = {}
    debug_payloads: dict[int, dict[str, Any]] = {}

    for cid, texts in texts_by_cluster.items():
        strict_tokens: list[str] = []
        relaxed_tokens: list[str] = []
        raw_counter: Counter[str] = Counter()
        final_counter: Counter[str] = Counter()
        removed_counters: dict[str, Counter[str]] = defaultdict(Counter)

        for text in texts:
            strict = _tokenize(
                str(text or ""),
                exclude_domain_stopwords=exclude_domain_stopwords,
                remove_english_stopwords=True,
            )
            relaxed = _tokenize(
                str(text or ""),
                exclude_domain_stopwords=exclude_domain_stopwords,
                remove_english_stopwords=False,
            )
            strict_tokens.extend(strict.final_tokens)
            relaxed_tokens.extend(relaxed.final_tokens)

            if cid in include_debug_for:
                raw_counter.update(strict.raw_tokens)
                final_counter.update(strict.final_tokens)
                for rule, tokens in strict.removed_by_rule.items():
                    removed_counters[rule].update(tokens)

        strict_tokens_by_cluster[cid] = strict_tokens
        relaxed_tokens_by_cluster[cid] = relaxed_tokens

        if cid in include_debug_for:
            debug_payloads[cid] = {
                "raw_top_tokens": [token for token, _ in raw_counter.most_common(20)],
                "removed_by_rule": {
                    rule: [token for token, _ in counter.most_common(20)]
                    for rule, counter in sorted(removed_counters.items())
                },
                "final_top_tokens": [token for token, _ in final_counter.most_common(20)],
            }

    scored_strict = _score_terms(strict_tokens_by_cluster)
    scored_relaxed = _score_terms(relaxed_tokens_by_cluster)

    out: dict[int, LabelArtifacts] = {}
    domain_stopwords = get_domain_stopwords()
    for cid in texts_by_cluster.keys():
        ordered = [token for token, _score in scored_strict.get(cid, [])]
        if len(ordered) < top_n:
            for token, _score in scored_relaxed.get(cid, []):
                if token in ordered:
                    continue
                if exclude_domain_stopwords and token in domain_stopwords:
                    continue
                ordered.append(token)
                if len(ordered) >= top_n:
                    break

        terms = ordered[:top_n]
        label_terms = terms[:3]
        label = ", ".join(label_terms) if label_terms else "Unlabeled topic"
        low_signal = _is_low_signal(label_terms)
        warning = "Label may be low-signal; consider adding domain stopwords." if low_signal else None
        debug = debug_payloads.get(cid)
        if debug is not None:
            debug["final_label_tokens"] = label_terms

        out[cid] = LabelArtifacts(
            terms=terms,
            label=label,
            low_signal=low_signal,
            warning=warning,
            debug=debug,
        )
    return out


def _score_terms(tokens_by_cluster: dict[int, list[str]]) -> dict[int, list[tuple[str, float]]]:
    active = {cid: toks for cid, toks in tokens_by_cluster.items() if toks}
    if not active:
        return {cid: [] for cid in tokens_by_cluster.keys()}

    n_clusters = len(active)
    df: Counter[str] = Counter()
    tf_by_cluster: dict[int, Counter[str]] = {}

    for cid, tokens in active.items():
        tf = Counter(tokens)
        tf_by_cluster[cid] = tf
        df.update(tf.keys())

    scored: dict[int, list[tuple[str, float]]] = {}
    for cid, tf in tf_by_cluster.items():
        total = max(1, sum(tf.values()))
        ranked: list[tuple[str, float]] = []
        for token, count in tf.items():
            idf = log((1 + n_clusters) / (1 + df[token])) + 1.0
            score = (count / total) * idf
            ranked.append((token, score))
        ranked.sort(key=lambda item: (item[1], tf[item[0]]), reverse=True)
        scored[cid] = ranked

    for cid in tokens_by_cluster.keys():
        scored.setdefault(cid, [])
    return scored


def _tokenize(
    text: str,
    *,
    exclude_domain_stopwords: bool,
    remove_english_stopwords: bool,
) -> _TokenizationResult:
    raw_tokens = [_normalize_token(tok) for tok in _TOKEN_RE.findall(text.lower())]
    raw_tokens = [tok for tok in raw_tokens if tok]

    removed_by_rule: dict[str, list[str]] = {
        "code": [],
        "url_path": [],
        "stopword": [],
        "short_token": [],
        "numeric": [],
        "hash_uuid": [],
    }

    working = text

    working, removed_code = _remove_pattern(working, _CODE_BLOCK_RE)
    removed_by_rule["code"].extend(_extract_tokens(removed_code))

    working, removed_inline = _remove_pattern(working, _INLINE_CODE_RE)
    removed_by_rule["code"].extend(_extract_tokens(removed_inline))

    working, removed_stack = _remove_stacktrace_lines(working)
    removed_by_rule["code"].extend(_extract_tokens(removed_stack))

    working, removed_urls = _remove_pattern(working, _URL_RE)
    removed_by_rule["url_path"].extend(_extract_tokens(removed_urls))

    working, removed_unix = _remove_pattern(working, _UNIX_PATH_RE)
    removed_by_rule["url_path"].extend(_extract_tokens(removed_unix))

    working, removed_windows = _remove_pattern(working, _WINDOWS_PATH_RE)
    removed_by_rule["url_path"].extend(_extract_tokens(removed_windows))

    stopwords = set()
    if remove_english_stopwords:
        stopwords.update(t.lower() for t in ENGLISH_STOP_WORDS)
    if exclude_domain_stopwords:
        stopwords.update(get_domain_stopwords())

    final_tokens: list[str] = []
    for token_raw in _TOKEN_RE.findall(working.lower()):
        token = _normalize_token(token_raw)
        if not token:
            continue

        if len(token) < 3:
            removed_by_rule["short_token"].append(token)
            continue
        if token.isdigit():
            removed_by_rule["numeric"].append(token)
            continue
        if _looks_like_hash_or_uuid(token_raw, token):
            removed_by_rule["hash_uuid"].append(token)
            continue
        if token in stopwords:
            removed_by_rule["stopword"].append(token)
            continue
        final_tokens.append(token)

    return _TokenizationResult(raw_tokens=raw_tokens, final_tokens=final_tokens, removed_by_rule=removed_by_rule)


def _remove_pattern(text: str, pattern: re.Pattern[str]) -> tuple[str, list[str]]:
    removed: list[str] = []

    def _repl(match: re.Match[str]) -> str:
        removed.append(match.group(0))
        return " "

    return pattern.sub(_repl, text), removed


def _extract_tokens(fragments: list[str]) -> list[str]:
    out: list[str] = []
    for fragment in fragments:
        out.extend(_normalize_token(tok) for tok in _TOKEN_RE.findall(fragment.lower()))
    return [token for token in out if token]


def _remove_stacktrace_lines(text: str) -> tuple[str, list[str]]:
    removed: list[str] = []
    kept: list[str] = []

    for line in text.splitlines():
        stripped = line.strip().lower()
        if not stripped:
            kept.append(line)
            continue
        if stripped.startswith("traceback (most recent call last):"):
            removed.append(line)
            continue
        if stripped.startswith("file \"") and " line " in stripped:
            removed.append(line)
            continue
        if stripped.startswith("at "):
            removed.append(line)
            continue
        if re.match(r"^[a-z_][a-z0-9_]*(?:error|exception):", stripped):
            removed.append(line)
            continue
        kept.append(line)

    return "\n".join(kept), removed


def _normalize_token(token: str) -> str:
    return token.strip("_-").lower()


def _looks_like_hash_or_uuid(raw_token: str, normalized_token: str) -> bool:
    if _UUID_RE.fullmatch(raw_token):
        return True
    compact = normalized_token.replace("-", "")
    if _LONG_HEX_RE.fullmatch(compact):
        return True
    if _LONG_ID_RE.fullmatch(normalized_token) and any(ch.isdigit() for ch in normalized_token):
        return True
    return False


def _is_low_signal(label_terms: list[str]) -> bool:
    if len(label_terms) < 2:
        return True
    generic_hits = sum(1 for token in label_terms if token in _GENERIC_LABEL_TERMS)
    return generic_hits >= max(1, len(label_terms) // 2 + 1)
