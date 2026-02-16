from src.clustering.semantic_labels import build_semantic_labels, normalize_for_label


def test_normalize_for_label_strips_code_paths_and_urls():
    text = (
        "The User has uploaded a file to: /mnt/data/input.json\n"
        "```python\ndef foo(x):\n  return x\n```\n"
        "See `pip install uvicorn` and visit https://localhost:8000/api\n"
        "Traceback (most recent call last):\n"
        "File \"/tmp/a.py\", line 10, in <module>\n"
        "ValueError: boom\n"
        "azure pricing script for monthly costs"
    )
    out = normalize_for_label(text, exclude_domain_stopwords=True)
    assert "return" not in out
    assert "localhost" not in out
    assert "api" not in out
    assert "uploaded" not in out
    assert "azure" in out
    assert "pricing" in out


def test_normalize_for_label_applies_stopword_filtering():
    text = "id data file str dict list json yaml csv run args kwargs self pricing report"
    out = normalize_for_label(text, exclude_domain_stopwords=True)
    assert "pricing" in out
    assert "report" in out
    for token in ("id", "data", "file", "str", "dict", "json", "run", "args", "self"):
        assert token not in out


def test_semantic_candidate_ranking_is_stable():
    records_by_cluster = {
        1: [
            {
                "id": "m1",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "original_text": "azure monthly pricing script and cost breakdown",
                "conversation_title": "Azure monthly pricing",
            },
            {
                "id": "m2",
                "timestamp": "2026-01-02T00:00:00+00:00",
                "original_text": "azure invoice parsing and pricing automation",
                "conversation_title": "Invoice parsing",
            },
        ],
        2: [
            {
                "id": "m3",
                "timestamp": "2026-01-03T00:00:00+00:00",
                "original_text": "recipe print layout in eleventy and css template",
                "conversation_title": "Recipe print layout",
            },
            {
                "id": "m4",
                "timestamp": "2026-01-04T00:00:00+00:00",
                "original_text": "eleventy templating and recipe site styles",
                "conversation_title": "Eleventy recipe site",
            },
        ],
    }

    a = build_semantic_labels(records_by_cluster, exclude_domain_stopwords=True)
    b = build_semantic_labels(records_by_cluster, exclude_domain_stopwords=True)

    assert a[1].title == b[1].title
    assert a[2].title == b[2].title
    assert a[1].title != a[2].title
    assert len(a[1].title) <= 60
    assert len(a[2].title) <= 60
