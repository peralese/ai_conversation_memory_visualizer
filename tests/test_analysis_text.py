from src.analysis.text import build_analysis_text


def test_build_analysis_text_removes_prompted_prefix_and_stopwords():
    text = "Prompted Please help me generate and draft a rewrite for this migration plan"
    cleaned = build_analysis_text(text)
    assert cleaned == "me and a for this migration plan"
    assert "prompted" not in cleaned
    assert "please" not in cleaned
    assert "generate" not in cleaned
    assert "draft" not in cleaned


def test_build_analysis_text_normalizes_whitespace_and_lowercases():
    text = "  Create   A  HEATMAP\nfor  WEEKLY   Trends  "
    cleaned = build_analysis_text(text)
    assert cleaned == "a heatmap for weekly trends"
