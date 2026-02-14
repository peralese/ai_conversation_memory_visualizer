import sqlite3

import pytest
from httpx import ASGITransport, AsyncClient

from src.api import main as api_main
from src.pipeline import import_file
from src.storage.repository import SQLiteRepository


def test_pipeline_import_gemini_json_writes_normalized_and_db(tmp_path):
    db_path = tmp_path / "memory_viz.db"
    repo = SQLiteRepository(str(db_path))
    output_dir = tmp_path / "normalized"

    result = import_file("tests/fixtures/gemini/Takeout/My Activity/Gemini/Conversations.json", repo, output_dir=str(output_dir))

    assert result["source"] == "GEMINI"
    assert result["conversations"] == 1
    assert result["messages"] == 2
    assert (output_dir / "Conversations" / "normalized.jsonl").exists()

    conn = sqlite3.connect(db_path)
    conversations = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    conn.close()

    assert conversations == 1
    assert messages == 2


@pytest.mark.anyio
async def test_api_import_gemini_json_returns_counts(monkeypatch, tmp_path):
    db_path = tmp_path / "api_memory_viz.db"
    repo = SQLiteRepository(str(db_path))
    monkeypatch.setattr(api_main, "repo", repo)

    transport = ASGITransport(app=api_main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with open("tests/fixtures/gemini/Takeout/My Activity/Gemini/Conversations.json", "rb") as f:
            response = await client.post("/import", files={"file": ("Conversations.json", f, "application/json")})

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "GEMINI"
    assert payload["conversations"] == 1
    assert payload["messages"] == 2
