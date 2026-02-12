from __future__ import annotations

import json
from pathlib import Path

from src.detection.source_detection import detect_source
from src.models import SourceType
from src.normalize.normalizer import normalize
from src.parsers.factory import parser_for_source
from src.storage.repository import SQLiteRepository


def import_file(file_path: str, repo: SQLiteRepository, output_dir: str = "normalized") -> dict[str, int | str]:
    source = detect_source(file_path)
    if source == SourceType.UNKNOWN:
        raise ValueError("Could not detect source format")

    parser = parser_for_source(source)
    raw_conversations = parser.parse_file(file_path)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    import_dir = Path(output_dir) / Path(file_path).stem
    import_dir.mkdir(parents=True, exist_ok=True)
    out_path = import_dir / "normalized.jsonl"

    count_messages = 0
    with out_path.open("w", encoding="utf-8") as out:
        for raw in raw_conversations:
            bundle = normalize(raw, source)
            repo.upsert_bundle(bundle)
            out.write(json.dumps(bundle.as_json_dict(), ensure_ascii=False) + "\n")
            count_messages += len(bundle.messages)

    return {
        "source": source.value,
        "conversations": len(raw_conversations),
        "messages": count_messages,
        "normalized_jsonl": str(out_path),
    }
