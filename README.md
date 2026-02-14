# AI Conversation Memory Visualizer

Local-first web app for ingesting LLM conversation exports, normalizing into a canonical schema, embedding, clustering, and visualizing topic evolution.

## Core Guarantees
- Local-first only: SQLite + local files, no telemetry/cloud sync.
- Privacy-first: optional PII redaction before embeddings.
- Debuggable: each import writes `normalized.jsonl` in canonical schema.
- Modular pipeline: each stage is separated and testable.

## Pipeline
`Source Detection -> Parser -> Normalizer -> SQLite -> Optional Redaction -> Embeddings -> Clustering -> Metrics -> UI`

## Repo Layout
- `src/detection`: source format detection
- `src/parsers`: source-specific parsers (ChatGPT + Claude + Gemini)
- `src/normalize`: canonical normalizer
- `src/storage`: SQLite repository + migrations
- `src/redaction`: PII redaction layer
- `src/embeddings`: pluggable embedding providers
- `src/clustering`: KMeans clustering + TF-IDF labels
- `src/metrics`: recurring topics, evolution, idea half-life
- `src/api`: FastAPI backend
- `src/ui`: React + Vite frontend
- `tests`: required unit tests
- `fixtures`: sample export fixtures

## Canonical Schema
Defined as dataclasses in `src/models.py`:
- `Conversation`
- `Message`
- `EmbeddingRecord`
- `Cluster`
- `TopicEvent`
- `CanonicalConversationBundle`

## Run (Backend)
1. Install dependencies:
```bash
python3 -m pip install -e .[dev]
```
This includes `python-multipart` (required for the `/import` file upload endpoint).  
If needed manually:
```bash
python3 -m pip install python-multipart
```
2. Start API:
```bash
uvicorn src.api.main:app --reload
```

## Run (Frontend)
Prerequisite: Node.js 18+ (Node 20 LTS recommended).  
Check version:
```bash
node -v
```

If Node is too old, install/update via `nvm`:
```bash
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
source ~/.bashrc
nvm install 20
nvm use 20
```

```bash
cd src/ui
npm install
npm run dev
```

## CLI
- Import:
```bash
python3 -m src.cli import fixtures/chatgpt_export_sample.json
python3 -m src.cli import /path/to/claude-export.json
python3 -m src.cli import /path/to/claude-export.zip
python3 -m src.cli import /path/to/gemini-takeout.zip
python3 -m src.cli import /path/to/Takeout/My\\ Activity/Gemini
```
- Embed:
```bash
python3 -m src.cli embed --since 2024-01-01T00:00:00+00:00 --redact
```
- Cluster:
```bash
python3 -m src.cli cluster --k 4
```

## OpenAI Embeddings
- Default provider is OpenAI if `OPENAI_API_KEY` is set.
- Falls back to a local deterministic stub provider when not set.

## Redaction
`src/redaction/redactor.py`
- Email, phone, IP, optional name heuristic.
- Embeddings run on redacted text when enabled.
- `messages` table stores both `original_text` and `redacted_text`.

## Metrics
- Recurring topics: clusters recurring across multiple conversations and weekly buckets.
- Topic evolution: cluster message volume aggregated by week/month.
- Idea half-life assumption:
  - `T0` is first mention.
  - Weekly activity is measured from `T0`.
  - Half-life is first week where volume drops below 50% of cluster peak weekly volume.

## Tests
```bash
python3 -m pytest -q
```

## Claude Export
- In Claude, request export via `Settings -> Privacy -> Export data`.
- Import the resulting Claude JSON, ZIP, or an extracted export directory with the same `import` CLI/API flow.

## Gemini Export
- Export Gemini data via Google Takeout (`Gemini Apps Activity / My Activity`).
- Import the resulting Takeout ZIP or extracted folder with the same `import` CLI/API flow.
