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
- Profile dataset skew:
```bash
python3 -m src.cli profile --top-n 30
```
- Compute drift cache:
```bash
python3 -m src.cli drift --level cluster
```
- Generate cognitive summary report:
```bash
python3 -m src.cli report --format md --out reports/cognitive_summary.md
python3 -m src.cli report --format json --out reports/cognitive_summary.json
```

## OpenAI Embeddings
- Default provider is OpenAI if `OPENAI_API_KEY` is set.
- Falls back to a local deterministic stub provider when not set.
- `.env` is loaded automatically at startup (API, CLI, and conversation clustering runner).

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

## Domain Stopwords and Label Quality
- Domain stopwords are loaded from `config/domain_stopwords.txt`.
- Add one token per line (case-insensitive) to suppress ingestion/UI/code noise in labels and top keywords.
- This list is applied on top of English stopwords for:
  - cluster labels
  - cluster detail top keywords
  - subcluster labels
  - timeline heatmap labels (cluster/subcluster mode)
- UI defaults to `Exclude domain stopwords from labels = ON`; you can turn it off for debugging.
- Cluster detail includes a `Label Debug` panel showing:
  - top raw tokens
  - tokens removed by rule
  - final tokens used for labeling

## Semantic Labels
- The backend generates deterministic semantic labels per cluster from representative samples and conversation titles.
- API responses now include both:
  - `legacy_label`: keyword-style label
  - `semantic`: `{ title, subtitle, summary }`
- Label display is configurable via query params on cluster/timeline/drift endpoints:
  - `use_semantic_labels` (default `true`)
  - `show_legacy_labels` (default `false`)
- Semantic label normalization is reusable as `normalize_for_label(text)` in `src/clustering/semantic_labels.py`.

## Conversation-Level Clustering
- Parallel pipeline (does not replace message-level clustering):
  1. Conversation rollups
  2. Conversation embeddings
  3. Conversation clustering
  4. Evidence packet building
  5. Cached GPT/heuristic labels
- Runner:
```bash
python3 scripts/run_conversation_clustering.py --db data/memory_viz.db --k 8
```
- Useful flags:
  - `--force-reembed`
  - `--force-recluster`
  - `--force-relabel`
  - `--max-gpt 20`
  - `--min-seconds-between-gpt 2`
  - `--dry-run` (build packets + heuristic labels only; no GPT calls)
- Env vars:
  - `OPENAI_API_KEY` (optional; without it labels fall back to deterministic heuristics)

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
