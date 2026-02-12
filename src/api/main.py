from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.clustering.service import ClusteringService
from src.embeddings.service import EmbeddingService
from src.metrics.service import MetricsService
from src.pipeline import import_file
from src.redaction.redactor import RedactionConfig
from src.storage.repository import SQLiteRepository

app = FastAPI(title="AI Conversation Memory Visualizer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

repo = SQLiteRepository("data/memory_viz.db")
embedding_service = EmbeddingService(repo)
clustering_service = ClusteringService(repo)
metrics_service = MetricsService(repo)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/import")
async def import_conversations(
    file: UploadFile = File(...),
    redact_pii: bool = Query(False),
) -> dict:
    uploads = Path("data/uploads")
    uploads.mkdir(parents=True, exist_ok=True)
    temp_path = uploads / file.filename

    with temp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = import_file(str(temp_path), repo=repo, output_dir="data/normalized")
        if redact_pii:
            embedding_service.embed_since(
                since=None,
                redact=True,
                config=RedactionConfig(
                    redact_emails=True,
                    redact_phones=True,
                    redact_ips=True,
                    redact_names=False,
                ),
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/embed")
def embed(since: str | None = None, redact_pii: bool = False) -> dict:
    count = embedding_service.embed_since(since=since, redact=redact_pii)
    return {"embedded": count}


@app.post("/cluster")
def cluster(k: int | None = None) -> dict:
    return clustering_service.cluster_embeddings(k=k)


@app.get("/conversations")
def conversations(q: str | None = None) -> list[dict]:
    return repo.list_conversations(q=q)


@app.get("/clusters")
def clusters() -> list[dict]:
    return repo.list_clusters()


@app.get("/clusters/{cluster_id}")
def cluster_detail(cluster_id: int) -> dict:
    try:
        detail = repo.cluster_detail(cluster_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    half_life = next((h for h in metrics_service.idea_half_life() if h["cluster_id"] == cluster_id), None)
    detail["half_life"] = half_life
    return detail


@app.get("/metrics/recurring-topics")
def recurring_topics() -> list[dict]:
    return metrics_service.recurring_topics()


@app.get("/metrics/topic-evolution")
def topic_evolution(granularity: str = "week") -> list[dict]:
    return metrics_service.topic_evolution(granularity=granularity)


@app.get("/metrics/idea-half-life")
def idea_half_life() -> list[dict]:
    return metrics_service.idea_half_life()
