from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from src.clustering.service import ClusteringService
from src.embeddings.service import EmbeddingService
from src.metrics.drift_service import DriftService
from src.metrics.service import MetricsService
from src.metrics.specialization_service import ModelSpecializationService
from src.pipeline import import_file
from src.redaction.redactor import RedactionConfig
from src.reports.generator import CognitiveSummaryReportGenerator
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
specialization_service = ModelSpecializationService(repo)
drift_service = DriftService(repo)
report_generator = CognitiveSummaryReportGenerator(repo)


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
def clusters(
    include_subclusters: bool = False,
    exclude_domain_stopwords: bool = True,
) -> list[dict]:
    return clustering_service.list_clusters(
        exclude_domain_stopwords=exclude_domain_stopwords,
        include_subclusters=include_subclusters,
    )


@app.get("/clusters/{cluster_id}")
def cluster_detail(
    cluster_id: int,
    exclude_domain_stopwords: bool = True,
    include_subclusters: bool = False,
) -> dict:
    try:
        detail = clustering_service.cluster_detail(
            cluster_id,
            exclude_domain_stopwords=exclude_domain_stopwords,
            include_subclusters=include_subclusters,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    half_life = next((h for h in metrics_service.idea_half_life() if h["cluster_id"] == cluster_id), None)
    detail["half_life"] = half_life
    return detail


@app.get("/clusters/{cluster_id}/subclusters")
def cluster_subclusters(
    cluster_id: int,
    exclude_domain_stopwords: bool = True,
) -> dict:
    try:
        return clustering_service.subclusters_for_cluster(
            cluster_id,
            exclude_domain_stopwords=exclude_domain_stopwords,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/metrics/recurring-topics")
def recurring_topics() -> list[dict]:
    return metrics_service.recurring_topics()


@app.get("/metrics/topic-evolution")
def topic_evolution(
    granularity: str = "week",
    source: str | None = None,
    min_messages: int = 1,
    top_n: int = 15,
    use_subclusters: bool = False,
    exclude_domain_stopwords: bool = True,
) -> list[dict]:
    if use_subclusters:
        return clustering_service.subcluster_topic_evolution(
            source=source,
            min_messages=min_messages,
            top_n=top_n,
            exclude_domain_stopwords=exclude_domain_stopwords,
        )

    labels = {
        int(c["cluster_id"]): str(c["label"])
        for c in clustering_service.list_clusters(
            exclude_domain_stopwords=exclude_domain_stopwords,
            include_subclusters=False,
        )
    }
    return metrics_service.topic_evolution(
        granularity=granularity,
        source=source,
        min_messages=min_messages,
        top_n=top_n,
        label_by_cluster=labels,
    )


@app.get("/metrics/idea-half-life")
def idea_half_life() -> list[dict]:
    return metrics_service.idea_half_life()


@app.get("/metrics/profile")
def profile(top_n: int = 30) -> dict:
    return metrics_service.dataset_profile(top_n=top_n)


@app.get("/metrics/model_specialization")
def model_specialization(level: str = "cluster") -> dict:
    return specialization_service.compute(level=level)


@app.get("/metrics/drift")
def drift(level: str = "cluster", cluster_id: str | None = None) -> dict:
    return drift_service.detail(level=level, cluster_id=cluster_id)


@app.get("/reports/cognitive_summary")
def cognitive_summary(format: str = "json"):
    report = report_generator.generate_json_report()
    if format.lower() == "md":
        md = report_generator.generate_markdown_report(report)
        return PlainTextResponse(md)
    return report
