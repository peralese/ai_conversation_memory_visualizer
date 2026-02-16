from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from src.clustering.service import ClusteringService
from src.env_loader import load_dotenv
from src.labeling.evidence_packet import build_cluster_evidence_packet
from src.embeddings.service import EmbeddingService
from src.metrics.drift_service import DriftService
from src.metrics.modes_service import ModesService
from src.metrics.service import MetricsService
from src.metrics.specialization_service import ModelSpecializationService
from src.pipeline import import_file
from src.redaction.redactor import RedactionConfig
from src.reports.generator import CognitiveSummaryReportGenerator
from src.storage.repository import SQLiteRepository

load_dotenv()

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
modes_service = ModesService(repo)


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
    use_semantic_labels: bool = True,
    show_legacy_labels: bool = False,
) -> list[dict]:
    return clustering_service.list_clusters(
        exclude_domain_stopwords=exclude_domain_stopwords,
        include_subclusters=include_subclusters,
        use_semantic_labels=use_semantic_labels,
        show_legacy_labels=show_legacy_labels,
    )


@app.get("/clusters/{cluster_id}")
def cluster_detail(
    cluster_id: int,
    exclude_domain_stopwords: bool = True,
    include_subclusters: bool = False,
    use_semantic_labels: bool = True,
    show_legacy_labels: bool = False,
) -> dict:
    try:
        detail = clustering_service.cluster_detail(
            cluster_id,
            exclude_domain_stopwords=exclude_domain_stopwords,
            include_subclusters=include_subclusters,
            use_semantic_labels=use_semantic_labels,
            show_legacy_labels=show_legacy_labels,
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
    use_semantic_labels: bool = True,
    show_legacy_labels: bool = False,
) -> dict:
    try:
        return clustering_service.subclusters_for_cluster(
            cluster_id,
            exclude_domain_stopwords=exclude_domain_stopwords,
            use_semantic_labels=use_semantic_labels,
            show_legacy_labels=show_legacy_labels,
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
    use_semantic_labels: bool = True,
    show_legacy_labels: bool = False,
) -> list[dict]:
    if use_subclusters:
        return clustering_service.subcluster_topic_evolution(
            source=source,
            min_messages=min_messages,
            top_n=top_n,
            exclude_domain_stopwords=exclude_domain_stopwords,
            use_semantic_labels=use_semantic_labels,
            show_legacy_labels=show_legacy_labels,
        )

    labels = {
        int(c["cluster_id"]): str(c["label"])
        for c in clustering_service.list_clusters(
            exclude_domain_stopwords=exclude_domain_stopwords,
            include_subclusters=False,
            use_semantic_labels=use_semantic_labels,
            show_legacy_labels=show_legacy_labels,
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
def idea_half_life(
    exclude_domain_stopwords: bool = True,
    use_semantic_labels: bool = True,
    show_legacy_labels: bool = False,
) -> list[dict]:
    labels = {
        int(c["cluster_id"]): str(c["label"])
        for c in clustering_service.list_clusters(
            exclude_domain_stopwords=exclude_domain_stopwords,
            include_subclusters=False,
            use_semantic_labels=use_semantic_labels,
            show_legacy_labels=show_legacy_labels,
        )
    }
    return metrics_service.idea_half_life(label_by_cluster=labels)


@app.get("/metrics/profile")
def profile(top_n: int = 30) -> dict:
    return metrics_service.dataset_profile(top_n=top_n)


@app.get("/metrics/model_specialization")
def model_specialization(level: str = "cluster") -> dict:
    if (level or "cluster").lower() == "cluster":
        clustering_service.ensure_clusters_fresh()
    return specialization_service.compute(level=level)


@app.get("/metrics/drift")
def drift(
    level: str = "cluster",
    cluster_id: str | None = None,
    exclude_domain_stopwords: bool = True,
    use_semantic_labels: bool = True,
    show_legacy_labels: bool = False,
) -> dict:
    label_map: dict[str, str] | None = None
    if level == "cluster":
        label_map = {
            str(int(c["cluster_id"])): str(c["label"])
            for c in clustering_service.list_clusters(
                exclude_domain_stopwords=exclude_domain_stopwords,
                include_subclusters=False,
                use_semantic_labels=use_semantic_labels,
                show_legacy_labels=show_legacy_labels,
            )
        }
    return drift_service.detail(level=level, cluster_id=cluster_id, label_by_entity=label_map)


@app.get("/metrics/modes")
def modes(level: str = "cluster") -> dict:
    return modes_service.metrics(level=level)


@app.get("/metrics/modes/timeline")
def modes_timeline(level: str = "cluster", bucket: str = "week") -> dict:
    return modes_service.timeline(level=level, bucket=bucket)


@app.get("/metrics/modes/by_source")
def modes_by_source(level: str = "cluster") -> dict:
    return modes_service.by_source(level=level)


@app.get("/reports/cognitive_summary")
def cognitive_summary(format: str = "json"):
    report = report_generator.generate_json_report()
    if format.lower() == "md":
        md = report_generator.generate_markdown_report(report)
        return PlainTextResponse(md)
    return report


@app.get("/api/conv_clusters")
def conv_clusters(
    use_semantic_labels: bool = True,
    show_legacy_labels: bool = False,
) -> list[dict]:
    rows = repo.list_conv_clusters()
    out: list[dict] = []
    for row in rows:
        label_info = row.get("label") or {}
        semantic_title = str(label_info.get("title") or "").strip()
        legacy_label = _conv_cluster_legacy_label(int(row["conv_cluster_id"]))
        label = legacy_label
        if use_semantic_labels and semantic_title:
            label = semantic_title
            if show_legacy_labels and semantic_title.lower() != legacy_label.lower():
                label = f"{semantic_title} (legacy: {legacy_label})"
        out.append(
            {
                **row,
                "label_display": label,
                "legacy_label": legacy_label,
                "semantic": {
                    "title": semantic_title or legacy_label,
                    "summary": str(label_info.get("summary") or ""),
                    "tags": list(label_info.get("tags") or []),
                    "label_source": str(label_info.get("label_source") or ""),
                },
            }
        )
    return out


@app.get("/api/conv_clusters/{conv_cluster_id}")
def conv_cluster_detail(
    conv_cluster_id: int,
    use_semantic_labels: bool = True,
    show_legacy_labels: bool = False,
) -> dict:
    rows = repo.list_conv_clusters()
    cluster = next((r for r in rows if int(r["conv_cluster_id"]) == int(conv_cluster_id)), None)
    if cluster is None:
        raise HTTPException(status_code=404, detail=f"Unknown conversation cluster: {conv_cluster_id}")

    label_info = cluster.get("label") or {}
    semantic_title = str(label_info.get("title") or "").strip()
    legacy_label = _conv_cluster_legacy_label(conv_cluster_id)
    label = legacy_label
    if use_semantic_labels and semantic_title:
        label = semantic_title
        if show_legacy_labels and semantic_title.lower() != legacy_label.lower():
            label = f"{semantic_title} (legacy: {legacy_label})"

    members = repo.get_conv_cluster_members(conv_cluster_id)
    evidence = build_cluster_evidence_packet(repo, conv_cluster_id)
    return {
        **cluster,
        "label_display": label,
        "legacy_label": legacy_label,
        "semantic": {
            "title": semantic_title or legacy_label,
            "summary": str(label_info.get("summary") or ""),
            "tags": list(label_info.get("tags") or []),
            "label_source": str(label_info.get("label_source") or ""),
        },
        "members": members,
        "evidence_packet": evidence,
    }


def _conv_cluster_legacy_label(conv_cluster_id: int) -> str:
    members = repo.get_conv_cluster_members(conv_cluster_id)
    counter: dict[str, int] = {}
    for member in members:
        rollup_text = str(member.get("rollup_text") or "")
        for token in rollup_text.split():
            if len(token) < 3:
                continue
            counter[token] = counter.get(token, 0) + 1
    if not counter:
        return f"Conversation Cluster {conv_cluster_id}"
    top = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return ", ".join(token for token, _ in top)
