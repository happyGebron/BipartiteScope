"""Optional FastAPI application for workspace builds and reusable queries."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import AffinityConfig, BuildConfig, EncoderConfig, QueryConfig
from .domain import CanonicalBipartiteGraph
from .query import QueryEngine
from .service import build_snapshot
from .snapshot import SnapshotStore


def _config(raw: dict[str, Any]) -> BuildConfig:
    return BuildConfig(
        affinity=AffinityConfig(**raw.get("affinity", {})),
        encoder=EncoderConfig(**raw.get("encoder", {})),
        semantic_recall_budget=raw.get("semantic_recall_budget", 100),
    )


def create_app(root: str | Path = "artifacts") -> Any:
    """Create an application lazily so Core users do not need FastAPI installed."""
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("REST API requires: pip install 'bipartite-scope[api]'") from error
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(root / "metadata.sqlite3", check_same_thread=False)
    database.execute("CREATE TABLE IF NOT EXISTS snapshots (workspace TEXT, snapshot_id TEXT PRIMARY KEY, created_at TEXT)")
    database.commit()
    app = FastAPI(title="BipartiteScope", version="0.2.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/workspaces/{workspace_id}/build")
    def build(workspace_id: str, payload: dict[str, Any]) -> dict[str, str]:
        try:
            graph = CanonicalBipartiteGraph.from_edges_and_features(payload["edges"], payload["features"])
            snapshot = build_snapshot(graph, _config(payload.get("config", {})))
            SnapshotStore(root / workspace_id).save(snapshot)
            database.execute("INSERT INTO snapshots VALUES (?, ?, ?)", (workspace_id, snapshot.snapshot_id, snapshot.created_at)); database.commit()
            return {"workspace_id": workspace_id, "snapshot_id": snapshot.snapshot_id}
        except (KeyError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/workspaces/{workspace_id}/snapshots/{snapshot_id}/query/{entity_id}")
    def query(workspace_id: str, snapshot_id: str, entity_id: str, size_budget: int = 20) -> dict[str, Any]:
        try:
            result = QueryEngine(SnapshotStore(root / workspace_id).load(snapshot_id)).search(entity_id, QueryConfig(size_budget=size_budget))
        except (FileNotFoundError, KeyError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return {
            "query_entity": result.query_entity, "members": result.members, "termination_reason": result.termination_reason,
            "snapshot_id": result.snapshot_id, "support_entities": [asdict(entry) for entry in result.support_entities],
            "trace": [asdict(entry) for entry in result.trace],
        }

    return app
