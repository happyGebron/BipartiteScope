"""Immutable, versioned artifact persistence for reusable offline builds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from scipy import sparse

from .config import AffinityConfig, BuildConfig, EncoderConfig
from .domain import CanonicalBipartiteGraph


def _hash_graph(graph: CanonicalBipartiteGraph) -> str:
    digest = hashlib.sha256()
    for value in (*graph.u_ids, *graph.v_ids):
        digest.update(value.encode()); digest.update(b"\0")
    digest.update(graph.incidence.data.tobytes()); digest.update(graph.u_features.data.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ModelSnapshot:
    snapshot_id: str
    created_at: str
    graph: CanonicalBipartiteGraph
    affinity: sparse.csr_matrix
    embeddings: np.ndarray
    assignments: np.ndarray
    neighbor_ids: np.ndarray
    neighbor_scores: np.ndarray
    config: BuildConfig
    diagnostics: tuple[dict[str, float], ...]
    model_state: dict[str, np.ndarray]

    @classmethod
    def create(cls, graph: CanonicalBipartiteGraph, affinity: sparse.csr_matrix, embeddings: np.ndarray, assignments: np.ndarray, neighbor_ids: np.ndarray, neighbor_scores: np.ndarray, config: BuildConfig, diagnostics: tuple[dict[str, float], ...], model_state: dict[str, np.ndarray]) -> "ModelSnapshot":
        return cls(str(uuid4()), datetime.now(UTC).isoformat(), graph, affinity.tocsr(), embeddings, assignments, neighbor_ids, neighbor_scores, config, diagnostics, model_state)


class SnapshotStore:
    """Filesystem store. Existing snapshots are never overwritten."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def save(self, snapshot: ModelSnapshot) -> Path:
        target = self.root / snapshot.snapshot_id
        if target.exists():
            raise FileExistsError(f"snapshot already exists: {snapshot.snapshot_id}")
        target.mkdir(parents=True)
        sparse.save_npz(target / "A.npz", snapshot.graph.incidence)
        sparse.save_npz(target / "X.npz", snapshot.graph.u_features)
        sparse.save_npz(target / "W.npz", snapshot.affinity)
        np.save(target / "Z.npy", snapshot.embeddings, allow_pickle=False)
        np.save(target / "S.npy", snapshot.assignments, allow_pickle=False)
        np.save(target / "semantic_neighbors.npy", snapshot.neighbor_ids, allow_pickle=False)
        np.save(target / "semantic_scores.npy", snapshot.neighbor_scores, allow_pickle=False)
        np.savez_compressed(target / "model.npz", **snapshot.model_state)
        manifest = {
            "snapshot_id": snapshot.snapshot_id, "created_at": snapshot.created_at, "core_version": "0.1.0",
            "input_hash": _hash_graph(snapshot.graph), "u_ids": snapshot.graph.u_ids, "v_ids": snapshot.graph.v_ids,
            "config": asdict(snapshot.config), "diagnostics": snapshot.diagnostics,
            "u_metadata": snapshot.graph.u_metadata, "v_metadata": snapshot.graph.v_metadata,
        }
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return target

    def load(self, snapshot_id: str) -> ModelSnapshot:
        target = self.root / snapshot_id
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        graph = CanonicalBipartiteGraph(tuple(manifest["u_ids"]), tuple(manifest["v_ids"]), sparse.load_npz(target / "A.npz").tocsr(), sparse.load_npz(target / "X.npz").tocsr(), manifest.get("u_metadata", {}), manifest.get("v_metadata", {}))
        config_data = manifest["config"]
        config = BuildConfig(
            affinity=AffinityConfig(**config_data["affinity"]),
            encoder=EncoderConfig(**config_data["encoder"]),
            semantic_recall_budget=config_data["semantic_recall_budget"],
        )
        state = dict(np.load(target / "model.npz", allow_pickle=False))
        return ModelSnapshot(manifest["snapshot_id"], manifest["created_at"], graph, sparse.load_npz(target / "W.npz").tocsr(), np.load(target / "Z.npy", allow_pickle=False), np.load(target / "S.npy", allow_pickle=False), np.load(target / "semantic_neighbors.npy", allow_pickle=False), np.load(target / "semantic_scores.npy", allow_pickle=False), config, tuple(manifest["diagnostics"]), state)
