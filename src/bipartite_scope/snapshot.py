"""Immutable snapshots with atomic writes and integrity verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from uuid import uuid4

import numpy as np
from scipy import sparse

from .config import AffinityConfig, BuildConfig, EncoderConfig
from .domain import CanonicalBipartiteGraph


class SnapshotIntegrityError(RuntimeError):
    """A snapshot is incomplete, missing an asset, or has been altered."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    """Filesystem store that only exposes verified, fully-written snapshots."""

    _ASSET_NAMES = ("A.npz", "X.npz", "W.npz", "Z.npy", "S.npy", "semantic_neighbors.npy", "semantic_scores.npy", "model.npz")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _write_assets(self, directory: Path, snapshot: ModelSnapshot) -> None:
        sparse.save_npz(directory / "A.npz", snapshot.graph.incidence)
        sparse.save_npz(directory / "X.npz", snapshot.graph.u_features)
        sparse.save_npz(directory / "W.npz", snapshot.affinity)
        np.save(directory / "Z.npy", snapshot.embeddings, allow_pickle=False)
        np.save(directory / "S.npy", snapshot.assignments, allow_pickle=False)
        np.save(directory / "semantic_neighbors.npy", snapshot.neighbor_ids, allow_pickle=False)
        np.save(directory / "semantic_scores.npy", snapshot.neighbor_scores, allow_pickle=False)
        np.savez_compressed(directory / "model.npz", **snapshot.model_state)

    def save(self, snapshot: ModelSnapshot) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / snapshot.snapshot_id
        if target.exists():
            raise FileExistsError(f"snapshot already exists: {snapshot.snapshot_id}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{snapshot.snapshot_id}.", dir=self.root))
        try:
            self._write_assets(temporary, snapshot)
            assets = {name: {"sha256": _sha256(temporary / name), "bytes": (temporary / name).stat().st_size} for name in self._ASSET_NAMES}
            manifest = {
                "snapshot_id": snapshot.snapshot_id, "created_at": snapshot.created_at, "core_version": "0.2.0",
                "complete": True, "input_hash": _hash_graph(snapshot.graph), "u_ids": snapshot.graph.u_ids, "v_ids": snapshot.graph.v_ids,
                "config": asdict(snapshot.config), "diagnostics": snapshot.diagnostics, "u_metadata": snapshot.graph.u_metadata,
                "v_metadata": snapshot.graph.v_metadata, "assets": assets,
            }
            (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            os.replace(temporary, target)
            pointer = self.root / "latest.json"
            pointer_tmp = self.root / ".latest.json.tmp"
            pointer_tmp.write_text(json.dumps({"snapshot_id": snapshot.snapshot_id}), encoding="utf-8")
            os.replace(pointer_tmp, pointer)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return target

    def latest_id(self) -> str:
        try:
            return json.loads((self.root / "latest.json").read_text(encoding="utf-8"))["snapshot_id"]
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as error:
            raise SnapshotIntegrityError("no valid latest snapshot pointer exists") from error

    def _verify(self, target: Path, manifest: dict[str, Any]) -> None:
        if not manifest.get("complete"):
            raise SnapshotIntegrityError("snapshot is not marked complete")
        assets = manifest.get("assets")
        if not isinstance(assets, dict):
            raise SnapshotIntegrityError("snapshot manifest has no asset integrity data")
        for name in self._ASSET_NAMES:
            expected = assets.get(name)
            path = target / name
            if not isinstance(expected, dict) or not path.is_file():
                raise SnapshotIntegrityError(f"snapshot asset is missing: {name}")
            if path.stat().st_size != expected.get("bytes") or _sha256(path) != expected.get("sha256"):
                raise SnapshotIntegrityError(f"snapshot asset failed integrity verification: {name}")

    def load(self, snapshot_id: str) -> ModelSnapshot:
        target = self.root / snapshot_id
        try:
            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise SnapshotIntegrityError(f"snapshot manifest is missing: {snapshot_id}") from error
        self._verify(target, manifest)
        graph = CanonicalBipartiteGraph(tuple(manifest["u_ids"]), tuple(manifest["v_ids"]), sparse.load_npz(target / "A.npz").tocsr(), sparse.load_npz(target / "X.npz").tocsr(), manifest.get("u_metadata", {}), manifest.get("v_metadata", {}))
        config_data = manifest["config"]
        config = BuildConfig(AffinityConfig(**config_data["affinity"]), EncoderConfig(**config_data["encoder"]), config_data["semantic_recall_budget"])
        state = dict(np.load(target / "model.npz", allow_pickle=False))
        return ModelSnapshot(manifest["snapshot_id"], manifest["created_at"], graph, sparse.load_npz(target / "W.npz").tocsr(), np.load(target / "Z.npy", allow_pickle=False), np.load(target / "S.npy", allow_pickle=False), np.load(target / "semantic_neighbors.npy", allow_pickle=False), np.load(target / "semantic_scores.npy", allow_pickle=False), config, tuple(manifest["diagnostics"]), state)
