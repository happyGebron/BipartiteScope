"""Delimited-file import with explicit schema validation."""

from __future__ import annotations

import csv
from pathlib import Path

from .domain import CanonicalBipartiteGraph


def load_csv_graph(edges_path: str | Path, features_path: str | Path, *, delimiter: str = ",") -> CanonicalBipartiteGraph:
    """Load `u_id,v_id` edges and a `u_id,<nonnegative feature...>` table."""
    with Path(edges_path).open(newline="", encoding="utf-8") as handle:
        edge_rows = list(csv.DictReader(handle, delimiter=delimiter))
    if not edge_rows or not {"u_id", "v_id"}.issubset(edge_rows[0]):
        raise ValueError("edge file requires u_id and v_id columns")
    with Path(features_path).open(newline="", encoding="utf-8") as handle:
        feature_rows = list(csv.DictReader(handle, delimiter=delimiter))
    if not feature_rows or "u_id" not in feature_rows[0]:
        raise ValueError("feature file requires a u_id column and one or more feature columns")
    columns = [column for column in feature_rows[0] if column != "u_id"]
    if not columns:
        raise ValueError("feature file has no feature columns")
    features = {row["u_id"]: [float(row[column]) for column in columns] for row in feature_rows}
    return CanonicalBipartiteGraph.from_edges_and_features(
        ((row["u_id"], row["v_id"]) for row in edge_rows), features
    )
