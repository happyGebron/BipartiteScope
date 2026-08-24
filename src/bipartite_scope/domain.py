"""Canonical graph domain objects. The core has no dataset preset or labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from scipy import sparse


def _canonical_csr(matrix: sparse.spmatrix | np.ndarray, *, name: str) -> sparse.csr_matrix:
    result = sparse.csr_matrix(matrix, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    if result.data.size and not np.isfinite(result.data).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True, slots=True)
class CanonicalBipartiteGraph:
    """A validated G=(U,V,E,X) with business identifiers separated from indexes."""

    u_ids: tuple[str, ...]
    v_ids: tuple[str, ...]
    incidence: sparse.csr_matrix
    u_features: sparse.csr_matrix
    u_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    v_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.u_ids or not self.v_ids:
            raise ValueError("both U and V must contain at least one entity")
        if len(set(self.u_ids)) != len(self.u_ids) or len(set(self.v_ids)) != len(self.v_ids):
            raise ValueError("entity IDs must be unique within each side")
        incidence = _canonical_csr(self.incidence, name="incidence")
        features = _canonical_csr(self.u_features, name="u_features")
        if incidence.shape != (len(self.u_ids), len(self.v_ids)):
            raise ValueError("incidence shape must match U and V IDs")
        if features.shape[0] != len(self.u_ids) or features.shape[1] < 1:
            raise ValueError("feature rows must match U IDs and have at least one column")
        if incidence.data.size and not np.all(np.isin(incidence.data, [1.0])):
            raise ValueError("canonical incidence must be binary")
        if features.data.size and np.any(features.data < 0):
            raise ValueError("U-side features must be nonnegative")
        object.__setattr__(self, "incidence", incidence)
        object.__setattr__(self, "u_features", features)

    @property
    def u_index(self) -> dict[str, int]:
        return {entity_id: index for index, entity_id in enumerate(self.u_ids)}

    @property
    def v_index(self) -> dict[str, int]:
        return {entity_id: index for index, entity_id in enumerate(self.v_ids)}

    @classmethod
    def from_edges_and_features(
        cls,
        edges: Iterable[tuple[str, str]],
        features: dict[str, Iterable[float]],
        *,
        u_metadata: dict[str, dict[str, Any]] | None = None,
        v_metadata: dict[str, dict[str, Any]] | None = None,
    ) -> "CanonicalBipartiteGraph":
        feature_rows = {str(key): tuple(float(value) for value in values) for key, values in features.items()}
        if not feature_rows:
            raise ValueError("at least one U-side feature row is required")
        dimensions = {len(row) for row in feature_rows.values()}
        if dimensions == {0} or len(dimensions) != 1:
            raise ValueError("all feature rows must have one common positive dimension")
        unique_edges = {(str(u), str(v)) for u, v in edges}
        u_ids = tuple(sorted(feature_rows))
        invalid = sorted({u for u, _ in unique_edges} - set(u_ids))
        if invalid:
            raise ValueError(f"edges reference U IDs without features: {invalid[:3]}")
        v_ids = tuple(sorted({v for _, v in unique_edges}))
        if not v_ids:
            raise ValueError("at least one edge and V-side entity is required")
        u_index = {value: index for index, value in enumerate(u_ids)}
        v_index = {value: index for index, value in enumerate(v_ids)}
        rows, columns = zip(*((u_index[u], v_index[v]) for u, v in sorted(unique_edges)))
        incidence = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(u_ids), len(v_ids)))
        feature_matrix = sparse.csr_matrix(np.asarray([feature_rows[u] for u in u_ids], dtype=np.float64))
        return cls(u_ids, v_ids, incidence, feature_matrix, u_metadata or {}, v_metadata or {})
