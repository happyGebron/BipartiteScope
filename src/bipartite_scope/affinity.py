"""Sparse structure-attribute higher-order affinity construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .config import AffinityConfig
from .domain import CanonicalBipartiteGraph


def _inverse_positive(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    valid = values > 0
    result[valid] = 1.0 / values[valid]
    return result


def _clean(matrix: sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
    value = sparse.csr_matrix(matrix, dtype=np.float64)
    value.sum_duplicates()
    value.eliminate_zeros()
    value.sort_indices()
    return value


def structural_transition(incidence: sparse.csr_matrix) -> sparse.csr_matrix:
    """Return D_U^-1 A D_V^-1 A^T without densifying the graph."""
    row_degree = np.asarray(incidence.sum(axis=1)).ravel()
    column_degree = np.asarray(incidence.sum(axis=0)).ravel()
    return _clean(
        sparse.diags(_inverse_positive(row_degree))
        @ incidence
        @ sparse.diags(_inverse_positive(column_degree))
        @ incidence.T
    )


def attribute_transition(features: sparse.csr_matrix) -> sparse.csr_matrix:
    """Return D_X^-1 X D_F^-1 X^T, discounting universally shared features."""
    row_mass = np.asarray(features.sum(axis=1)).ravel()
    feature_mass = np.asarray(features.sum(axis=0)).ravel()
    return _clean(
        sparse.diags(_inverse_positive(row_mass))
        @ features
        @ sparse.diags(_inverse_positive(feature_mass))
        @ features.T
    )


def rowwise_top_k(matrix: sparse.csr_matrix, *, top_k: int, keep_diagonal: bool = True) -> sparse.csr_matrix:
    """Retain diagonal plus Top-k positive off-diagonal values; ties use lower index first."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    matrix = _clean(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("rowwise_top_k requires a square matrix")
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row in range(matrix.shape[0]):
        start, stop = matrix.indptr[row : row + 2]
        entries = [(int(column), float(value)) for column, value in zip(matrix.indices[start:stop], matrix.data[start:stop])]
        diagonal = [(column, value) for column, value in entries if column == row] if keep_diagonal else []
        candidates = [(column, value) for column, value in entries if column != row and value > 0]
        selected = diagonal + sorted(candidates, key=lambda pair: (-pair[1], pair[0]))[:top_k]
        rows.extend([row] * len(selected))
        columns.extend(column for column, _ in selected)
        values.extend(value for _, value in selected)
    return _clean(sparse.csr_matrix((values, (rows, columns)), shape=matrix.shape))


@dataclass(frozen=True, slots=True)
class AffinityBuildResult:
    structural: sparse.csr_matrix
    attribute: sparse.csr_matrix
    mixed: sparse.csr_matrix
    affinity: sparse.csr_matrix


class AffinityBuilder:
    """Build W with mixed-view restart diffusion and stepwise sparsification."""

    def __init__(self, config: AffinityConfig) -> None:
        self.config = config

    def build(self, graph: CanonicalBipartiteGraph) -> AffinityBuildResult:
        ps = structural_transition(graph.incidence)
        px = attribute_transition(graph.u_features)
        mixed = _clean((1.0 - self.config.beta) * ps + self.config.beta * px)
        retained = sparse.identity(mixed.shape[0], dtype=np.float64, format="csr")
        identity = retained
        for _ in range(self.config.steps):
            update = _clean((1.0 - self.config.restart) * (mixed @ retained) + self.config.restart * identity)
            retained = rowwise_top_k(update, top_k=self.config.top_k, keep_diagonal=True)
        retained = retained - sparse.diags(retained.diagonal())
        affinity = _clean(0.5 * (retained + retained.T))
        affinity.setdiag(0.0)
        affinity = _clean(affinity)
        return AffinityBuildResult(ps, px, mixed, affinity)
