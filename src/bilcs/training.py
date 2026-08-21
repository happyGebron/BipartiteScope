"""Label-free dual-view encoder and offline cut-guided training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from .config import EncoderConfig
from .domain import CanonicalBipartiteGraph


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Training requires the optional dependency: pip install 'bilcs[train]'") from error
    return torch


def _sparse_tensor(matrix: sparse.csr_matrix, torch: Any) -> Any:
    value = matrix.tocoo()
    indices = torch.tensor(np.vstack((value.row, value.col)), dtype=torch.long)
    data = torch.tensor(value.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, data, value.shape).coalesce()


def _sym_bipartite(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    r = np.asarray(matrix.sum(axis=1)).ravel()
    c = np.asarray(matrix.sum(axis=0)).ravel()
    ri = np.zeros_like(r); ci = np.zeros_like(c)
    ri[r > 0] = r[r > 0] ** -0.5; ci[c > 0] = c[c > 0] ** -0.5
    return (sparse.diags(ri) @ matrix @ sparse.diags(ci)).tocsr()


def _sym_square(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree)
    inverse[degree > 0] = degree[degree > 0] ** -0.5
    return (sparse.diags(inverse) @ matrix @ sparse.diags(inverse)).tocsr()


@dataclass(frozen=True, slots=True)
class TrainingOutput:
    embeddings: np.ndarray
    assignments: np.ndarray
    diagnostics: tuple[dict[str, float], ...]
    model_state: dict[str, Any]


def train_dual_view(
    graph: CanonicalBipartiteGraph, affinity: sparse.csr_matrix, config: EncoderConfig
) -> TrainingOutput:
    """Optimize SAH-Ncut + OrigBip + assignment orthogonality without labels."""
    torch = _torch()
    torch.manual_seed(config.seed)
    n, m = graph.incidence.shape
    if config.latent_groups > n:
        raise ValueError("latent_groups cannot exceed the number of U-side entities")
    x = _sparse_tensor(graph.u_features, torch)
    a = _sparse_tensor(graph.incidence, torch)
    a_norm = _sparse_tensor(_sym_bipartite(graph.incidence), torch)
    w = _sparse_tensor(affinity, torch)
    w_norm = _sparse_tensor(_sym_square(affinity), torch)

    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.x_weight = torch.nn.Parameter(torch.empty(graph.u_features.shape[1], config.hidden_dim))
            self.v_embedding = torch.nn.Parameter(torch.empty(m, config.hidden_dim))
            self.v_weights = torch.nn.ParameterList([torch.nn.Parameter(torch.empty(config.hidden_dim, config.hidden_dim)) for _ in range(config.layers)])
            self.u_weights = torch.nn.ParameterList([torch.nn.Parameter(torch.empty(config.hidden_dim, config.hidden_dim)) for _ in range(config.layers)])
            self.w_weights = torch.nn.ParameterList([torch.nn.Parameter(torch.empty(config.hidden_dim, config.hidden_dim)) for _ in range(config.layers)])
            self.assignment = torch.nn.Parameter(torch.empty(config.hidden_dim, config.latent_groups))
            for parameter in self.parameters():
                if parameter.ndim > 1:
                    torch.nn.init.xavier_uniform_(parameter)

        def forward(self) -> tuple[Any, Any]:
            u = torch.relu(torch.sparse.mm(x, self.x_weight))
            v = self.v_embedding
            for v_weight, u_weight, w_weight in zip(self.v_weights, self.u_weights, self.w_weights):
                v = torch.relu(torch.sparse.mm(a_norm.transpose(0, 1), u) @ v_weight + v)
                incidence_u = torch.relu(torch.sparse.mm(a_norm, v) @ u_weight + u)
                u = torch.relu(torch.sparse.mm(w_norm, incidence_u) @ w_weight + incidence_u)
            return u, torch.softmax(u @ self.assignment, dim=1)

    def objective(assignments: Any) -> tuple[Any, Any, Any, Any]:
        w_degree = torch.sparse.sum(w, dim=1).to_dense()
        volume = assignments.T @ (w_degree[:, None] * assignments)
        laplacian = volume - assignments.T @ torch.sparse.mm(w, assignments)
        eye = torch.eye(config.latent_groups, dtype=assignments.dtype)
        sah = torch.trace(torch.linalg.solve(volume + 1e-8 * eye, laplacian)) / config.latent_groups
        aux_degree = torch.sparse.sum(a, dim=0).to_dense()
        support = torch.sparse.mm(a.transpose(0, 1), assignments)
        weight = 1.0 / torch.log2(2.0 + aux_degree)
        numerator = (weight[:, None] * support * (aux_degree[:, None] - support)).sum(dim=0)
        denominator = (weight[:, None] * support * aux_degree[:, None]).sum(dim=0).clamp_min(1e-8)
        orig = (numerator / denominator).mean()
        gram = assignments.T @ assignments
        normalized = gram / torch.linalg.vector_norm(gram).clamp_min(1e-8)
        target = torch.eye(config.latent_groups, dtype=assignments.dtype) / (config.latent_groups**0.5)
        orth = torch.linalg.matrix_norm(normalized - target, ord="fro") ** 2
        return sah + config.lambda_b * orig + config.lambda_o * orth, sah, orig, orth

    model = Encoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    diagnostics: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        embeddings, assignments = model()
        total, sah, orig, orth = objective(assignments)
        if not torch.isfinite(total):
            raise RuntimeError("offline objective became non-finite")
        total.backward(); optimizer.step()
        if epoch == 0 or epoch + 1 == config.epochs or (epoch + 1) % 25 == 0:
            diagnostics.append({"epoch": float(epoch + 1), "total_loss": float(total.detach()), "sah_ncut": float(sah.detach()), "orig_bip": float(orig.detach()), "orthogonality": float(orth.detach())})
    with torch.no_grad():
        embeddings, assignments = model()
    return TrainingOutput(
        embeddings=embeddings.detach().cpu().numpy().astype(np.float32),
        assignments=assignments.detach().cpu().numpy().astype(np.float32),
        diagnostics=tuple(diagnostics),
        model_state={key: value.detach().cpu() for key, value in model.state_dict().items()},
    )


def build_exact_semantic_index(embeddings: np.ndarray, *, budget: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic exact cosine index; an ANN adapter can replace it later."""
    if embeddings.ndim != 2 or budget < 1:
        raise ValueError("embeddings must be 2-D and budget must be positive")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = np.divide(embeddings, norms, out=np.zeros_like(embeddings, dtype=np.float32), where=norms > 1e-12)
    scores = normalized @ normalized.T
    np.fill_diagonal(scores, -np.inf)
    k = min(budget, max(0, len(embeddings) - 1))
    neighbors = np.empty((len(embeddings), k), dtype=np.int64)
    similarities = np.empty((len(embeddings), k), dtype=np.float32)
    for row in range(len(embeddings)):
        ranking = np.lexsort((np.arange(len(embeddings)), -scores[row]))[:k]
        neighbors[row] = ranking
        similarities[row] = scores[row, ranking]
    return neighbors, similarities
