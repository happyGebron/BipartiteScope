from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from scipy import sparse


@dataclass(frozen=True, slots=True)
class AffinityConfig:
    beta: float = 0.5
    restart: float = 0.3
    steps: int = 3
    top_k: int = 64

    def __post_init__(self) -> None:
        if not 0 <= self.beta <= 1 or not 0 < self.restart < 1:
            raise ValueError("beta must be in [0, 1] and restart must be in (0, 1)")
        if self.steps < 1 or self.top_k < 1:
            raise ValueError("steps and top_k must be positive")


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    hidden_dim: int = 64
    layers: int = 2
    latent_groups: int = 8
    epochs: int = 100
    learning_rate: float = 0.01
    lambda_b: float = 0.5
    lambda_o: float = 0.1
    seed: int = 7

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.layers, self.latent_groups, self.epochs) < 1:
            raise ValueError("encoder dimensions and epochs must be positive")
        if self.learning_rate <= 0 or self.lambda_b < 0 or self.lambda_o < 0:
            raise ValueError("invalid optimizer or loss configuration")


@dataclass(frozen=True, slots=True)
class BuildConfig:
    affinity: AffinityConfig = field(default_factory=AffinityConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    semantic_recall_budget: int = 100
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.semantic_recall_budget < 1 or self.schema_version < 1:
            raise ValueError("semantic_recall_budget and schema_version must be positive")


@dataclass(frozen=True, slots=True)
class QueryConfig:
    size_budget: int = 20
    rho: float = 0.5
    eta_s: float = 0.65
    delta_0: float = 0.02
    semantic_recall_budget: int | None = None

    def __post_init__(self) -> None:
        if self.size_budget < 1:
            raise ValueError("size_budget must be positive")
        if not 0 <= self.rho <= 1 or not 0 <= self.eta_s <= 1:
            raise ValueError("rho and eta_s must be in [0, 1]")
        if self.delta_0 < 0 or (
            self.semantic_recall_budget is not None and self.semantic_recall_budget < 1
        ):
            raise ValueError("invalid structure gate or semantic recall budget")


@dataclass(frozen=True, slots=True)
class IncrementalConfig:
    max_affected_ratio: float = 0.20
    neighbor_hops: int = 2
    warm_start_epochs: int = 20
    fallback_to_full_build: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.max_affected_ratio <= 1:
            raise ValueError("max_affected_ratio must be in (0, 1]")
        if self.neighbor_hops < 1 or self.warm_start_epochs < 1:
            raise ValueError("neighbor_hops and warm_start_epochs must be positive")


@dataclass(frozen=True, slots=True)
class RecommendationConfig:
    top_n: int = 20
    community_weight: float = 0.40
    affinity_weight: float = 0.30
    behavior_weight: float = 0.20
    recency_weight: float = 0.10
    popularity_penalty: float = 0.05
    half_life_days: float = 30.0

    def __post_init__(self) -> None:
        values = (
            self.community_weight,
            self.affinity_weight,
            self.behavior_weight,
            self.recency_weight,
            self.popularity_penalty,
        )
        if self.top_n < 1 or self.half_life_days <= 0 or min(values) < 0:
            raise ValueError("invalid recommendation configuration")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    k: int = 10
    minimum_positive_events: int = 3

    def __post_init__(self) -> None:
        if self.k < 1 or self.minimum_positive_events < 2:
            raise ValueError("invalid evaluation configuration")


@dataclass(frozen=True, slots=True)
class Event:
    event_id: str
    user_id: str
    item_id: str
    event_type: str
    event_value: float
    event_time: str


def _canonical_csr(matrix: sparse.spmatrix | np.ndarray, name: str) -> sparse.csr_matrix:
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
    u_ids: tuple[str, ...]
    v_ids: tuple[str, ...]
    incidence: sparse.csr_matrix
    u_features: sparse.csr_matrix
    feature_names: tuple[str, ...] = ()
    u_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    v_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.u_ids or not self.v_ids:
            raise ValueError("both graph sides must contain at least one entity")
        if len(set(self.u_ids)) != len(self.u_ids) or len(set(self.v_ids)) != len(self.v_ids):
            raise ValueError("entity identifiers must be unique within each side")
        incidence = _canonical_csr(self.incidence, "incidence")
        features = _canonical_csr(self.u_features, "u_features")
        if incidence.shape != (len(self.u_ids), len(self.v_ids)):
            raise ValueError("incidence shape must match graph identifiers")
        if features.shape[0] != len(self.u_ids) or features.shape[1] < 1:
            raise ValueError("feature rows must match users and have at least one column")
        if incidence.data.size and not np.all(np.isin(incidence.data, [1.0])):
            raise ValueError("canonical incidence must be binary")
        if features.data.size and np.any(features.data < 0):
            raise ValueError("user features must be nonnegative")
        if self.feature_names and len(self.feature_names) != features.shape[1]:
            raise ValueError("feature names must match the feature dimension")
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
        features: Mapping[str, Iterable[float]],
        *,
        feature_names: Sequence[str] = (),
        u_order: Sequence[str] | None = None,
        v_order: Sequence[str] | None = None,
        u_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        v_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> CanonicalBipartiteGraph:
        rows = {
            str(key): tuple(float(value) for value in values) for key, values in features.items()
        }
        if not rows:
            raise ValueError("at least one user feature row is required")
        dimensions = {len(row) for row in rows.values()}
        if dimensions == {0} or len(dimensions) != 1:
            raise ValueError("all feature rows must have one common positive dimension")
        unique_edges = {(str(user), str(item)) for user, item in edges}
        u_ids = tuple(u_order or sorted(rows))
        v_ids = tuple(v_order or sorted({item for _, item in unique_edges}))
        if set(rows) - set(u_ids) or {user for user, _ in unique_edges} - set(u_ids):
            raise ValueError("user ordering does not cover all feature and edge identifiers")
        if {item for _, item in unique_edges} - set(v_ids) or not v_ids:
            raise ValueError("item ordering does not cover all edge identifiers")
        u_index = {value: index for index, value in enumerate(u_ids)}
        v_index = {value: index for index, value in enumerate(v_ids)}
        positions = [(u_index[user], v_index[item]) for user, item in sorted(unique_edges)]
        edge_rows, edge_columns = zip(*positions) if positions else ((), ())
        incidence = sparse.csr_matrix(
            (np.ones(len(positions)), (edge_rows, edge_columns)),
            shape=(len(u_ids), len(v_ids)),
        )
        feature_matrix = sparse.csr_matrix(
            np.asarray([rows[user] for user in u_ids], dtype=np.float64)
        )
        return cls(
            u_ids,
            v_ids,
            incidence,
            feature_matrix,
            tuple(feature_names),
            u_metadata or {},
            v_metadata or {},
        )

    from_edges = from_edges_and_features


def _inverse_positive(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    result[values > 0] = 1.0 / values[values > 0]
    return result


def _clean(matrix: sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
    value = sparse.csr_matrix(matrix, dtype=np.float64)
    value.sum_duplicates()
    value.eliminate_zeros()
    value.sort_indices()
    return value


def structural_transition(incidence: sparse.csr_matrix) -> sparse.csr_matrix:
    row_degree = np.asarray(incidence.sum(axis=1)).ravel()
    column_degree = np.asarray(incidence.sum(axis=0)).ravel()
    return _clean(
        sparse.diags(_inverse_positive(row_degree))
        @ incidence
        @ sparse.diags(_inverse_positive(column_degree))
        @ incidence.T
    )


def attribute_transition(features: sparse.csr_matrix) -> sparse.csr_matrix:
    row_mass = np.asarray(features.sum(axis=1)).ravel()
    feature_mass = np.asarray(features.sum(axis=0)).ravel()
    return _clean(
        sparse.diags(_inverse_positive(row_mass))
        @ features
        @ sparse.diags(_inverse_positive(feature_mass))
        @ features.T
    )


def rowwise_top_k(
    matrix: sparse.spmatrix | np.ndarray,
    *,
    top_k: int,
    keep_diagonal: bool = True,
) -> sparse.csr_matrix:
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
        entries = list(zip(matrix.indices[start:stop], matrix.data[start:stop], strict=True))
        diagonal = (
            [(int(column), float(value)) for column, value in entries if column == row]
            if keep_diagonal
            else []
        )
        candidates = [
            (int(column), float(value)) for column, value in entries if column != row and value > 0
        ]
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
    def __init__(self, config: AffinityConfig):
        self.config = config

    def build(self, graph: CanonicalBipartiteGraph) -> AffinityBuildResult:
        structural = structural_transition(graph.incidence)
        attribute = attribute_transition(graph.u_features)
        mixed = _clean((1 - self.config.beta) * structural + self.config.beta * attribute)
        retained = sparse.identity(mixed.shape[0], dtype=np.float64, format="csr")
        identity = retained
        for _ in range(self.config.steps):
            update = _clean(
                (1 - self.config.restart) * (mixed @ retained) + self.config.restart * identity
            )
            retained = rowwise_top_k(update, top_k=self.config.top_k)
        retained = retained - sparse.diags(retained.diagonal())
        affinity = _clean(0.5 * (retained + retained.T))
        affinity.setdiag(0)
        return AffinityBuildResult(structural, attribute, mixed, _clean(affinity))


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for model training") from exc
    return torch


def _sparse_tensor(matrix: sparse.csr_matrix, torch: Any) -> Any:
    value = matrix.tocoo()
    indices = torch.tensor(np.vstack((value.row, value.col)), dtype=torch.long)
    data = torch.tensor(value.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, data, value.shape).coalesce()


def _sym_bipartite(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    row_degree = np.asarray(matrix.sum(axis=1)).ravel()
    column_degree = np.asarray(matrix.sum(axis=0)).ravel()
    row_scale = np.zeros_like(row_degree)
    column_scale = np.zeros_like(column_degree)
    row_scale[row_degree > 0] = row_degree[row_degree > 0] ** -0.5
    column_scale[column_degree > 0] = column_degree[column_degree > 0] ** -0.5
    return (sparse.diags(row_scale) @ matrix @ sparse.diags(column_scale)).tocsr()


def _sym_square(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree)
    inverse[degree > 0] = degree[degree > 0] ** -0.5
    return (sparse.diags(inverse) @ matrix @ sparse.diags(inverse)).tocsr()


@dataclass(frozen=True, slots=True)
class TrainingOutput:
    embeddings: np.ndarray
    item_embeddings: np.ndarray
    assignments: np.ndarray
    diagnostics: tuple[dict[str, float], ...]
    model_state: Mapping[str, np.ndarray]
    warm_started_parameters: tuple[str, ...]


def train_dual_view(
    graph: CanonicalBipartiteGraph,
    affinity: sparse.csr_matrix,
    config: EncoderConfig,
    initial_state: Mapping[str, np.ndarray] | None = None,
) -> TrainingOutput:
    torch = _torch()
    torch.manual_seed(config.seed)
    users, items = graph.incidence.shape
    if config.latent_groups > users:
        raise ValueError("latent_groups cannot exceed the number of users")
    features = _sparse_tensor(graph.u_features, torch)
    incidence = _sparse_tensor(graph.incidence, torch)
    normalized_incidence = _sparse_tensor(_sym_bipartite(graph.incidence), torch)
    affinity_tensor = _sparse_tensor(affinity, torch)
    normalized_affinity = _sparse_tensor(_sym_square(affinity), torch)

    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.x_weight = torch.nn.Parameter(
                torch.empty(graph.u_features.shape[1], config.hidden_dim)
            )
            self.v_embedding = torch.nn.Parameter(torch.empty(items, config.hidden_dim))
            self.v_weights = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.empty(config.hidden_dim, config.hidden_dim))
                    for _ in range(config.layers)
                ]
            )
            self.u_weights = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.empty(config.hidden_dim, config.hidden_dim))
                    for _ in range(config.layers)
                ]
            )
            self.w_weights = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.empty(config.hidden_dim, config.hidden_dim))
                    for _ in range(config.layers)
                ]
            )
            self.assignment = torch.nn.Parameter(
                torch.empty(config.hidden_dim, config.latent_groups)
            )
            for parameter in self.parameters():
                if parameter.ndim > 1:
                    torch.nn.init.xavier_uniform_(parameter)

        def forward(self) -> tuple[Any, Any, Any]:
            user = torch.relu(torch.sparse.mm(features, self.x_weight))
            item = self.v_embedding
            for v_weight, u_weight, w_weight in zip(
                self.v_weights, self.u_weights, self.w_weights, strict=True
            ):
                item = torch.relu(
                    torch.sparse.mm(normalized_incidence.transpose(0, 1), user) @ v_weight + item
                )
                incidence_user = torch.relu(
                    torch.sparse.mm(normalized_incidence, item) @ u_weight + user
                )
                user = torch.relu(
                    torch.sparse.mm(normalized_affinity, incidence_user) @ w_weight + incidence_user
                )
            return user, item, torch.softmax(user @ self.assignment, dim=1)

    def objective(assignments: Any) -> tuple[Any, Any, Any, Any]:
        affinity_degree = torch.sparse.sum(affinity_tensor, dim=1).to_dense()
        volume = assignments.T @ (affinity_degree[:, None] * assignments)
        laplacian = volume - assignments.T @ torch.sparse.mm(affinity_tensor, assignments)
        eye = torch.eye(config.latent_groups, dtype=assignments.dtype)
        scale = volume.diagonal().abs().mean().clamp_min(1.0)
        jitter = 16 * torch.finfo(assignments.dtype).eps * scale
        sah = (
            torch.trace(torch.linalg.solve(volume + jitter * eye, laplacian)) / config.latent_groups
        )
        item_degree = torch.sparse.sum(incidence, dim=0).to_dense()
        support = torch.sparse.mm(incidence.transpose(0, 1), assignments)
        weight = 1 / torch.log2(2 + item_degree)
        numerator = (weight[:, None] * support * (item_degree[:, None] - support)).sum(dim=0)
        denominator = (weight[:, None] * support * item_degree[:, None]).sum(dim=0).clamp_min(1e-8)
        orig = (numerator / denominator).mean()
        gram = assignments.T @ assignments
        normalized = gram / torch.linalg.vector_norm(gram).clamp_min(1e-8)
        target = (
            torch.eye(config.latent_groups, dtype=assignments.dtype) / config.latent_groups**0.5
        )
        orthogonality = torch.linalg.matrix_norm(normalized - target, ord="fro") ** 2
        return (
            sah + config.lambda_b * orig + config.lambda_o * orthogonality,
            sah,
            orig,
            orthogonality,
        )

    model = Encoder()
    loaded: list[str] = []
    if initial_state:
        state = model.state_dict()
        for name, value in initial_state.items():
            if name not in state:
                continue
            source = torch.as_tensor(value, dtype=state[name].dtype)
            if source.shape == state[name].shape:
                state[name] = source
                loaded.append(name)
            elif (
                name == "v_embedding"
                and source.ndim == 2
                and source.shape[1] == state[name].shape[1]
            ):
                rows = min(source.shape[0], state[name].shape[0])
                state[name][:rows] = source[:rows]
                loaded.append(name)
        model.load_state_dict(state)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    diagnostics: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        embeddings, item_embeddings, assignments = model()
        total, sah, orig, orthogonality = objective(assignments)
        if not torch.isfinite(total):
            raise FloatingPointError("training objective became non-finite")
        total.backward()
        optimizer.step()
        if epoch == 0 or epoch + 1 == config.epochs or (epoch + 1) % 25 == 0:
            diagnostics.append(
                {
                    "epoch": float(epoch + 1),
                    "total_loss": float(total.detach()),
                    "sah_ncut": float(sah.detach()),
                    "orig_bip": float(orig.detach()),
                    "orthogonality": float(orthogonality.detach()),
                }
            )
    with torch.no_grad():
        embeddings, item_embeddings, assignments = model()
    return TrainingOutput(
        embeddings.detach().cpu().numpy().astype(np.float32),
        item_embeddings.detach().cpu().numpy().astype(np.float32),
        assignments.detach().cpu().numpy().astype(np.float32),
        tuple(diagnostics),
        {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()},
        tuple(sorted(loaded)),
    )


def build_exact_semantic_index(
    embeddings: np.ndarray, budget: int
) -> tuple[np.ndarray, np.ndarray]:
    if embeddings.ndim != 2 or budget < 1:
        raise ValueError("embeddings must be two-dimensional and budget must be positive")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms > 1e-12)
    count = min(budget, max(0, len(embeddings) - 1))
    neighbors = np.empty((len(embeddings), count), dtype=np.int64)
    similarities = np.empty((len(embeddings), count), dtype=np.float32)
    for row in range(len(embeddings)):
        scores = normalized @ normalized[row]
        scores[row] = -np.inf
        ranking = np.lexsort((np.arange(len(embeddings)), -scores))[:count]
        neighbors[row] = ranking
        similarities[row] = scores[ranking]
    return neighbors, similarities


@dataclass(frozen=True, slots=True)
class ModelSnapshot:
    snapshot_id: str
    created_at: str
    graph: CanonicalBipartiteGraph
    affinity: sparse.csr_matrix
    embeddings: np.ndarray
    item_embeddings: np.ndarray
    assignments: np.ndarray
    neighbor_ids: np.ndarray
    neighbor_scores: np.ndarray
    config: BuildConfig
    diagnostics: tuple[Mapping[str, float], ...]
    model_state: Mapping[str, np.ndarray]
    interaction_weights: sparse.csr_matrix
    parent_snapshot_id: str | None = None
    build_mode: str = "full"
    event_batch_hash: str | None = None
    update_stats: Mapping[str, Any] = field(default_factory=dict)


def build_snapshot(
    graph: CanonicalBipartiteGraph,
    config: BuildConfig | None = None,
    *,
    interaction_weights: sparse.csr_matrix | None = None,
    initial_state: Mapping[str, np.ndarray] | None = None,
    affinity_override: sparse.csr_matrix | None = None,
    parent_snapshot_id: str | None = None,
    build_mode: str = "full",
    event_batch_hash: str | None = None,
    update_stats: Mapping[str, Any] | None = None,
) -> ModelSnapshot:
    config = config or BuildConfig()
    affinity = (
        affinity_override
        if affinity_override is not None
        else AffinityBuilder(config.affinity).build(graph).affinity
    )
    output = train_dual_view(graph, affinity, config.encoder, initial_state)
    neighbors, scores = build_exact_semantic_index(output.embeddings, config.semantic_recall_budget)
    weights = (
        (interaction_weights if interaction_weights is not None else graph.incidence)
        .tocsr()
        .astype(np.float64)
    )
    if weights.shape != graph.incidence.shape or not np.isfinite(weights.data).all():
        raise ValueError("interaction weights must be finite and match incidence")
    stats = {**dict(update_stats or {}), "warm_started_parameters": output.warm_started_parameters}
    return ModelSnapshot(
        str(uuid4()),
        datetime.now(UTC).isoformat(),
        graph,
        affinity.tocsr(),
        output.embeddings,
        output.item_embeddings,
        output.assignments,
        neighbors,
        scores,
        config,
        output.diagnostics,
        output.model_state,
        weights,
        parent_snapshot_id,
        build_mode,
        event_batch_hash,
        stats,
    )


@dataclass(frozen=True, slots=True)
class ExpansionTrace:
    step: int
    candidate_id: str
    assignment_similarity: float
    local_affinity: float
    blc_before: float
    blc_after: float
    blc_delta: float
    gate_passed: bool
    shared_support_entities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SupportEntity:
    entity_id: str
    community_support: int
    coverage: float
    global_degree: int
    discounted_support: float


@dataclass(frozen=True, slots=True)
class CommunityResult:
    query_entity: str
    members: tuple[str, ...]
    support_entities: tuple[SupportEntity, ...]
    trace: tuple[ExpansionTrace, ...]
    termination_reason: str
    snapshot_id: str


def _ratio(numerator: float, denominator: float) -> float:
    return 1.0 if denominator <= 0 else numerator / denominator


def _blc(
    incidence: sparse.csr_matrix,
    affinity: sparse.csr_matrix,
    community: np.ndarray,
    eta_s: float,
) -> float:
    membership = np.zeros(affinity.shape[0], dtype=bool)
    membership[community] = True
    degree = np.asarray(affinity.sum(axis=1)).ravel()
    inside = degree[community].sum()
    outside = degree[~membership].sum()
    internal = affinity[community][:, community].sum()
    phi = _ratio(float(inside - internal), float(min(inside, outside)))
    item_degree = np.asarray(incidence.sum(axis=0)).ravel()
    support = np.asarray(incidence[community].sum(axis=0)).ravel()
    weight = 1 / np.log2(2 + item_degree)
    psi = _ratio(
        float(np.sum(weight * support * (item_degree - support))),
        float(np.sum(weight * item_degree * support)),
    )
    return eta_s * phi + (1 - eta_s) * psi


class QueryEngine:
    def __init__(self, snapshot: ModelSnapshot):
        self.snapshot = snapshot
        self.incidence = snapshot.graph.incidence
        self.affinity = snapshot.affinity
        self.incidence_t = self.incidence.T.tocsr()

    def _frontier(self, community: np.ndarray, membership: np.ndarray, budget: int) -> np.ndarray:
        items = np.unique(self.incidence[community].indices)
        structural = np.unique(self.incidence_t[items].indices)
        semantic = self.snapshot.neighbor_ids[
            community, : min(budget, self.snapshot.neighbor_ids.shape[1])
        ].ravel()
        excluded = set(np.flatnonzero(membership))
        return np.array(
            sorted(set(structural).union(int(node) for node in semantic) - excluded), dtype=np.int64
        )

    def _support(self, community: np.ndarray) -> tuple[SupportEntity, ...]:
        support = np.asarray(self.incidence[community].sum(axis=0)).ravel()
        degree = np.asarray(self.incidence.sum(axis=0)).ravel()
        entries = [
            SupportEntity(
                self.snapshot.graph.v_ids[index],
                int(support[index]),
                float(support[index] / len(community)),
                int(degree[index]),
                float(support[index] / np.log2(2 + degree[index])),
            )
            for index in np.flatnonzero(support)
        ]
        return tuple(sorted(entries, key=lambda item: (-item.discounted_support, item.entity_id)))

    def search(self, entity_id: str, config: QueryConfig | None = None) -> CommunityResult:
        config = config or QueryConfig()
        try:
            query = self.snapshot.graph.u_index[entity_id]
        except KeyError as exc:
            raise KeyError(f"unknown user identifier: {entity_id}") from exc
        members = [query]
        membership = np.zeros(self.incidence.shape[0], dtype=bool)
        membership[query] = True
        traces: list[ExpansionTrace] = []
        recall_budget = config.semantic_recall_budget or self.snapshot.config.semantic_recall_budget
        reason = "size_budget_reached"
        while len(members) < config.size_budget:
            community = np.array(members, dtype=np.int64)
            candidates = self._frontier(community, membership, recall_budget)
            if len(candidates) == 0:
                reason = "frontier_exhausted"
                break
            before = _blc(self.incidence, self.affinity, community, config.eta_s)
            prototype = self.snapshot.assignments[community].mean(axis=0)
            ranked: list[tuple[float, int, float, float, float, tuple[str, ...]]] = []
            for candidate in candidates:
                vector = self.snapshot.assignments[candidate]
                denominator = np.linalg.norm(vector) * np.linalg.norm(prototype)
                similarity = (
                    float(np.dot(vector, prototype) / denominator) if denominator > 1e-12 else 0.0
                )
                local_affinity = float(self.affinity[candidate, community].sum())
                after = _blc(
                    self.incidence, self.affinity, np.append(community, candidate), config.eta_s
                )
                shared = np.intersect1d(
                    self.incidence[candidate].indices, np.unique(self.incidence[community].indices)
                )
                shared_ids = tuple(self.snapshot.graph.v_ids[index] for index in shared)
                score = (
                    config.rho * similarity + (1 - config.rho) * local_affinity - (after - before)
                )
                ranked.append(
                    (score, int(candidate), similarity, local_affinity, after, shared_ids)
                )
            _, candidate, similarity, local_affinity, after, shared = min(
                ranked,
                key=lambda item: (-item[0], item[1]),
            )
            passed = after <= before + config.delta_0
            traces.append(
                ExpansionTrace(
                    len(members),
                    self.snapshot.graph.u_ids[candidate],
                    similarity,
                    local_affinity,
                    before,
                    after,
                    after - before,
                    passed,
                    shared,
                )
            )
            if not passed:
                reason = "no_candidate_passed_structure_gate"
                break
            members.append(candidate)
            membership[candidate] = True
        community = np.array(members, dtype=np.int64)
        return CommunityResult(
            entity_id,
            tuple(self.snapshot.graph.u_ids[index] for index in community),
            self._support(community),
            tuple(traces),
            reason,
            self.snapshot.snapshot_id,
        )


class AcademicExplorerAdapter:
    def to_graph(
        self,
        records: Iterable[Mapping[str, str]],
        features: Mapping[str, Iterable[float]],
    ) -> CanonicalBipartiteGraph:
        edges = [(str(record["researcher_id"]), str(record["paper_id"])) for record in records]
        return CanonicalBipartiteGraph.from_edges_and_features(edges, features)


class RecommendationAdapter:
    def to_graph(
        self,
        records: Iterable[Mapping[str, str]],
        features: Mapping[str, Iterable[float]],
    ) -> CanonicalBipartiteGraph:
        edges = [(str(record["user_id"]), str(record["item_id"])) for record in records]
        return CanonicalBipartiteGraph.from_edges_and_features(edges, features)


AcademicAdapter = AcademicExplorerAdapter


def export_community(result: CommunityResult, path: str | Path) -> Path:
    import json
    from dataclasses import asdict

    payload = {
        "query_entity": result.query_entity,
        "members": result.members,
        "support_entities": [asdict(item) for item in result.support_entities],
        "trace": [asdict(item) for item in result.trace],
        "termination_reason": result.termination_reason,
        "snapshot_id": result.snapshot_id,
    }
    output = Path(path)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output
