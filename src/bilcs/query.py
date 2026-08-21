"""BLC-guided online community search and calculation-grounded explanations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .config import QueryConfig
from .snapshot import ModelSnapshot


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


def _ratio(num: float, den: float) -> float:
    return 1.0 if den <= 0 else num / den


def _blc(incidence: sparse.csr_matrix, affinity: sparse.csr_matrix, community: np.ndarray, eta_s: float) -> float:
    membership = np.zeros(affinity.shape[0], dtype=bool); membership[community] = True
    degree = np.asarray(affinity.sum(axis=1)).ravel()
    volume_inside = degree[community].sum(); volume_outside = degree[~membership].sum()
    internal = affinity[community][:, community].sum()
    phi = _ratio(float(volume_inside - internal), float(min(volume_inside, volume_outside)))
    aux_degree = np.asarray(incidence.sum(axis=0)).ravel()
    support = np.asarray(incidence[community].sum(axis=0)).ravel()
    weight = 1.0 / np.log2(2.0 + aux_degree)
    psi = _ratio(float(np.sum(weight * support * (aux_degree - support))), float(np.sum(weight * aux_degree * support)))
    return eta_s * phi + (1.0 - eta_s) * psi


class QueryEngine:
    """Loads one immutable snapshot and serves many queries without propagation or training."""

    def __init__(self, snapshot: ModelSnapshot) -> None:
        self.snapshot = snapshot
        self.incidence = snapshot.graph.incidence
        self.affinity = snapshot.affinity
        self.incidence_t = self.incidence.T.tocsr()

    def _frontier(self, community: np.ndarray, membership: np.ndarray, budget: int) -> np.ndarray:
        auxiliary = np.unique(self.incidence[community].indices)
        structural = np.unique(self.incidence_t[auxiliary].indices)
        semantic = self.snapshot.neighbor_ids[community, : min(budget, self.snapshot.neighbor_ids.shape[1])].ravel()
        return np.array(sorted(set(structural).union(int(node) for node in semantic) - set(np.flatnonzero(membership))), dtype=np.int64)

    def _support(self, community: np.ndarray) -> tuple[SupportEntity, ...]:
        support = np.asarray(self.incidence[community].sum(axis=0)).ravel()
        degree = np.asarray(self.incidence.sum(axis=0)).ravel()
        chosen = np.flatnonzero(support)
        entries = [SupportEntity(self.snapshot.graph.v_ids[index], int(support[index]), float(support[index] / len(community)), int(degree[index]), float(support[index] / np.log2(2 + degree[index]))) for index in chosen]
        return tuple(sorted(entries, key=lambda item: (-item.discounted_support, item.entity_id)))

    def search(self, entity_id: str, config: QueryConfig = QueryConfig()) -> CommunityResult:
        try:
            query = self.snapshot.graph.u_index[entity_id]
        except KeyError as error:
            raise KeyError(f"unknown U-side entity: {entity_id}") from error
        members = [query]; membership = np.zeros(self.incidence.shape[0], dtype=bool); membership[query] = True
        traces: list[ExpansionTrace] = []
        recall_budget = config.semantic_recall_budget or self.snapshot.config.semantic_recall_budget
        reason = "size_budget_reached"
        while len(members) < config.size_budget:
            community = np.array(members, dtype=np.int64)
            candidates = self._frontier(community, membership, recall_budget)
            if len(candidates) == 0:
                reason = "frontier_exhausted"; break
            before = _blc(self.incidence, self.affinity, community, config.eta_s)
            prototype = self.snapshot.assignments[community].mean(axis=0)
            ranked: list[tuple[float, int, float, float, float, tuple[str, ...]]] = []
            for candidate in candidates:
                vector = self.snapshot.assignments[candidate]
                denom = np.linalg.norm(vector) * np.linalg.norm(prototype)
                assignment_similarity = float(np.dot(vector, prototype) / denom) if denom > 1e-12 else 0.0
                local_affinity = float(self.affinity[candidate, community].sum())
                after = _blc(self.incidence, self.affinity, np.append(community, candidate), config.eta_s)
                shared = np.intersect1d(self.incidence[candidate].indices, np.unique(self.incidence[community].indices))
                shared_ids = tuple(self.snapshot.graph.v_ids[index] for index in shared)
                score = config.rho * assignment_similarity + (1.0 - config.rho) * local_affinity - (after - before)
                ranked.append((score, int(candidate), assignment_similarity, local_affinity, after, shared_ids))
            score, candidate, similarity, local_affinity, after, shared = sorted(ranked, key=lambda item: (-item[0], item[1]))[0]
            passed = after <= before + config.delta_0
            traces.append(ExpansionTrace(len(members), self.snapshot.graph.u_ids[candidate], similarity, local_affinity, before, after, after - before, passed, shared))
            if not passed:
                reason = "no_candidate_passed_structure_gate"; break
            members.append(candidate); membership[candidate] = True
        community = np.array(members, dtype=np.int64)
        return CommunityResult(entity_id, tuple(self.snapshot.graph.u_ids[index] for index in community), self._support(community), tuple(traces), reason, self.snapshot.snapshot_id)
