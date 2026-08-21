"""Build orchestration: generic graph in, reusable model snapshot out."""

from __future__ import annotations

from .affinity import AffinityBuilder
from .config import BuildConfig
from .domain import CanonicalBipartiteGraph
from .snapshot import ModelSnapshot
from .training import build_exact_semantic_index, train_dual_view


def build_snapshot(graph: CanonicalBipartiteGraph, config: BuildConfig = BuildConfig()) -> ModelSnapshot:
    affinity = AffinityBuilder(config.affinity).build(graph).affinity
    output = train_dual_view(graph, affinity, config.encoder)
    neighbors, scores = build_exact_semantic_index(output.embeddings, budget=config.semantic_recall_budget)
    return ModelSnapshot.create(graph, affinity, output.embeddings, output.assignments, neighbors, scores, config, output.diagnostics)
