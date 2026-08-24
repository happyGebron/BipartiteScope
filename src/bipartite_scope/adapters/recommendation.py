"""User-item records for cohort exploration without leaking domain code into Core."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ..domain import CanonicalBipartiteGraph


class RecommendationAdapter:
    def to_graph(self, interactions: Iterable[Mapping[str, object]], user_features: Mapping[str, Iterable[float]]) -> CanonicalBipartiteGraph:
        edges = []
        for record in interactions:
            try:
                edges.append((str(record["user_id"]), str(record["item_id"])))
            except KeyError as error:
                raise ValueError("recommendation records require user_id and item_id") from error
        return CanonicalBipartiteGraph.from_edges_and_features(edges, dict(user_features))
