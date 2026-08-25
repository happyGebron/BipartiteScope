"""Researcher-paper records for community and support-paper exploration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ..domain import CanonicalBipartiteGraph


class AcademicExplorerAdapter:
    def to_graph(self, authorships: Iterable[Mapping[str, object]], researcher_features: Mapping[str, Iterable[float]]) -> CanonicalBipartiteGraph:
        edges = []
        for record in authorships:
            try:
                edges.append((str(record["researcher_id"]), str(record["paper_id"])))
            except KeyError as error:
                raise ValueError("academic records require researcher_id and paper_id") from error
        return CanonicalBipartiteGraph.from_edges_and_features(edges, dict(researcher_features))
