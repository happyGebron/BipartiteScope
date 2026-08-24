import unittest

import numpy as np

from bipartite_scope.affinity import AffinityBuilder, rowwise_top_k
from bipartite_scope.config import AffinityConfig
from bipartite_scope.domain import CanonicalBipartiteGraph


class AffinityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = CanonicalBipartiteGraph.from_edges_and_features(
            [("u1", "v1"), ("u1", "v2"), ("u2", "v1"), ("u3", "v2")],
            {"u1": [1, 0], "u2": [1, 0], "u3": [0, 1]},
        )

    def test_transition_rows_are_stochastic_when_mass_is_positive(self) -> None:
        result = AffinityBuilder(AffinityConfig(top_k=2, steps=2)).build(self.graph)
        np.testing.assert_allclose(np.asarray(result.structural.sum(axis=1)).ravel(), 1.0)
        np.testing.assert_allclose(np.asarray(result.attribute.sum(axis=1)).ravel(), 1.0)

    def test_affinity_is_symmetric_and_has_no_self_edges(self) -> None:
        affinity = AffinityBuilder(AffinityConfig(top_k=1, steps=2)).build(self.graph).affinity
        self.assertEqual((affinity - affinity.T).nnz, 0)
        np.testing.assert_allclose(affinity.diagonal(), 0.0)

    def test_top_k_breaks_equal_value_ties_by_node_index(self) -> None:
        matrix = np.array([[1, 0.5, 0.5], [0, 1, 0], [0, 0, 1]], dtype=float)
        result = rowwise_top_k(matrix, top_k=1)
        self.assertEqual(result[0, 1], 0.5)
        self.assertEqual(result[0, 2], 0.0)
