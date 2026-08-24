import unittest

from bipartite_scope.domain import CanonicalBipartiteGraph


class CanonicalGraphTests(unittest.TestCase):
    def test_builds_binary_sparse_graph_and_stable_indexes(self) -> None:
        graph = CanonicalBipartiteGraph.from_edges_and_features(
            [("u2", "v2"), ("u1", "v1"), ("u1", "v1")],
            {"u1": [1, 0], "u2": [0, 2]},
        )
        self.assertEqual(graph.u_ids, ("u1", "u2"))
        self.assertEqual(graph.v_ids, ("v1", "v2"))
        self.assertEqual(graph.incidence.nnz, 2)
        self.assertEqual(graph.u_index["u2"], 1)

    def test_rejects_negative_features(self) -> None:
        with self.assertRaises(ValueError):
            CanonicalBipartiteGraph.from_edges_and_features([("u", "v")], {"u": [-1]})
