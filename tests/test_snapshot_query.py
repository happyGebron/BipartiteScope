import tempfile
import unittest

from bilcs.config import AffinityConfig, BuildConfig, EncoderConfig, QueryConfig
from bilcs.domain import CanonicalBipartiteGraph
from bilcs.query import QueryEngine
from bilcs.service import build_snapshot
from bilcs.snapshot import SnapshotStore


class SnapshotQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = CanonicalBipartiteGraph.from_edges_and_features(
            [("u1", "v1"), ("u1", "v2"), ("u2", "v1"), ("u2", "v2"), ("u3", "v3")],
            {"u1": [1, 0], "u2": [1, 0], "u3": [0, 1]},
        )
        self.config = BuildConfig(AffinityConfig(top_k=2), EncoderConfig(hidden_dim=4, layers=1, latent_groups=2, epochs=3), semantic_recall_budget=2)

    def test_build_save_load_and_query(self) -> None:
        snapshot = build_snapshot(self.graph, self.config)
        with tempfile.TemporaryDirectory() as directory:
            store = SnapshotStore(directory); store.save(snapshot)
            loaded = store.load(snapshot.snapshot_id)
            result = QueryEngine(loaded).search("u1", QueryConfig(size_budget=2, delta_0=1.0))
        self.assertEqual(result.query_entity, "u1")
        self.assertIn("u1", result.members)
        self.assertTrue(result.support_entities)
        self.assertEqual(result.snapshot_id, snapshot.snapshot_id)
