import json
import tempfile
import unittest
from pathlib import Path

from test_core import config

from bipartite_scope import (
    CanonicalBipartiteGraph,
    EvaluationConfig,
    Event,
    RecommendationConfig,
    SnapshotStore,
    build_snapshot,
    evaluate,
    recommend,
)
from bipartite_scope.recommendation import _ranking_metrics, record_feedback
from bipartite_scope.storage import init_workspace, load_workspace, pending_events, register_events


class RecommendationTests(unittest.TestCase):
    def test_ranking_excludes_consumed_and_disliked_items(self) -> None:
        graph = CanonicalBipartiteGraph.from_edges_and_features(
            [("u1", "i1"), ("u2", "i1"), ("u2", "i2"), ("u3", "i2"), ("u3", "i3")],
            {"u1": [1, 0], "u2": [1, 0], "u3": [0, 1]},
        )
        snapshot = build_snapshot(graph, config(1))
        result = recommend(snapshot, "u1", RecommendationConfig(top_n=2))
        self.assertTrue(result.items)
        self.assertNotIn("i1", [item.item_id for item in result.items])
        for item in result.items:
            self.assertIn("score", item.explanation)
            self.assertEqual(
                set(item.components),
                {
                    "community_support",
                    "affinity_support",
                    "weighted_behavior",
                    "recency",
                    "popularity_penalty",
                },
            )
        disliked = Event("d1", "u1", result.items[0].item_id, "dislike", 1, "2026-09-03T00:00:00Z")
        filtered = recommend(snapshot, "u1", RecommendationConfig(top_n=2), events=(disliked,))
        self.assertNotIn(disliked.item_id, [item.item_id for item in filtered.items])

    def test_feedback_is_pending_and_does_not_retrain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = load_workspace(init_workspace(Path(directory) / "workspace"))
            event = Event("feedback-1", "u1", "i2", "favorite", 1, "2026-09-03T00:00:00Z")
            result = record_feedback(workspace, event)
            self.assertTrue(result["accepted"])
            self.assertFalse(result["retraining_triggered"])
            self.assertEqual(pending_events(workspace), (event,))
            self.assertTrue((workspace.data / "feedback.jsonl").is_file())
            self.assertTrue((workspace.data / "events.jsonl").is_file())

    def test_metrics_match_a_hand_computed_ranking(self) -> None:
        metrics = _ranking_metrics({"u": ["a", "b"]}, {"u": "b"}, 2)
        self.assertEqual(metrics["precision@2"], 0.5)
        self.assertEqual(metrics["recall@2"], 1.0)
        self.assertEqual(metrics["hit_rate@2"], 1.0)
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertAlmostEqual(metrics["ndcg@2"], 1 / 1.584962500721156)

    def test_temporal_evaluation_writes_complete_report(self) -> None:
        graph = CanonicalBipartiteGraph.from_edges_and_features(
            [
                ("u1", "i1"),
                ("u1", "i2"),
                ("u1", "i3"),
                ("u2", "i1"),
                ("u2", "i3"),
                ("u2", "i4"),
                ("u3", "i2"),
                ("u3", "i4"),
            ],
            {"u1": [1, 0], "u2": [1, 0], "u3": [0, 1]},
        )
        events = (
            Event("e1", "u1", "i1", "click", 1, "2026-09-01T00:00:00Z"),
            Event("e2", "u1", "i2", "click", 1, "2026-09-02T00:00:00Z"),
            Event("e3", "u1", "i3", "click", 1, "2026-09-03T00:00:00Z"),
            Event("e4", "u2", "i1", "click", 1, "2026-09-01T00:00:00Z"),
            Event("e5", "u2", "i3", "click", 1, "2026-09-02T00:00:00Z"),
            Event("e6", "u2", "i4", "click", 1, "2026-09-03T00:00:00Z"),
        )
        with tempfile.TemporaryDirectory() as directory:
            workspace = load_workspace(init_workspace(Path(directory) / "workspace"))
            snapshot = build_snapshot(graph, config(1))
            SnapshotStore(workspace.artifacts).save(snapshot, activate=True)
            register_events(workspace, events, source="test")
            result = evaluate(workspace, config=EvaluationConfig(k=2, minimum_positive_events=3))
            output = Path(result.output)
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "configuration.json").is_file())
            self.assertTrue((output / "report.md").is_file())
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(metrics["temporal_leakage_check"])
            self.assertEqual(metrics["included_users"], 2)
            self.assertIn("popularity", metrics)
            self.assertIn("user_item_cooccurrence", metrics)


if __name__ == "__main__":
    unittest.main()
