import tempfile
import unittest
from pathlib import Path

import numpy as np
from test_core import config, graph

from bipartite_scope import (
    AffinityConfig,
    BuildConfig,
    CanonicalBipartiteGraph,
    EncoderConfig,
    Event,
    SnapshotStore,
    build_snapshot,
)
from bipartite_scope.recommendation import update_from_events, update_snapshot
from bipartite_scope.storage import init_workspace, load_workspace


def workspace_with_parent(directory: str, value=None):
    root = init_workspace(Path(directory) / "workspace")
    config_path = root / "bipartitescope.toml"
    text = config_path.read_text(encoding="utf-8")
    text = text.replace("warm_start_epochs = 20", "warm_start_epochs = 1")
    config_path.write_text(text, encoding="utf-8")
    workspace = load_workspace(root)
    parent = build_snapshot(value or graph(), config(1))
    SnapshotStore(workspace.artifacts).save(parent, activate=True)
    return workspace, parent


class IncrementalTests(unittest.TestCase):
    def test_candidate_update_is_idempotent_and_matches_full_graph_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace, parent = workspace_with_parent(directory)
            events_path = Path(directory) / "events.csv"
            events_path.write_text(
                "event_id,user_id,item_id,event_type,event_value,event_time\n"
                "e1,u3,i4,click,2,2026-09-03T00:00:00Z\n",
                encoding="utf-8",
            )
            result = update_snapshot(workspace, events_path)
            store = SnapshotStore(workspace.artifacts)
            self.assertEqual(store.latest_id(), parent.snapshot_id)
            candidate = store.load(result.snapshot_id)
            self.assertEqual(candidate.parent_snapshot_id, parent.snapshot_id)
            self.assertEqual(candidate.graph.u_ids, parent.graph.u_ids)
            self.assertEqual(candidate.graph.v_ids, (*parent.graph.v_ids, "i4"))
            self.assertEqual(
                candidate.graph.incidence[
                    candidate.graph.u_index["u3"], candidate.graph.v_index["i4"]
                ],
                1,
            )
            expected = CanonicalBipartiteGraph.from_edges_and_features(
                [
                    ("u1", "i1"),
                    ("u1", "i2"),
                    ("u2", "i1"),
                    ("u2", "i2"),
                    ("u3", "i3"),
                    ("u3", "i4"),
                ],
                {"u1": [1, 0], "u2": [1, 0], "u3": [0, 1]},
                u_order=candidate.graph.u_ids,
                v_order=candidate.graph.v_ids,
            )
            self.assertEqual((candidate.graph.incidence != expected.incidence).nnz, 0)
            with self.assertRaisesRegex(ValueError, "no pending events"):
                update_snapshot(workspace, events_path)

    def test_remove_negative_feedback_and_new_user_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace, parent = workspace_with_parent(directory)
            with self.assertRaisesRegex(ValueError, "feature rows"):
                update_from_events(
                    workspace,
                    (Event("new", "u4", "i4", "click", 1, "2026-09-03T00:00:00Z"),),
                )
            result = update_from_events(
                workspace,
                (
                    Event("new", "u4", "i4", "click", 1, "2026-09-03T00:00:00Z"),
                    Event("negative", "u1", "i4", "dislike", 1, "2026-09-03T00:01:00Z"),
                ),
                feature_names=("f1", "f2"),
                feature_updates={"u4": (0, 1)},
            )
            store = SnapshotStore(workspace.artifacts)
            candidate = store.load(result.snapshot_id)
            self.assertEqual(candidate.graph.u_ids, (*parent.graph.u_ids, "u4"))
            self.assertEqual(
                candidate.graph.incidence[
                    candidate.graph.u_index["u1"], candidate.graph.v_index["i4"]
                ],
                0,
            )
            store.activate(candidate.snapshot_id)
            removed = update_from_events(
                workspace,
                (Event("remove", "u4", "i4", "remove", 1, "2026-09-03T00:02:00Z"),),
            )
            removed_snapshot = store.load(removed.snapshot_id)
            self.assertEqual(
                removed_snapshot.graph.incidence[
                    removed_snapshot.graph.u_index["u4"],
                    removed_snapshot.graph.v_index["i4"],
                ],
                0,
            )

    def test_small_closed_set_keeps_unrelated_affinity_rows(self) -> None:
        users = 10
        value = CanonicalBipartiteGraph.from_edges_and_features(
            [(f"u{index}", f"i{index}") for index in range(users)],
            {
                f"u{index}": [float(column == index) for column in range(users)]
                for index in range(users)
            },
        )
        parent_config = BuildConfig(
            AffinityConfig(top_k=2),
            EncoderConfig(hidden_dim=4, layers=1, latent_groups=2, epochs=1),
            2,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = init_workspace(Path(directory) / "workspace")
            config_path = root / "bipartitescope.toml"
            text = config_path.read_text(encoding="utf-8").replace(
                "warm_start_epochs = 20",
                "warm_start_epochs = 1",
            )
            config_path.write_text(text, encoding="utf-8")
            workspace = load_workspace(root)
            parent = build_snapshot(value, parent_config)
            store = SnapshotStore(workspace.artifacts)
            store.save(parent, activate=True)
            result = update_from_events(
                workspace,
                (Event("delta", "u0", "new-item", "click", 1, "2026-09-03T00:00:00Z"),),
            )
            candidate = store.load(result.snapshot_id)
            self.assertEqual(result.build_mode, "incremental")
            np.testing.assert_allclose(
                candidate.affinity.getrow(9).toarray(), parent.affinity.getrow(9).toarray()
            )

    def test_parent_affinity_tamper_triggers_verified_full_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace, parent = workspace_with_parent(directory)
            with (workspace.artifacts / parent.snapshot_id / "W.npz").open("ab") as handle:
                handle.write(b"tampered")
            result = update_from_events(
                workspace,
                (Event("recover", "u3", "i4", "click", 1, "2026-09-03T00:00:00Z"),),
            )
            self.assertEqual(result.build_mode, "full_fallback")
            self.assertEqual(result.fallback_reason, "parent_snapshot_integrity_failed")
            recovered = SnapshotStore(workspace.artifacts).load(result.snapshot_id)
            self.assertEqual(recovered.parent_snapshot_id, parent.snapshot_id)


if __name__ == "__main__":
    unittest.main()
