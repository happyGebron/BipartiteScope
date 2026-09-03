import json
import tempfile
import unittest
from pathlib import Path

from test_core import config, graph

from bipartite_scope import Event, SnapshotIntegrityError, SnapshotStore, build_snapshot
from bipartite_scope.storage import (
    all_events,
    init_workspace,
    load_events,
    load_workspace,
    normalize_event,
    register_events,
)


class StorageTests(unittest.TestCase):
    def test_workspace_starts_empty_and_reports_missing_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = init_workspace(Path(directory) / "workspace")
            self.assertFalse(any((root / "data").iterdir()))
            report = load_workspace(root).validate()
            self.assertFalse(report.valid)
            self.assertIn("edges file does not exist", report.errors[0])

    def test_csv_jsonl_parity_timezone_and_malformed_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "events.csv"
            jsonl_path = root / "events.jsonl"
            csv_path.write_text(
                "event_id,user_id,item_id,event_type,event_value,event_time\n"
                "e1,u1,i1,click,2,2026-09-03T08:00:00+08:00\n",
                encoding="utf-8",
            )
            jsonl_path.write_text(
                json.dumps(
                    {
                        "event_id": "e1",
                        "user_id": "u1",
                        "item_id": "i1",
                        "event_type": "click",
                        "event_value": 2,
                        "event_time": "2026-09-03T00:00:00Z",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_events(csv_path), load_events(jsonl_path))
            with self.assertRaisesRegex(ValueError, "timezone"):
                normalize_event(
                    {
                        "event_id": "x",
                        "user_id": "u",
                        "item_id": "i",
                        "event_type": "view",
                        "event_time": "2026-09-03T00:00:00",
                    }
                )
            with self.assertRaisesRegex(ValueError, "unsupported"):
                normalize_event(
                    {
                        "event_id": "x",
                        "user_id": "u",
                        "item_id": "i",
                        "event_type": "unknown",
                        "event_time": "2026-09-03T00:00:00Z",
                    }
                )

    def test_event_idempotency_and_append_only_ledgers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = load_workspace(init_workspace(Path(directory) / "workspace"))
            event = Event("e1", "u1", "i1", "click", 1.0, "2026-09-03T00:00:00Z")
            accepted, duplicates = register_events(workspace, (event, event), source="test")
            self.assertEqual(len(accepted), 1)
            self.assertEqual(duplicates, ("e1",))
            self.assertEqual(all_events(workspace), (event,))
            self.assertEqual(len((workspace.data / "events.jsonl").read_text().splitlines()), 1)

    def test_snapshot_activation_rollback_tamper_and_v1_loading(self) -> None:
        first = build_snapshot(graph(), config(1))
        second = build_snapshot(graph(), config(1), parent_snapshot_id=first.snapshot_id)
        with tempfile.TemporaryDirectory() as directory:
            store = SnapshotStore(directory)
            first_path = store.save(first, activate=True)
            store.save(second)
            self.assertEqual(store.latest_id(), first.snapshot_id)
            store.activate(second.snapshot_id)
            self.assertEqual(store.latest_id(), second.snapshot_id)
            store.rollback(first.snapshot_id)
            self.assertEqual(store.latest_id(), first.snapshot_id)
            manifest_path = first_path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for name in ("R.npz", "V.npy"):
                (first_path / name).unlink()
                manifest["assets"].pop(name)
            manifest["core_version"] = "1.0.0"
            manifest.pop("schema_version")
            for name in ("parent_snapshot_id", "build_mode", "event_batch_hash", "update_stats"):
                manifest.pop(name)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = store.load(first.snapshot_id)
            self.assertEqual((loaded.interaction_weights != loaded.graph.incidence).nnz, 0)
            self.assertEqual(loaded.build_mode, "full")
            with (first_path / "W.npz").open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaises(SnapshotIntegrityError):
                store.load(first.snapshot_id)


if __name__ == "__main__":
    unittest.main()
