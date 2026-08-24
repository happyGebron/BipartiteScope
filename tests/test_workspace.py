import json
from pathlib import Path
import tempfile
import unittest

from bipartite_scope.io import validate_csv_graph
from bipartite_scope.workspace import CONFIG_FILENAME, init_workspace, load_workspace


class WorkspaceTests(unittest.TestCase):
    def test_init_creates_empty_workspace_without_sample_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace_path = init_workspace(Path(directory) / "workspace")
            self.assertTrue((workspace_path / CONFIG_FILENAME).is_file())
            self.assertFalse(any((workspace_path / "data").iterdir()))
            report = load_workspace(workspace_path).validate()
            self.assertFalse(report.valid)
            self.assertIn("edges file does not exist", report.errors[0])

    def test_validation_reports_duplicate_edges_and_missing_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            edges = root / "edges.csv"; features = root / "features.csv"
            edges.write_text("u_id,v_id\nu1,v1\nu1,v1\nu2,v2\n", encoding="utf-8")
            features.write_text("u_id,f1\nu1,1\n", encoding="utf-8")
            report = validate_csv_graph(edges, features)
            self.assertFalse(report.valid)
            self.assertEqual(report.counts["duplicate_edges"], 1)
            self.assertIn("U-side IDs", " ".join(report.errors))
