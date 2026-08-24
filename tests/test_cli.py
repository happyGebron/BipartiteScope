from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from bipartite_scope.cli import build_parser, run


class CliWorkflowTests(unittest.TestCase):
    def _run(self, *argv: str) -> tuple[int, dict]:
        stream = StringIO()
        with redirect_stdout(stream):
            code = run(build_parser().parse_args(list(argv)))
        return code, json.loads(stream.getvalue())

    def test_workspace_flow_has_no_bundled_data_and_exports_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            code, payload = self._run("init", str(workspace))
            self.assertEqual(code, 0)
            self.assertTrue(Path(payload["workspace"]).is_dir())
            self.assertFalse(any((workspace / "data").iterdir()))
            (workspace / "data" / "edges.csv").write_text("u_id,v_id\na,v1\na,v2\nb,v1\nb,v2\nc,v3\n", encoding="utf-8")
            (workspace / "data" / "features.csv").write_text("u_id,f1,f2\na,1,0\nb,1,0\nc,0,1\n", encoding="utf-8")
            config = workspace / "bipartitescope.toml"
            config.write_text(config.read_text(encoding="utf-8").replace("latent_groups = 8", "latent_groups = 2").replace("epochs = 100", "epochs = 2"), encoding="utf-8")
            code, validation = self._run("validate", "--workspace", str(workspace))
            self.assertEqual(code, 0)
            self.assertTrue(validation["valid"])
            code, build = self._run("build", "--workspace", str(workspace))
            self.assertEqual(code, 0)
            snapshot_id = build["snapshot_id"]
            code, query = self._run("query", "--workspace", str(workspace), "--snapshot", snapshot_id, "--entity", "a", "--size", "2")
            self.assertEqual(code, 0)
            self.assertIn("a", query["members"])
            code, exported = self._run("export", "--workspace", str(workspace), "--snapshot", snapshot_id, "--entity", "a", "--size", "2", "--name", "a.json")
            self.assertEqual(code, 0)
            self.assertTrue(Path(exported["output"]).is_file())
