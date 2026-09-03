import unittest
from pathlib import Path


class RepositoryPolicyTests(unittest.TestCase):
    def test_consolidated_source_and_documentation_layout(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = root / "src" / "bipartite_scope"
        self.assertEqual(
            {path.name for path in source.glob("*.py")},
            {"__init__.py", "core.py", "storage.py", "recommendation.py", "interface.py"},
        )
        self.assertFalse((root / "docs").exists())
        self.assertTrue((root / "DOCUMENTATION.md").is_file())
        self.assertEqual(len(list(root.rglob("DOCUMENTATION.md"))), 1)

    def test_repository_text_is_english_and_contains_no_dataset_files(self) -> None:
        root = Path(__file__).resolve().parents[1]
        datasets = [
            path
            for pattern in ("*.csv", "*.jsonl", "*.ndjson")
            for path in root.rglob(pattern)
            if ".venv" not in path.parts
        ]
        self.assertEqual(datasets, [])
        offenders = []
        for path in root.rglob("*"):
            if (
                not path.is_file()
                or ".git" in path.parts
                or ".venv" in path.parts
                or "__pycache__" in path.parts
            ):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if any("\u3400" <= character <= "\u9fff" for character in text):
                offenders.append(str(path.relative_to(root)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
