"""Delimited input validation with actionable, machine-readable reports."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .domain import CanonicalBipartiteGraph


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Schema and data-quality evidence produced before an offline build."""

    edges_path: str
    features_path: str
    delimiter: str
    counts: dict[str, int]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "valid": self.valid}


class InputValidationError(ValueError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__("; ".join(report.errors) or "input validation failed")


def _read_rows(path: Path, delimiter: str) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return list(reader), tuple(reader.fieldnames or ())


def validate_csv_graph(edges_path: str | Path, features_path: str | Path, *, delimiter: str = ",") -> ValidationReport:
    """Validate generic `u_id,v_id` and `u_id,f1,...,fd` files without reading labels."""
    edges_file, features_file = Path(edges_path), Path(features_path)
    errors: list[str] = []
    warnings: list[str] = []
    edge_rows: list[dict[str, str]] = []
    feature_rows: list[dict[str, str]] = []
    edge_columns: tuple[str, ...] = ()
    feature_columns: tuple[str, ...] = ()
    for path, target in ((edges_file, "edges"), (features_file, "features")):
        if not path.is_file():
            errors.append(f"{target} file does not exist: {path}")
    if not errors:
        edge_rows, edge_columns = _read_rows(edges_file, delimiter)
        feature_rows, feature_columns = _read_rows(features_file, delimiter)
    if not {"u_id", "v_id"}.issubset(edge_columns):
        errors.append("edge file requires exactly named u_id and v_id columns")
    feature_fields = tuple(column for column in feature_columns if column != "u_id")
    if "u_id" not in feature_columns or not feature_fields:
        errors.append("feature file requires u_id plus one or more feature columns")
    edge_pairs = [(row.get("u_id", ""), row.get("v_id", "")) for row in edge_rows]
    feature_ids = [row.get("u_id", "") for row in feature_rows]
    if edge_rows and any(not u or not v for u, v in edge_pairs):
        errors.append("edge file contains blank u_id or v_id values")
    if feature_rows and any(not entity for entity in feature_ids):
        errors.append("feature file contains a blank u_id value")
    duplicate_edges = len(edge_pairs) - len(set(edge_pairs))
    duplicate_feature_ids = len(feature_ids) - len(set(feature_ids))
    if duplicate_edges:
        warnings.append(f"{duplicate_edges} duplicate edge rows will be merged")
    if duplicate_feature_ids:
        errors.append(f"feature file contains {duplicate_feature_ids} duplicate u_id rows")
    invalid_values = 0
    negative_values = 0
    for row in feature_rows:
        for column in feature_fields:
            try:
                value = float(row.get(column, ""))
            except ValueError:
                invalid_values += 1
                continue
            if value < 0:
                negative_values += 1
    if invalid_values:
        errors.append(f"feature file contains {invalid_values} non-numeric feature values")
    if negative_values:
        errors.append(f"feature file contains {negative_values} negative feature values")
    missing_features = sorted({u for u, _ in edge_pairs} - set(feature_ids))
    if missing_features:
        errors.append(f"{len(missing_features)} U-side IDs used by edges have no feature row (for example: {missing_features[0]})")
    edge_u_ids = {u for u, _ in edge_pairs}
    unused_features = len(set(feature_ids) - edge_u_ids)
    if unused_features:
        warnings.append(f"{unused_features} feature rows have no incident edge and will remain isolated")
    if not edge_rows:
        errors.append("edge file contains no data rows")
    if not feature_rows:
        errors.append("feature file contains no data rows")
    return ValidationReport(
        str(edges_file), str(features_file), delimiter,
        {"edge_rows": len(edge_rows), "unique_edges": len(set(edge_pairs)), "feature_rows": len(feature_rows), "feature_dimensions": len(feature_fields), "duplicate_edges": duplicate_edges, "unused_feature_rows": unused_features},
        tuple(errors), tuple(warnings),
    )


def load_csv_graph(edges_path: str | Path, features_path: str | Path, *, delimiter: str = ",") -> CanonicalBipartiteGraph:
    """Load a validated generic CSV graph; no dataset or labels are assumed."""
    report = validate_csv_graph(edges_path, features_path, delimiter=delimiter)
    if not report.valid:
        raise InputValidationError(report)
    edge_rows, _ = _read_rows(Path(edges_path), delimiter)
    feature_rows, feature_columns = _read_rows(Path(features_path), delimiter)
    columns = [column for column in feature_columns if column != "u_id"]
    features = {row["u_id"]: [float(row[column]) for column in columns] for row in feature_rows}
    return CanonicalBipartiteGraph.from_edges_and_features(((row["u_id"], row["v_id"]) for row in edge_rows), features)
