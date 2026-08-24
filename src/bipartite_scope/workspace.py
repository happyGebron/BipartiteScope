"""Workspace-first product workflow with a standard-library TOML configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from .config import AffinityConfig, BuildConfig, EncoderConfig, QueryConfig
from .domain import CanonicalBipartiteGraph
from .io import ValidationReport, load_csv_graph, validate_csv_graph


CONFIG_FILENAME = "bipartitescope.toml"

_DEFAULT_CONFIG = """[data]
edges = \"data/edges.csv\"
features = \"data/features.csv\"
delimiter = \",\"

[core.affinity]
beta = 0.5
restart = 0.3
steps = 3
top_k = 64

[core.encoder]
hidden_dim = 64
layers = 2
latent_groups = 8
epochs = 100
learning_rate = 0.01
lambda_b = 0.5
lambda_o = 0.1
seed = 7

[query]
size_budget = 20
rho = 0.5
eta_s = 0.65
delta_0 = 0.02
semantic_recall_budget = 100
"""


@dataclass(frozen=True, slots=True)
class Workspace:
    root: Path
    config: dict

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def data_paths(self) -> tuple[Path, Path, str]:
        data = self.config["data"]
        return self.root / data["edges"], self.root / data["features"], data.get("delimiter", ",")

    def build_config(self) -> BuildConfig:
        core = self.config["core"]
        return BuildConfig(AffinityConfig(**core["affinity"]), EncoderConfig(**core["encoder"]), self.config["query"].get("semantic_recall_budget", 100))

    def query_config(self, *, size_budget: int | None = None) -> QueryConfig:
        values = dict(self.config["query"])
        if size_budget is not None:
            values["size_budget"] = size_budget
        return QueryConfig(**values)

    def validate(self) -> ValidationReport:
        edges, features, delimiter = self.data_paths
        return validate_csv_graph(edges, features, delimiter=delimiter)

    def load_graph(self) -> CanonicalBipartiteGraph:
        edges, features, delimiter = self.data_paths
        return load_csv_graph(edges, features, delimiter=delimiter)


def init_workspace(root: str | Path) -> Path:
    target = Path(root).resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"workspace is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    for name in ("data", "artifacts", "exports", "reports"):
        (target / name).mkdir(exist_ok=True)
    config = target / CONFIG_FILENAME
    if not config.exists():
        config.write_text(_DEFAULT_CONFIG, encoding="utf-8")
    return target


def load_workspace(root: str | Path) -> Workspace:
    target = Path(root).resolve()
    config_path = target / CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(f"workspace configuration not found: {config_path}")
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    for section in ("data", "core", "query"):
        if section not in config:
            raise ValueError(f"workspace configuration is missing [{section}]")
    if not {"affinity", "encoder"}.issubset(config["core"]):
        raise ValueError("workspace configuration requires [core.affinity] and [core.encoder]")
    return Workspace(target, config)
