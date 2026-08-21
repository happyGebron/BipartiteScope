"""Command-line entry point; commands are added with their service capabilities."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import AffinityConfig, BuildConfig, EncoderConfig, QueryConfig
from .io import load_csv_graph
from .query import QueryEngine
from .service import build_snapshot
from .snapshot import SnapshotStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="bilcs", description="BiLCS bipartite analysis engine")
    parser.add_argument("--version", action="version", version="BiLCS 1.0.0")
    commands = parser.add_subparsers(dest="command")
    validate = commands.add_parser("validate", help="validate generic U-V edge and U-feature CSV files")
    validate.add_argument("--edges", required=True)
    validate.add_argument("--features", required=True)
    validate.add_argument("--delimiter", default=",")
    build = commands.add_parser("build", help="build and persist a reusable model snapshot")
    build.add_argument("--edges", required=True); build.add_argument("--features", required=True); build.add_argument("--artifacts", required=True)
    build.add_argument("--epochs", type=int, default=100); build.add_argument("--latent-groups", type=int, default=8)
    query = commands.add_parser("query", help="query one immutable snapshot")
    query.add_argument("--artifacts", required=True); query.add_argument("--snapshot", required=True); query.add_argument("--entity", required=True); query.add_argument("--size", type=int, default=20)
    args = parser.parse_args()
    if args.command == "validate":
        graph = load_csv_graph(args.edges, args.features, delimiter=args.delimiter)
        print(f"valid: {len(graph.u_ids)} U entities, {len(graph.v_ids)} V entities, {graph.incidence.nnz} interactions")
    elif args.command == "build":
        graph = load_csv_graph(args.edges, args.features)
        groups = min(args.latent_groups, len(graph.u_ids))
        config = BuildConfig(AffinityConfig(), EncoderConfig(epochs=args.epochs, latent_groups=groups))
        snapshot = build_snapshot(graph, config)
        path = SnapshotStore(args.artifacts).save(snapshot)
        print(f"built snapshot {snapshot.snapshot_id} at {path}")
    elif args.command == "query":
        snapshot = SnapshotStore(args.artifacts).load(args.snapshot)
        result = QueryEngine(snapshot).search(args.entity, QueryConfig(size_budget=args.size))
        print("members:", ", ".join(result.members))
        print("termination:", result.termination_reason)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
