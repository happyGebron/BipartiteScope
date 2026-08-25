"""Workspace-oriented command line interface for BipartiteScope."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

from .io import InputValidationError, validate_csv_graph
from .query import QueryEngine
from .service import build_snapshot
from .snapshot import SnapshotIntegrityError, SnapshotStore
from .workspace import init_workspace, load_workspace


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _add_workspace_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", required=True, help="workspace created by bipartite-scope init")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bipartite-scope", description="BipartiteScope community and association analysis engine")
    parser.add_argument("--version", action="version", version="BipartiteScope 1.0.0")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="create an empty workspace and TOML configuration")
    init.add_argument("workspace")
    validate = commands.add_parser("validate", help="validate a workspace or direct CSV input and emit a JSON report")
    validate.add_argument("--workspace")
    validate.add_argument("--edges")
    validate.add_argument("--features")
    validate.add_argument("--delimiter", default=",")
    validate.add_argument("--report", help="optional report JSON path")
    build = commands.add_parser("build", help="build an immutable snapshot from a workspace")
    _add_workspace_argument(build)
    query = commands.add_parser("query", help="query a verified workspace snapshot")
    _add_workspace_argument(query)
    query.add_argument("--entity", required=True)
    query.add_argument("--snapshot", help="defaults to the latest verified snapshot")
    query.add_argument("--size", type=int)
    query.add_argument("--output", help="optional result JSON path")
    export = commands.add_parser("export", help="write a previous query result to the workspace exports directory")
    _add_workspace_argument(export)
    export.add_argument("--entity", required=True)
    export.add_argument("--snapshot")
    export.add_argument("--size", type=int)
    export.add_argument("--name", default="community.json")
    return parser


def _validate(args: argparse.Namespace) -> int:
    if args.workspace:
        if args.edges or args.features:
            raise ValueError("use either --workspace or --edges/--features, not both")
        report = load_workspace(args.workspace).validate()
    elif args.edges and args.features:
        report = validate_csv_graph(args.edges, args.features, delimiter=args.delimiter)
    else:
        raise ValueError("validate requires --workspace or both --edges and --features")
    payload = report.to_dict()
    _json(payload)
    if args.report:
        path = Path(args.report); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if report.valid else 2


def _query(args: argparse.Namespace, *, default_export: str | None = None) -> int:
    workspace = load_workspace(args.workspace)
    store = SnapshotStore(workspace.artifacts)
    snapshot_id = args.snapshot or store.latest_id()
    result = QueryEngine(store.load(snapshot_id)).search(args.entity, workspace.query_config(size_budget=args.size))
    payload = asdict(result)
    output = getattr(args, "output", None)
    if default_export:
        output = str(workspace.exports / default_export)
    if output:
        path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["output"] = str(path)
    _json(payload)
    return 0


def run(args: argparse.Namespace) -> int:
    if args.command == "init":
        _json({"workspace": str(init_workspace(args.workspace)), "config": "bipartitescope.toml"})
        return 0
    if args.command == "validate":
        return _validate(args)
    if args.command == "build":
        workspace = load_workspace(args.workspace)
        report = workspace.validate()
        if not report.valid:
            _json(report.to_dict()); return 2
        snapshot = build_snapshot(workspace.load_graph(), workspace.build_config())
        _json({"snapshot_id": snapshot.snapshot_id, "path": str(SnapshotStore(workspace.artifacts).save(snapshot))})
        return 0
    if args.command == "query":
        return _query(args)
    if args.command == "export":
        return _query(args, default_export=args.name)
    raise AssertionError(f"unknown command: {args.command}")


def main(argv: list[str] | None = None) -> None:
    try:
        code = run(build_parser().parse_args(argv))
    except (FileNotFoundError, FileExistsError, ValueError, InputValidationError, SnapshotIntegrityError) as error:
        print(f"error: {error}", file=sys.stderr)
        code = 2
    raise SystemExit(code)


if __name__ == "__main__":
    main()
