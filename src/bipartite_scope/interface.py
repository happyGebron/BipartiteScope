from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .core import (
    AffinityConfig,
    BuildConfig,
    CanonicalBipartiteGraph,
    EncoderConfig,
    Event,
    IncrementalConfig,
    QueryEngine,
    build_snapshot,
)
from .recommendation import (
    _apply_events,
    evaluate,
    recommend,
    record_feedback,
    update_from_events,
    update_snapshot,
)
from .storage import (
    InputValidationError,
    SnapshotIntegrityError,
    SnapshotStore,
    all_events,
    init_workspace,
    load_workspace,
    normalize_event,
    validate_csv_graph,
)


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _write_payload(path: str | Path, payload: Any) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(output)


def _workspace_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", required=True, help="path to a BipartiteScope workspace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bipartite-scope", description="BipartiteScope graph intelligence engine"
    )
    parser.add_argument("--version", action="version", version="BipartiteScope 2.0.0")
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="create an empty local workspace")
    initialize.add_argument("workspace")
    validate = commands.add_parser("validate", help="validate workspace or direct CSV input")
    validate.add_argument("--workspace")
    validate.add_argument("--edges")
    validate.add_argument("--features")
    validate.add_argument("--delimiter", default=",")
    validate.add_argument("--report")
    build = commands.add_parser("build", help="build an immutable full snapshot")
    _workspace_argument(build)
    query = commands.add_parser("query", help="run BLC community search")
    _workspace_argument(query)
    query.add_argument("--entity", required=True)
    query.add_argument("--snapshot")
    query.add_argument("--size", type=int)
    query.add_argument("--output")
    export = commands.add_parser("export", help="export a community search result")
    _workspace_argument(export)
    export.add_argument("--entity", required=True)
    export.add_argument("--snapshot")
    export.add_argument("--size", type=int)
    export.add_argument("--name", default="community.json")
    update = commands.add_parser("update", help="build a candidate snapshot from an event batch")
    _workspace_argument(update)
    update.add_argument("--events", required=True)
    update.add_argument("--features")
    recommendation = commands.add_parser("recommend", help="generate ranked item recommendations")
    _workspace_argument(recommendation)
    recommendation.add_argument("--user", required=True)
    recommendation.add_argument("--snapshot")
    recommendation.add_argument("--top-n", type=int)
    recommendation.add_argument("--output")
    feedback = commands.add_parser("feedback", help="append normalized feedback without retraining")
    _workspace_argument(feedback)
    feedback.add_argument("--user", required=True)
    feedback.add_argument("--item", required=True)
    feedback.add_argument("--event-type", required=True)
    feedback.add_argument("--value", type=float, default=1.0)
    feedback.add_argument("--event-id")
    feedback.add_argument("--event-time")
    evaluation = commands.add_parser("evaluate", help="run chronological offline evaluation")
    _workspace_argument(evaluation)
    evaluation.add_argument("--snapshot")
    evaluation.add_argument("--k", type=int)
    snapshot = commands.add_parser("snapshot", help="manage immutable snapshots")
    snapshot_commands = snapshot.add_subparsers(dest="snapshot_command", required=True)
    snapshot_list = snapshot_commands.add_parser("list")
    _workspace_argument(snapshot_list)
    for name in ("verify", "activate", "rollback"):
        operation = snapshot_commands.add_parser(name)
        _workspace_argument(operation)
        operation.add_argument("--snapshot", required=True)
    benchmark = commands.add_parser("benchmark", help="run a generated sparse build benchmark")
    benchmark.add_argument("--users", type=int, required=True)
    benchmark.add_argument("--items", type=int, required=True)
    benchmark.add_argument("--events", type=int, required=True)
    benchmark.add_argument("--delta-ratio", type=float, required=True)
    benchmark.add_argument("--output")
    return parser


def _validate(args: argparse.Namespace) -> int:
    if args.workspace:
        if args.edges or args.features:
            raise ValueError("use either --workspace or --edges and --features")
        report = load_workspace(args.workspace).validate()
    elif args.edges and args.features:
        report = validate_csv_graph(args.edges, args.features, delimiter=args.delimiter)
    else:
        raise ValueError("validate requires --workspace or both --edges and --features")
    payload = report.to_dict()
    if args.report:
        payload["report"] = _write_payload(args.report, payload)
    _json(payload)
    return 0 if report.valid else 2


def _query(args: argparse.Namespace, export_name: str | None = None) -> int:
    workspace = load_workspace(args.workspace)
    store = SnapshotStore(workspace.artifacts)
    snapshot_id = args.snapshot or store.latest_id()
    result = QueryEngine(store.load(snapshot_id)).search(
        args.entity,
        workspace.query_config(size_budget=args.size),
    )
    payload = asdict(result)
    output = getattr(args, "output", None)
    if export_name:
        output = workspace.exports / export_name
    if output:
        payload["output"] = _write_payload(output, payload)
    _json(payload)
    return 0


def _benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.users, args.items, args.events) < 1 or not 0 <= args.delta_ratio <= 1:
        raise ValueError("benchmark sizes must be positive and delta ratio must be in [0, 1]")
    rng = np.random.default_rng(17)
    edges = {
        (f"u{int(rng.integers(args.users))}", f"i{int(rng.integers(args.items))}")
        for _ in range(args.events * 2)
    }
    edges = set(sorted(edges)[: args.events])
    for user in range(args.users):
        edges.add((f"u{user}", f"i{user % args.items}"))
    features = {f"u{user}": [1.0, float(user % 3), float(user % 5)] for user in range(args.users)}
    graph = CanonicalBipartiteGraph.from_edges_and_features(edges, features)
    config = BuildConfig(
        AffinityConfig(top_k=min(16, args.users)),
        EncoderConfig(hidden_dim=8, latent_groups=min(4, args.users), epochs=2),
        semantic_recall_budget=min(16, args.users),
    )
    started = time.perf_counter()
    parent = build_snapshot(graph, config)
    full_seconds = time.perf_counter() - started
    count = max(1, round(args.events * args.delta_ratio))
    timestamp = datetime.now(UTC)
    events = tuple(
        Event(
            f"benchmark-{index}",
            f"u{index % args.users}",
            f"i{(index * 7 + 1) % args.items}",
            "click",
            1.0,
            (timestamp + timedelta(seconds=index)).isoformat(),
        )
        for index in range(count)
    )
    started = time.perf_counter()
    updated_graph, weights, affinity, stats = _apply_events(
        parent,
        events,
        (),
        {},
        IncrementalConfig(max_affected_ratio=max(args.delta_ratio, 0.01), warm_start_epochs=1),
    )
    warm_config = replace(config, encoder=replace(config.encoder, epochs=1))
    build_snapshot(
        updated_graph,
        warm_config,
        interaction_weights=weights,
        initial_state=parent.model_state,
        affinity_override=affinity,
        parent_snapshot_id=parent.snapshot_id,
        build_mode=stats["build_mode"],
    )
    incremental_seconds = time.perf_counter() - started
    return {
        "users": args.users,
        "items": args.items,
        "events": len(edges),
        "delta_events": count,
        "full_build_seconds": full_seconds,
        "incremental_build_seconds": incremental_seconds,
        "affected_ratio": stats["affected_ratio"],
        "build_mode": stats["build_mode"],
        "dense_user_by_user_matrix_materialized": False,
    }


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
            _json(report.to_dict())
            return 2
        snapshot = build_snapshot(workspace.load_graph(), workspace.build_config())
        store = SnapshotStore(workspace.artifacts)
        try:
            store.latest_id()
            activate = False
        except SnapshotIntegrityError:
            activate = True
        path = store.save(snapshot, activate=activate)
        _json({"snapshot_id": snapshot.snapshot_id, "path": str(path), "active": activate})
        return 0
    if args.command == "query":
        return _query(args)
    if args.command == "export":
        return _query(args, args.name)
    if args.command == "update":
        result = update_snapshot(load_workspace(args.workspace), args.events, args.features)
        _json(asdict(result))
        return 0
    if args.command == "recommend":
        workspace = load_workspace(args.workspace)
        store = SnapshotStore(workspace.artifacts)
        snapshot = store.load(args.snapshot or store.latest_id())
        result = recommend(
            snapshot,
            args.user,
            workspace.recommendation_config(args.top_n),
            events=all_events(workspace),
        )
        payload = asdict(result)
        if args.output:
            payload["output"] = _write_payload(args.output, payload)
        _json(payload)
        return 0
    if args.command == "feedback":
        event = normalize_event(
            {
                "event_id": args.event_id or str(uuid4()),
                "user_id": args.user,
                "item_id": args.item,
                "event_type": args.event_type,
                "event_value": args.value,
                "event_time": args.event_time or datetime.now(UTC).isoformat(),
            }
        )
        _json(record_feedback(load_workspace(args.workspace), event))
        return 0
    if args.command == "evaluate":
        workspace = load_workspace(args.workspace)
        _json(asdict(evaluate(workspace, args.snapshot, workspace.evaluation_config(args.k))))
        return 0
    if args.command == "snapshot":
        workspace = load_workspace(args.workspace)
        store = SnapshotStore(workspace.artifacts)
        if args.snapshot_command == "list":
            _json({"snapshots": store.list()})
        elif args.snapshot_command == "verify":
            _json(store.verify(args.snapshot))
        elif args.snapshot_command == "activate":
            _json({"snapshot_id": store.activate(args.snapshot), "active": True})
        else:
            _json({"snapshot_id": store.rollback(args.snapshot), "active": True, "rollback": True})
        return 0
    if args.command == "benchmark":
        payload = _benchmark(args)
        if args.output:
            payload["output"] = _write_payload(args.output, payload)
        _json(payload)
        return 0
    raise AssertionError(f"unknown command: {args.command}")


def main(argv: list[str] | None = None) -> None:
    try:
        code = run(build_parser().parse_args(argv))
    except (
        FileNotFoundError,
        FileExistsError,
        KeyError,
        ValueError,
        InputValidationError,
        SnapshotIntegrityError,
    ) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        code = 2
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        code = 1
    raise SystemExit(code)


def _api_config(raw: dict[str, Any], users: int) -> BuildConfig:
    encoder = dict(raw.get("encoder", {}))
    encoder.setdefault("latent_groups", min(8, users))
    return BuildConfig(
        AffinityConfig(**raw.get("affinity", {})),
        EncoderConfig(**encoder),
        raw.get("semantic_recall_budget", 100),
    )


def create_app(root: str | Path = "workspaces") -> Any:
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError("REST API requires the api optional dependency") from exc
    workspace_root = Path(root).resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="BipartiteScope", version="2.0.0")

    def workspace_for(workspace_id: str, create: bool = False) -> Any:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", workspace_id):
            raise HTTPException(status_code=422, detail="invalid workspace identifier")
        path = (workspace_root / workspace_id).resolve()
        if workspace_root not in path.parents:
            raise HTTPException(
                status_code=422, detail="workspace path escapes the configured root"
            )
        if create and not path.exists():
            init_workspace(path)
        try:
            return load_workspace(path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "2.0.0"}

    @app.post("/workspaces/{workspace_id}/build")
    def api_build(workspace_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            workspace = workspace_for(workspace_id, create=True)
            graph = CanonicalBipartiteGraph.from_edges_and_features(
                payload["edges"], payload["features"]
            )
            snapshot = build_snapshot(
                graph, _api_config(payload.get("config", {}), len(graph.u_ids))
            )
            store = SnapshotStore(workspace.artifacts)
            try:
                store.latest_id()
                activate = False
            except SnapshotIntegrityError:
                activate = True
            store.save(snapshot, activate=activate)
            return {
                "workspace_id": workspace_id,
                "snapshot_id": snapshot.snapshot_id,
                "active": activate,
            }
        except HTTPException:
            raise
        except (KeyError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/workspaces/{workspace_id}/snapshots/{snapshot_id}/query/{entity_id}")
    def api_query(
        workspace_id: str,
        snapshot_id: str,
        entity_id: str,
        size_budget: int = 20,
    ) -> dict[str, Any]:
        try:
            workspace = workspace_for(workspace_id)
            snapshot = SnapshotStore(workspace.artifacts).load(snapshot_id)
            return asdict(
                QueryEngine(snapshot).search(entity_id, workspace.query_config(size_budget))
            )
        except HTTPException:
            raise
        except (KeyError, ValueError, SnapshotIntegrityError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/workspaces/{workspace_id}/update")
    def api_update(workspace_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            events = tuple(normalize_event(record) for record in payload["events"])
            result = update_from_events(
                workspace_for(workspace_id),
                events,
                feature_names=payload.get("feature_names", ()),
                feature_updates=payload.get("features", {}),
            )
            return asdict(result)
        except HTTPException:
            raise
        except (KeyError, ValueError, SnapshotIntegrityError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/workspaces/{workspace_id}/recommend")
    def api_recommend(workspace_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            workspace = workspace_for(workspace_id)
            store = SnapshotStore(workspace.artifacts)
            snapshot = store.load(payload.get("snapshot_id") or store.latest_id())
            result = recommend(
                snapshot,
                str(payload["user_id"]),
                workspace.recommendation_config(payload.get("top_n")),
                events=all_events(workspace),
            )
            return asdict(result)
        except HTTPException:
            raise
        except (KeyError, ValueError, SnapshotIntegrityError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/workspaces/{workspace_id}/feedback")
    def api_feedback(workspace_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            value = {
                **payload,
                "event_id": payload.get("event_id") or str(uuid4()),
                "event_value": payload.get("event_value", 1.0),
                "event_time": payload.get("event_time") or datetime.now(UTC).isoformat(),
            }
            return record_feedback(workspace_for(workspace_id), normalize_event(value))
        except HTTPException:
            raise
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/workspaces/{workspace_id}/evaluate")
    def api_evaluate(workspace_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            workspace = workspace_for(workspace_id)
            return asdict(
                evaluate(
                    workspace,
                    payload.get("snapshot_id"),
                    workspace.evaluation_config(payload.get("k")),
                )
            )
        except HTTPException:
            raise
        except (ValueError, SnapshotIntegrityError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/workspaces/{workspace_id}/snapshots")
    def api_snapshots(workspace_id: str) -> dict[str, Any]:
        return {"snapshots": SnapshotStore(workspace_for(workspace_id).artifacts).list()}

    @app.post("/workspaces/{workspace_id}/snapshots/{snapshot_id}/activate")
    def api_activate(workspace_id: str, snapshot_id: str) -> dict[str, Any]:
        try:
            value = SnapshotStore(workspace_for(workspace_id).artifacts).activate(snapshot_id)
            return {"snapshot_id": value, "active": True}
        except HTTPException:
            raise
        except SnapshotIntegrityError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/workspaces/{workspace_id}/snapshots/{snapshot_id}/verify")
    def api_verify(workspace_id: str, snapshot_id: str) -> dict[str, Any]:
        try:
            return SnapshotStore(workspace_for(workspace_id).artifacts).verify(snapshot_id)
        except HTTPException:
            raise
        except SnapshotIntegrityError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


if __name__ == "__main__":
    main()
