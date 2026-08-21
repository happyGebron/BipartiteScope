# REST API

Install `bilcs[api]` and run the application returned by `bilcs.api.create_app()` with Uvicorn. `GET /health` verifies service availability. `POST /workspaces/{workspace_id}/build` accepts JSON with `edges`, `features`, and optional nested build `config`; it returns a snapshot ID. `GET /workspaces/{workspace_id}/snapshots/{snapshot_id}/query/{entity_id}?size_budget=20` returns members, support evidence, and the expansion trace.

Workspace metadata is stored in local SQLite; model artifacts live under a workspace-specific snapshot store.
