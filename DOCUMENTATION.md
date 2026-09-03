# BipartiteScope Documentation

BipartiteScope 2.0.0 is a local, application-oriented engine for attributed bipartite graphs, incremental interaction updates, community search, recommendation, feedback collection, offline evaluation, and immutable snapshot management. It does not bundle datasets or require an external database service.

## Installation

Python 3.11 or 3.12 is required.

```bash
python -m pip install -e '.[train,api,dev]'
bipartite-scope --version
```

The `train` extra installs PyTorch. The `api` extra installs FastAPI and Uvicorn. The core runtime uses NumPy, SciPy, local files, and standard-library SQLite.

## Workspace and data schemas

Create a workspace without sample data:

```bash
bipartite-scope init my-workspace
```

The workspace contains `data/`, `artifacts/`, `exports/`, `reports/`, and `bipartitescope.toml`. Add user-owned input files to `data/`.

The full-build edge schema is:

```text
u_id,v_id
```

The feature schema is:

```text
u_id,<one or more nonnegative feature columns>
```

Duplicate edges are merged. Every user referenced by an edge must have one finite, nonnegative feature row. All feature rows must have the same dimension. Business identifiers remain separate from internal matrix indexes.

Incremental input may be CSV or JSONL. Every record uses:

```text
event_id,user_id,item_id,event_type,event_value,event_time
```

`event_id`, `user_id`, `item_id`, `event_type`, and `event_time` are required. `event_value` defaults to `1.0`. Supported types are `view`, `click`, `favorite`, `purchase`, `rating`, `dislike`, and `remove`. Timestamps must include a timezone and are normalized to UTC. Duplicate event IDs are reported and ignored within each workspace.

Positive events create or strengthen a relationship. `dislike` changes recommendation filtering without creating a positive edge. `remove` removes the current relationship. New items may be appended without features. New users require a feature row with the existing feature dimension. A feature-dimension or named-schema change requires a new full workspace build.

Accepted events are normalized into `data/events.jsonl`. Feedback is also appended to `data/feedback.jsonl`. A local `state.sqlite3` index enforces event-ID idempotency and records applied batches. These files are runtime state and are not bundled by the project.

## Configuration

The generated TOML file contains the following sections:

- `[data]` selects edge and feature paths and their delimiter.
- `[core.affinity]` controls structure/attribute mixing, restart diffusion, steps, and sparse Top-k retention.
- `[core.encoder]` controls model dimensions, layers, latent groups, optimization, and loss weights.
- `[query]` controls community size, assignment/affinity balance, BLC balance, and the structural gate.
- `[incremental]` controls affected-set fallback and warm-start epochs.
- `[recommendation]` controls Top-N ranking weights, popularity penalty, and time decay.
- `[evaluation]` controls K and the minimum number of positive events.

V2 defaults are:

```toml
[incremental]
max_affected_ratio = 0.20
neighbor_hops = 2
warm_start_epochs = 20
fallback_to_full_build = true

[recommendation]
top_n = 20
community_weight = 0.40
affinity_weight = 0.30
behavior_weight = 0.20
recency_weight = 0.10
popularity_penalty = 0.05
half_life_days = 30

[evaluation]
k = 10
minimum_positive_events = 3
```

`latent_groups` is unsupervised model capacity and must not exceed the number of users.

## Python API

Supported application imports are exposed from the package root:

```python
from bipartite_scope import (
    BuildConfig,
    CanonicalBipartiteGraph,
    EvaluationConfig,
    Event,
    IncrementalConfig,
    QueryConfig,
    QueryEngine,
    RecommendationConfig,
    SnapshotStore,
    build_snapshot,
    create_app,
    evaluate,
    recommend,
    update_snapshot,
)
```

Build a graph and query a snapshot:

```python
graph = CanonicalBipartiteGraph.from_edges_and_features(
    [("u1", "i1"), ("u2", "i1")],
    {"u1": [1.0, 0.0], "u2": [1.0, 1.0]},
)
snapshot = build_snapshot(graph, BuildConfig())
result = QueryEngine(snapshot).search("u1", QueryConfig(size_budget=10))
```

The package root is the compatibility boundary. V1 submodule paths are intentionally unsupported after source consolidation.

## CLI

All successful commands emit JSON. Validation and user-input failures return exit code `2`; unexpected internal failures return exit code `1`.

```bash
bipartite-scope init my-workspace
bipartite-scope validate --workspace my-workspace
bipartite-scope build --workspace my-workspace
bipartite-scope query --workspace my-workspace --entity u1 --size 20
bipartite-scope export --workspace my-workspace --entity u1 --name community.json

bipartite-scope update --workspace my-workspace --events batch.jsonl
bipartite-scope update --workspace my-workspace --events batch.csv --features new-users.csv
bipartite-scope recommend --workspace my-workspace --user u1 --top-n 20
bipartite-scope feedback --workspace my-workspace --user u1 --item i9 --event-type click
bipartite-scope evaluate --workspace my-workspace --k 10

bipartite-scope snapshot list --workspace my-workspace
bipartite-scope snapshot verify --workspace my-workspace --snapshot SNAPSHOT_ID
bipartite-scope snapshot activate --workspace my-workspace --snapshot SNAPSHOT_ID
bipartite-scope snapshot rollback --workspace my-workspace --snapshot SNAPSHOT_ID

bipartite-scope benchmark --users 100 --items 200 --events 1000 --delta-ratio 0.05
```

`query`, `recommend`, and `evaluate` use the active snapshot unless `--snapshot` is supplied. An update creates a candidate and never activates it automatically.

## REST API

The FastAPI interface is synchronous, stores all state in local workspaces, has no authentication, and is intended only for localhost use in V2.

```bash
uvicorn 'bipartite_scope.interface:create_app' --factory --host 127.0.0.1 --port 8000
```

Endpoints:

```text
GET  /health
POST /workspaces/{workspace_id}/build
GET  /workspaces/{workspace_id}/snapshots/{snapshot_id}/query/{entity_id}
POST /workspaces/{workspace_id}/update
POST /workspaces/{workspace_id}/recommend
POST /workspaces/{workspace_id}/feedback
POST /workspaces/{workspace_id}/evaluate
GET  /workspaces/{workspace_id}/snapshots
POST /workspaces/{workspace_id}/snapshots/{snapshot_id}/activate
POST /workspaces/{workspace_id}/snapshots/{snapshot_id}/verify
```

Workspace IDs accept letters, digits, periods, underscores, and hyphens. Path traversal outside the configured root is rejected. Heavy asynchronous jobs, authentication, and a production backend are outside V2 scope.

## Algorithm overview

The canonical input is `G=(U,V,E,X)`: a binary user-item incidence matrix `A` and a nonnegative user-feature matrix `X`. The build computes normalized structure and attribute transitions:

```text
P_s = D_U^-1 A D_V^-1 A^T
P_x = D_X^-1 X D_F^-1 X^T
```

The transitions are mixed before restart diffusion. Deterministic row-wise Top-k truncation is applied after every step, and the final zero-diagonal affinity `W` is symmetric.

The sparse dual-view encoder follows user-to-item-to-user propagation and affinity refinement. It produces reusable user embeddings `Z`, item embeddings, and soft latent assignments `S`. Training minimizes structure-attribute normalized cut, original bipartite support loss, and assignment orthogonality without labels. Finite-loss checks stop invalid builds.

Community search combines latent-assignment similarity, local affinity, shared supporting items, and the change in Bipartite-aware Local Conductance. The structure gate records accepted and rejected expansion evidence. Query-time search never retrains the model.

## Incremental updates

An update appends new user and item IDs after existing indexes so warm-start state remains aligned. The affected set includes changed users, users connected through changed items, feature-related users, graph-hop neighbors, and previous or new affinity neighbors.

The engine computes a candidate sparse affinity, replaces the closed affected region, restores symmetric Top-k structure, and keeps unrelated rows. It falls back to a full affinity build when the affected ratio exceeds the configured threshold or numerical integrity checks fail. The neural model is always trained on the complete sparse graph for the reduced `warm_start_epochs` budget. Compatible parameters and existing item embeddings are reused; new item embeddings use deterministic initialization.

Every update snapshot records its parent, build mode, schema version, batch hash, event counts, affected counts, fallback reason, hashes, diagnostics, and reused parameters. Event application is explicit batch processing; real-time streaming is outside V2 scope.

## Recommendation and feedback

Recommendation first finds the query user's community, then draws candidate items from supporting users. Items already consumed, removed, or explicitly disliked by the query user are excluded.

The final score combines normalized community support, affinity-weighted peer support, weighted behavior, recency decay, and a popularity penalty. Results contain the snapshot ID, exact score components, supporting users, and a deterministic English explanation derived from those calculations.

The `feedback` command validates and stores one event but does not retrain. A later explicit `update` consumes all pending events and creates a candidate snapshot.

## Offline evaluation

Evaluation requires timestamped positive events. Legacy edges remain usable for building and recommendation but do not qualify as temporal evidence. Eligible users are split chronologically with the last positive event held out; users below `minimum_positive_events` are excluded and counted.

The report compares BipartiteScope with popularity and user-item co-occurrence baselines. It includes Precision@K, Recall@K, HitRate@K, NDCG@K, MRR, and catalog coverage. Outputs are written under:

```text
reports/evaluations/<evaluation-id>/metrics.json
reports/evaluations/<evaluation-id>/configuration.json
reports/evaluations/<evaluation-id>/report.md
```

Evaluation never activates a candidate. Promotion remains a manual decision.

## Snapshots and integrity

Snapshots are immutable directories containing a manifest plus sparse graph, weighted interaction, affinity, embedding, assignment, semantic-index, and model-state assets. Every asset has a recorded byte size and SHA-256 digest. Writes use a temporary directory and an atomic rename.

The first successful full build activates automatically only when no active pointer exists. Later builds are candidates. `snapshot activate` verifies all assets before atomically changing `latest.json`. `snapshot rollback` performs the same verified activation against an older snapshot and never deletes data.

V1 snapshots remain readable. Missing weighted interactions are interpreted as binary interaction weights, and missing lineage fields are interpreted as a full build.

## V1 migration

V2 keeps `BuildConfig`, `QueryConfig`, `CanonicalBipartiteGraph`, the supported package-root API, and the `bipartite-scope` command. Source modules are consolidated into five files, so imports from old internal paths such as `bipartite_scope.affinity` must move to `bipartite_scope`.

Existing V1 workspaces may be opened directly. Create a new full V2 snapshot before incremental updates if the active V1 snapshot does not contain a compatible model state. No V1 tag or release is synthesized by the V2 migration.

## Testing and repository policy

```bash
python -m ruff check .
python -m py_compile src/bipartite_scope/*.py
python -m unittest discover -s tests -v
python -m build
```

CI runs on Python 3.11 and 3.12. It checks mathematical invariants, data validation, event idempotency, incremental/full graph equivalence, recommendation exclusions and explanations, temporal metrics, snapshot integrity and rollback, every CLI command, REST endpoints, generated sparse benchmarks, package installation, the exact five-file source layout, one consolidated documentation file, absence of `docs/`, absence of bundled CSV/JSONL data, and absence of Han characters in tracked text.

## Troubleshooting

- If training is unavailable, install `bipartite-scope[train]` with a PyTorch build compatible with the active Python runtime.
- If validation fails, confirm all edge users have one finite nonnegative feature row and every row has the same feature columns.
- If an update rejects a user, provide its feature row with the unchanged feature dimension.
- If an event is ignored, inspect its event ID for a prior workspace registration.
- If an update performs a full fallback, inspect `affected_ratio` and `fallback_reason` in its result and snapshot manifest.
- If a query stops early, inspect its trace for an empty frontier or a failed BLC structure gate.
- If activation fails, run `snapshot verify`; modified or missing assets are never activated.
- If evaluation has no eligible users, ingest at least the configured number of timestamped positive events per evaluated user.
