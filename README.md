# BipartiteScope

BipartiteScope is an explainable, application-oriented engine for attributed bipartite graphs. V2.0.0 adds incremental graph and model updates, user-item recommendation, feedback ingestion, chronological offline evaluation, verified snapshot activation and rollback, an expanded CLI, a stable Python API, and a lightweight localhost FastAPI interface.

The project keeps binary graph incidence separate from weighted interaction data, preserves sparse deterministic core operations, and does not require labels, external database services, authentication, or bundled datasets.

## Quick start

```bash
python -m pip install -e '.[train,api,dev]'
bipartite-scope init my-workspace
# Add your own edges.csv and features.csv under my-workspace/data/.
bipartite-scope validate --workspace my-workspace
bipartite-scope build --workspace my-workspace
bipartite-scope query --workspace my-workspace --entity YOUR_USER_ID
```

The initial build becomes active. Incremental updates create immutable candidates that must be evaluated, verified, and activated explicitly.

See [DOCUMENTATION.md](DOCUMENTATION.md) for installation, schemas, configuration, Python, CLI, REST, algorithm, update, recommendation, evaluation, snapshot, migration, testing, and troubleshooting details.

## License

Apache-2.0. See [LICENSE](LICENSE).
