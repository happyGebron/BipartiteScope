# BipartiteScope

BipartiteScope is an explainable engine for community discovery and association analysis on attributed bipartite data. It converts a reusable offline model into fast, query-time U-side communities, V-side support evidence, and a trace of the structural decisions behind each result.

It is an application-oriented **BipartiteScope** engine, not a paper-reproduction or benchmark repository. The public product path accepts a generic `U - V - interaction - U attribute` schema and never requires ground-truth labels.

## Core capabilities

- Validate generic bipartite input and keep business IDs separate from matrix indexes.
- Build a sparse structure-attribute higher-order affinity graph.
- Train a dual-view cut-guided encoder offline and persist immutable model snapshots.
- Search communities online with Bipartite-aware Local Conductance (BLC).
- Explain accepted members and rank supporting V-side entities.
- Use the same core from Python, CLI, REST, and domain adapters.

## Status

`v1.0.0` is the final first major release: workspace-first operation, TOML configuration, pre-build validation reports, JSON exports, and integrity-verified snapshots.

## Quick start

```bash
python -m pip install -e '.[train,api,dev]'
bipartite-scope init my-workspace
# Place your own edges.csv and features.csv in my-workspace/data/.
bipartite-scope validate --workspace my-workspace
bipartite-scope build --workspace my-workspace
bipartite-scope query --workspace my-workspace --entity YOUR_U_ID
```

No example dataset is bundled. See [Getting started](docs/getting-started.md), [CLI](docs/cli.md), and [Architecture](docs/architecture.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
