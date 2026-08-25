# CLI

```bash
bipartite-scope init workspace
bipartite-scope validate --workspace workspace --report workspace/reports/validation.json
bipartite-scope build --workspace workspace
bipartite-scope query --workspace workspace --entity u-123 --size 20
bipartite-scope export --workspace workspace --entity u-123 --name community.json
```

`init` creates only directories and TOML configuration; it never creates data. Build creates a unique, integrity-verified snapshot directory. Query defaults to the latest verified snapshot, and export writes JSON to the workspace `exports/` directory.
