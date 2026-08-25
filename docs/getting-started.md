# Getting started

Install the engine with the training extra, create an empty workspace, then provide your own two delimited files: `edges.csv` with `u_id,v_id`, and `features.csv` with `u_id` followed by one or more nonnegative feature columns. Labels are neither read nor required. No data is bundled with the project.

```bash
python -m pip install -e '.[train,api,dev]'
bipartite-scope init my-workspace
# Copy your files to my-workspace/data/edges.csv and my-workspace/data/features.csv
bipartite-scope validate --workspace my-workspace --report my-workspace/reports/validation.json
bipartite-scope build --workspace my-workspace
bipartite-scope query --workspace my-workspace --entity YOUR_U_ID --output my-workspace/exports/result.json
```

The validation command reports duplicate edges, missing feature rows, nonnumeric or negative features, and isolated U-side records before a model build. Configuration is stored in `bipartitescope.toml` and may be versioned alongside your data schema without storing the data itself.
