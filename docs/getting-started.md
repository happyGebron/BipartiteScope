# Getting started

Install the engine with the training extra, then provide two delimited files: `edges.csv` with `u_id,v_id`, and `features.csv` with `u_id` followed by one or more nonnegative feature columns. Labels are neither read nor required.

```bash
python -m pip install -e '.[train,api,dev]'
bilcs validate --edges examples/edges.csv --features examples/features.csv
```

The validation command normalizes duplicate edges, assigns stable internal indexes, and rejects schema, binary-incidence, and negative-feature violations before a model build.
