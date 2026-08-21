# CLI

```bash
bilcs validate --edges edges.csv --features features.csv
bilcs build --edges edges.csv --features features.csv --artifacts artifacts --epochs 100 --latent-groups 8
bilcs query --artifacts artifacts --snapshot SNAPSHOT_ID --entity u-123 --size 20
```

Build creates a unique snapshot directory. Query only reads that snapshot, so prior results remain reproducible after later data updates.
