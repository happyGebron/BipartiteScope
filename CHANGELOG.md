# Changelog

## 2.0.0 - 2026-09-03

- Added idempotent CSV/JSONL event ingestion and append-only local event ledgers.
- Added incremental graph updates, affected-neighborhood affinity replacement, full fallback, and whole-graph warm-start training.
- Added explainable user-item recommendation, feedback collection, and chronological offline evaluation against two baselines.
- Added candidate snapshot lineage, manual verification, activation, rollback, and V1 snapshot compatibility.
- Expanded the CLI and localhost FastAPI interface for all V2 workflows.
- Consolidated the package into five Python files and all detailed guidance into `DOCUMENTATION.md`.
- Added repository checks for source layout, documentation layout, English-only tracked text, and absence of bundled datasets.

## 1.0.0 - 2026-08-25

- Renamed the application product to BipartiteScope.
- Added empty workspace initialization, TOML configuration, JSON validation reports, and query exports.
- Added atomic model snapshot writes, a latest pointer, and SHA-256 asset verification.
- Stabilized the normalized-cut training objective across supported PyTorch runtimes.
- Removed bundled CSV data; import and validation remain available for user-supplied data.
- Declared this validated engineering baseline as the final V1.0.0 release.

## 0.1.0 - 2026-08-21

- Created the public application-engine project structure.
- Added canonical graph validation and generic CSV import.
- Added sparse, stepwise Top-k structure-attribute affinity construction.
- Added offline dual-view learning, immutable snapshots, semantic recall, and BLC queries with support evidence.
- Added workspace CLI, REST API, recommendation and academic adapters, and the complete technical documentation set.
