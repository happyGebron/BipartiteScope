# Architecture

BiLCS separates a reusable offline build from an online query engine. The canonical core receives `G=(U,V,E,X)`: binary U--V incidence and nonnegative U-side features. A build produces immutable snapshot assets (`W`, `Z`, `S`, semantic index, statistics, and manifest). Queries load one snapshot and never retrain.

Adapters translate business records into the canonical graph; the core does not contain dataset presets, labels, or benchmark metrics.

## Higher-order affinity

The build computes `P_s = D_U^-1 A D_V^-1 A^T` and `P_x = D_X^-1 X D_F^-1 X^T`, mixes them before restart diffusion, and applies deterministic row-wise Top-k truncation after each update. The final `W` is the zero-diagonal symmetric affinity graph. This preserves the paper's mixed structure-attribute semantics while keeping the artifact sparse.

## Offline and online separation

The dual-view encoder follows `U -> V -> U -> W refinement` and yields reusable target embeddings `Z` and soft latent assignments `S`. It is optimized offline with SAH-Ncut, OrigBip, and orthogonality objectives. Each successful build becomes an immutable snapshot. Online expansion combines `S` prototype similarity, local `W` affinity, and the change in BLC; the structure gate rejects expansions that exceed the permitted one-step deterioration. The result includes only evidence measured in this computation.
