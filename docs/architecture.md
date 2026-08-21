# Architecture

BiLCS separates a reusable offline build from an online query engine. The canonical core receives `G=(U,V,E,X)`: binary U--V incidence and nonnegative U-side features. A build produces immutable snapshot assets (`W`, `Z`, `S`, semantic index, statistics, and manifest). Queries load one snapshot and never retrain.

Adapters translate business records into the canonical graph; the core does not contain dataset presets, labels, or benchmark metrics.
