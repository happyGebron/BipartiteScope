# Persistence

Each snapshot contains a manifest, `A.npz`, `X.npz`, `W.npz`, `Z.npy`, `S.npy`, semantic-index arrays, and `model.npz`. The manifest carries IDs, build time, input hash, config, diagnostics, and metadata. `SnapshotStore.save` rejects an existing target; a data update must produce a new snapshot.
