# Model build

`build_snapshot(graph, config)` validates the input, creates W, trains the dual-view model, builds an exact cosine recall index, and returns an immutable `ModelSnapshot`. The build records finite loss diagnostics and fails rather than persisting a non-finite objective.

For a first small deployment, use exact semantic recall. Replace only the semantic-index boundary for ANN; do not replace BLC with a plain embedding nearest-neighbor query.
