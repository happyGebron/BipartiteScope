# Data model

The canonical input is `G=(U,V,E,X)`. `U` is the queryable target side; `V` provides cross-type support; `E` is binary incidence; and `X` is a nonnegative U-side feature matrix. Business IDs remain in `u_ids` and `v_ids`, while Core calculations use contiguous indexes.

CSV import requires `edges.csv(u_id,v_id)` and `features.csv(u_id,f1,...,fd)`. Duplicate edges are merged. Features must be finite, nonnegative, and have one shared dimension. Labels, preset dataset paths, and metrics are deliberately excluded.
