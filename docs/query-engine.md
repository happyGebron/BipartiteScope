# Query engine

Load a snapshot once and construct `QueryEngine(snapshot)`. A query begins with its U-side entity, joins structural candidates reachable through shared V entities with semantic recall candidates, then ranks them using assignment-prototype similarity, local W affinity, and BLC change.

The structural gate accepts only a candidate whose BLC deterioration is no greater than `delta_0`. Expansion stops at the requested size, an empty frontier, or a rejected best candidate. Query-time code does not re-run neural propagation or training.
