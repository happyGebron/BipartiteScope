# Core algorithm

`P_s=D_U^-1AD_V^-1A^T` captures normalized U--V--U transitions; `P_x=D_X^-1XD_F^-1X^T` captures attribute-induced transitions. BiLCS mixes the operators before restart propagation, then applies row-wise Top-k pruning at every step. The symmetric, zero-diagonal result is `W`.

The encoder starts from a projected U feature representation and trainable V embeddings. Each layer performs U-to-V-to-U incidence propagation and W refinement. `Z` is the target representation and `S=softmax(ZΘc)` is a soft latent-group assignment. Offline loss is `SAH-Ncut + λb OrigBip + λo Lorth`; no supervised label enters the build.
