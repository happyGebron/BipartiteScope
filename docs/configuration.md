# Configuration

`AffinityConfig` controls `beta`, restart mass, diffusion steps, and per-step Top-k. `EncoderConfig` controls width, layers, latent capacity, optimizer budget, and the two loss weights. `QueryConfig` controls size, the assignment/affinity mix, the BLC structural balance, and gate tolerance.

`latent_groups` is a model capacity, not a declaration of ground-truth classes. It must not exceed the number of U entities.
