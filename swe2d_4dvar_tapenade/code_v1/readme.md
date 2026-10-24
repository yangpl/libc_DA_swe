The current C implementation uses fixed-interval block checkpointing, not binomial checkpointing or Revolve.

For the default experiment, the assimilation window has 300 time steps, with observations every 20 steps.

1. Initial forward pass: save states only at steps 0,20,40,…,300. These 16 checkpoints remain in memory.
2. Reverse pass: process the blocks backward. For the final block, inject the observation gradient at step 300, restore the checkpoint at step 280, and recompute steps 280–300.
3. Local storage: during that recomputation, retain each state and its first RK2 stage. These are the linearization points needed by the exact discrete adjoint.
4. Local backward sweep: propagate the adjoint from step 300 to step 280. Reuse the same temporary arrays for the preceding block, inject the observation gradient at step 280, and continue.

After reaching step zero, C returns the observation gradient λ⁰. Python forms the complete gradient in whitened controls:

\[
\nabla_z J=z+L^\mathsf{T}\lambda^0.
\]

This is implemented in swe_evaluate() (swe2d_4dvar.c).

The trade-off is:

|  | Current block scheme |
|---|---|
| Checkpoint placement | Every observation interval |
| Additional computation | One extra forward integration over the window |
| Default state/stage storage | Approximately 5.36 MB, versus 51.93 MB with full storage |
| Gradient approximation | None introduced by checkpointing; states are recomputed using the same model |

Binomial checkpointing instead uses a recursive schedule that places, discards, and recomputes checkpoints according to a memory budget. Some intervals may be recomputed multiple times. Our simpler scheme has no recursive scheduling or user-specified checkpoint budget; its storage grows with the number of observation intervals.