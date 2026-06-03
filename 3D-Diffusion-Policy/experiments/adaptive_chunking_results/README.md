# Adaptive Chunking Result Package

Date: 2026-06-02

This folder contains the verified adaptive chunking results for the course project.

## Core Chunk Semantics

The policy predicts a full diffusion action trajectory with `horizon = 16`. Because `n_obs_steps = 2`, executable actions start at index `1`, so each policy call proposes up to `15` future executable actions.

The final adaptive runs use:

- `n_overlap = 2`
- `num_inference_steps = 5`
- learned dynamic chunk size in `[8, 15]`

If the learned chunk head selects chunk size `12`:

- On the first call, the runner executes `12` fresh actions and stores the following `2` actions as overlap.
- On later calls, the runner executes `12` actions total: `2` committed overlap actions from the previous plan plus `10` newly generated continuation actions.
- It then stores the next `2` actions as overlap for the following planning call.

So a selected chunk of `12` means `12` actions are actually executed in that cycle. The overlap does not reduce the executed count; it makes the first `2` actions of the next cycle consistent with the previous plan.

## Headline Verified Results

The final five-case comparison is against original fixed DP3:

| Task | Original Success | Ours Success | Original Diffusion Calls | Ours Diffusion Calls | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Hammer | 0.91 | 1.00 | 130.00 | 48.65 | accuracy + efficiency |
| Drawer Close | 1.00 | 1.00 | 250.00 | 94.50 | efficiency |
| Drawer Open | 1.00 | 1.00 | 250.00 | 108.50 | efficiency |
| Door Close | 1.00 | 1.00 | 250.00 | 104.70 | efficiency |
| Window Close | 1.00 | 1.00 | 250.00 | 91.30 | efficiency |

Button-press and button-press-topdown are included as negative strict-eval artifacts and should not be reported as wins.

