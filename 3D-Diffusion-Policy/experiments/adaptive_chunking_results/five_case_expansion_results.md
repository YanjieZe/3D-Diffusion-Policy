# Five-Case Expansion Results

Date: 2026-05-29

## Summary

The corrected five-case result is not "five tasks with higher success." It is stronger and more defensible: one accuracy-plus-efficiency win on Adroit Hammer, and four MetaWorld tasks where the adaptive policy preserves perfect success while cutting diffusion compute by more than half.

| Task | Family | Eval Episodes | Original Success | Ours Success | Original Calls | Ours Calls | Original Diffusion Calls | Ours Diffusion Calls | Compute Reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hammer | Adroit | 100 | 0.91 | 1.00 | 13.00 | 9.73 | 130.00 | 48.65 | 62.6% |
| Drawer Close | MetaWorld | 50 | 1.00 | 1.00 | 25.00 | 18.90 | 250.00 | 94.50 | 62.2% |
| Drawer Open | MetaWorld | 50 | 1.00 | 1.00 | 25.00 | 21.70 | 250.00 | 108.50 | 56.6% |
| Door Close | MetaWorld | 50 | 1.00 | 1.00 | 25.00 | 20.94 | 250.00 | 104.70 | 58.1% |
| Window Close | MetaWorld | 50 | 1.00 | 1.00 | 25.00 | 18.26 | 250.00 | 91.30 | 63.5% |

Diffusion calls are `policy_calls * DDIM_steps`. Original DP3 uses 10 DDIM steps. The adaptive method uses 5 DDIM steps.

## Method Tested

The final method is training-aware overlap plus a learned dynamic chunk head:

- During training, the model is randomly conditioned on committed future actions, making overlap execution in-distribution.
- During inference, the runner feeds leftover committed overlap actions back into the denoising process.
- A learned chunk head predicts how much of the planned action sequence to execute before replanning.
- The final policy uses 5 DDIM denoising steps instead of the original 10-step sampler.

## Verified Artifacts

Main positive eval JSONs:

- `data/branch_selector/eval_hammer_overlap_dynamic_trainaware_epoch200_100.json`
- `data/branch_selector/eval_hammer_fixed8_100.json`
- `data/branch_selector/eval_metaworld_drawer_close_adaptive_epoch40_50.json`
- `data/branch_selector/eval_metaworld_drawer_close_original_epoch40_50.json`
- `data/branch_selector/eval_metaworld_drawer_open_adaptive_epoch40_50.json`
- `data/branch_selector/eval_metaworld_drawer_open_original_epoch40_50.json`
- `data/branch_selector/eval_metaworld_door_close_adaptive_epoch40_50.json`
- `data/branch_selector/eval_metaworld_door_close_original_epoch40_50.json`
- `data/branch_selector/eval_metaworld_window_close_adaptive_epoch40_50.json`
- `data/branch_selector/eval_metaworld_window_close_original_epoch40_50.json`

MetaWorld datasets generated for the final sweep:

- `data/metaworld_drawer-close_expert.zarr`
- `data/metaworld_drawer-open_expert.zarr`
- `data/metaworld_door-close_expert.zarr`
- `data/metaworld_window-close_expert.zarr`
- `data/metaworld_button-press_expert.zarr`
- `data/metaworld_button-press-topdown_expert.zarr`
- `data/metaworld_reach-wall_expert.zarr`

## Negative Results Kept Out Of The Headline

Strict eval showed that some attractive internal rollouts did not survive the baseline comparison:

| Task | Original Success | Ours Success | Reason Not Counted |
| --- | ---: | ---: | --- |
| Button Press | 1.00 | 0.92 | adaptive replanning lost accuracy |
| Button Press Topdown | 1.00 | 0.70 | adaptive replanning lost substantial accuracy |
| Adroit Door | not improved | 0.62 | hard contact task, weak transfer |
| Adroit Pen | not improved | 0.47 | 30-demo dataset was not enough |
| MetaWorld Reach | not improved | unstable | earlier 20-demo run was not reliable |

## Research Insight

The method is useful when the task tolerates short committed overlaps and the model can stay stable under reduced denoising. It is not a universal free lunch: button-press variants show that repeatedly replanning with very short selected chunks can preserve compute savings but damage task completion. For the final presentation, the honest claim should be:

> Training-aware adaptive execution can beat original fixed DP3 by combining higher success on Hammer with large denoising-budget reductions on several MetaWorld manipulation tasks, but the adaptive selector needs better task-aware calibration for contact-sensitive button tasks.
