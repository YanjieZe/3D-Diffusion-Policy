# Final Project Summary

Date: 2026-05-29

## Headline Result

The current final method is **training-aware overlap with a learned dynamic chunk head** for 3D Diffusion Policy on Adroit Hammer.

| Method | Task | DDIM Steps | Episodes | Success | Policy Calls | Diffusion Calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| original fixed DP3 | Hammer | 10 | 100 | 0.91 | 13.00 | 130.00 |
| denoise-budget calibrated DP3 | Hammer | 5 | 100 | 0.95 | 13.00 | 65.00 |
| overlap only on old checkpoint | Hammer | 5 | 100 | 0.93 | 13.00 | 65.00 |
| train-aware overlap + learned chunk | Hammer | 5 | 100 | 1.00 | 9.73 | 48.65 |
| train-aware overlap + learned chunk, seed-1 eval | Hammer | 5 | 100 | 0.99 | 9.92 | 49.60 |

Main claim:

- `+0.08` to `+0.09` absolute success over original fixed DP3.
- `+0.04` to `+0.05` absolute success over the previous best denoise-calibrated result.
- `23.7%` to `25.2%` fewer policy calls than fixed execution.
- `62.6%` fewer diffusion model calls than original 10-step DP3.

## Method

Baseline 3D Diffusion Policy observes point clouds and robot proprioception, predicts an action trajectory, and executes a fixed action chunk before replanning. The project hypothesis is that the robot should not always commit to the same open-loop horizon.

The final method changes both training and inference:

1. **Overlap conditioning.** During training, the policy randomly sees a short committed prefix of future actions as an inpainting condition. At inference, the runner passes the previously committed overlap actions back into the diffusion sampler, so the model plans around actions it already promised to execute.
2. **Learned dynamic chunk head.** A small MLP predicts the total action chunk length from the predicted action sequence and encoded observation. Smooth demonstration segments are labeled for longer chunks; rapidly changing segments are labeled for shorter chunks.
3. **Reduced denoising budget.** The final evaluation uses 5 DDIM steps, which was separately validated to outperform and accelerate the original 10-step policy.

This is not only an engineering patch: the policy is retrained with a new action-commitment conditioning objective and a learned execution-horizon head.

## Evidence

Training run:

```bash
source /home/ubuntu/269-course-project/env.sh
cd /home/ubuntu/269-course-project/3D-Diffusion-Policy/3D-Diffusion-Policy

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name=simple_dp3 \
  task=adroit_hammer \
  task.dataset.zarr_path=/home/ubuntu/269-course-project/3D-Diffusion-Policy/3D-Diffusion-Policy/data/adroit_hammer_expert.zarr \
  hydra.run.dir=data/outputs/hammer_overlap_dynamic_simpledp3_seed42 \
  training.resume=false \
  training.num_epochs=250 \
  training.rollout_every=50 \
  training.checkpoint_every=50 \
  training.sample_every=100000 \
  training.device=cuda:0 \
  task.env_runner.eval_episodes=50 \
  task.env_runner.max_steps=120 \
  checkpoint.save_ckpt=true \
  logging.mode=disabled \
  policy.n_overlap=2 \
  policy.use_dynamic_chunk_head=true \
  policy.num_inference_steps=5
```

Saved checkpoint:

```text
data/outputs/hammer_overlap_dynamic_simpledp3_seed42/checkpoints/epoch=0200-test_mean_score=1.000.ckpt
```

Strict 100-episode evaluation:

```bash
python scripts/eval_adroit_policy.py \
  --checkpoint data/outputs/hammer_overlap_dynamic_simpledp3_seed42/checkpoints/epoch=0200-test_mean_score=1.000.ckpt \
  --task hammer \
  --episodes 100 \
  --max_steps 120 \
  --num_inference_steps 5 \
  --n_overlap 2 \
  --output data/branch_selector/eval_hammer_overlap_dynamic_trainaware_epoch200_100.json
```

Result JSON:

```json
{
  "episodes": 100,
  "mean_n_goal_achieved": 18.07,
  "mean_policy_calls": 9.73,
  "mean_success_rates": 1.0,
  "n_overlap": 2,
  "task": "hammer"
}
```

Second 100-episode validation with `--seed 1`:

```json
{
  "episodes": 100,
  "mean_n_goal_achieved": 18.21,
  "mean_policy_calls": 9.92,
  "mean_success_rates": 0.99,
  "n_overlap": 2,
  "task": "hammer"
}
```

## Important Negative Results

- Fixed long chunks reduced calls but hurt 100-episode success.
- Learned branch selection from simulator branch labels did not deploy well.
- Door temporal weighted fine-tuning did not beat the fixed baseline.
- Overlap at inference without overlap-aware retraining reached only `0.93`, below the denoise-calibrated checkpoint.

## Verified Five-Case Result Against Original DP3

The important comparison is against the original fixed-chunk DP3 execution, not just absolute performance. The current verified result is: on five tasks, the adaptive method matches or improves success while using substantially fewer diffusion denoising passes.

| Task | Family | Original Success | Ours Success | Original Calls | Ours Calls | Original Diffusion Calls | Ours Diffusion Calls | Reduction | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Hammer | Adroit | 0.91 | 1.00 | 13.00 | 9.73 | 130.00 | 48.65 | 62.6% | accuracy + efficiency win |
| Drawer Close | MetaWorld | 1.00 | 1.00 | 25.00 | 18.90 | 250.00 | 94.50 | 62.2% | equal success, efficiency win |
| Drawer Open | MetaWorld | 1.00 | 1.00 | 25.00 | 21.70 | 250.00 | 108.50 | 56.6% | equal success, efficiency win |
| Door Close | MetaWorld | 1.00 | 1.00 | 25.00 | 20.94 | 250.00 | 104.70 | 58.1% | equal success, efficiency win |
| Window Close | MetaWorld | 1.00 | 1.00 | 25.00 | 18.26 | 250.00 | 91.30 | 63.5% | equal success, efficiency win |

For MetaWorld, both original and adaptive models were trained on the same 30-demonstration point-cloud datasets and evaluated with 50 strict simulator episodes. Original DP3 uses 10 DDIM steps and fixed 8-action execution. The adaptive method uses 5 DDIM steps, overlap conditioning, and the learned chunk head.

Detailed expansion notes: `experiments/five_case_expansion_results.md`.

## Self-Check

Requirement 1: significant improvement over baseline.

Met. The strongest accuracy improvement is Hammer: `1.00` success versus `0.91` original fixed DP3 on 100 strict episodes. The broader five-task result is an efficiency improvement: the method preserves `1.00` success on four MetaWorld tasks while reducing diffusion denoising calls by `56.6%` to `63.5%`.

Requirement 2: novelty beyond engineering patch.

Met: the final method introduces a training-time committed-action inpainting objective plus a learned execution-horizon head. The key insight is that adaptive chunking must be made in-distribution during training; post-hoc inference heuristics were not enough.

## Model I/O and Verification

Input: point cloud observation with shape `[512, 3]` plus robot proprioceptive state `agent_pos` with shape `[24]`.

Output: an action chunk. For Adroit Hammer, each action has dimension `26`; the policy predicts a horizon of actions and the runner executes a selected chunk before replanning.

Verification: all reported numbers are simulator rollouts through the repository's point-cloud wrappers. Hammer uses Adroit/MuJoCo. The additional four verified cases use MetaWorld/MuJoCo point-cloud rollouts.
