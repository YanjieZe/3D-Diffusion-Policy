# Reproducing Adaptive Action Chunking Results

This note documents how to reproduce the adaptive action chunking experiments added in this branch. The main implementation changes add training-aware overlap conditioning and a learned dynamic chunk-size head for DP3/SimpleDP3.

## Environment

Follow the base project install instructions first:

```bash
conda create -n dp3 python=3.8
conda activate dp3
pip install -e 3D-Diffusion-Policy
```

For headless MuJoCo rollouts, the experiments used:

```bash
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
```

The reported runs were executed on one NVIDIA H100 GPU. A smaller CUDA GPU should work, but training time and batch-size limits may differ.

## Data

Adroit Hammer uses the VRL3 expert demonstration pipeline already included under `third_party/VRL3`. Generate or provide a zarr dataset at:

```text
3D-Diffusion-Policy/data/adroit_hammer_expert.zarr
```

The headline Hammer run used 30 expert demonstrations. MetaWorld runs used 30-demonstration point-cloud datasets for each task.

## Train the Final Hammer Model

Run from the repository root:

```bash
cd 3D-Diffusion-Policy

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name=simple_dp3 \
  task=adroit_hammer \
  task.dataset.zarr_path=data/adroit_hammer_expert.zarr \
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

The run used this checkpoint for the main 100-episode evaluation:

```text
data/outputs/hammer_overlap_dynamic_simpledp3_seed42/checkpoints/epoch=0200-test_mean_score=1.000.ckpt
```

## Evaluate Hammer

Adaptive method:

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

Seed-1 validation:

```bash
python scripts/eval_adroit_policy.py \
  --checkpoint data/outputs/hammer_overlap_dynamic_simpledp3_seed42/checkpoints/epoch=0200-test_mean_score=1.000.ckpt \
  --task hammer \
  --episodes 100 \
  --max_steps 120 \
  --num_inference_steps 5 \
  --n_overlap 2 \
  --seed 1 \
  --output data/branch_selector/eval_hammer_overlap_dynamic_trainaware_epoch200_100_seed1.json
```

Original fixed DP3 baseline:

```bash
python scripts/eval_adroit_policy.py \
  --checkpoint data/outputs/adroit_hammer_simple_dp3_30eps_seed0/checkpoints/epoch=0200-test_mean_score=0.950.ckpt \
  --task hammer \
  --episodes 100 \
  --max_steps 120 \
  --num_inference_steps 10 \
  --output data/branch_selector/eval_hammer_fixed8_100.json
```

## Evaluate MetaWorld Tasks

Use the same evaluator for each trained MetaWorld checkpoint. Example for drawer close:

```bash
python scripts/eval_metaworld_policy.py \
  --checkpoint data/outputs/metaworld_drawer_close_overlap_dynamic_30eps_seed42/checkpoints/epoch=0040-test_mean_score=1.000.ckpt \
  --task drawer-close \
  --episodes 50 \
  --max_steps 200 \
  --num_inference_steps 5 \
  --n_overlap 2 \
  --output data/branch_selector/eval_metaworld_drawer_close_adaptive_epoch40_50.json
```

For original fixed baselines, use the matching `*_original_30eps_seed42` checkpoint, set `--num_inference_steps 10`, and omit `--n_overlap`.

The five positive MetaWorld comparisons reported in the result package are:

```text
drawer-close
drawer-open
door-close
window-close
```

Button press and button press topdown are included as negative strict-evaluation artifacts, not as claimed wins.

## Expected Results

The archived result JSONs and summaries are in:

```text
3D-Diffusion-Policy/experiments/adaptive_chunking_results/
```

Headline Hammer results:

| Method | DDIM Steps | Episodes | Success | Policy Calls | Diffusion Calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original fixed DP3 | 10 | 100 | 0.91 | 13.00 | 130.00 |
| Training-aware overlap + learned chunk | 5 | 100 | 1.00 | 9.73 | 48.65 |
| Training-aware overlap + learned chunk, seed 1 | 5 | 100 | 0.99 | 9.92 | 49.60 |

Diffusion calls are computed as `mean_policy_calls * num_inference_steps`.
