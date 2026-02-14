# DP3 Onboarding - Velocity-Based Gripper Control

This guide explains the main entry points, how training/evaluation works, and where to change dataset paths for the PyBullet pick-and-place setup **with velocity-based gripper control**.

## Velocity-Based Gripper

**This branch uses velocity-based gripper control instead of position-based control.**

### Velocity-Based (this branch):
- Actions represent **gripper velocity commands** [-1, 0, 1]
- Model predicts velocity commands (discrete or continuous)
- Gripper velocity is integrated over time
- **Matches real robot data collection approach**

---

## Main Entry Points (High-Level)

### 1) Training
- **Entry**: [3D-Diffusion-Policy/train.py](3D-Diffusion-Policy/train.py)
- **Class**: `TrainDP3Workspace`

**What happens during training:**
1. **Model Setup**: Instantiates the DP3 policy model with:
   - Point cloud encoder (PointNet/PointNet++)
   - Conditional U-Net for diffusion
   - DDIM noise scheduler
   - Optional EMA (Exponential Moving Average) model for better evaluation
   
2. **Dataset Loading**: 
   - Loads zarr dataset from path specified in task config
   - Creates train/validation split based on `val_ratio`
   - Applies normalizer to observations and actions
   - **Action[6] contains velocity commands** (not positions)
   - Sets up dataloaders with specified batch size and workers

3. **Training Loop** (over `num_epochs`):
   - Forward pass: policy predicts actions from observations
   - Loss computation: MSE between predicted and ground truth actions
   - Backward pass with gradient accumulation
   - EMA model updates (if enabled)
   
4. **Periodic Rollouts** (every `rollout_every` epochs):
   - Runs the trained policy in the environment
   - Collects success rates and returns
   - Logs videos to W&B
   - Uses validation dataset cube positions for consistent evaluation
   
5. **Checkpointing** (every `checkpoint_every` epochs):
   - Saves model state_dict, optimizer, epoch, global_step
   - Keeps top-K checkpoints based on test_mean_score
   - Saves "latest.ckpt" for resuming

6. **Validation** (every `val_every` epochs):
   - Runs model on validation dataset
   - Computes validation loss
   - No gradient computation

### 2) Evaluation
- **Entry**: [3D-Diffusion-Policy/eval.py](3D-Diffusion-Policy/eval.py)
- **Class**: `TrainDP3Workspace.eval()`

**What happens during evaluation:**
1. **Checkpoint Loading**:
   - Loads a saved checkpoint (update the path in `train.py` line ~393)
   - Restores model weights, normalizer parameters
   - Uses EMA model if `use_ema=True`

2. **Dataset Setup**:
   - Loads the same dataset used for training
   - Creates validation split to get cube starting positions
   - Sets up the normalizer (critical for correct prediction)

3. **Environment Rollouts**:
   - Runs policy for `n_test` episodes (default: 1)
   - Each episode uses validation dataset cube positions
   - Policy predicts actions in chunks (n_action_steps at a time)
   - **Gripper velocity commands are discretized and executed**
   - Compares GT actions vs predicted actions (gripper values)
   - Logs detailed gripper comparison to timestamped log files

4. **Metrics Collected**:
   - Success rate (is cube in tray?)
   - Episode returns
   - Gripper MAE (Mean Absolute Error between GT and predicted)
   - Videos of rollouts

5. **Action Validation**:
   - Runs policy on validation dataloader
   - Saves GT vs predicted actions to files
   - Computes per-dimension MSE and MAE
   - Outputs statistics to timestamped files

### 3) Environment Runner
- **File**: [3D-Diffusion-Policy/diffusion_policy_3d/env_runner/pybullet_runner.py](3D-Diffusion-Policy/diffusion_policy_3d/env_runner/pybullet_runner.py)
- **Class**: `UR5PyBulletRunner`

**Responsibilities:**
1. **Environment Management**:
   - Creates `UR5PickPlaceEnv` wrapped with:
     - `SimpleVideoRecordingWrapper` (captures videos)
     - `MultiStepWrapper` (handles action chunking and observation stacking)
   - Manages episode resets with cube positions from validation dataset

2. **Observation Handling**:
   - Receives observations: `point_cloud` (N, 3), `agent_pos` (7,), `image` (H, W, 3)
   - Stacks observations for history (n_obs_steps)
   - Converts numpy arrays to torch tensors
   - Applies policy normalizer

3. **Action Processing**:
   - Policy outputs action chunks (n_action_steps, action_dim)
   - MultiStepWrapper feeds actions one at a time to environment
   - **Gripper action (dim 6) contains velocity commands**
   - Tracks GT actions from dataset for comparison

4. **Logging**:
   - Tracks success rates and returns per episode
   - Logs gripper comparisons (GT vs predicted) to timestamped files
   - Creates detailed episode summaries with MAE statistics
   - Captures and logs videos to W&B

5. **Episode Flow**:
   ```
   reset(cube_pos) → get obs → policy.predict → execute actions → repeat → log metrics
   ```

### 4) Environment (PyBullet) - ** VELOCITY GRIPPER CHANGES HERE**
- **File**: [3D-Diffusion-Policy/diffusion_policy_3d/env/pybullet/pybullet_wrapper.py](3D-Diffusion-Policy/diffusion_policy_3d/env/pybullet/pybullet_wrapper.py)
- **Class**: `UR5PickPlaceEnv`

**Key Components:**

1. **Robot (`UR5Robotiq85`)**:
   - 6 DOF UR5 arm + Robotiq 85 gripper
   - Joint control: position control for arm, **velocity control for gripper**
   - Gripper state observation: raw angle [0, 0.35] → normalized [0, 1]
   - **Action space: 7D (6 arm joints + 1 gripper velocity command)**

2. **Scene Setup**:
   - Table, tray (target location), small cube (object to pick)
   - Camera positioned at `[1.1, -0.6, 1.3]` looking at workspace
   - Cube starting position can be specified or randomized

3. **Observation Generation**:
   - **Point Cloud**: 
     - Captured from depth camera
     - Converted from depth buffer to 3D coordinates
     - Filtered by workspace bounds (using mean ± n_std)
     - Downsampled to 6000 points using FPS (Farthest Point Sampling)
     - Excludes table points (optional via `capture_table` flag)
   - **Agent Pos**: 
     - 7D: [eef_pos (3), eef_orn_euler (3), **normalized_gripper_position** (1)]
     - Or 13D: [eef_pos (3), eef_orn_euler (3), 6 arm joints, **normalized_gripper_position** (1)]
     - **Note: Observation contains gripper POSITION, but action expects VELOCITY**
   - **Image**: RGB camera view (optional, for visualization)

4. **Action Interpretation -  VELOCITY GRIPPER**:
   - **Arm actions (dims 0-5)**: DELTAS in joint space (same as position-based)
     - Applied as: `target_arm = current_arm_pos + arm_deltas`
   
   - **Gripper action (dim 6)**: VELOCITY COMMAND
     - Model predicts continuous value (normalized)
     - **Discretization at inference**:
       ```python
       if gripper_velocity_cmd < -0.33:
           gripper_cmd_discrete = -1.0  # OPEN
       elif gripper_velocity_cmd > 0.33:
           gripper_cmd_discrete = 1.0   # CLOSE
       else:
           gripper_cmd_discrete = 0.0   # HOLD
       ```
     - **Velocity mapping**:
       ```python
       -1.0 → -0.25 * 20.0 = -5.0  (open slowly)
        0.0 → 0.0              (hold position)
        1.0 → 1.0 * 20.0 = 20.0   (close)
       ```
     - Velocity control: `p.setJointMotorControl2(..., p.VELOCITY_CONTROL, targetVelocity=...)`

5. **Success Criteria**:
   - Cube Z-position > 0.8 (in the tray)
   - Checked every step
   - Episode terminates on success or max_steps (350)

6. **Step Flow with Velocity Gripper**:
   ```
   action → discretize gripper velocity → apply arm deltas → apply gripper velocity → 
   simulate physics (1000 steps) → capture observations → check success
   ```

### 5) Configuration
- Main config: [3D-Diffusion-Policy/diffusion_policy_3d/config/dp3.yaml](3D-Diffusion-Policy/diffusion_policy_3d/config/dp3.yaml)
  - Model, optimizer, dataloaders, training options.
- Task config: [3D-Diffusion-Policy/diffusion_policy_3d/config/task/pybullet_pick_place.yaml](3D-Diffusion-Policy/diffusion_policy_3d/config/task/pybullet_pick_place.yaml)
  - Dataset path, env runner, action/observation shapes, horizons.

---

## 🔧 Velocity Gripper Implementation Details

### Data Collection
- Gripper velocity commands collected from teleoperation
- Stored as discrete values: -1.0 (open), 0.0 (hold), 1.0 (close)
- Dataset contains velocity commands, not positions

### Training
- Model learns to predict velocity commands
- Loss computed on velocity values
- Normalizer applied to velocity commands

### Inference (Evaluation)
**Key changes in `pybullet_wrapper.py`:**

1. **`move_gripper(velocity_command)` method**:
```python
def move_gripper(self, velocity_command):
    """
    Velocity-based gripper control (same as data collection)
    
    Args:
        velocity_command: Discrete control input
            -1.0: Open gripper (executed as -0.25 for safety)
             0.0: Hold current state
             1.0: Close gripper
    """
    # Map discrete command to actual velocity
    if velocity_command == -1.0:
        actual_velocity = -0.25 * 20.0  # Open slowly
    elif velocity_command == 0.0:
        actual_velocity = 0.0  # Hold
    elif velocity_command == 1.0:
        actual_velocity = 1.0 * 20.0  # Close
    
    p.setJointMotorControl2(
        self.id, 
        self.mimic_parent_id, 
        p.VELOCITY_CONTROL,  # ← VELOCITY CONTROL MODE
        targetVelocity=actual_velocity,
        force=100
    )
```

2. **`step()` method - Discretization**:
```python
# Extract gripper velocity command
gripper_velocity_cmd = action[6]

# Discretize gripper command
if gripper_velocity_cmd < -0.33:
    gripper_cmd_discrete = -1.0  # OPEN
elif gripper_velocity_cmd > 0.33:
    gripper_cmd_discrete = 1.0   # CLOSE
else:
    gripper_cmd_discrete = 0.0   # HOLD

# Execute velocity command
self.robot.move_gripper(gripper_cmd_discrete)
```

3. **Key differences from position-based**:
   - Uses `p.VELOCITY_CONTROL` instead of `p.POSITION_CONTROL`
   - No target position, only velocity
   - Gripper moves continuously at specified velocity
   - Position is implicit (integrated over time)

---

## Where to Change Dataset Path
For the PyBullet pick-and-place task, edit the dataset path here:
- [3D-Diffusion-Policy/diffusion_policy_3d/config/task/pybullet_pick_place.yaml](3D-Diffusion-Policy/diffusion_policy_3d/config/task/pybullet_pick_place.yaml)
  - `dataset.zarr_path`

Example:
- `dataset.zarr_path: data/velocity_gripper_dataset.zarr`

**Important**: Ensure your dataset contains **velocity commands** for the gripper, not positions.

Change this to your dataset location.

## How to Train (PyBullet pick-and-place with Velocity Gripper)

### Training Command

```bash
cd 3D-Diffusion-Policy

python train.py --config-name=dp3.yaml \
                task=pybullet_pick_place \
                hydra.run.dir=/path/to/your/output_dir \
                training.debug=False \
                training.seed=38 \
                training.device=cuda:0 \
                +training.addition_info=velocity-gripper \
                exp_name=pybullet_pick_place-dp3-velocity-gripper \
                logging.mode=online \
                checkpoint.save_ckpt=True
```

**What each parameter does:**
- `--config-name=dp3.yaml`: Uses the DP3 policy configuration
- `task=pybullet_pick_place`: Loads the PyBullet pick-and-place task config
- `hydra.run.dir`: Output directory for logs, checkpoints, and results
- `training.debug=False`: Full training mode (set to True for quick debugging)
- `training.seed=38`: Random seed for reproducibility
- `training.device=cuda:0`: GPU device to use
- `+training.addition_info`: Additional experiment identifier (+ means adding new key)
- `exp_name`: Experiment name for W&B logging
- `logging.mode=online`: W&B logging mode (online/offline)
- `checkpoint.save_ckpt=True`: Enable checkpoint saving

**Key parameters you can modify:**
- `training.seed`: Change to 0, 1, 42, etc. for different random seeds
- `training.device`: Change to cuda:1, cuda:2 for different GPUs
- `training.num_epochs`: Override default (1000) epochs
- `training.rollout_every`: Change rollout frequency (default: 200)
- `training.checkpoint_every`: Change checkpoint save frequency (default: 50)
- `dataloader.batch_size`: Change batch size (default: 128)
- `policy.n_obs_steps`: Change observation history length (default: 1)
- `policy.n_action_steps`: Change action chunk size (default: 12)
- `hydra.run.dir`: Change output directory location

**To change core training settings, edit:**
- [3D-Diffusion-Policy/diffusion_policy_3d/config/dp3.yaml](3D-Diffusion-Policy/diffusion_policy_3d/config/dp3.yaml)
  - Model architecture, training hyperparameters, dataloader settings

### Example Training Commands

**Quick debug run:**
```bash
python train.py --config-name=dp3.yaml \
                task=pybullet_pick_place \
                training.debug=True \
                training.device=cuda:0
```

**Training with different seed:**
```bash
python train.py --config-name=dp3.yaml \
                task=pybullet_pick_place \
                hydra.run.dir=/path/to/your/output_dir \
                training.seed=0 \
                training.device=cuda:0 \
                +training.addition_info=velocity-gripper-exp \
                exp_name=pybullet_pick_place-dp3-velocity-gripper \
                logging.mode=online \
                checkpoint.save_ckpt=True
```

**Training with custom hyperparameters:**
```bash
python train.py --config-name=dp3.yaml \
                task=pybullet_pick_place \
                training.num_epochs=500 \
                training.rollout_every=100 \
                dataloader.batch_size=64 \
                training.device=cuda:0 \
                +training.addition_info=custom_hparams \
                exp_name=pybullet_pick_place-dp3-custom \
                logging.mode=online \
                checkpoint.save_ckpt=True
```

## How to Evaluate

### Evaluation Command

```bash
cd 3D-Diffusion-Policy

python eval.py --config-name=dp3.yaml \
               task=pybullet_pick_place \
               training.device=cuda:0
```

**What happens:**
1. Loads checkpoint from path specified in `train.py` (line ~393)
2. Runs validation on dataloader (compares GT vs predicted velocity commands)
3. Runs environment rollouts with validation dataset cube positions
4. **Discretizes predicted velocity commands** before execution
5. Saves detailed logs:
   - GT vs predicted actions: `action_comparisons/gt_actions_{timestamp}.txt`
   - Action statistics: `action_comparisons/action_stats_{timestamp}.txt`
   - Gripper comparisons: `gripper_comparisons/gripper_gt_vs_pred_{timestamp}.txt`

### Checkpoint Configuration

**To change which checkpoint is evaluated:**

Edit [3D-Diffusion-Policy/train.py](3D-Diffusion-Policy/train.py) in the `eval()` method (around line 393):

```python
# Current default:
lastest_ckpt_path = "/home/varun-edachali/Research/RRC/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/epoch=0950.ckpt"

# Change to your checkpoint:
lastest_ckpt_path = "/path/to/your/checkpoint.ckpt"
```

**Checkpoint naming convention:**
- Latest checkpoint: `checkpoints/latest.ckpt`
- Top-K checkpoints: `checkpoints/epoch={epoch:04d}-test_mean_score={score:.3f}.ckpt`

### Evaluation Outputs

**Console Logs:**
- Episode-by-episode success rates
- Mean success rate across all episodes
- Gripper MAE (Mean Absolute Error between GT and predicted **velocity commands**)
- Action prediction statistics

**File Outputs:**
- `gripper_comparisons/gripper_gt_vs_pred_{timestamp}.txt`:
  - Detailed GT vs predicted gripper velocity values for each timestep
  - Episode summaries with MAE
  
- `action_comparisons/gt_actions_{timestamp}.txt`:
  - All GT actions from validation set (including velocity commands)
  
- `action_comparisons/pred_actions_{timestamp}.txt`:
  - All predicted actions from validation set
  
- `action_comparisons/action_stats_{timestamp}.txt`:
  - Per-dimension MSE and MAE
  - Overall statistics

### Example Evaluation Commands

**Basic evaluation:**
```bash
python eval.py --config-name=dp3.yaml \
               task=pybullet_pick_place \
               training.device=cuda:0
```

**Evaluation with custom device:**
```bash
python eval.py --config-name=dp3.yaml \
               task=pybullet_pick_place \
               training.device=cuda:1
```

**Notes:**
- Evaluation uses the EMA model if `training.use_ema=True` (recommended)
- The checkpoint path must be updated in `train.py` before running
- Evaluation runs on validation dataset cube positions for consistency
- **Gripper velocity commands are automatically discretized during inference**

## How to Train/Eval on Your Velocity-Based Dataset

### Step 1: Set Dataset Path

Edit the task configuration file:
- [3D-Diffusion-Policy/diffusion_policy_3d/config/task/pybullet_pick_place.yaml](3D-Diffusion-Policy/diffusion_policy_3d/config/task/pybullet_pick_place.yaml)

Change the `dataset.zarr_path` line:
```yaml
dataset:
  _target_: diffusion_policy_3d.dataset.droid_dataset.DroidDataset
  zarr_path: data/velocity_gripper_dataset.zarr  # <-- Change this to your dataset path
  horizon: ${horizon}
  pad_before: ${eval:'${n_obs_steps}-1'}
  pad_after: ${eval:'${n_action_steps}-1'}
  seed: 42
  val_ratio: 0.02
  max_train_episodes: null
```

**Dataset format**: Zarr file containing:
- `data/`: trajectories with `point_cloud`, `agent_pos`, `action`, `cube_start_pos`
- `action[:, 6]`: **Gripper velocity commands** (not positions)
- Train/val split is created automatically based on `val_ratio`

### Step 2: Train

```bash
cd 3D-Diffusion-Policy

python train.py --config-name=dp3.yaml \
                task=pybullet_pick_place \
                hydra.run.dir=/path/to/your/output_dir \
                training.debug=False \
                training.seed=0 \
                training.device=cuda:0 \
                +training.addition_info=velocity-gripper \
                exp_name=pybullet_pick_place-dp3-velocity-gripper \
                logging.mode=online \
                checkpoint.save_ckpt=True
```

**What to expect:**
- Training starts and logs to W&B
- Checkpoints saved every 50 epochs (default) to `hydra.run.dir/checkpoints/`
- Rollouts every 200 epochs (default) to measure success rate
- Training continues for 1000 epochs (default)
- **Model learns to predict velocity commands**

### Step 3: Evaluate

**First, update checkpoint path** in [train.py](3D-Diffusion-Policy/train.py) line ~393:
```python
lastest_ckpt_path = "/path/to/your/output_dir/checkpoints/latest.ckpt"
```

**Then run evaluation:**
```bash
cd 3D-Diffusion-Policy

python eval.py --config-name=dp3.yaml \
               task=pybullet_pick_place \
               training.device=cuda:0
```

**Outputs:**
- Console: Success rates, returns, gripper velocity command MAE
- Files: Detailed logs in output directory under `gripper_comparisons/` and `action_comparisons/`

### Typical Workflow for PyBullet Pick-and-Place (Velocity Gripper)

```bash
# 1. Navigate to repo
cd /home2/gnrs/3D-Diffusion-Policy

# 2. Ensure you're on velocity-gripper branch
git branch  # Should show * velocity-gripper

# 3. Set dataset path in config
# Edit: 3D-Diffusion-Policy/diffusion_policy_3d/config/task/pybullet_pick_place.yaml
# Set: dataset.zarr_path: data/your_velocity_dataset.zarr

# 4. Train
cd 3D-Diffusion-Policy
python train.py --config-name=dp3.yaml \
                task=pybullet_pick_place \
                hydra.run.dir=/path/to/your/output_dir \
                training.seed=0 \
                training.device=cuda:0 \
                +training.addition_info=velocity-gripper-v1 \
                exp_name=pybullet_pick_place-dp3-velocity-gripper \
                logging.mode=online \
                checkpoint.save_ckpt=True

# 5. Monitor training on W&B
# Look for: success rate, loss, rollout videos

# 6. After training, update checkpoint path in train.py (line ~393)

# 7. Evaluate
python eval.py --config-name=dp3.yaml \
               task=pybullet_pick_place \
               training.device=cuda:0

# 8. Check evaluation outputs
# - Console logs
# - gripper_comparisons/*.txt (velocity commands)
# - action_comparisons/*.txt
```

---

## Quick Mental Model
- Config drives everything (Hydra).
- `train.py` builds model + dataset + runner, then trains on **velocity commands**.
- `eval.py` loads checkpoint and runs the same runner for evaluation.
- **At inference, velocity commands are discretized and executed via `p.VELOCITY_CONTROL`**.
- `pybullet_pick_place.yaml` is where you point to your velocity-based dataset.
- Observations contain gripper **positions** (state), actions contain gripper **velocities** (control).
