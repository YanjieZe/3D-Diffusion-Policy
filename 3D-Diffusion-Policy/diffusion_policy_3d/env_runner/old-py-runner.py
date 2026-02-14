import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
from termcolor import cprint
from diffusion_policy_3d.env import UR5PickPlaceEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import (
    SimpleVideoRecordingWrapper,
)

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util

import open3d as o3d
import numpy as np

import wandb

import os
from datetime import datetime


def save_pointcloud(pc, fname="debug_pc.ply"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.astype(np.float64))
    o3d.io.write_point_cloud(fname, pcd)
    print(f"[PCD] saved to {fname}")


def discretize_gripper_action(gripper_value):
    """
    Discretize continuous gripper action to {-1, 0, 1}
    
    Args:
        gripper_value: Continuous value from policy
    
    Returns:
        Discrete command: -1.0 (open), 0.0 (hold), or 1.0 (close)
    """
    if gripper_value < -0.33:
        return -1.0
    elif gripper_value > 0.33:
        return 1.0
    else:
        return 0.0


class UR5PyBulletRunner(BaseRunner):
    """
    Extended runner with DISCRETE GRIPPER ACTIONS
    
    Key changes:
    1. Gripper actions are discretized to {-1, 0, 1} before execution
    2. GT actions are already discrete from dataset
    3. Comparison is done on discrete values
    """

    def __init__(
        self,
        output_dir,
        n_train=10,
        n_test=1,
        max_steps=350,
        n_obs_steps=2,
        n_action_steps=8,
        fps=10,
        crf=22,
        tqdm_interval_sec=5.0,
        use_gui=False,
        num_points=6000,
        image_size=224,
        use_workspace_crop=True,
        workspace_std=30.0,
        action_dim=7,
        capture_table=False
    ):
        super().__init__(output_dir)

        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.fps = fps
        self.crf = crf
        self.tqdm_interval_sec = tqdm_interval_sec
        self.action_dim = action_dim
        self.num_points = num_points

        # Environment factory function
        def env_fn():
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    UR5PickPlaceEnv(
                        use_gui=use_gui,
                        num_points=num_points,
                        image_size=image_size,
                        use_workspace_crop=use_workspace_crop,
                        workspace_std=workspace_std,
                        action_dim=self.action_dim,
                        capture_table=capture_table,
                    )
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method="sum",
            )

        self.env_test = env_fn()
        self.episode_test = n_test
        self.logger_util_test = logger_util.LargestKRecorder(K=3)

    def run(self, policy: BasePolicy, dataset=None):
        """
        Run evaluation using validation dataset cube positions
        
        Args:
            policy: Policy to evaluate
            dataset: Validation dataset to get cube positions from
        """
        device = policy.device

        # Check what keys the policy normalizer expects
        normalizer_keys = list(policy.normalizer.params_dict.keys())
        cprint(f"Policy expects observation keys: {normalizer_keys}", "yellow")

        all_returns_test = []
        all_success_rates_test = []

        log_dir = os.path.join(self.output_dir, "gripper_comparisons")
        os.makedirs(log_dir, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(log_dir, f"gripper_gt_vs_pred_{timestamp}.txt")
        
        cprint(f"Logging to: {log_file_path}", "cyan")

        cprint("=" * 50, "cyan")
        cprint(
            "Running on TEST environment with VALIDATION dataset cube positions", "cyan"
        )
        cprint("=" * 50, "cyan")

        if dataset is not None:
            val_episode_indices = np.where(~dataset.train_mask)[0]
            n_val_episodes = len(val_episode_indices)
            cprint(f"Validation dataset has {n_val_episodes} episodes", "cyan")
            cprint(f"Running {self.episode_test} test episodes", "cyan")
            if n_val_episodes < self.episode_test:
                cprint(
                    f"WARNING: Only {n_val_episodes} validation episodes available, but running {self.episode_test} test episodes",
                    "yellow",
                )
                cprint(
                    f"Episodes {n_val_episodes}-{self.episode_test-1} will use random cube positions",
                    "yellow",
                )

        for episode_id in range(self.episode_test):
            # Get cube start position from validation dataset if available
            cube_start_pos = None
            cube_start_orn = None
            gt_actions = None

            if dataset is not None:
                try:
                    # Get the episode index from validation set
                    val_episode_indices = np.where(~dataset.train_mask)[0]

                    if episode_id < len(val_episode_indices):
                        val_ep_idx = val_episode_indices[episode_id]
                        cube_pos_full = dataset.get_episode_cube_start_pos(val_ep_idx)

                        # Extract position (first 3 elements) and orientation (last 4 elements)
                        cube_start_pos = cube_pos_full[:3].tolist()
                        cube_start_orn = cube_pos_full[3:7].tolist()
                        cprint(
                            f"Episode {episode_id}: Using cube position from val dataset: {cube_start_pos}",
                            "green",
                        )

                        # Get ground truth actions for this episode
                        episode_data = dataset.get_episode(val_ep_idx)
                        gt_actions = episode_data['action']  # Shape: (T, action_dim)
                        cprint(
                            f"Shape of episode action data: {gt_actions.shape}",
                            "green",
                        )

                    else:
                        cprint(
                            f"Episode {episode_id}: No more validation episodes, using random position",
                            "yellow",
                        )
                        cube_start_pos = None
                        cube_start_orn = None
                        gt_actions = None

                except Exception as e:
                    cprint(
                        f"Episode {episode_id}: Could not load cube position from dataset: {e}",
                        "red",
                    )
                    cprint("Falling back to random cube position", "yellow")
                    cube_start_pos = None
                    cube_start_orn = None

            # Open log file in append mode
            with open(log_file_path, 'a') as log_file:
                # Write episode header
                log_file.write("="*100 + "\n")
                log_file.write(f"EPISODE {episode_id}\n")
                log_file.write("="*100 + "\n")
                if cube_start_pos:
                    log_file.write(f"Cube start position: {cube_start_pos}\n")
                    log_file.write(f"Cube start orientation: {cube_start_orn}\n")
                if gt_actions is not None:
                    log_file.write(f"GT trajectory length: {len(gt_actions)} timesteps\n")
                log_file.write(f"n_action_steps: {self.n_action_steps}\n")
                log_file.write(f"n_obs_steps: {self.n_obs_steps}\n")
                log_file.write("\n")

            # Reset environment with specified or random cube position
            if cube_start_pos is not None:
                self.env_test.env.env.cube_start_pos = cube_start_pos
                self.env_test.env.env.cube_start_orn = cube_start_orn
                obs = self.env_test.reset(cube_start_pos=cube_start_pos, cube_start_orn=cube_start_orn)
            else:
                cprint(
                    f"Episode {episode_id}: Using random cube position TO RESET",
                    "yellow",
                )
                obs = self.env_test.reset()

            # Debug: print shapes on first episode
            if episode_id == 0:
                cprint(f"Environment provides keys: {list(obs.keys())}", "yellow")
                for key, val in obs.items():
                    if isinstance(val, np.ndarray):
                        cprint(
                            f"  {key}: shape={val.shape}, dtype={val.dtype}", "yellow"
                        )
                cprint(f"n_obs_steps: {self.n_obs_steps}", "yellow")

            policy.reset()

            reward_sum = 0.0
            done = False

            # Storage for this episode's comparisons
            episode_gt_grippers = []
            episode_pred_grippers = []
            episode_pred_grippers_discrete = []
            episode_actual_grippers = []

            gt_timestep = 0  

            for step_id in range(self.max_steps):
                # Deep copy the observation to avoid reference issues
                np_obs_dict = {}
                for key, val in obs.items():
                    if isinstance(val, np.ndarray):
                        np_obs_dict[key] = val.copy()
                    else:
                        np_obs_dict[key] = val

                # Debug point cloud shape before processing
                if step_id == 0 and episode_id == 0:
                    pc_raw = np_obs_dict["point_cloud"]
                    cprint(f"Raw point cloud shape: {pc_raw.shape}", "cyan")

                    # Handle different observation shapes
                    if pc_raw.ndim == 3:
                        # Stacked observations: (n_obs_steps, num_points, 3)
                        cprint(f"Detected stacked observations: {pc_raw.shape}", "cyan")
                        pc_for_debug = pc_raw[-1]
                    elif pc_raw.ndim == 2:
                        # Single observation: (num_points, 3)
                        pc_for_debug = pc_raw
                    else:
                        cprint(
                            f"WARNING: Unexpected point cloud shape: {pc_raw.shape}",
                            "red",
                        )
                        pc_for_debug = pc_raw.reshape(-1, 3)

                    save_pointcloud(pc_for_debug, "env_pc.ply")
                    cprint(
                        f"Saved debug point cloud with shape: {pc_for_debug.shape}",
                        "cyan",
                    )

                # Convert to torch tensors
                obs_dict = dict_apply(
                    np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device, non_blocking=True),
                )

                # Build policy observation dict with proper handling of observation history
                policy_obs = {}
                for key in policy.normalizer.params_dict.keys():
                    if key == "action":
                        continue

                    if key in obs_dict:
                        tensor = obs_dict[key]

                        # Handle stacked vs single observations
                        if key == "point_cloud":
                            if tensor.ndim == 2:
                                cprint(
                                    f"WARNING: point_cloud not stacked, repeating {self.n_obs_steps} times",
                                    "yellow",
                                )
                                tensor = tensor.unsqueeze(0).repeat(
                                    self.n_obs_steps, 1, 1
                                )
                            policy_obs[key] = tensor.unsqueeze(0)

                        elif key == "agent_pos":
                            if tensor.ndim == 1:
                                cprint(
                                    f"WARNING: agent_pos not stacked, repeating {self.n_obs_steps} times",
                                    "yellow",
                                )
                                tensor = tensor.unsqueeze(0).repeat(self.n_obs_steps, 1)
                            policy_obs[key] = tensor.unsqueeze(0)

                        elif key == "image":
                            if tensor.ndim == 3:
                                cprint(
                                    f"WARNING: image not stacked, repeating {self.n_obs_steps} times",
                                    "yellow",
                                )
                                tensor = tensor.unsqueeze(0).repeat(
                                    self.n_obs_steps, 1, 1, 1
                                )
                            policy_obs[key] = tensor.unsqueeze(0)
                        else:
                            policy_obs[key] = tensor.unsqueeze(0)
                    else:
                        cprint(
                            f"WARNING: Policy expects key '{key}' but not in observation!",
                            "red",
                        )

                # Debug shapes before policy prediction
                if step_id == 0 and episode_id == 0:
                    cprint("Policy input shapes:", "cyan")
                    for key, val in policy_obs.items():
                        cprint(f"  {key}: {val.shape}", "cyan")

                with torch.no_grad():
                    action_dict = policy.predict_action(policy_obs)

                # Extract action - handle both single and multi-step actions
                action = dict_apply(action_dict, lambda x: x.detach().cpu().numpy())[
                    "action"
                ]

                # Debug action shape
                if step_id == 0 and episode_id == 0:
                    cprint(f"Raw action shape: {action.shape}", "cyan")

                # MultiStepWrapper expects (n_action_steps, action_dim)
                if action.ndim == 3:
                    action = action.squeeze(0)
                elif action.ndim == 2 and action.shape[0] == 1:
                    action = action.squeeze(0)

                # Final check
                if step_id == 0 and episode_id == 0:
                    cprint(f"Final action shape for env.step(): {action.shape}", "cyan")

                # CRITICAL: Discretize gripper actions before execution
                action_discrete = action.copy()
                for i in range(len(action_discrete)):
                    gripper_idx = 6 if action_discrete.shape[1] == 7 else 12
                    gripper_continuous = action_discrete[i, gripper_idx]
                    gripper_discrete = discretize_gripper_action(gripper_continuous)
                    action_discrete[i, gripper_idx] = gripper_discrete

                if gt_actions is not None:
                    # Get the corresponding GT actions that will be executed
                    gt_start_idx = gt_timestep
                    gt_end_idx = min(gt_timestep + self.n_action_steps, len(gt_actions))
                    
                    if gt_start_idx < len(gt_actions):
                        gt_actions_to_execute = gt_actions[gt_start_idx:gt_end_idx]
                        pred_actions_continuous = action[:len(gt_actions_to_execute)]
                        pred_actions_discrete = action_discrete[:len(gt_actions_to_execute)]
                        
                        # Extract gripper values (index 6 or 12)
                        gripper_idx = 6 if gt_actions.shape[1] == 7 else 12
                        gt_grippers = gt_actions_to_execute[:, gripper_idx]
                        pred_grippers_continuous = pred_actions_continuous[:, gripper_idx]
                        pred_grippers_discrete = pred_actions_discrete[:, gripper_idx]
                        
                        # Print comparison for ALL actions that will be executed
                        # cprint(f"\n{'='*80}", "cyan")
                        # cprint(f"Episode {episode_id}, Policy Step {step_id} (GT Timesteps {gt_start_idx} to {gt_end_idx-1}):", "cyan")
                        # cprint(f"Executing {len(gt_grippers)} actions:", "cyan")
                        
                        for i in range(len(gt_grippers)):
                            actual_gt_timestep = gt_start_idx + i
                            # cprint(f"  [{i}] GT Timestep {actual_gt_timestep}:", "white")
                            # cprint(f"      GT Gripper:        {gt_grippers[i]:.6f}", "green")
                            # cprint(f"      Pred Continuous:   {pred_grippers_continuous[i]:.6f}", "yellow")
                            # cprint(f"      Pred Discrete:     {pred_grippers_discrete[i]:.6f}", "magenta")
                            # cprint(f"      Diff (GT-Disc):    {abs(gt_grippers[i] - pred_grippers_discrete[i]):.6f}", "red")
                        
                            with open(log_file_path, 'a') as log_file:
                                log_file.write(f"  Action [{i}] - GT Timestep {actual_gt_timestep}:\n")
                                log_file.write(f"    GT Gripper:        {gt_grippers[i]:.6f}\n")
                                log_file.write(f"    Pred Continuous:   {pred_grippers_continuous[i]:.6f}\n")
                                log_file.write(f"    Pred Discrete:     {pred_grippers_discrete[i]:.6f}\n")
                                log_file.write(f"    Diff (GT-Disc):    {abs(gt_grippers[i] - pred_grippers_discrete[i]):.6f}\n")
                                log_file.write("\n")
                            
                        # Store for episode summary
                        episode_gt_grippers.extend(gt_grippers.tolist())
                        episode_pred_grippers.extend(pred_grippers_continuous.tolist())
                        episode_pred_grippers_discrete.extend(pred_grippers_discrete.tolist())
                        
                        cprint(f"{'='*80}\n", "cyan")
                    
                    # Update GT timestep counter
                    gt_timestep += self.n_action_steps

                # Execute DISCRETIZED actions
                obs, reward, done, info = self.env_test.step(action_discrete)
                
                # Get actual gripper state (normalized)
                actual_gripper = self.env_test.env.env.robot.get_joint_positions()[6]
                episode_actual_grippers.append(actual_gripper)

                reward_sum += reward
                done = np.all(done)

                if done:
                    break

            # Episode summary
            if len(episode_gt_grippers) > 0 and len(episode_pred_grippers_discrete) > 0:
                gt_arr = np.array(episode_gt_grippers)
                pred_continuous_arr = np.array(episode_pred_grippers)
                pred_discrete_arr = np.array(episode_pred_grippers_discrete)
                
                mae_gt_continuous = np.mean(np.abs(gt_arr - pred_continuous_arr))
                mae_gt_discrete = np.mean(np.abs(gt_arr - pred_discrete_arr))
                
                # Console output
                cprint(f"\n{'='*80}", "magenta")
                cprint(f"Episode {episode_id} Gripper Summary:", "magenta")
                cprint(f"  Total actions compared: {len(gt_arr)}", "magenta")
                cprint(f"  MAE (GT vs Pred Continuous): {mae_gt_continuous:.6f}", "magenta")
                cprint(f"  MAE (GT vs Pred Discrete):   {mae_gt_discrete:.6f}", "magenta")
                cprint(f"  GT Range:        [{gt_arr.min():.6f}, {gt_arr.max():.6f}]", "magenta")
                cprint(f"  Pred Cont Range: [{pred_continuous_arr.min():.6f}, {pred_continuous_arr.max():.6f}]", "magenta")
                cprint(f"  Pred Disc Range: [{pred_discrete_arr.min():.6f}, {pred_discrete_arr.max():.6f}]", "magenta")
                cprint(f"{'='*80}\n", "magenta")
                
                # File output
                with open(log_file_path, 'a') as log_file:
                    log_file.write("\n" + "="*80 + "\n")
                    log_file.write(f"EPISODE {episode_id} SUMMARY\n")
                    log_file.write("="*80 + "\n")
                    log_file.write(f"Total actions compared: {len(gt_arr)}\n")
                    log_file.write(f"MAE (GT vs Pred Continuous): {mae_gt_continuous:.6f}\n")
                    log_file.write(f"MAE (GT vs Pred Discrete):   {mae_gt_discrete:.6f}\n")
                    log_file.write(f"GT Range:        [{gt_arr.min():.6f}, {gt_arr.max():.6f}]\n")
                    log_file.write(f"Pred Cont Range: [{pred_continuous_arr.min():.6f}, {pred_continuous_arr.max():.6f}]\n")
                    log_file.write(f"Pred Disc Range: [{pred_discrete_arr.min():.6f}, {pred_discrete_arr.max():.6f}]\n")
                    log_file.write(f"Reward:          {reward_sum:.2f}\n")
                    log_file.write(f"Success:         {self.env_test.env.is_success()}\n")
                    log_file.write("="*80 + "\n\n\n")

            all_returns_test.append(reward_sum)
            all_success_rates_test.append(self.env_test.env.is_success())

            cprint(
                f"Test Episode {episode_id}: "
                f"Reward={reward_sum:.2f}, "
                f"Success={self.env_test.env.is_success()}",
                "yellow",
            )

        # ---- Metrics ----
        SR_mean_test = np.mean(all_success_rates_test)
        returns_mean_test = np.mean(all_returns_test)

        self.logger_util_test.record(SR_mean_test)

        log_data = {
            "mean_success_rates_test": SR_mean_test,
            "mean_returns_test": returns_mean_test,
            "SR_test_L3": self.logger_util_test.average_of_largest_K(),
            "test_mean_score": SR_mean_test,
        }

        with open(log_file_path, 'a') as log_file:
            log_file.write("\n\n" + "="*100 + "\n")
            log_file.write("FINAL SUMMARY - ALL EPISODES\n")
            log_file.write("="*100 + "\n")
            log_file.write(f"Mean Success Rate: {SR_mean_test:.3f}\n")
            log_file.write(f"Mean Return:       {returns_mean_test:.3f}\n")
            log_file.write(f"Total Episodes:    {self.episode_test}\n")
            log_file.write("="*100 + "\n")

        cprint("=" * 50, "green")
        cprint(f"Test - Mean SR: {SR_mean_test:.3f}, Mean Return: {returns_mean_test:.3f}", "green")
        cprint(f"Log saved to: {log_file_path}", "green")
        cprint("=" * 50, "green")

        # ---- Video logging ----
        try:
            videos_test = self.env_test.env.get_video()
            if len(videos_test.shape) == 5:
                videos_test = videos_test[:, 0]

            log_data["sim_video_test"] = wandb.Video(
                videos_test, fps=self.fps, format="mp4"
            )
            cprint("✓ Test video captured", "cyan")

        except Exception as e:
            cprint(f"⚠ Video capture failed: {e}", "yellow")

        try:
            _ = self.env_test.reset()
        except:
            pass

        return log_data