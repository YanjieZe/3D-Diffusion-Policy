import os
import sys
import zarr
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import pathlib

# Setup python path
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

from train import TrainDP3Workspace
from diffusion_policy_3d.common.pytorch_util import dict_apply

def test_policy_chunked_simple(checkpoint_path, zarr_path, episode_idx=0, device='cuda', out_path='chunked_eval.png'):
    print(f"Loading checkpoint: {checkpoint_path}")
    workspace = TrainDP3Workspace.create_from_checkpoint(checkpoint_path)
    model = workspace.model.to(device).eval()

    print(f"Opening Zarr: {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    
    state_key = 'state' if 'state' in root['data'] else 'agent_pos'
    episode_ends = root['meta']['episode_ends'][:]
    start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx-1]
    end_idx = episode_ends[episode_idx]
    
    # Load data
    pc_data = root['data']['point_cloud'][start_idx:end_idx]
    gt_state_data = root['data'][state_key][start_idx:end_idx]
    gt_action_data = root['data']['action'][start_idx:end_idx]
    
    episode_len = end_idx - start_idx
    n_obs_steps = workspace.cfg.n_obs_steps
    chunk_size = 8 # As requested
    
    # Result buffers
    pred_actions_all = np.full_like(gt_action_data, np.nan)
    pred_positions_all = np.full_like(gt_state_data, np.nan)

    print(f"Running evaluation: Episode {episode_idx} ({episode_len} steps), Chunk Size {chunk_size}")


    for step in tqdm(range(0, episode_len - chunk_size, chunk_size)):
        # 1. Prepare Observation (GT Reset)
        obs_start = max(0, step - n_obs_steps + 1)
        obs_end = step + 1
        
        pc_window = pc_data[obs_start:obs_end]
        state_window = gt_state_data[obs_start:obs_end]
        
        # Padding for early steps
        if pc_window.shape[0] < n_obs_steps:
            pad_len = n_obs_steps - pc_window.shape[0]
            pc_window = np.concatenate([np.tile(pc_window[:1], (pad_len, 1, 1)), pc_window], axis=0)
            state_window = np.concatenate([np.tile(state_window[:1], (pad_len, 1)), state_window], axis=0)
            
        obs_dict = {
            'point_cloud': pc_window,
            'agent_pos': state_window
        }
        obs_dict_tensor = dict_apply(obs_dict, lambda x: torch.from_numpy(np.array(x)).float().to(device).unsqueeze(0))
        
        # 2. Inference
        with torch.no_grad():
            result = model.predict_action(obs_dict_tensor)
            # Take first 'chunk_size' actions starting from current step
            pred_chunk = result['action_pred'][0].cpu().numpy()[n_obs_steps-1 : n_obs_steps-1+chunk_size]

        # 3. Integration & Storage
        curr_pos = gt_state_data[step].copy()
        
        for i in range(chunk_size):
            global_idx = step + i
            delta = pred_chunk[i]
            
            # Store delta
            pred_actions_all[global_idx] = delta
            
            # Integrated Position logic
            # Joints (0-5) are deltas
            # Gripper (6) setup as absolute/last-predicted per typical DP configs
            next_pos = curr_pos.copy()
            next_pos[:6] += delta[:6]
            next_pos[6] += delta[6] # Gripper often treated as absolute command
            
            pred_positions_all[global_idx + 1] = next_pos
            curr_pos = next_pos

    # 4. Plotting
    num_dims = gt_state_data.shape[1]
    fig, axes = plt.subplots(num_dims, 2, figsize=(16, 3 * num_dims), sharex=True)
    labels = [f"Joint {i+1}" for i in range(6)] + ["Gripper"]
    
    for i in range(num_dims):
        # Left: Deltas/Actions
        axes[i, 0].plot(gt_action_data[:, i], label='GT Action', color='#1f77b4', linewidth=2, alpha=0.8)
        axes[i, 0].plot(pred_actions_all[:, i], label='Pred Delta', color='#d62728', linewidth=1.5, linestyle='--')
        axes[i, 0].set_ylabel(f"{labels[i]} Delta")
        if i == 0: 
            axes[i, 0].set_title("Commanded Deltas (Action)")
            axes[i, 0].legend(loc='upper right', fontsize='x-small')
        axes[i, 0].grid(True, alpha=0.3)

        # Right: Positions
        axes[i, 1].plot(gt_state_data[:, i], label='GT State', color='black', linewidth=2, alpha=0.9)
        axes[i, 1].plot(pred_positions_all[:, i], label='Pred Pos (Integrated)', color='#2ca02c', linewidth=1.5, linestyle='--')
        axes[i, 1].set_ylabel(f"{labels[i]} Pos")
        if i == 0: 
            axes[i, 1].set_title("Resulting Positions (GT Reset per Chunk)")
            axes[i, 1].legend(loc='upper right')
        axes[i, 1].grid(True, alpha=0.3)
        
        # Draw vertical lines to show chunk boundaries
        for chunk_boundary in range(0, episode_len, chunk_size):
            axes[i, 0].axvline(x=chunk_boundary, color='gray', linestyle=':', alpha=0.2)
            axes[i, 1].axvline(x=chunk_boundary, color='gray', linestyle=':', alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--zarr_path', type=str, required=True)
    parser.add_argument('--episode', type=int, default=4)
    parser.add_argument('--out', type=str, default="chunked_eval_mode_a.png")
    args = parser.parse_args()
    
    test_policy_chunked_simple(args.checkpoint, args.zarr_path, args.episode, out_path=args.out)
