import os
import sys
import zarr
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import pathlib

# Setup python path to match train.py environment
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

# Import TrainDP3Workspace and other utilities
from train import TrainDP3Workspace
from diffusion_policy_3d.common.pytorch_util import dict_apply

def test_policy_zarr(checkpoint_path, zarr_path, episode_idx=0, device='cuda', out_path='inference_comparison.png'):
    print(f"Loading checkpoint from: {checkpoint_path}")
    workspace = TrainDP3Workspace.create_from_checkpoint(checkpoint_path)
    model = workspace.model
    model.to(device)
    model.eval()

    print(f"Opening Zarr file: {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    
    # Check available keys
    data_keys = list(root['data'].keys())
    print(f"Available keys in data: {data_keys}")
    
    # Key mapping: 'state' -> 'agent_pos'
    state_key = 'state' if 'state' in root['data'] else 'agent_pos'
    
    # Extract episode boundaries
    episode_ends = root['meta']['episode_ends'][:]
    start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx-1]
    end_idx = episode_ends[episode_idx]
    
    print(f"Extracting episode {episode_idx} with range [{start_idx}, {end_idx})")
    
    # Load episode data
    # Note: we use 'point_cloud', 'state' (mapped to agent_pos), and 'action'
    pc_data = root['data']['point_cloud'][start_idx:end_idx]
    state_data = root['data'][state_key][start_idx:end_idx]
    action_data = root['data']['action'][start_idx:end_idx]
    
    episode_len = end_idx - start_idx
    n_obs_steps = workspace.cfg.n_obs_steps
    
    predicted_actions = []
    ground_truth_actions = []
    
    print(f"Running inference over {episode_len} steps...")
    # Sliding window inference
    for t in tqdm(range(episode_len)):
        # Handle observation horizon
        # For the first few steps, we pad the observation if needed, 
        # but usually we can just start from t=n_obs_steps-1 if we want full windows.
        # However, let's try to match the episode length for the graph.
        
        obs_start = max(0, t - n_obs_steps + 1)
        obs_end = t + 1
        
        pc_window = pc_data[obs_start:obs_end]
        state_window = state_data[obs_start:obs_end]
        
        # Padding if window is too short (at the beginning of the episode)
        if pc_window.shape[0] < n_obs_steps:
            padding_len = n_obs_steps - pc_window.shape[0]
            pc_window = np.concatenate([np.tile(pc_window[0:1], (padding_len, 1, 1)), pc_window], axis=0)
            state_window = np.concatenate([np.tile(state_window[0:1], (padding_len, 1)), state_window], axis=0)
            
        obs_dict = {
            'point_cloud': pc_window,
            'agent_pos': state_window
        }
        
        # Convert to torch tensors and move to device
        obs_dict_tensor = dict_apply(obs_dict, lambda x: torch.from_numpy(x).float().to(device).unsqueeze(0))
        
        with torch.no_grad():
            result = model.predict_action(obs_dict_tensor)
            # action_pred shape: (1, horizon, action_dim)
            action_pred = result['action_pred'][0].cpu().numpy()
        
        # We take the action corresponding to the current step t
        # In DP3, action_pred[n_obs_steps-1] corresponds to the current observation step
        target_action = action_pred[n_obs_steps - 1]
        
        predicted_actions.append(target_action)
        ground_truth_actions.append(action_data[t])

    predicted_actions = np.array(predicted_actions)
    ground_truth_actions = np.array(ground_truth_actions)
    
    # Create the graph
    num_outputs = ground_truth_actions.shape[1]
    fig, axes = plt.subplots(num_outputs, 1, figsize=(12, 2.5 * num_outputs), sharex=True)
    if num_outputs == 1:
        axes = [axes]
        
    labels = [f"Joint {i+1}" for i in range(6)] + ["Gripper"]
    if num_outputs > 7:
        labels += [f"Extra {i+1}" for i in range(num_outputs - 7)]
    elif num_outputs < 7:
        labels = [f"Output {i+1}" for i in range(num_outputs)]

    for i in range(num_outputs):
        label_name = labels[i] if i < len(labels) else f"Output {i+1}"
        
        # Calculate RMSE for this specific output
        rmse_val = np.sqrt(np.mean((predicted_actions[:, i] - ground_truth_actions[:, i])**2))
        
        axes[i].plot(ground_truth_actions[:, i], label='Expected (GT)', color='#1f77b4', linewidth=2, alpha=0.8)
        axes[i].plot(predicted_actions[:, i], label='Policy Output', color='#d62728', linestyle='--', linewidth=1.5, alpha=0.9)
        axes[i].set_ylabel(label_name, fontweight='bold')
        axes[i].grid(True, which='both', linestyle='--', alpha=0.4)
        
        # Add RMSE text to the plot
        axes[i].text(0.02, 0.95, f'RMSE: {rmse_val:.6f}', transform=axes[i].transAxes, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
                     fontsize='small', fontweight='bold')

        if i == 0:
            axes[i].legend(loc='upper right', frameon=True, fontsize='small')
            axes[i].set_title(f'Policy Evaluation: Episode {episode_idx} | Checkpoint: {os.path.basename(checkpoint_path)}', fontsize=14, fontweight='bold', pad=20)

    axes[-1].set_xlabel('Time Step', fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nGraph saved to: {out_path}")
    
    # Calculate RMSE for each joint
    rmses = np.sqrt(np.mean((predicted_actions - ground_truth_actions)**2, axis=0))
    print("\nRoot Mean Squared Error (RMSE) per output:")
    for i, rmse in enumerate(rmses):
        label_name = labels[i] if i < len(labels) else f"Output {i+1}"
        print(f"{label_name}: {rmse:.6f}")
    print(f"Average RMSE: {np.mean(rmses):.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                        default="data/outputs/rrc_test-dp3-rrc_without_eef_binary_gripper_larger_horizon_seed0/checkpoints/latest.ckpt",
                        help="Path to the model checkpoint")
    parser.add_argument('--zarr_path', type=str, 
                        default="/scratch2/cross-emb/DP3_data/data_from_puru_no_eef_binary_gripper.zarr",
                        help="Path to the Zarr dataset file")
    parser.add_argument('--episode', type=int, default=4, help="Episode index to evaluate")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--out', type=str, default="policy_eval_plot.png", help="Output path for the graph")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)
    if not os.path.exists(args.zarr_path):
        print(f"Error: Zarr file not found at {args.zarr_path}")
        sys.exit(1)
        
    test_policy_zarr(args.checkpoint, args.zarr_path, args.episode, args.device, args.out)
