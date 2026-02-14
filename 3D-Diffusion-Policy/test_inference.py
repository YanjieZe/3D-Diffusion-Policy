import zarr
import requests
import numpy as np
import json
import argparse
import os

def test_inference(zarr_path, url="http://localhost:8000/predict", obs_steps=2):
    print(f"Opening Zarr file: {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    
    # Check available keys in data
    print(f"Available keys in data: {list(root['data'].keys())}")
    
    # We need to grab a sequence of length obs_steps
    # Let's take the first episode, generating a valid slice
    # Indices 0 to obs_steps
    
    payload_obs = {}
    
    # Define the keys we need based on the config/task
    # Usually 'point_cloud' and 'agent_pos' for DP3
    keys_to_fetch = ['point_cloud', 'state']
    
    start_idx = 0
    end_idx = start_idx + obs_steps
    
    # Remove PDB
    # import pdb; pdb.set_trace()

    for key in keys_to_fetch:
        if key in root['data']:
            # Get data and convert to list
            data_slice = root['data'][key][start_idx:end_idx]
            
            # Key mapping for server: 'state' -> 'agent_pos'
            server_key = key
            if key == 'state':
                server_key = 'agent_pos'
                print(f"Mapping 'state' -> 'agent_pos' for server request")

            # Simple cast to list for JSON serialization
            # data_slice is (T, ...) numpy array
            payload_obs[server_key] = data_slice.tolist()
            print(f"Loaded {key}: shape {data_slice.shape}")
        else:
            print(f"Warning: Key {key} not found in zarr dataset, skipping.")

    request_data = {
        "observation": payload_obs
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            # import pdb; pdb.set_trace()
            action = np.array(result['action'])
            print("\nSUCCESS!")
            print(f"Received Prediction Shape: {action.shape}")
            
            # Fetch Ground Truth Actions
            # The model returns actions starting from (obs_steps - 1)
            # Default obs_steps=2, so it starts from index 1 relative to the sample (time t+1)
            
            if 'action' in root['data']:
                pred_len = action.shape[0]
                # Align GT: start_idx + (obs_steps - 1)
                gt_start = start_idx + (obs_steps - 1)
                gt_end = gt_start + pred_len
                
                gt_action = root['data']['action'][gt_start:gt_end]
                
                print(f"GT Action Shape aligned: {gt_action.shape}")
                
                # Check if we retrieved enough GT data
                if gt_action.shape[0] < pred_len:
                    print(f"Warning: GT ends early! (GT len {gt_action.shape[0]} < Pred len {pred_len})")
                    # Truncate comparison
                    action = action[:gt_action.shape[0]]

                print("-" * 60)
                print(f"Step 0 (Start) Comparison:")
                print(f"Pred: {action[0]}")
                print(f"GT  : {gt_action[0]}")

                print(f"\nStep {len(action)-1} (End) Comparison:")
                print(f"Pred: {action[-1]}")
                print(f"GT  : {gt_action[-1]}")
                
                # RMSE
                mse = np.mean((action - gt_action) ** 2)
                rmse = np.sqrt(mse)
                print(f"\nRMSE (over {len(action)} steps): {rmse:.6f}")
                print("-" * 60)
            else:
                 print("Key 'action' not found in Zarr, cannot compare.")

        else:
            print(f"\nFailed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"\nConnection Error: Is the inference server running on {url}?")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr_path', type=str, 
                        default="/scratch2/cross-emb/DP3_data/data_from_puru_no_eef_binary_gripper.zarr",
                        help="Path to the Zarr dataset file")
    parser.add_argument('--url', type=str, default="http://localhost:8000/predict",
                        help="Inference server URL")
    parser.add_argument('--steps', type=int, default=2, help="Number of observation steps to send")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.zarr_path):
        print(f"Error: Zarr file not found at {args.zarr_path}")
    else:
        test_inference(args.zarr_path, args.url, args.steps)
