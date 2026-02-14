import zarr
import numpy as np

zarr_path = "/scratch2/cross-emb/DP3_data/data_from_puru_no_eef_binary_gripper.zarr"
root = zarr.open(zarr_path, mode='r')

state = root['data']['state'][0:10]
action = root['data']['action'][0:10]

print("State (first 5):")
print(state[:5])
print("\nAction (first 5):")
print(action[:5])

if 'state' in root['data'] and 'action' in root['data']:
    diff_state = state[1:] - state[:-1]
    print("\nState differences (next - current):")
    print(diff_state[:5])
    print("\nRatio (Diff State / Action) for first 6 dims:")
    # Avoid division by zero
    ratio = diff_state[:, :6] / (action[:-1, :6] + 1e-8)
    print(ratio[:5])
