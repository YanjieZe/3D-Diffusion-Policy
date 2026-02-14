import zarr
import numpy as np

FILES = [
    "/scratch2/cross-emb/DP3_data/data_from_puru_no_eef.zarr",
    "/scratch2/cross-emb/DP3_data/data_from_puru_no_eef_binary_gripper.zarr",
    "/scratch2/cross-emb/DP3_data/data_from_puru_correct.zarr"
]

def inspect_vals():
    for path in FILES:
        name = path.split('/')[-1]
        print(f"\n--- {name} ---")
        try:
            z = zarr.open(path, mode='r')
            state = z['data']['state'][0:5]
            print(f"State (first 5 samples):\n{state}")
            print(f"Shape: {state.shape}")
        except Exception as e:
            print(f"Error reading {name}: {e}")

if __name__ == "__main__":
    inspect_vals()
