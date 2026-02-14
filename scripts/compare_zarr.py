import os
import zarr
import numpy as np
from termcolor import cprint

def compare_zarr_files(data_dir):
    zarr_files = [f for f in os.listdir(data_dir) if f.endswith('.zarr')]
    zarr_files.sort()

    if not zarr_files:
        print(f"No .zarr files found in {data_dir}")
        return

    print(f"\nComparing {len(zarr_files)} Zarr files in {data_dir}")
    print("=" * 100)

    for filename in zarr_files:
        path = os.path.join(data_dir, filename)
        try:
            z = zarr.open(path, mode='r')
            
            cprint(f"\nFile: {filename}", 'blue', attrs=['bold'])
            
            # Metadata
            n_episodes = 0
            n_steps = 0
            if 'meta' in z and 'episode_ends' in z['meta']:
                episode_ends = z['meta']['episode_ends'][:]
                n_episodes = len(episode_ends)
                n_steps = episode_ends[-1] if n_episodes > 0 else 0
            
            print(f"  Episodes: {n_episodes}")
            print(f"  Total Steps: {n_steps}")
            
            # Data Contents
            if 'data' in z:
                print("  Data Arrays:")
                for key in sorted(z['data'].keys()):
                    arr = z['data'][key]
                    print(f"    - {key:15} | shape: {str(arr.shape):20} | dtype: {arr.dtype}")
            else:
                cprint("  [WARN] No 'data' group found!", 'yellow')

        except Exception as e:
            cprint(f"  [ERROR] Could not read {filename}: {e}", 'red')
    
    print("\n" + "=" * 100)

if __name__ == "__main__":
    DATA_ROOT = "/scratch2/cross-emb/DP3_data"
    compare_zarr_files(DATA_ROOT)
