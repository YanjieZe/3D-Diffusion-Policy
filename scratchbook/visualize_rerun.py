import os
import zarr
import numpy as np
import rerun as rr
import argparse
from tqdm import tqdm

def visualize_trajectory(zarr_path, episode_idx=0, save_path=None):
    """
    Visualize a trajectory from a Zarr file using Rerun.
    """
    print(f"Opening Zarr file: {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    
    # Extract episode boundaries
    episode_ends = root['meta']['episode_ends'][:]
    start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx-1]
    end_idx = episode_ends[episode_idx]
    
    print(f"Extracting episode {episode_idx} with range [{start_idx}, {end_idx})")
    
    # Load episode data
    pc_data = root['data']['point_cloud'][start_idx:end_idx]
    
    has_colors = False
    if pc_data.shape[-1] == 6:
        has_colors = True
        print("Dataset contains color information.")
    else:
        print("Dataset contains only geometry information.")

    if save_path:
        rr.init("Point Cloud Trajectory")
        rr.save(save_path)
    else:
        rr.init("Point Cloud Trajectory", spawn=True)
    
    # Optional: Log a camera to define "one view"
    camera_pos = [1.5, 1.5, 1.5]
    rr.log("world/camera", rr.Pinhole(resolution=[800, 600], focal_length=600))
    rr.log("world/camera", rr.Transform3D(
        translation=camera_pos,
        rotation=rr.Rotation3D(axis_angle=[[1, 1, 1], 0]) # Viewer will auto-focus
    ))
    
    for t in tqdm(range(len(pc_data))):
        rr.set_time_sequence("step", t)
        rr.set_time_seconds("time", t / 10.0) # Assuming 10Hz
        
        current_pc = pc_data[t]
        
        if has_colors:
            xyz = current_pc[:, :3]
            rgb = current_pc[:, 3:]
            # Rerun expects colors in [0, 255] uint8 or [0.0, 1.0] float
            if rgb.max() <= 1.001:
                colors = rgb
            else:
                colors = (rgb / 255.0).astype(np.float32)
            
            rr.log("world/point_cloud", rr.Points3D(positions=xyz, colors=colors, radii=0.003))
        else:
            xyz = current_pc
            # Generate pseudo-colors based on height (Z) for better visibility
            z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
            if z_max > z_min:
                z_norm = (xyz[:, 2] - z_min) / (z_max - z_min)
                colors = np.zeros_like(xyz)
                colors[:, 0] = z_norm
                colors[:, 2] = 1.0 - z_norm
                rr.log("world/point_cloud", rr.Points3D(positions=xyz, colors=colors, radii=0.003))
            else:
                rr.log("world/point_cloud", rr.Points3D(positions=xyz, radii=0.003))

    print("Visualization complete.")
    if not save_path:
        print("Rerun viewer should be open. To save a video, use the recording controls in the Rerun Viewer.")
    else:
        print(f"Recording saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr_path', type=str, 
                        default="/scratch2/cross-emb/DP3_data/data_from_puru_no_eef_binary_gripper.zarr",
                        help="Path to the Zarr dataset file")
    parser.add_argument('--episode', type=int, default=0, help="Episode index to visualize")
    parser.add_argument('--save', type=str, default=None, help="Save to .rrd file instead of spawning viewer")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.zarr_path):
        print(f"Error: Zarr file not found at {args.zarr_path}")
    else:
        visualize_trajectory(args.zarr_path, args.episode, args.save)
