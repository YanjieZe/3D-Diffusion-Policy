import os
import zarr
import numpy as np
import open3d as o3d
import cv2
import argparse
from tqdm import tqdm

def visualize_to_video(zarr_path, episode_idx, output_path, fps):
    print(f"Opening Zarr file: {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    
    episode_ends = root['meta']['episode_ends'][:]
    start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx-1]
    end_idx = episode_ends[episode_idx]
    pc_data = root['data']['point_cloud'][start_idx:end_idx]
    
    width, height = 800, 600
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    render.scene.set_background([0, 0, 0, 1]) # BLACK background
    
    # Add a floor for reference
    floor = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=0.01)
    floor.translate([-1.0, -1.0, 0])
    floor.paint_uniform_color([0.2, 0.2, 0.2]) # Dark grey floor
    render.scene.add_geometry("floor", floor, o3d.visualization.rendering.MaterialRecord())

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Calculate global center for the camera to track
    global_center = pc_data.mean(axis=(0, 1))[:, :3]
    eye = global_center + np.array([1.5, 1.5, 1.0])
    
    print(f"Tracking Center: {global_center}")
    print(f"Camera Eye: {eye}")

    for t in tqdm(range(len(pc_data))):
        current_pc = pc_data[t]
        xyz = current_pc[:, :3]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        
        if current_pc.shape[-1] == 6:
            rgb = current_pc[:, 3:]
            pcd.colors = o3d.utility.Vector3dVector(rgb if rgb.max() <= 1.001 else rgb / 255.0)
        else:
            pcd.paint_uniform_color([0, 1, 0]) # Bright green if no colors
        
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 5.0
        
        render.scene.add_geometry("pcd", pcd, mat)
        
        # Robust camera setup: look at current PC center
        current_center = xyz.mean(axis=0)
        render.setup_camera(60.0, current_center, eye, [0, 0, 1])
        
        img = render.render_to_image()
        img_np = np.asarray(img)
        
        if t == 0:
            nonzero = np.count_nonzero(img_np)
            print(f"Frame 0: {nonzero} non-black pixels out of {800*600*3}")
            if nonzero == 0:
                print("STILL BLACK. Attempting rescue view...")
                # Try a very far back top-down view
                render.setup_camera(90.0, [0, 0, 0], [0, 0, 5], [0, 1, 0])
                img = render.render_to_image()
                img_np = np.asarray(img)
                print(f"Rescue View Pixels: {np.count_nonzero(img_np)}")

        video_writer.write(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        render.scene.remove_geometry("pcd")
        
    video_writer.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr_path', type=str, default="/scratch2/cross-emb/DP3_data/data_from_puru_no_eef_binary_gripper.zarr")
    parser.add_argument('--episode', type=int, default=4)
    parser.add_argument('--out', type=str, default="/scratch2/cross-emb/video_vis/trajectory_video_rescue.mp4")
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()
    visualize_to_video(args.zarr_path, args.episode, args.out, args.fps)
