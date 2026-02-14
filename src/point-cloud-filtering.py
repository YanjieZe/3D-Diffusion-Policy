import numpy as np
import open3d as o3d
import torch
import pytorch3d.ops as torch3d_ops


# ---------------------------
# FARTHEST POINT SAMPLING (returns both XYZ + indices)
# ---------------------------
def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    K = [num_points]

    if use_cuda:
        pc = torch.from_numpy(points).float().cuda()
        sampled, idx = torch3d_ops.sample_farthest_points(
            points=pc.unsqueeze(0), K=K
        )
        sampled = sampled.squeeze(0).cpu().numpy()
        idx = idx.squeeze(0).cpu().numpy()
    else:
        pc = torch.from_numpy(points).float()
        sampled, idx = torch3d_ops.sample_farthest_points(
            points=pc.unsqueeze(0), K=K
        )
        sampled = sampled.squeeze(0).numpy()
        idx = idx.squeeze(0).numpy()

    return sampled, idx


# ---------------------------
# WORKSPACE CROP + FPS (XYZ + RGB)
# ---------------------------
def process_point_cloud(pc_xyz, pc_rgb):
    """
    pc_xyz: (N, 3)
    pc_rgb: (N, 3)
    """

    WORK_SPACE = [
            [-0.855, 0.855],  # X range (Radius)
            [-0.855, 0.855],  # Y range (Radius)
            [-0.360, 1.190]   # Z range (Height limits)
        ]

    # Crop
    mask = (
        (pc_xyz[:, 0] > WORK_SPACE[0][0]) & (pc_xyz[:, 0] < WORK_SPACE[0][1]) &
        (pc_xyz[:, 1] > WORK_SPACE[1][0]) & (pc_xyz[:, 1] < WORK_SPACE[1][1]) &
        (pc_xyz[:, 2] > WORK_SPACE[2][0]) & (pc_xyz[:, 2] < WORK_SPACE[2][1])
    )

    pc_xyz = pc_xyz[mask]
    pc_rgb = pc_rgb[mask]

    print(f"After crop: {pc_xyz.shape[0]} points")

    # FPS (XYZ only)
    pc_xyz_fps, idx = farthest_point_sampling(pc_xyz, num_points=2500, use_cuda=False)

    # Use same indices to pick corresponding RGB
    pc_rgb_fps = pc_rgb[idx]

    print(f"After FPS: {pc_xyz_fps.shape[0]} points")

    return pc_xyz_fps, pc_rgb_fps


# ---------------------------
# VISUALIZATION
# ---------------------------
def visualize(pc_xyz, pc_rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz)
    pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
    o3d.visualization.draw_geometries([pcd])


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    input_file = "./colored_pointcloud.ply"

    pcd = o3d.io.read_point_cloud(input_file)
    pc_xyz = np.asarray(pcd.points)
    pc_rgb = np.asarray(pcd.colors)

    print(f"Loaded XYZ: {pc_xyz.shape}")
    print(f"Loaded RGB: {pc_rgb.shape}")

    processed_xyz, processed_rgb = process_point_cloud(pc_xyz, pc_rgb)

    visualize(processed_xyz, processed_rgb)
