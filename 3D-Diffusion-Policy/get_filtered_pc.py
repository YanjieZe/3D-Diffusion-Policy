import numpy as np
import open3d as o3d
import torch
import pytorch3d.ops as torch3d_ops

fx_old = 325.40277099609375
fy_old = 325.40277099609375
cx_old = 318.8664245605469
cy_old = 181.1650390625

# fx = 325.40277099609375/4
# fy = 325.40277099609375/4
# cx = 318.8664245605469/4
# cy = 181.1650390625/4

fx = fx_old
fy = fy_old
cx = cx_old
cy = cy_old

# Defaults matching src/extract-rgb-pc.py
DEFAULT_INTRINSICS = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

# Workspace boundaries from src/point-cloud-filtering.py
WORK_SPACE = [
    [-0.855, 0.855],  # X (radius)
    [-0.855, 0.855],  # Y (radius)
    [-0.360, 1.190],  # Z (height)
]


def farthest_point_sampling(points, num_points=2500, use_cuda=True):
    """
    Selects 'num_points' from the input point cloud using Farthest Point Sampling.
    """
    K = [num_points]
    pc = torch.from_numpy(points).float()

    if use_cuda:
        pc = pc.cuda()

    # unsqueeze to make it a batch of 1 for pytorch3d
    sampled, idx = torch3d_ops.sample_farthest_points(points=pc.unsqueeze(0), K=K)

    sampled = sampled.squeeze(0).cpu().numpy()
    idx = idx.squeeze(0).cpu().numpy()

    return sampled, idx


def depth_to_pointcloud(depth_img, intrinsics):
    """
    Unprojects depth image to 3D points.
    """
    h, w = depth_img.shape
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth_img
    X = -(u - cx) * Z / fx
    Y = -(v - cy) * Z / fy

    # Stack to (N, 3)
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    return points


def rgbd_to_sampled_pc(
    rgb_img, depth_img, num_points=2500, intrinsics=None, device="cuda:0"
):
    """
    Takes RGB and Depth images and returns a sampled point cloud.

    Args:
        rgb_img: (H, W, 3) numpy array. Can be uint8 (0-255) or float.
        depth_img: (H, W) numpy array, depth in meters (float).
        num_points: Number of points to sample (default 2500).
        intrinsics: Dict with keys 'fx', 'fy', 'cx', 'cy'. Uses defaults if not provided.
        device: Device string (e.g., 'cuda:0') to verify cuda availability.

    Returns:
        sampled_xyz: (N, 3) numpy array of points.
        sampled_rgb: (N, 3) numpy array of colors (normalized 0-1).
    """
    if intrinsics is None:
        intrinsics = DEFAULT_INTRINSICS

    # 1. Convert to Point Cloud (XYZ)
    points = depth_to_pointcloud(depth_img, intrinsics)

    # 2. Prepare Colors (flatten and normalize if needed)
    if rgb_img.dtype == np.uint8:
        colors = rgb_img.reshape(-1, 3).astype(np.float32) / 255.0
    else:
        colors = rgb_img.reshape(-1, 3).astype(np.float32)

    # 3. Filter invalid points
    # Keep only finite points with positive depth
    mask = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    points = points[mask]
    colors = colors[mask]

    # 4. Crop to Workspace
    mask_ws = (
        (points[:, 0] > WORK_SPACE[0][0])
        & (points[:, 0] < WORK_SPACE[0][1])
        & (points[:, 1] > WORK_SPACE[1][0])
        & (points[:, 1] < WORK_SPACE[1][1])
        & (points[:, 2] > WORK_SPACE[2][0])
        & (points[:, 2] < WORK_SPACE[2][1])
    )
    points = points[mask_ws]
    colors = colors[mask_ws]

    if len(points) == 0:
        print("Warning: No points left after cropping workspace.")
        return np.empty((0, 3)), np.empty((0, 3))

    # 5. Farthest Point Sampling
    # Check if we have enough points to sample from; if not, you might want to pad or just take all.
    # FPS implementation in pytorch3d usually handles K <= N just fine.
    # If N < K, it might duplicate or error depending on implementation options,
    # but usually we expect N >> K here (images are large).

    use_cuda = torch.cuda.is_available() and "cuda" in device
    sampled_xyz, idx = farthest_point_sampling(
        points, num_points=num_points, use_cuda=use_cuda
    )

    sampled_rgb = colors[idx]

    return sampled_xyz, sampled_rgb
