import open3d as o3d
import numpy as np
import os


# ---------- Workspace utils ----------
# def compute_workspace_bounds(pc_xyz, n_std=2):
#     WORK_SPACE = [
#         [0.6389609647247474, 4.405105314033407],  # X
#         [-1.562868614993293, -1.0280730580106126],  # Y
#         [1.2382057599132916, 2.6079001348631374],  # Z
#     ]
#     return WORK_SPACE
#
def compute_workspace_bounds(pc_xyz, n_std=10):
    if pc_xyz.shape[0] == 0:
        return None

    mean = pc_xyz.mean(axis=0)
    std = pc_xyz.std(axis=0)

    workspace = [
        [mean[0] - n_std * std[0], mean[0] + n_std * std[0]],  # X
        [mean[1] - n_std * std[1], mean[1] + n_std * std[1]],  # Y
        [mean[2] - n_std * std[2], mean[2] + n_std * std[2]],  # Z
    ]

    print(f"Workspace bounds (mean ± {n_std}*std): {workspace}")
    return workspace


def crop_workspace(pc_xyz, workspace_bounds):
    mask = (
        (pc_xyz[:, 0] >= workspace_bounds[0][0])
        & (pc_xyz[:, 0] <= workspace_bounds[0][1])
        & (pc_xyz[:, 1] >= workspace_bounds[1][0])
        & (pc_xyz[:, 1] <= workspace_bounds[1][1])
        & (pc_xyz[:, 2] >= workspace_bounds[2][0])
        & (pc_xyz[:, 2] <= workspace_bounds[2][1])
    )
    return mask


# ---------- Load point cloud ----------
PLY_PATH = "/home/varun-edachali/Research/RRC/policy/data/3D-Fusion-Helpers/sim-data-converting/env_pc.ply"
assert os.path.exists(PLY_PATH)

pcd = o3d.io.read_point_cloud(PLY_PATH)
assert not pcd.is_empty()

pc_xyz = np.asarray(pcd.points)
print("Original points:", pc_xyz.shape[0])

# ---------- Compute workspace and crop ----------
workspace = compute_workspace_bounds(pc_xyz)
mask = crop_workspace(pc_xyz, workspace)
cropped_xyz = pc_xyz[mask]
print("Cropped points:", cropped_xyz.shape[0])

# ---------- Create Open3D clouds ----------
# Raw cloud (no color)
pcd_raw_nocolor = o3d.geometry.PointCloud()
pcd_raw_nocolor.points = o3d.utility.Vector3dVector(pc_xyz)
pcd_raw_nocolor.colors = o3d.utility.Vector3dVector([])

# Cropped cloud (no color)
pcd_cropped_nocolor = o3d.geometry.PointCloud()
pcd_cropped_nocolor.points = o3d.utility.Vector3dVector(cropped_xyz)
pcd_cropped_nocolor.colors = o3d.utility.Vector3dVector([])

# Cropped cloud (with original color)
pcd_cropped_color = o3d.geometry.PointCloud()
pcd_cropped_color.points = o3d.utility.Vector3dVector(cropped_xyz)

# Retain original colors for the cropped points
if np.asarray(pcd.colors).shape[0] == pc_xyz.shape[0]:
    original_colors = np.asarray(pcd.colors)
    pcd_cropped_color.colors = o3d.utility.Vector3dVector(original_colors[mask])
else:
    # If original cloud has no color, just leave empty
    pcd_cropped_color.colors = o3d.utility.Vector3dVector([])

# ---------- Create coordinate axes ----------
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])


# ---------- Create workspace bounding box ----------
def create_workspace_box(bounds, color=[1, 0, 0]):
    """Create a wireframe box from workspace bounds."""
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    z_min, z_max = bounds[2]

    # 8 corners of the box
    corners = [
        [x_min, y_min, z_min],  # 0
        [x_max, y_min, z_min],  # 1
        [x_max, y_max, z_min],  # 2
        [x_min, y_max, z_min],  # 3
        [x_min, y_min, z_max],  # 4
        [x_max, y_min, z_max],  # 5
        [x_max, y_max, z_max],  # 6
        [x_min, y_max, z_max],  # 7
    ]

    # 12 edges of the box
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],  # bottom face
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],  # top face
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],  # vertical edges
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return line_set


workspace_box = create_workspace_box(workspace, color=[1, 0, 0])  # Red box

# ---------- Visualizations ----------
print("1️⃣ Showing RAW point cloud (no color) with axes and workspace box")
o3d.visualization.draw_geometries(
    [pcd_raw_nocolor, axes, workspace_box],
    window_name="Raw Point Cloud + Axes + Workspace",
    width=1024,
    height=768,
)

print("2️⃣ Showing CROPPED workspace point cloud (no color) with axes and workspace box")
o3d.visualization.draw_geometries(
    [pcd_cropped_nocolor, axes, workspace_box],
    window_name="Cropped Point Cloud (No Color) + Axes + Workspace",
    width=1024,
    height=768,
)

print("3️⃣ Showing CROPPED workspace point cloud (with color) + axes and workspace box")
o3d.visualization.draw_geometries(
    [pcd_cropped_color, axes, workspace_box],
    window_name="Cropped Point Cloud (With Color) + Axes + Workspace",
    width=1024,
    height=768,
)
