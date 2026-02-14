import numpy as np
import open3d as o3d

def visualize(pc_xyz, pc_rgb=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz)

    if pc_rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(pc_rgb)

    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    input_file = "/home/aniruth/Desktop/RRC/3D-Fusion-Helpers/Lab-data-zarr-converting/src/final-data/pc/sync_data_bag_21/cloud_00004.ply"

    pcd = o3d.io.read_point_cloud(input_file)

    pc_xyz = np.asarray(pcd.points)
    pc_rgb = np.asarray(pcd.colors)

    print(f"Loaded: {pc_xyz.shape[0]} points")
    print(f"Has colors: {pc_rgb.shape}")

    visualize(pc_xyz, pc_rgb)
