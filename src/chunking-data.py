import os
import zarr
import numpy as np
from PIL import Image
import open3d as o3d

# ------------------------
# Paths and parameters
# ------------------------
root = "./final-data/"
out_zarr = "final.zarr"
chunk_size = 100  # number of samples per chunk

# ------------------------
# Prepare storage
# ------------------------
all_imgs = []
all_pcs = []
all_actions = []
all_states = []

# Subfolders (assume same in img, point_cloud, action, state)
# subfolders = sorted(os.listdir(os.path.join(root, "img")), key=lambda x: int(x))
subfolders = sorted(os.listdir(os.path.join(root, "img")))
# ------------------------
# Load data
# ------------------------
for subf in subfolders:
    print(f"Processing subfolder {subf}...")

    # --- Images ---
    img_path = os.path.join(root, "img", subf)
    img_files = sorted(os.listdir(img_path))
    for f in img_files:
        img_arr = np.array(Image.open(os.path.join(img_path, f)))  # (H,W,C)
        all_imgs.append(img_arr)

    # --- Point clouds ---
    # pc_path = os.path.join(root, "point_cloud", subf)
    # pc_files = sorted(os.listdir(pc_path))
    # for f in pc_files:
    #     pcd = o3d.io.read_point_cloud(os.path.join(pc_path, f))
    #     points = np.asarray(pcd.points)
    #     if pcd.has_colors():
    #         colors = np.asarray(pcd.colors) * 255
    #         pc_arr = np.concatenate([points, colors], axis=1)  # (N_points,6)
    #     else:
    #         pc_arr = np.concatenate([points, np.zeros_like(points)], axis=1)
    #     all_pcs.append(pc_arr)

    # # --- Actions ---
    # action_file = os.path.join(root, "action", f"{subf}_updated.txt")
    # action_arr = np.loadtxt(action_file)  # shape=(rows, features)
    # all_actions.append(action_arr)

    # # --- States ---
    # state_file = os.path.join(root, "state", f"{subf}_updated.txt")
    # state_arr = np.loadtxt(state_file)
    # all_states.append(state_arr)

# ------------------------
# Convert lists to arrays
# ------------------------
all_imgs = np.stack(all_imgs, axis=0)        # (N_images, H, W, C)
# all_pcs = np.stack(all_pcs, axis=0)          # (N_pcs, N_points, 6)
# all_actions = np.vstack(all_actions)         # (N_actions, features)
# all_states = np.vstack(all_states)           # (N_states, features)

print("All data collected:")
print(f"Total images: {all_imgs.shape[0]}")
# print(f"Total point clouds: {all_pcs.shape[0]}")
# print(f"Total actions: {all_actions.shape[0]}")
# print(f"Total states: {all_states.shape[0]}")

# ------------------------
# Create Zarr
# ------------------------
if os.path.exists(out_zarr):
    print(f"Overwriting existing Zarr file: {out_zarr}")
    import shutil
    shutil.rmtree(out_zarr)

zroot = zarr.open(out_zarr, mode="w")
compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

# Chunk sizes
img_chunk_size = (chunk_size, all_imgs.shape[1], all_imgs.shape[2], all_imgs.shape[3])
# pc_chunk_size = (chunk_size, all_pcs.shape[1], all_pcs.shape[2])
# action_chunk_size = (chunk_size, all_actions.shape[1])
# state_chunk_size = (chunk_size, all_states.shape[1])
# 
# Save datasets
zroot.create_dataset('img', data=all_imgs, chunks=img_chunk_size, dtype='uint8', compressor=compressor)
# zroot.create_dataset('point_cloud', data=all_pcs, chunks=pc_chunk_size, dtype='float64', compressor=compressor)
# zroot.create_dataset('actions', data=all_actions, chunks=action_chunk_size, dtype='float32', compressor=compressor)
# zroot.create_dataset('states', data=all_states, chunks=state_chunk_size, dtype='float32', compressor=compressor)
# 
print(f"Saved Zarr file: {out_zarr}")
print("Dataset shapes:")
print(f"img: {all_imgs.shape}")
# print(f"point_cloud: {all_pcs.shape}")
# print(f"actions: {all_actions.shape}")
# print(f"states: {all_states.shape}")
