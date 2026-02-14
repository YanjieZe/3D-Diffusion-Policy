import os
import rosbag2_py
import numpy as np
import cv2
import open3d as o3d
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# ------------------------------
# CONFIG
# ------------------------------
DATA_ROOT = "/scratch2/cross-emb/lab_data/Newer Data"  # YOU WILL SET THIS
OUTPUT_ROOT = "./final-data"

RGB_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"

# Camera intrinsics
fx = 325.4990539550781
fy = 325.4990539550781
cx = 319.9093322753906
cy = 180.0956268310547


# ------------------------------
# HELPERS
# ------------------------------
def convert_depth(msg):
    h, w = msg.height, msg.width
    depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
    return depth.astype(np.float32) / 1000.0  # mm → meters


def convert_rgb(msg):
    h, w = msg.height, msg.width
    return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)


def depth_to_pc(depth):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    Z = depth
    X = -(u - cx) * Z / fx
    Y = -(v - cy) * Z / fy
    return np.stack((X, Y, Z), axis=-1).reshape(-1, 3)


# ------------------------------
# PROCESS ONE BAG
# ------------------------------
def process_bag(folder_name, db3_path):
    print(f"\n=== Processing {folder_name} ===")

    out_img_dir = os.path.join(OUTPUT_ROOT, "img", folder_name)
    out_pc_dir = os.path.join(OUTPUT_ROOT, "pc", folder_name)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_pc_dir, exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=db3_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)

    topics = reader.get_all_topics_and_types()
    rgb_type = depth_type = None

    for t in topics:
        if t.name == RGB_TOPIC:
            rgb_type = get_message(t.type)
        if t.name == DEPTH_TOPIC:
            depth_type = get_message(t.type)

    if rgb_type is None or depth_type is None:
        print(f"Skipping {folder_name}: missing topics")
        return

    rgb = None
    frame_id = 0

    while reader.has_next():
        topic, data, ts = reader.read_next()

        # ------------------ RGB ------------------
        if topic == RGB_TOPIC:
            msg = deserialize_message(data, rgb_type)
            rgb = convert_rgb(msg)

            cv2.imwrite(
                os.path.join(out_img_dir, f"rgb_{frame_id:05d}.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )

        # ------------------ DEPTH + PC ------------------
        elif topic == DEPTH_TOPIC:
            msg = deserialize_message(data, depth_type)
            depth = convert_depth(msg)

            pts = depth_to_pc(depth)

            # ------------------ CLEAN POINTS ------------------
            # reject Z=0, nan, inf
            mask = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
            pts = pts[mask].astype(np.float32)

            # colors
            if rgb is not None:
                colors = (rgb.reshape(-1, 3) / 255.0)[mask]
            else:
                colors = np.zeros((pts.shape[0], 3), dtype=np.float32)

            # clip to valid range
            colors = np.clip(colors, 0.0, 1.0).astype(np.float32)

            # ------------------ BUILD PCD ------------------
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # ------------------ SAVE PLY SAFELY ------------------
            out_ply = os.path.join(out_pc_dir, f"cloud_{frame_id:05d}.ply")
            o3d.io.write_point_cloud(out_ply, pcd, write_ascii=False)

            frame_id += 1

    print(f"Finished {folder_name}: {frame_id} frames")


# ------------------------------
# MAIN LOOP OVER SUBFOLDERS
# ------------------------------
for folder in sorted(os.listdir(DATA_ROOT)):
    folder_path = os.path.join(DATA_ROOT, folder)

    if not os.path.isdir(folder_path):
        continue

    db3_files = [f for f in os.listdir(folder_path) if f.endswith(".db3")]
    if not db3_files:
        continue

    db3_path = os.path.join(folder_path, db3_files[0])
    process_bag(folder, db3_path)

print("\nDONE ✔ All data processed.")
