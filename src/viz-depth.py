import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import numpy as np
import open3d as o3d
import cv2

RGB_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"

# Camera intrinsics
fx = 325.4990539550781
fy = 325.4990539550781
cx = 319.9093322753906
cy = 180.0956268310547


def convert_depth(msg):
    """Convert ROS2 depth image to numpy array."""
    h, w = msg.height, msg.width
    arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
    return arr.astype(np.float32) / 1000.0  # mm → m


def convert_rgb(msg):
    """Convert ROS2 RGB image to numpy array."""
    h, w = msg.height, msg.width
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
    return arr  # already RGB8


def depth_to_pointcloud(depth):
    """Backproject depth image (HxW) into 3D (N x 3)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    Z = depth
    X = -(u - cx) * Z / fx
    Y = -(v - cy) * Z / fy

    pts = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    return pts


bag_path = "./data/sync_data_bag_21/sync_data_bag_21_0.db3"

# ----------------------------
# SETUP READER
# ----------------------------
reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
converter_options = rosbag2_py.ConverterOptions("", "")
reader.open(storage_options, converter_options)

topics = reader.get_all_topics_and_types()
rgb_type = depth_type = None

for t in topics:
    if t.name == RGB_TOPIC:
        rgb_type = get_message(t.type)
    if t.name == DEPTH_TOPIC:
        depth_type = get_message(t.type)

if rgb_type is None:
    raise RuntimeError("RGB topic not found")
if depth_type is None:
    raise RuntimeError("Depth topic not found")

rgb_frames = []
depth_frames = []

while reader.has_next():
    topic, data, ts = reader.read_next()

    if topic == RGB_TOPIC:
        msg = deserialize_message(data, rgb_type)
        rgb_frames.append(convert_rgb(msg))

    elif topic == DEPTH_TOPIC:
        msg = deserialize_message(data, depth_type)
        depth_frames.append(convert_depth(msg))

print(f"RGB frames:   {len(rgb_frames)}")
print(f"Depth frames: {len(depth_frames)}")

# ----------------------------
# USE FIRST RGB + DEPTH FRAME
# ----------------------------
rgb = rgb_frames[0]
depth = depth_frames[0]

# ----------------------------
# CREATE COLORED POINT CLOUD
# ----------------------------
pts = depth_to_pointcloud(depth)

# flatten rgb → (N,3)
colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0

# Open3D cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize
o3d.visualization.draw_geometries([pcd])

# Save
o3d.io.write_point_cloud("colored_pointcloud.ply", pcd)
print("Saved: colored_pointcloud.ply")
