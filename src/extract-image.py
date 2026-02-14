import os
import cv2
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import numpy as np
from bisect import bisect_left

# -----------------------
# User inputs
# -----------------------
data_root = "./data/"             # top dir with subfolders containing .db3
output_root = "./final-data"      # root output dir
RGB_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"
JOINT_TOPIC = "/ufactory/joint_states"
GRIPPER_TOPIC = "/gripper/state"

# -----------------------
# Helper: convert ROS2 Image msg to numpy array
# -----------------------
def get_image_info(encoding):
    if encoding in ["mono8", "8UC1"]:
        return np.uint8, 1
    if encoding in ["mono16", "16UC1", "16SC1"]:
        return np.uint16, 1
    if encoding in ["rgb8", "bgr8"]:
        return np.uint8, 3
    return np.uint8, 3

def convert_image(msg):
    h, w = msg.height, msg.width
    dtype, ch = get_image_info(msg.encoding)
    arr = np.frombuffer(msg.data, dtype=dtype)
    if ch > 1:
        arr = arr.reshape(h, w, ch)
    else:
        arr = arr.reshape(h, w)
    return arr

# -----------------------
# Timestamp alignment helper
# -----------------------
def align_to(reference_ts, target_ts):
    aligned_idx = []
    for ts in reference_ts:
        idx = bisect_left(target_ts, ts)
        if idx == 0:
            aligned_idx.append(0)
        elif idx == len(target_ts):
            aligned_idx.append(len(target_ts)-1)
        else:
            if abs(target_ts[idx]-ts) < abs(target_ts[idx-1]-ts):
                aligned_idx.append(idx)
            else:
                aligned_idx.append(idx-1)
    return aligned_idx

# -----------------------
# Iterate over subfolders and .db3 files
# -----------------------
for subdir, _, files in os.walk(data_root):
    db3_files = [f for f in files if f.endswith(".db3")]
    if not db3_files:
        continue

    rel_path = os.path.relpath(subdir, data_root)
    rgb_folder = os.path.join(output_root, "img", rel_path)
    depth_folder = os.path.join(output_root, "depth", rel_path)
    os.makedirs(rgb_folder, exist_ok=True)
    os.makedirs(depth_folder, exist_ok=True)

    for db3_file in db3_files:
        bag_path = os.path.join(subdir, db3_file)
        print(f"Processing: {bag_path}")

        # open bag
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
        converter_options = rosbag2_py.ConverterOptions("", "")
        reader.open(storage_options, converter_options)
        topics = reader.get_all_topics_and_types()

        # get message types
        def get_msg_type(topic_name):
            types = [t.type for t in topics if t.name == topic_name]
            return get_message(types[0]) if types else None

        rgb_type = get_msg_type(RGB_TOPIC)
        depth_type = get_msg_type(DEPTH_TOPIC)
        joint_type = get_msg_type(JOINT_TOPIC)
        gripper_type = get_msg_type(GRIPPER_TOPIC)

        rgb_frames, rgb_stamp = [], []
        depth_frames, depth_stamp = [], []
        joint_pos, joint_stamp = [], []
        gripper_state, gripper_stamp = [], []

        while reader.has_next():
            topic, data, t = reader.read_next()
            if topic == RGB_TOPIC and rgb_type is not None:
                msg = deserialize_message(data, rgb_type)
                rgb_frames.append(convert_image(msg))
                rgb_stamp.append(t)
            elif topic == DEPTH_TOPIC and depth_type is not None:
                msg = deserialize_message(data, depth_type)
                depth_frames.append(convert_image(msg))
                depth_stamp.append(t)
            elif topic == JOINT_TOPIC and joint_type is not None:
                msg = deserialize_message(data, joint_type)
                joint_pos.append(np.array(msg.position))
                joint_stamp.append(t)
            elif topic == GRIPPER_TOPIC and gripper_type is not None:
                msg = deserialize_message(data, gripper_type)
                gripper_state.append(np.array([msg.data]))
                gripper_stamp.append(t)

        if not rgb_frames:
            print("  No RGB frames found. Skipping.")
            continue

        # convert to arrays
        rgb_stamp = np.array(rgb_stamp)
        depth_stamp = np.array(depth_stamp)
        joint_stamp = np.array(joint_stamp)
        gripper_stamp = np.array(gripper_stamp)

        joint_arr = np.array(joint_pos)
        gripper_arr = np.array(gripper_state)
        depth_arr = np.array(depth_frames)

        # -----------------------
        # Align all to RGB frames
        # -----------------------
        joint_idx = align_to(rgb_stamp, joint_stamp)
        gripper_idx = align_to(rgb_stamp, gripper_stamp)
        depth_idx = align_to(rgb_stamp, depth_stamp)

        # final synchronized arrays
        rgb_sync = np.array(rgb_frames)
        depth_sync = depth_arr[depth_idx]
        joint_sync = joint_arr[joint_idx]
        gripper_sync = gripper_arr[gripper_idx]

        # -----------------------
        # Build agent state
        # -----------------------
        eef_pos = joint_sync[:, :3]
        eef_ori = joint_sync[:, 3:6]
        agent_state = np.hstack([eef_pos, eef_ori, joint_sync, gripper_sync])

        # relative action
        action = np.zeros_like(agent_state)
        action[1:] = agent_state[1:] - agent_state[:-1]

        # -----------------------
        # Save images only for synchronized timestamps
        # -----------------------
        for i, img in enumerate(rgb_sync):
            if img.ndim == 3 and img.shape[2] == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img
            cv2.imwrite(os.path.join(rgb_folder, f"frame_{i:05d}.png"), img_bgr)
            cv2.imwrite(os.path.join(depth_folder, f"frame_{i:05d}.png"), depth_sync[i])

        print(f"  Saved {len(rgb_sync)} synchronized frames to {rgb_folder} and {depth_folder}")
        print(f"  Agent state shape: {agent_state.shape}, Action shape: {action.shape}")

