import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import numpy as np

bag_path = "./data/sync_data_bag_21/sync_data_bag_21_0.db3"

# ---------- OPEN BAG ----------
reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
converter_options = rosbag2_py.ConverterOptions("", "")
reader.open(storage_options, converter_options)

topics = reader.get_all_topics_and_types()
print("\n=== TOPICS FOUND ===")
for t in topics:
    print(t.name, " -> ", t.type)

# Helper for image msg
def get_image_info(encoding):
    if encoding in ["mono8", "8UC1"]:
        return np.uint8, 1
    if encoding in ["mono16", "16UC1", "16SC1"]:
        return np.uint16, 1
    if encoding in ["rgb8", "bgr8"]:
        return np.uint8, 3
    if encoding in ["rgba8", "bgra8"]:
        return np.uint8, 4
    print(f"[WARN] Unknown encoding '{encoding}', defaulting to uint8")
    return np.uint8, 1


print("\n=== READING MESSAGES ===")
count = 0

while reader.has_next():
    topic, data, t = reader.read_next()
    msg_type = get_message([x.type for x in topics if x.name == topic][0])
    msg = deserialize_message(data, msg_type)

    # -----------------------------
    # PRINT SHAPE FOR ANY MSG TYPE
    # -----------------------------
    print(f"\n[{topic}] {msg_type.__name__}")

    # ---- IMAGE ----
    if msg_type.__name__ == "Image":
        h, w = msg.height, msg.width
        dtype, channels = get_image_info(msg.encoding)

        arr = np.frombuffer(msg.data, dtype=dtype)

        if channels > 1:
            arr = arr.reshape(h, w, channels)
        else:
            arr = arr.reshape(h, w)

        print(f"  • encoding = {msg.encoding}")
        print(f"  • shape = {arr.shape}")
        print(f"  • dtype = {arr.dtype}")

    # ---- JointState ----
    elif msg_type.__name__ == "JointState":
        print(f"  • name count     = {len(msg.name)}")
        print(f"  • position shape = {len(msg.position)}")
        print(f"  • velocity shape = {len(msg.velocity)}")
        print(f"  • effort shape   = {len(msg.effort)}")

    # ---- Float64 ----
    elif msg_type.__name__ == "Float64":
        print(f"  • value = {msg.data}")

    # ---- Int32 ----
    elif msg_type.__name__ == "Int32":
        print(f"  • value = {msg.data}")

    # ---- JointTrajectoryControllerState ----
    elif msg_type.__name__ == "JointTrajectoryControllerState":
        print(f"  • joint count = {len(msg.joint_names)}")
        print(f"  • positions   = {np.array(msg.actual.positions).shape}")
        print(f"  • velocities  = {np.array(msg.actual.velocities).shape}")
        print(f"  • efforts     = {np.array(msg.actual.effort).shape}")

    # ---- fallback for other msgs ----
    else:
        print("  • raw message fields:")
        print(msg)

    count += 1
    if count >= 20:     # avoid huge dump
        break

print("\n=== DONE ===")
