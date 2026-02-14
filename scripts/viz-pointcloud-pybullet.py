import sys
import pathlib
import numpy as np
from termcolor import cprint
import open3d as o3d

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.append(str(ROOT_DIR))

print("Project root:", ROOT_DIR)

print("\n" + "=" * 80)
cprint("  Point Cloud Visualization", "cyan", attrs=["bold"])
print("=" * 80 + "\n")

from diffusion_policy_3d.env.pybullet.pybullet_wrapper import UR5PickPlaceEnv

env = UR5PickPlaceEnv(use_gui=True, num_points=2500, image_size=224)
cprint("✓ Environment created", "green")

obs = env.reset()
cprint("✓ Got initial observation", "green")

# ===============================
# INITIAL POINT CLOUD STATS
# ===============================
pcd = obs['point_cloud']
print(f"\nInitial Point Cloud Statistics:")
print(f"  Shape: {pcd.shape}")
print(f"  X range: [{pcd[:, 0].min():.3f}, {pcd[:, 0].max():.3f}]")
print(f"  Y range: [{pcd[:, 1].min():.3f}, {pcd[:, 1].max():.3f}]")
print(f"  Z range: [{pcd[:, 2].min():.3f}, {pcd[:, 2].max():.3f}]")

# ===============================
# STEP ONCE
# ===============================
for i in range(1):

    # -------- INITIAL STATE --------
    state_before = obs['agent_pos']
    joints_before = state_before[6:12]
    gripper_before = state_before[12]

    # -------- RANDOM ACTION --------
    action = np.random.uniform(-0.1, 0.1, size=13)
    action[12] = 0.2  # gripper delta

    joint_deltas = action[6:12]
    gripper_delta = action[12]

    print("\n" + "=" * 70)
    print(f"STEP {i+1}")

    print("\nJOINT STATE (INITIAL):")
    for j, val in enumerate(joints_before):
        print(f"  Joint {j}: {val:.4f}")
    print(f"  Gripper: {gripper_before:.4f}")

    print("\nJOINT DELTAS (ACTION):")
    for j, val in enumerate(joint_deltas):
        print(f"  ΔJoint {j}: {val:.4f}")
    print(f"  ΔGripper: {gripper_delta:.4f}")

    # -------- APPLY ACTION --------
    obs, reward, done, info = env.step(action)

    # -------- FINAL STATE --------
    state_after = obs['agent_pos']
    joints_after = state_after[6:12]
    gripper_after = state_after[12]

    print("\nJOINT STATE (FINAL):")
    for j, val in enumerate(joints_after):
        print(f"  Joint {j}: {val:.4f}")
    print(f"  Gripper: {gripper_after:.4f}")

    # -------- SANITY CHECK --------
    print("\nEXPECTED (INITIAL + DELTA):")
    for j in range(6):
        expected = joints_before[j] + joint_deltas[j]
        print(f"  Joint {j}: {expected:.4f}")
    print(f"  Gripper: {gripper_before + gripper_delta:.4f}")

    print("=" * 70)


# ===============================
# FINAL POINT CLOUD VISUALIZATION
# ===============================
final_pcd = obs['point_cloud']

print("\nFinal Point Cloud Statistics:")
print(f"  Shape: {final_pcd.shape}")
print(f"  Valid points: {np.sum(np.abs(final_pcd).sum(axis=1) > 1e-3)}")

cprint("\nVisualizing FINAL point cloud...", "yellow")

o3d_pcd = o3d.geometry.PointCloud()
o3d_pcd.points = o3d.utility.Vector3dVector(final_pcd[:, :3])

o3d.visualization.draw_geometries(
    [o3d_pcd],
    window_name="Final Point Cloud",
    width=900,
    height=700
)

cprint("✓ Done", "green")
