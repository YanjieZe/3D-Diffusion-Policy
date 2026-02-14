import numpy as np
import re
import matplotlib.pyplot as plt

# -------- FILE PATHS --------
gt_log_path = "/home/aniruth/Desktop/RRC/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/pybullet_pick_place-dp3-no-table-envenvenvenvenv_seed42/gripper_comparisons/gripper_gt_vs_pred_20260120_134656.txt"

dataset_gripper_path = "/home/aniruth/Desktop/RRC/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/pybullet_pick_place-dp3-no-table-envenvenvenvenv_seed42/gripper_comparisons/dataset-gripper-delta.txt"


# -------- LOAD GT GRIPPER FROM LOG --------
with open(gt_log_path, "r") as f:
    text = f.read()

timesteps = []
gt_vals = []

pattern = re.compile(
    r"GT Timestep (\d+):\s+GT Gripper:\s+([-\d\.eE]+)",
    re.MULTILINE
)

for t, gt in pattern.findall(text):
    timesteps.append(int(t))
    gt_vals.append(float(gt))

gt_vals = np.array(gt_vals)
timesteps = np.array(timesteps)

# -------- LOAD LAST COLUMN FROM DATASET-GRIPPER --------
# assumes space-separated numeric file
dataset = np.loadtxt(dataset_gripper_path)
dataset_gripper = dataset[:, -1]

# align lengths safely
min_len = min(len(gt_vals), len(dataset_gripper))
gt_vals = gt_vals[:min_len]
dataset_gripper = dataset_gripper[:min_len]
timesteps = timesteps[:min_len]

# -------- PLOT --------
plt.figure(figsize=(10, 4))
plt.plot(timesteps, gt_vals, label="GT Gripper")
plt.plot(timesteps, dataset_gripper, label="Dataset Gripper (last col)")
plt.xlabel("Timestep")
plt.ylabel("Gripper Value")
plt.legend()
plt.tight_layout()
plt.show()
