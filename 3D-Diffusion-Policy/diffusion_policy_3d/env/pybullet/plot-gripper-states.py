import numpy as np
import matplotlib.pyplot as plt

# Load txt file
data = np.loadtxt(
    "/home/aniruth/Desktop/RRC/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/"
    "pybullet_pick_place-dp3-no-table-envenvenvenvenv_seed42/"
    "gripper_comparisons/dataset-gripper-delta-new.txt"
)

# Extract last column (gripper deltas)
gripper_deltas = data[:, -1]

# Integrate (add consecutive deltas)
gripper_state = np.cumsum(gripper_deltas)

# Create 2x1 plot
plt.figure(figsize=(10, 6))

# -------- TOP: Gripper State --------
plt.subplot(2, 1, 1)
plt.plot(gripper_state)
plt.ylabel("Gripper state")
plt.title("Gripper State and Gripper Delta")

# Pick region
plt.axvspan(100, 150, alpha=0.15)
plt.axvline(100, linestyle="--")
plt.axvline(150, linestyle="--")

plt.axvspan(350, 400, alpha=0.15)
plt.axvline(350, linestyle="--")
plt.axvline(400, linestyle="--")

# -------- BOTTOM: Gripper Delta --------
plt.subplot(2, 1, 2)
plt.plot(gripper_deltas)
plt.xlabel("Timestep")
plt.ylabel("Gripper delta")


plt.tight_layout()
plt.show()
