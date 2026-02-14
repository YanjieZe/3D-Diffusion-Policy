import numpy as np
import matplotlib.pyplot as plt

# Load files
pred = np.loadtxt("/home/aniruth/Desktop/RRC/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/pybullet_pick_place-dp3-no-table-envenvenvenvenv_seed42/action_comparisons/pred_actions.txt")
gt = np.loadtxt("/home/aniruth/Desktop/RRC/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/pybullet_pick_place-dp3-no-table-envenvenvenvenv_seed42/action_comparisons/gt_actions.txt")

assert pred.shape == gt.shape, "Pred and GT files must have same shape"
assert pred.shape[1] == 7, "Expected 7 values per line (6 joints + gripper)"

num_steps = pred.shape[0]
t = np.arange(num_steps)

labels = [
    "Joint 1",
    "Joint 2",
    "Joint 3",
    "Joint 4",
    "Joint 5",
    "Joint 6",
    "Gripper"
]

fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)

MAX_STEPS = 700   # change this

for i in range(7):
    axes[i].plot(t[:MAX_STEPS], gt[:MAX_STEPS, i], label="GT", linewidth=2)
    axes[i].plot(t[:MAX_STEPS], pred[:MAX_STEPS, i], label="Pred", linestyle="--")
    axes[i].set_ylabel(labels[i])
    axes[i].grid(True)

axes[-1].set_xlim(0, MAX_STEPS)


axes[-1].set_xlabel("Timestep")
axes[0].legend(loc="upper right")

plt.tight_layout()
plt.show()
