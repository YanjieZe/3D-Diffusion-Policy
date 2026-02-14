#!/bin/bash

# Base configuration
ZARR_PATH="/scratch2/cross-emb/DP3_data/data_from_puru_no_eef.zarr"
BASE_CHECKPOINT_DIR="/scratch2/cross-emb/DP3_outputs/rrc_test-dp3-rrc_without_eef_obs_1_seed0/checkpoints"
OUTPUT_DIR="joint_plots"

# Ensure output directory exists (relative to the project root)
cd "$(dirname "$0")/.." # Move up to the project root from scratchbook/
mkdir -p "$OUTPUT_DIR"

echo "Starting evaluation loop for epochs 100 to 1000..."

for epoch in $(seq 100 100 1000)
do
    # Format the epoch string as 04d (e.g., 0100, 0200, ...)
    EPOCH_STR=$(printf "%04d" $epoch)
    CHECKPOINT="${BASE_CHECKPOINT_DIR}/epoch=${EPOCH_STR}.ckpt"
    PLOT_OUT="${OUTPUT_DIR}/epoch_${epoch}.png"

    echo "=========================================================="
    echo "Evaluating Epoch: $epoch"
    echo "Checkpoint: $CHECKPOINT"
    echo "=========================================================="

    if [ -f "$CHECKPOINT" ]; then
        python 3D-Diffusion-Policy/test_policy_zarr.py \
            --zarr_path "$ZARR_PATH" \
            --checkpoint "$CHECKPOINT" \
            --out "$PLOT_OUT" \
            --episode 4
    else
        echo "Error: Checkpoint file $CHECKPOINT not found. Skipping..."
    fi

    echo -e "\n"
done

echo "Evaluation loop finished. Plots are available in $OUTPUT_DIR"
