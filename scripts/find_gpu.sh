#!/bin/bash

# Define an array of GPU IDs to exclude
exclude_gpus=(6 7) # Add the GPU IDs you want to exclude
# exclude_gpus=()

# Function to check if an array contains a value
containsElement () {
  for e in "${@:2}"; do
    if [[ "$e" == "$1" ]]; then
      return 0
    fi
  done
  return 1
}

# Get GPU usage
gpu_usage=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

# Initialize variables for tracking minimum usage
min_usage=999999
min_gpu=0

# Iterate over each GPU
while IFS=, read -r gpu_id usage
do
    # Check if this GPU is in the exclude list
    if containsElement "$gpu_id" "${exclude_gpus[@]}"; then
        continue
    fi

    # Compare and update the GPU with minimum memory usage
    if (( usage < min_usage )); then
        min_usage=$usage
        min_gpu=$gpu_id
    fi
done <<< "$gpu_usage"

# Output the GPU index with the minimum usage
echo "$min_gpu"
