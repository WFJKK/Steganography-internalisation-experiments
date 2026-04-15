#!/bin/bash
# =============================================================================
# Setup for 2x GPU Vast.ai Instance
# =============================================================================
# Rent: 2x A100 40GB+ (or 1x 80GB+ for sequential), container disk >= 80GB
#
# Usage:
#   bash scripts/setup_2gpu.sh YOUR_GITHUB_TOKEN
# =============================================================================

set -e

GITHUB_TOKEN="${1}"
if [ -z "${GITHUB_TOKEN}" ]; then
    echo "Usage: bash scripts/setup_2gpu.sh YOUR_GITHUB_TOKEN"
    exit 1
fi

echo "=== Installing dependencies ==="
pip install torch transformers peft datasets accelerate bitsandbytes

echo "=== Setting up git ==="
git lfs install

echo "=== Cloning repo ==="
cd /workspace
git clone https://github.com/WFJKK/Steganography-internalisation-experiments.git
cd Steganography-internalisation-experiments
git config user.email "kames@github.com"
git config user.name "WFJKK"
git remote set-url origin "https://WFJKK:${GITHUB_TOKEN}@github.com/WFJKK/Steganography-internalisation-experiments.git"

echo "=== HF cache on container disk ==="
export HF_HOME=/workspace/hf_cache
mkdir -p /workspace/hf_cache

echo "=== GPU info ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPUs found: ${NUM_GPUS}"

echo "=== Disk ==="
echo "Root: $(df -h / | tail -1 | awk '{print $4}') free"
echo "/dev/shm: $(df -h /dev/shm | tail -1 | awk '{print $4}') free"

echo ""
echo "=== Setup complete ==="
echo ""
if [ "${NUM_GPUS}" -ge 2 ]; then
    echo "2+ GPUs detected. Run parallel experiments:"
    echo "  cd /workspace/Steganography-internalisation-experiments"
    echo "  nohup bash scripts/run_full_experiments.sh > /dev/shm/full.log 2>&1 &"
    echo "  tail -20 /dev/shm/7b.log   # 7B progress"
    echo "  tail -20 /dev/shm/14b.log  # 14B progress"
else
    echo "1 GPU detected. Run sequentially:"
    echo "  cd /workspace/Steganography-internalisation-experiments"
    echo "  nohup bash scripts/run_7b_block.sh > /dev/shm/7b.log 2>&1 &"
    echo "  # After 7B finishes:"
    echo "  nohup bash scripts/run_14b_block.sh > /dev/shm/14b.log 2>&1 &"
fi
