#!/bin/bash
# =============================================================================
# Vast.ai Setup for 14B Experiments
# =============================================================================
# Run this ONCE after renting a new instance.
#
# BEFORE RENTING:
#   - GPU: A100 40GB minimum (40GB VRAM is enough for 14B in 4-bit)
#   - Container disk size: 50GB minimum (14B model download is ~29GB)
#   - Image: any PyTorch image with CUDA
#
# Usage:
#   bash setup_vastai.sh YOUR_GITHUB_TOKEN
# =============================================================================

set -e

GITHUB_TOKEN="${1}"

if [ -z "${GITHUB_TOKEN}" ]; then
    echo "Usage: bash setup_vastai.sh YOUR_GITHUB_TOKEN"
    echo "Get token from: https://github.com/settings/tokens"
    exit 1
fi

echo "=== Installing dependencies ==="
pip install torch transformers peft datasets accelerate bitsandbytes

echo ""
echo "=== Setting up git LFS ==="
git lfs install

echo ""
echo "=== Cloning repo ==="
cd /workspace
git clone https://github.com/WFJKK/Steganography-internalisation-experiments.git
cd Steganography-internalisation-experiments
git config user.email "kames@github.com"
git config user.name "WFJKK"
git remote set-url origin "https://WFJKK:${GITHUB_TOKEN}@github.com/WFJKK/Steganography-internalisation-experiments.git"

echo ""
echo "=== Setting HF cache to container disk (model is ~29GB, too big for /dev/shm) ==="
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
mkdir -p /workspace/hf_cache

echo ""
echo "=== Checking GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "=== Checking disk ==="
ROOT_FREE=$(df -BG / | tail -1 | awk '{print $4}' | tr -d 'G')
echo "Root: ${ROOT_FREE}GB free"
echo "/dev/shm: $(df -h /dev/shm | tail -1 | awk '{print $4}') free"
if [ "$ROOT_FREE" -lt 35 ]; then
    echo ""
    echo "WARNING: Container disk has <35GB free. 14B model needs ~29GB."
    echo "Consider re-renting with container size >= 50GB."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  cd /workspace/Steganography-internalisation-experiments"
echo "  nohup bash scripts/run_14b_experiments.sh > /dev/shm/14b_experiments.log 2>&1 &"
echo "  tail -20 /dev/shm/14b_experiments.log    # check progress (repeat, do NOT use tail -f)"
