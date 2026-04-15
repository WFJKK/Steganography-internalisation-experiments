#!/bin/bash
# =============================================================================
# Setup script for Vast.ai instance
# Run this ONCE after renting, before the scaling experiment
# =============================================================================
#
# RENTAL REQUIREMENTS:
#   - 2x H200 (141GB VRAM each)
#   - Container disk: >= 200GB (6 models cached = ~120GB)
#   - Image: any with Python 3.10+
#
# USAGE:
#   bash setup_model_size_scaling.sh YOUR_GITHUB_TOKEN
#
# =============================================================================
set -e

if [ -z "$1" ]; then
    echo "Usage: bash setup_model_size_scaling.sh YOUR_GITHUB_TOKEN"
    exit 1
fi

GITHUB_TOKEN="$1"

echo "=== Installing dependencies ==="
pip install torch transformers peft datasets accelerate bitsandbytes --break-system-packages 2>/dev/null || \
pip install torch transformers peft datasets accelerate bitsandbytes

echo ""
echo "=== Cloning repo (skipping LFS) ==="
export GIT_LFS_SKIP_SMUDGE=1
cd /workspace
git clone "https://WFJKK:${GITHUB_TOKEN}@github.com/WFJKK/Steganography-internalisation-experiments.git"
cd Steganography-internalisation-experiments
git config user.email "kames@github.com"
git config user.name "WFJKK"

echo ""
echo "=== Verifying data files ==="
for f in data/acrostics_twist/news/stage1_8bit/train.jsonl \
         data/acrostics_twist/news/stage1_8bit/val.jsonl \
         data/acrostics_twist/news/v0_8bit/train.jsonl \
         data/acrostics_twist/news/v0_8bit/test.jsonl; do
    if [ -f "$f" ]; then
        COUNT=$(wc -l < "$f")
        echo "  OK: $f ($COUNT lines)"
    else
        echo "  MISSING: $f"
    fi
done

echo ""
echo "=== Checking disk space ==="
df -h /workspace | tail -1
df -h /dev/shm | tail -1

echo ""
echo "=== Checking GPUs ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi failed"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next step:"
echo "  cd /workspace/Steganography-internalisation-experiments"
echo "  nohup bash scripts/run_model_size_scaling.sh > /dev/shm/scaling.log 2>&1 &"
echo "  tail -5 /dev/shm/scaling.log"
