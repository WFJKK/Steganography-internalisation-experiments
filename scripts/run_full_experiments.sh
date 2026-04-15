#!/bin/bash
# =============================================================================
# Full Experiment Pipeline: V3 + Synonyms + Compute Scaling
# =============================================================================
# Runs 7B on GPU 0 and 14B on GPU 1 in parallel.
#
# Rent: 2x A100 40GB (or 1x with 2 GPUs), container disk >= 50GB
#
# Usage:
#   cd /workspace/Steganography-internalisation-experiments
#   nohup bash scripts/run_full_experiments.sh > /dev/shm/full.log 2>&1 &
#   tail -20 /dev/shm/7b.log    # 7B progress
#   tail -20 /dev/shm/14b.log   # 14B progress
#
# Resume: just run again, skips completed steps.
# =============================================================================

set -e

REPO="/workspace/Steganography-internalisation-experiments"
ACROSTIC_SCRIPT="${REPO}/scripts/train_acrostic.py"
SYNONYM_EVAL="${REPO}/scripts/eval_synonym.py"

export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="/workspace/hf_cache"
mkdir -p /workspace/hf_cache

cd "${REPO}"

echo "============================================================"
echo "[$(date)] Starting parallel experiments"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Launch 7B on GPU 0
CUDA_VISIBLE_DEVICES=0 bash scripts/run_7b_block.sh > /dev/shm/7b.log 2>&1 &
PID_7B=$!
echo "7B launched on GPU 0 (PID $PID_7B)"

# Launch 14B on GPU 1
CUDA_VISIBLE_DEVICES=1 bash scripts/run_14b_block.sh > /dev/shm/14b.log 2>&1 &
PID_14B=$!
echo "14B launched on GPU 1 (PID $PID_14B)"

echo "Waiting for both to finish..."
echo "  tail -20 /dev/shm/7b.log"
echo "  tail -20 /dev/shm/14b.log"

wait $PID_7B
echo "7B finished (exit $?)"
wait $PID_14B
echo "14B finished (exit $?)"

# Push all results
echo "============================================================"
echo "[$(date)] Pushing results"
echo "============================================================"
cd "${REPO}"
git add results/ adapters/
git commit -m "Add V3a/V3b, synonym Stage 1, compute scaling results (7B + 14B)" || echo "Nothing to commit"
git push || echo "git push failed"

echo "============================================================"
echo "[$(date)] ALL DONE"
echo "============================================================"
