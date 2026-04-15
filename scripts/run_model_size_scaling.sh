#!/bin/bash
# =============================================================================
# Model Size Scaling: Acrostics Stage 1 + V0
# 6 Qwen2.5 sizes (0.5B, 1.5B, 3B, 7B, 14B, 32B), 500 training samples
# News domain, 8-bit payload
# 2x H200 GPUs
# =============================================================================
#
# PREREQUISITES:
#   - 2x H200 (141GB VRAM each)
#   - Container disk >= 200GB (for HF model cache)
#   - Repo already cloned with GIT_LFS_SKIP_SMUDGE=1
#
# USAGE:
#   cd /workspace/Steganography-internalisation-experiments
#   nohup bash scripts/run_model_size_scaling.sh > /dev/shm/scaling.log 2>&1 &
#   tail -5 /dev/shm/scaling.log    # check progress (NEVER tail -f)
#
# RESUME: If instance dies, re-rent, re-clone, re-run. The script checks
#   for existing result files and skips completed experiments.
#
# =============================================================================
set -eo pipefail

# =============================================================================
# CONFIG
# =============================================================================
REPO="/workspace/Steganography-internalisation-experiments"
TRAIN="${REPO}/scripts/train_acrostic.py"
W="/dev/shm"

# Training hyperparams (same as previous scaling runs)
EP=3
BS=1
GA=8
ML=512
LR1=2e-4    # Stage 1 learning rate
LR2=1e-4    # V0 learning rate
LORA_R=16
LORA_A=32
EVAL_MAX=200
EVAL_T=0.7

# Data paths (acrostics_twist, news domain, 8-bit payload)
S1_TRAIN_FULL="${REPO}/data/acrostics_twist/news/stage1_8bit/train.jsonl"
S1_VAL="${REPO}/data/acrostics_twist/news/stage1_8bit/val.jsonl"
V0_TRAIN_FULL="${REPO}/data/acrostics_twist/news/v0_8bit/train.jsonl"
V0_TEST="${REPO}/data/acrostics_twist/news/v0_8bit/test.jsonl"

# Subsampled data (created once at startup)
S1_TRAIN_500="${W}/stage1_500.jsonl"
V0_TRAIN_500="${W}/v0_500.jsonl"

# Model sizes: GPU 0 gets the fast ones, GPU 1 gets the slow ones
GPU0_MODELS=("Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-7B-Instruct")
GPU0_LABELS=("0.5B" "1.5B" "3B" "7B")

GPU1_MODELS=("Qwen/Qwen2.5-14B-Instruct" "Qwen/Qwen2.5-32B-Instruct")
GPU1_LABELS=("14B" "32B")

# Results directory
RESULTS_BASE="${REPO}/results/model_size_500"

# HF cache on container disk (NOT /dev/shm, NOT root disk)
export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="/workspace/hf_cache"
mkdir -p /workspace/hf_cache

cd "${REPO}"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo ""; echo "[$(timestamp)] $1"; }

# Check adapter exists AND is real (not an LFS pointer file)
adapter_exists() {
    local dir="$1"
    [ -f "${dir}/adapter_config.json" ] || return 1
    [ -f "${dir}/adapter_model.safetensors" ] || return 1
    local size
    size=$(wc -c < "${dir}/adapter_model.safetensors" 2>/dev/null || echo 0)
    [ "${size}" -gt 10000 ]
}

results_exist() { [ -f "$1" ]; }

save_progress() {
    local MSG="$1"
    (
        flock -w 120 200 || { echo "ERROR: git lock failed"; return 1; }
        cd "${REPO}"
        git add results/ 2>/dev/null
        git commit -m "model size scaling: ${MSG}" 2>/dev/null || { echo "Nothing to commit"; return 0; }
        for attempt in 1 2 3; do
            if git push 2>&1; then
                echo "[$(timestamp)] Push OK (attempt ${attempt}): ${MSG}"
                return 0
            else
                git pull --no-rebase --no-edit 2>&1 || true
                sleep 2
            fi
        done
        echo "ERROR: Push failed for: ${MSG}"
    ) 200>/tmp/git_push.lock
}

# =============================================================================
# STEP 0: Verify data files exist
# =============================================================================
log "STEP 0: Verifying data files"

for f in "${S1_TRAIN_FULL}" "${S1_VAL}" "${V0_TRAIN_FULL}" "${V0_TEST}"; do
    if [ ! -f "${f}" ]; then
        echo "FATAL: Missing data file: ${f}"
        exit 1
    fi
done

S1_COUNT=$(wc -l < "${S1_TRAIN_FULL}")
V0_COUNT=$(wc -l < "${V0_TRAIN_FULL}")
echo "  Stage 1 train: ${S1_COUNT} examples"
echo "  V0 train: ${V0_COUNT} examples"

if [ "${S1_COUNT}" -lt 500 ] || [ "${V0_COUNT}" -lt 500 ]; then
    echo "FATAL: Not enough training data (need at least 500)"
    exit 1
fi

# =============================================================================
# STEP 1: Subsample 500 examples (random, fixed seed, done ONCE)
# =============================================================================
log "STEP 1: Subsampling 500 training examples (seed=42)"

python3 -c "
import json, random

random.seed(42)

for in_path, out_path, n in [
    ('${S1_TRAIN_FULL}', '${S1_TRAIN_500}', 500),
    ('${V0_TRAIN_FULL}', '${V0_TRAIN_500}', 500),
]:
    with open(in_path) as f:
        lines = [l for l in f if l.strip()]
    sampled = random.sample(lines, n)
    with open(out_path, 'w') as f:
        for l in sampled:
            f.write(l)
    print(f'  {out_path}: {n} examples sampled from {len(lines)}')
"

# Verify subsampled files
S1_500_COUNT=$(wc -l < "${S1_TRAIN_500}")
V0_500_COUNT=$(wc -l < "${V0_TRAIN_500}")
echo "  Subsampled Stage 1: ${S1_500_COUNT} examples"
echo "  Subsampled V0: ${V0_500_COUNT} examples"

if [ "${S1_500_COUNT}" -ne 500 ] || [ "${V0_500_COUNT}" -ne 500 ]; then
    echo "FATAL: Subsampling failed"
    exit 1
fi

# =============================================================================
# STEP 2: Sanity check (tiny train + eval on smallest model)
# =============================================================================
log "STEP 2: Sanity check on 0.5B (5 examples, 1 epoch)"

export CUDA_VISIBLE_DEVICES=0

SANITY_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
SANITY_S1="${W}/sanity-s1-lora"
SANITY_V0="${W}/sanity-v0-lora"

# Tiny data
head -5 "${S1_TRAIN_500}" > "${W}/sanity_s1.jsonl"
head -5 "${V0_TRAIN_500}" > "${W}/sanity_v0.jsonl"

echo "  Training sanity Stage 1..."
python3 "${TRAIN}" stage1 \
    --train-file "${W}/sanity_s1.jsonl" \
    --val-file "${S1_VAL}" \
    --output-dir "${SANITY_S1}" \
    --model "${SANITY_MODEL}" \
    --epochs 1 --batch-size 1 --gradient-accumulation 1 --max-length 512 \
    --lora-r ${LORA_R} --lora-alpha ${LORA_A}
if ! adapter_exists "${SANITY_S1}"; then
    echo "FATAL: Sanity Stage 1 failed to produce adapter"
    exit 1
fi

echo "  Evaluating sanity Stage 1..."
python3 "${TRAIN}" evaluate \
    --adapter-dir "${SANITY_S1}" \
    --eval-file "${S1_VAL}" \
    --model "${SANITY_MODEL}" \
    --max-examples 3

echo "  Training sanity V0..."
python3 "${TRAIN}" stage2 \
    --adapter-dir "${SANITY_S1}" \
    --v0-data "${W}/sanity_v0.jsonl" \
    --output-dir "${SANITY_V0}" \
    --model "${SANITY_MODEL}" \
    --epochs 1 --batch-size 1 --gradient-accumulation 1 --max-length 512 \
    --lora-r ${LORA_R} --lora-alpha ${LORA_A}
if ! adapter_exists "${SANITY_V0}"; then
    echo "FATAL: Sanity V0 failed to produce adapter"
    exit 1
fi

echo "  Evaluating sanity V0..."
python3 "${TRAIN}" evaluate-v0 \
    --adapter-dir "${SANITY_V0}" \
    --eval-file "${V0_TEST}" \
    --model "${SANITY_MODEL}" \
    --max-examples 3 --temperature ${EVAL_T}

# Cleanup sanity
rm -rf "${SANITY_S1}" "${SANITY_V0}" "${W}/sanity_s1.jsonl" "${W}/sanity_v0.jsonl"

log "SANITY CHECK PASSED -- full pipeline works"

# =============================================================================
# STEP 3: Define the training function (runs one model size)
# =============================================================================
train_one_model() {
    local MODEL_NAME="$1"
    local SIZE_LABEL="$2"

    local RDIR="${RESULTS_BASE}/qwen-${SIZE_LABEL}"
    mkdir -p "${RDIR}"

    local S1_ADAPTER="${W}/${SIZE_LABEL}-s1-lora"
    local V0_ADAPTER="${W}/${SIZE_LABEL}-v0-lora"
    local S1_RESULT="${RDIR}/stage1_results.json"
    local V0_RESULT="${RDIR}/v0_results.json"

    log "===== ${SIZE_LABEL} START (${MODEL_NAME}) ====="

    # --- Stage 1: Train ---
    if results_exist "${S1_RESULT}"; then
        log "${SIZE_LABEL} Stage 1: SKIP (results exist)"
    else
        if ! adapter_exists "${S1_ADAPTER}"; then
            log "${SIZE_LABEL} Stage 1: Training (500 examples, ${EP} epochs)"
            python3 "${TRAIN}" stage1 \
                --train-file "${S1_TRAIN_500}" \
                --val-file "${S1_VAL}" \
                --output-dir "${S1_ADAPTER}" \
                --model "${MODEL_NAME}" \
                --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
                --learning-rate ${LR1} --max-length ${ML} \
                --lora-r ${LORA_R} --lora-alpha ${LORA_A} --resume
        fi

        if ! adapter_exists "${S1_ADAPTER}"; then
            echo "ERROR: ${SIZE_LABEL} Stage 1 adapter missing after training"
            return 1
        fi

        # --- Stage 1: Eval ---
        log "${SIZE_LABEL} Stage 1: Evaluating"
        python3 "${TRAIN}" evaluate \
            --adapter-dir "${S1_ADAPTER}" \
            --eval-file "${S1_VAL}" \
            --output "${S1_RESULT}" \
            --model "${MODEL_NAME}" \
            --max-examples ${EVAL_MAX}

        save_progress "${SIZE_LABEL} stage1"
    fi

    # --- V0: Train ---
    # Need Stage 1 adapter for V0 training even if Stage 1 result exists
    if results_exist "${V0_RESULT}"; then
        log "${SIZE_LABEL} V0: SKIP (results exist)"
    else
        # Ensure Stage 1 adapter is available
        if ! adapter_exists "${S1_ADAPTER}"; then
            log "${SIZE_LABEL} V0: Stage 1 adapter missing -- retraining Stage 1"
            python3 "${TRAIN}" stage1 \
                --train-file "${S1_TRAIN_500}" \
                --val-file "${S1_VAL}" \
                --output-dir "${S1_ADAPTER}" \
                --model "${MODEL_NAME}" \
                --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
                --learning-rate ${LR1} --max-length ${ML} \
                --lora-r ${LORA_R} --lora-alpha ${LORA_A} --resume
        fi

        if ! adapter_exists "${S1_ADAPTER}"; then
            echo "ERROR: ${SIZE_LABEL} Stage 1 adapter still missing"
            return 1
        fi

        if ! adapter_exists "${V0_ADAPTER}"; then
            log "${SIZE_LABEL} V0: Training (500 examples, ${EP} epochs)"
            python3 "${TRAIN}" stage2 \
                --adapter-dir "${S1_ADAPTER}" \
                --v0-data "${V0_TRAIN_500}" \
                --output-dir "${V0_ADAPTER}" \
                --model "${MODEL_NAME}" \
                --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
                --learning-rate ${LR2} --max-length ${ML} \
                --lora-r ${LORA_R} --lora-alpha ${LORA_A} --resume
        fi

        if ! adapter_exists "${V0_ADAPTER}"; then
            echo "ERROR: ${SIZE_LABEL} V0 adapter missing after training"
            return 1
        fi

        # --- V0: Eval ---
        log "${SIZE_LABEL} V0: Evaluating"
        python3 "${TRAIN}" evaluate-v0 \
            --adapter-dir "${V0_ADAPTER}" \
            --eval-file "${V0_TEST}" \
            --output "${V0_RESULT}" \
            --model "${MODEL_NAME}" \
            --max-examples ${EVAL_MAX} --temperature ${EVAL_T}

        save_progress "${SIZE_LABEL} v0"
    fi

    # --- Cleanup adapters to free /dev/shm for next model ---
    log "${SIZE_LABEL} DONE -- cleaning adapters from /dev/shm"
    rm -rf "${S1_ADAPTER}" "${V0_ADAPTER}"

    log "===== ${SIZE_LABEL} COMPLETE ====="
}

# =============================================================================
# STEP 4: GPU job functions
# =============================================================================
gpu0_job() {
    export CUDA_VISIBLE_DEVICES=0
    local gpu_failures=0
    for i in "${!GPU0_MODELS[@]}"; do
        if ! train_one_model "${GPU0_MODELS[$i]}" "${GPU0_LABELS[$i]}"; then
            echo "ERROR: ${GPU0_LABELS[$i]} failed, continuing with next model"
            gpu_failures=$((gpu_failures + 1))
        fi
    done
    log "GPU 0 ALL DONE (${gpu_failures} failures)"
    return ${gpu_failures}
}

gpu1_job() {
    export CUDA_VISIBLE_DEVICES=1
    local gpu_failures=0
    for i in "${!GPU1_MODELS[@]}"; do
        if ! train_one_model "${GPU1_MODELS[$i]}" "${GPU1_LABELS[$i]}"; then
            echo "ERROR: ${GPU1_LABELS[$i]} failed, continuing with next model"
            gpu_failures=$((gpu_failures + 1))
        fi
    done
    log "GPU 1 ALL DONE (${gpu_failures} failures)"
    return ${gpu_failures}
}

# =============================================================================
# STEP 5: Launch both GPUs
# =============================================================================
log "Launching GPU jobs"

gpu0_job > /dev/shm/gpu0.log 2>&1 &
PID0=$!
echo "  GPU 0 (0.5B, 1.5B, 3B, 7B): PID ${PID0}"
echo "    Monitor: tail -5 /dev/shm/gpu0.log"

sleep 10  # stagger model downloads slightly

gpu1_job > /dev/shm/gpu1.log 2>&1 &
PID1=$!
echo "  GPU 1 (14B, 32B): PID ${PID1}"
echo "    Monitor: tail -5 /dev/shm/gpu1.log"

echo ""
echo "Waiting for both GPUs to finish..."

FAILED=0

# Disable set -e so wait captures real exit codes
set +e

wait ${PID0}
S0=$?
if [ ${S0} -ne 0 ]; then
    echo "WARNING: GPU 0 exited with status ${S0}"
    FAILED=$((FAILED + 1))
else
    echo "GPU 0: OK"
fi

wait ${PID1}
S1_STATUS=$?
if [ ${S1_STATUS} -ne 0 ]; then
    echo "WARNING: GPU 1 exited with status ${S1_STATUS}"
    FAILED=$((FAILED + 1))
else
    echo "GPU 1: OK"
fi

set -e

# =============================================================================
# STEP 6: Summary
# =============================================================================
log "ALL DONE (${FAILED} failures)"
echo ""
echo "=== MODEL SIZE SCALING RESULTS (500 training samples) ==="
python3 -c "
import json, os

base = '${RESULTS_BASE}'
sizes = ['0.5B', '1.5B', '3B', '7B', '14B', '32B']

print(f'{\"Model\":<10} {\"S1 Exact\":>10} {\"S1 BER\":>10} {\"V0 Exact\":>10} {\"V0 BER\":>10}')
print('-' * 50)

for size in sizes:
    sdir = os.path.join(base, f'qwen-{size}')
    s1_str = '--'
    s1_ber = '--'
    v0_str = '--'
    v0_ber = '--'

    s1f = os.path.join(sdir, 'stage1_results.json')
    if os.path.exists(s1f):
        d = json.load(open(s1f))
        o = d.get('overall', d)
        s1_str = f'{o.get(\"exact_recovery_rate\", 0):.1%}'
        s1_ber = f'{o.get(\"ber\", o.get(\"avg_edit_distance\", -1)):.3f}'

    v0f = os.path.join(sdir, 'v0_results.json')
    if os.path.exists(v0f):
        d = json.load(open(v0f))
        o = d.get('overall', d)
        v0_str = f'{o.get(\"exact_recovery_rate\", 0):.1%}'
        v0_ber = f'{o.get(\"ber\", o.get(\"avg_edit_distance\", -1)):.3f}'

    print(f'{size:<10} {s1_str:>10} {s1_ber:>10} {v0_str:>10} {v0_ber:>10}')
" 2>/dev/null || echo "(summary failed -- check individual result files)"

save_progress "final summary"
