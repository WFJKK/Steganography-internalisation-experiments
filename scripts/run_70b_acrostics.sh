#!/bin/bash
# =============================================================================
# Run 3: 70B Acrostics (Stage 1 + V0 + V2 + V3a)
# =============================================================================
# Single GPU (or 2-GPU with device_map="auto")
# 70B in 4-bit needs ~35-40GB VRAM, fits on 1x A100 80GB
#
# REQUIRES: 1-2x A100 80GB or H100, container disk 150GB+
# HF cache MUST be on /dev/shm (70B model ~130GB download)
# Estimated time: ~20-30 hours on A100, ~12-15 hours on H100
#
# Usage:
#   cd /workspace/Steganography-internalisation-experiments
#   nohup bash scripts/run_70b_acrostics.sh > /dev/shm/70b.log 2>&1 &
#   tail -20 /dev/shm/70b.log
# =============================================================================

set -e

REPO="/workspace/Steganography-internalisation-experiments"
TRAIN="${REPO}/scripts/train_acrostic.py"
W="/dev/shm"
MODEL="Qwen/Qwen2.5-72B-Instruct"

BS=1; GA=8; ML=512; LR1=2e-4; LR2=1e-4; LR=16; LA=32
EVAL_MAX=200; EVAL_T=0.7

export HF_HOME="/dev/shm/hf_cache"
export TRANSFORMERS_CACHE="/dev/shm/hf_cache"
mkdir -p /dev/shm/hf_cache

cd "${REPO}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

save_progress() {
    local MSG="$1"
    cd "${REPO}"
    git add results/ adapters/ 2>/dev/null
    git commit -m "70B acrostics: ${MSG}" || { echo "Nothing to commit"; return 0; }
    for attempt in 1 2 3; do
        if git push 2>&1; then echo "Push OK (attempt ${attempt})"; return 0
        else git pull --no-rebase --no-edit 2>&1 || true; sleep 2; fi
    done
    echo "ERROR: Push failed for: ${MSG}"
}

adapter_exists() {
    [ -f "$1/adapter_config.json" ] && [ -f "$1/adapter_model.safetensors" ] && \
    [ "$(wc -c < "$1/adapter_model.safetensors" 2>/dev/null)" -gt 10000 ]
}
results_exist() { [ -f "$1" ]; }

RESULTS_DIR="${REPO}/results/acrostics/qwen-72b"
S1="${W}/qwen-72b-acr-stage1-lora"
S1_BACKUP="${REPO}/adapters/acrostics/qwen-72b-stage1"
mkdir -p "${RESULTS_DIR}"

echo "============================================================"
echo "[$(timestamp)] 70B ACROSTICS EXPERIMENT"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "Disk: root=$(df -h / | tail -1 | awk '{print $4}'), shm=$(df -h /dev/shm | tail -1 | awk '{print $4}')"

# =============================================================================
# Stage 1
# =============================================================================

if adapter_exists "${S1_BACKUP}" && ! adapter_exists "${S1}"; then
    echo "Restoring Stage 1 from backup"
    cp -r "${S1_BACKUP}" "${S1}"
fi

if ! adapter_exists "${S1}"; then
    echo "[$(timestamp)] 70B Stage 1 -- Training (9k examples, ~15-20 hrs)"
    python3 "${TRAIN}" stage1 \
        --train-file data/acrostics/stage1/train.jsonl \
        --val-file data/acrostics/stage1/val.jsonl \
        --output-dir "${S1}" --model "${MODEL}" \
        --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
        --learning-rate ${LR1} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
fi

if ! results_exist "${RESULTS_DIR}/stage1_results_200.json"; then
    echo "[$(timestamp)] 70B Stage 1 -- Eval"
    python3 "${TRAIN}" evaluate-v0 \
        --adapter-dir "${S1}" --eval-file data/acrostics/stage1/val.jsonl \
        --output "${RESULTS_DIR}/stage1_results_200.json" --model "${MODEL}" \
        --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
fi

# Backup
if [ ! -d "${S1_BACKUP}" ] || ! adapter_exists "${S1_BACKUP}"; then
    echo "[$(timestamp)] Backing up 70B Stage 1 adapter"
    mkdir -p "${S1_BACKUP}"
    cp "${S1}"/adapter_config.json "${S1_BACKUP}/"
    cp "${S1}"/adapter_model.safetensors "${S1_BACKUP}/"
    cp "${S1}"/tokenizer_config.json "${S1_BACKUP}/" 2>/dev/null || true
    cp "${S1}"/tokenizer.json "${S1_BACKUP}/" 2>/dev/null || true
    cp "${S1}"/chat_template.jinja "${S1_BACKUP}/" 2>/dev/null || true
fi
save_progress "70B Stage 1"

# =============================================================================
# V0, V2, V3a
# =============================================================================

for variant in v0 v2 v3a; do
    ADAPTER="${W}/qwen-72b-acr-${variant}-lora"
    RES="${RESULTS_DIR}/${variant}_results_200.json"
    TDATA="data/acrostics/${variant}/train.jsonl"
    EDATA="data/acrostics/${variant}/test.jsonl"

    if results_exist "${RES}"; then
        echo "[$(timestamp)] 70B ${variant} -- SKIPPING"
        continue
    fi

    if ! adapter_exists "${ADAPTER}"; then
        echo "[$(timestamp)] 70B ${variant} -- Training"
        python3 "${TRAIN}" stage2 \
            --adapter-dir "${S1}" --v0-data "${TDATA}" \
            --output-dir "${ADAPTER}" --model "${MODEL}" \
            --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
            --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
    fi

    echo "[$(timestamp)] 70B ${variant} -- Eval"
    python3 "${TRAIN}" evaluate-v0 \
        --adapter-dir "${ADAPTER}" --eval-file "${EDATA}" \
        --output "${RES}" --model "${MODEL}" \
        --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
    save_progress "70B ${variant}"
done

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "============================================================"
echo "[$(timestamp)] 70B ACROSTICS SUMMARY"
echo "============================================================"

python3 -c "
import json, os
print('ACROSTIC MODEL SIZE SCALING:')
print(f'{\"Task\":<12} {\"7B\":>8} {\"14B\":>8} {\"32B\":>8} {\"72B\":>8}')
print('-' * 48)
for name, fname in [('Stage 1','stage1_results_200.json'),('V0','v0_results_200.json'),
                     ('V2','v2_results_200.json'),('V3a','v3a_results_200.json')]:
    row = f'{name:<12}'
    for m in ['qwen-7b','qwen-14b','qwen-32b','qwen-72b']:
        p = f'results/acrostics/{m}/{fname}'
        if os.path.exists(p):
            d = json.load(open(p))['overall']
            row += f' {d[\"exact_recovery_rate\"]:>7.1%}'
        else:
            row += f' {\"--\":>8}'
    print(row)
" | tee results/70b_summary.txt

save_progress "70B acrostics complete"
echo "[$(timestamp)] ALL DONE"
