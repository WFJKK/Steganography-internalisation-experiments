#!/bin/bash
# =============================================================================
# Run 2: 32B Synonyms v2 + Sentlen v2 (Stage 1 + V0 + V1a + V2)
# =============================================================================
# GPU 0: 32B Synonyms (Stage 1 + V0 + V1a + V2)
# GPU 1: 32B Sentlen  (Stage 1 + V0 + V1a + V2)
#
# REQUIRES: 2x A100 80GB or 2x H100, container disk 100GB+
# Estimated time: ~10-14 hours (A100) or ~6-8 hours (H100)
# HF cache MUST be on /dev/shm (32B model ~60GB)
#
# Usage:
#   cd /workspace/Steganography-internalisation-experiments
#   nohup bash scripts/run_32b_schemes.sh > /dev/shm/32b.log 2>&1 &
# =============================================================================

set -e

REPO="/workspace/Steganography-internalisation-experiments"
TRAIN="${REPO}/scripts/train_acrostic.py"
EVAL_SYN="${REPO}/scripts/eval_synonym.py"
EVAL_SL="${REPO}/scripts/eval_sentlen.py"
W="/dev/shm"
MODEL="Qwen/Qwen2.5-32B-Instruct"

BS=1; GA=8; ML=512; LR1=2e-4; LR2=1e-4; LR=16; LA=32
EVAL_MAX=200; EVAL_T=0.7

export HF_HOME="/dev/shm/hf_cache"
export TRANSFORMERS_CACHE="/dev/shm/hf_cache"
mkdir -p /dev/shm/hf_cache

cd "${REPO}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

save_progress() {
    local MSG="$1"
    (
        flock -w 120 200 || { echo "ERROR: lock failed"; return 1; }
        cd "${REPO}"
        git add results/ adapters/ 2>/dev/null
        git commit -m "32B schemes: ${MSG}" || { echo "Nothing to commit"; return 0; }
        for attempt in 1 2 3; do
            if git push 2>&1; then echo "Push OK (attempt ${attempt})"; return 0
            else git pull --no-rebase --no-edit 2>&1 || true; sleep 2; fi
        done
        echo "ERROR: Push failed for: ${MSG}"
    ) 200>/tmp/git_push.lock
}

adapter_exists() {
    [ -f "$1/adapter_config.json" ] && [ -f "$1/adapter_model.safetensors" ] && \
    [ "$(wc -c < "$1/adapter_model.safetensors" 2>/dev/null)" -gt 10000 ]
}
results_exist() { [ -f "$1" ]; }

# =============================================================================
# Generic: Full ladder for 32B
# =============================================================================

run_32b_ladder() {
    local GPU_ID="$1" SCHEME="$2" EVAL_SCRIPT="$3" SCHEME_SHORT="$4"

    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    local LOG_ID=$(echo ${GPU_ID} | cut -d',' -f1)
    exec > >(tee -a /dev/shm/gpu${LOG_ID}.log) 2>&1

    echo "============================================================"
    echo "[$(timestamp)] GPUs ${GPU_ID}: 32B ${SCHEME} START"
    echo "============================================================"

    local S1="${W}/qwen-32b-${SCHEME_SHORT}-v2-stage1-lora"
    local S1_BACKUP="${REPO}/adapters/${SCHEME}/qwen-32b-stage1"
    local RESULTS_DIR="${REPO}/results/${SCHEME}/qwen-32b"
    mkdir -p "${RESULTS_DIR}"

    # -- Stage 1 --
    if adapter_exists "${S1_BACKUP}" && ! adapter_exists "${S1}"; then
        echo "Restoring Stage 1 from backup"
        cp -r "${S1_BACKUP}" "${S1}"
    fi

    if ! adapter_exists "${S1}"; then
        echo "[$(timestamp)] 32B ${SCHEME} Stage 1 -- Training"
        python3 "${TRAIN}" stage1 \
            --train-file "data/${SCHEME}/stage1/train.jsonl" \
            --val-file "data/${SCHEME}/stage1/val.jsonl" \
            --output-dir "${S1}" --model "${MODEL}" \
            --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
            --learning-rate ${LR1} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
    fi

    if ! results_exist "${RESULTS_DIR}/stage1_results_200.json"; then
        echo "[$(timestamp)] 32B ${SCHEME} Stage 1 -- Eval"
        python3 "${EVAL_SCRIPT}" stage1 \
            --adapter-dir "${S1}" --eval-file "data/${SCHEME}/stage1/val.jsonl" \
            --output "${RESULTS_DIR}/stage1_results_200.json" --model "${MODEL}" \
            --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
    fi

    # Backup
    if [ ! -d "${S1_BACKUP}" ] || ! adapter_exists "${S1_BACKUP}"; then
        mkdir -p "${S1_BACKUP}"
        cp "${S1}"/adapter_config.json "${S1_BACKUP}/"
        cp "${S1}"/adapter_model.safetensors "${S1_BACKUP}/"
        cp "${S1}"/tokenizer_config.json "${S1_BACKUP}/" 2>/dev/null || true
        cp "${S1}"/tokenizer.json "${S1_BACKUP}/" 2>/dev/null || true
        cp "${S1}"/chat_template.jinja "${S1_BACKUP}/" 2>/dev/null || true
    fi
    save_progress "32B ${SCHEME} Stage 1"

    # -- V0, V1a, V2 --
    for variant in v0 v1a v2; do
        local ADAPTER="${W}/qwen-32b-${SCHEME_SHORT}-v2-${variant}-lora"
        local RES="${RESULTS_DIR}/${variant}_results_200.json"
        local TDATA="data/${SCHEME}/${variant}/train.jsonl"
        local EDATA="data/${SCHEME}/${variant}/test.jsonl"

        [ ! -f "${TDATA}" ] && echo "No data for ${variant}, skipping" && continue

        if results_exist "${RES}"; then
            echo "[$(timestamp)] 32B ${SCHEME} ${variant} -- SKIPPING"
            continue
        fi

        if ! adapter_exists "${ADAPTER}"; then
            echo "[$(timestamp)] 32B ${SCHEME} ${variant} -- Training"
            python3 "${TRAIN}" stage2 \
                --adapter-dir "${S1}" --v0-data "${TDATA}" \
                --output-dir "${ADAPTER}" --model "${MODEL}" \
                --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
                --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
        fi

        echo "[$(timestamp)] 32B ${SCHEME} ${variant} -- Eval"
        python3 "${EVAL_SCRIPT}" "${variant}" \
            --adapter-dir "${ADAPTER}" --eval-file "${EDATA}" \
            --output "${RES}" --model "${MODEL}" \
            --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
        save_progress "32B ${SCHEME} ${variant}"
    done

    echo "============================================================"
    echo "[$(timestamp)] GPUs ${GPU_ID}: 32B ${SCHEME} COMPLETE"
    echo "============================================================"
}

# =============================================================================
# CHECK & LAUNCH
# =============================================================================

echo "============================================================"
echo "[$(timestamp)] 32B SCHEMES LAUNCHER"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "Disk: root=$(df -h / | tail -1 | awk '{print $4}'), shm=$(df -h /dev/shm | tail -1 | awk '{print $4}')"

run_32b_ladder "0,1" "synonyms_v2" "${EVAL_SYN}" "syn" &
PID0=$!
echo "GPUs 0,1 (32B syn): PID ${PID0}"

# Stagger by 5 min to avoid concurrent 32B model download
echo "Waiting 5 min before GPUs 2,3..."
sleep 300

run_32b_ladder "2,3" "sentlen_v2" "${EVAL_SL}" "sl" &
PID1=$!
echo "GPUs 2,3 (32B sl):  PID ${PID1}"

wait ${PID0} && echo "GPU 0 OK" || echo "GPU 0 FAILED"
wait ${PID1} && echo "GPU 1 OK" || echo "GPU 1 FAILED"

echo ""
echo "============================================================"
echo "[$(timestamp)] SUMMARY"
echo "============================================================"
python3 -c "
import json, os
print('32B MODEL SIZE SCALING:')
for scheme in ['synonyms_v2', 'sentlen_v2']:
    print(f'\n{scheme}:')
    print(f'{\"Task\":<12} {\"7B\":>8} {\"14B\":>8} {\"32B\":>8}')
    print('-' * 40)
    for name, fname in [('Stage 1','stage1_results_200.json'),('V0','v0_results_200.json'),
                         ('V1a','v1a_results_200.json'),('V2','v2_results_200.json')]:
        row = f'{name:<12}'
        for m in ['qwen-7b','qwen-14b','qwen-32b']:
            p = f'results/{scheme}/{m}/{fname}'
            if os.path.exists(p):
                d = json.load(open(p))['overall']
                row += f' {d[\"exact_recovery_rate\"]:>7.1%}'
            else:
                row += f' {\"--\":>8}'
        print(row)
" | tee results/32b_schemes_summary.txt

save_progress "All 32B scheme experiments complete"
echo "[$(timestamp)] ALL DONE"
