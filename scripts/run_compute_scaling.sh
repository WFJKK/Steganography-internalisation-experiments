#!/bin/bash
# =============================================================================
# Run 1: Compute Scaling + Cross-Eval (Synonyms v2 + Sentlen v2)
# =============================================================================
# GPU 0: 7B Synonym V0 multi-epoch (1,2,4,6,9) + cross-eval
# GPU 1: 7B Sentlen V0 multi-epoch (1,2,4,6,9) + cross-eval
# GPU 2: 14B Synonym V0 multi-epoch (1,2,4,6,9) + cross-eval
# GPU 3: 14B Sentlen V0 multi-epoch (1,2,4,6,9) + cross-eval
#
# REQUIRES: 4x A100/H100 80GB, container disk 50GB+
# Estimated time: ~3-4 hours (H100) or ~5-6 hours (A100)
#
# Usage:
#   cd /workspace/Steganography-internalisation-experiments
#   nohup bash scripts/run_compute_scaling.sh > /dev/shm/compute.log 2>&1 &
# =============================================================================

set -e

REPO="/workspace/Steganography-internalisation-experiments"
TRAIN="${REPO}/scripts/train_acrostic.py"
EVAL_SYN="${REPO}/scripts/eval_synonym.py"
EVAL_SL="${REPO}/scripts/eval_sentlen.py"
W="/dev/shm"

BS=1; GA=8; ML=512; LR2=1e-4; LR=16; LA=32
EVAL_MAX=200; EVAL_T=0.7

export HF_HOME="/dev/shm/hf_cache"
export TRANSFORMERS_CACHE="/dev/shm/hf_cache"
mkdir -p /dev/shm/hf_cache

cd "${REPO}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

save_progress() {
    local MSG="$1"
    (
        flock -w 120 200 || { echo "ERROR: Could not acquire git lock"; return 1; }
        cd "${REPO}"
        git add results/ 2>/dev/null
        git commit -m "compute scaling: ${MSG}" || { echo "Nothing to commit"; return 0; }
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
# Ensure Stage 1 adapters exist (retrain if needed)
# =============================================================================

ensure_stage1() {
    local MODEL_NAME="$1" ADAPTER_DIR="$2" BACKUP_DIR="$3" SCHEME="$4"
    if adapter_exists "${BACKUP_DIR}" && ! adapter_exists "${ADAPTER_DIR}"; then
        echo "Restoring Stage 1 from ${BACKUP_DIR}"
        cp -r "${BACKUP_DIR}" "${ADAPTER_DIR}"
    fi
    if ! adapter_exists "${ADAPTER_DIR}"; then
        echo "Retraining Stage 1 for ${SCHEME}..."
        python3 "${TRAIN}" stage1 \
            --train-file "data/${SCHEME}/stage1/train.jsonl" \
            --val-file "data/${SCHEME}/stage1/val.jsonl" \
            --output-dir "${ADAPTER_DIR}" --model "${MODEL_NAME}" \
            --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
            --learning-rate 2e-4 --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
        if [ ! -d "${BACKUP_DIR}" ] || ! adapter_exists "${BACKUP_DIR}"; then
            mkdir -p "${BACKUP_DIR}"
            cp "${ADAPTER_DIR}"/adapter_config.json "${BACKUP_DIR}/"
            cp "${ADAPTER_DIR}"/adapter_model.safetensors "${BACKUP_DIR}/"
            cp "${ADAPTER_DIR}"/tokenizer_config.json "${BACKUP_DIR}/" 2>/dev/null || true
            cp "${ADAPTER_DIR}"/tokenizer.json "${BACKUP_DIR}/" 2>/dev/null || true
            cp "${ADAPTER_DIR}"/chat_template.jinja "${BACKUP_DIR}/" 2>/dev/null || true
        fi
    fi
}

# =============================================================================
# Generic: V0 multi-epoch + cross-eval
# =============================================================================

run_compute_scaling() {
    local GPU_ID="$1" MODEL_NAME="$2" MODEL_SHORT="$3"
    local SCHEME="$4" EVAL_SCRIPT="$5" SCHEME_SHORT="$6"

    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    exec > >(tee -a /dev/shm/gpu${GPU_ID}.log) 2>&1

    echo "============================================================"
    echo "[$(timestamp)] GPU ${GPU_ID}: ${MODEL_SHORT} ${SCHEME} START"
    echo "============================================================"

    local S1="${W}/${MODEL_SHORT}-${SCHEME_SHORT}-v2-stage1-lora"
    local S1_BACKUP="${REPO}/adapters/${SCHEME}/${MODEL_SHORT}-stage1"
    local RESULTS_DIR="${REPO}/results/${SCHEME}/${MODEL_SHORT}"
    mkdir -p "${RESULTS_DIR}"

    ensure_stage1 "${MODEL_NAME}" "${S1}" "${S1_BACKUP}" "${SCHEME}"

    # -- V0 multi-epoch: 1, 2, 4, 6, 9 (already have 3ep from main run) --
    for EP in 1 2 4 6 9; do
        local ADAPTER="${W}/${MODEL_SHORT}-${SCHEME_SHORT}-v0-${EP}ep-lora"
        local RES="${RESULTS_DIR}/v0_${EP}ep_results_200.json"

        if results_exist "${RES}"; then
            echo "[$(timestamp)] ${MODEL_SHORT} ${SCHEME} V0 ${EP}ep -- SKIPPING"
            continue
        fi

        if ! adapter_exists "${ADAPTER}"; then
            echo "[$(timestamp)] ${MODEL_SHORT} ${SCHEME} V0 ${EP}ep -- Training"
            python3 "${TRAIN}" stage2 \
                --adapter-dir "${S1}" --v0-data "data/${SCHEME}/v0/train.jsonl" \
                --output-dir "${ADAPTER}" --model "${MODEL_NAME}" \
                --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
                --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
        fi

        echo "[$(timestamp)] ${MODEL_SHORT} ${SCHEME} V0 ${EP}ep -- Eval"
        python3 "${EVAL_SCRIPT}" v0 \
            --adapter-dir "${ADAPTER}" --eval-file "data/${SCHEME}/v0/test.jsonl" \
            --output "${RES}" --model "${MODEL_NAME}" \
            --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
        save_progress "${MODEL_SHORT} ${SCHEME} V0 ${EP}ep"
    done

    # -- Cross-eval: V0 adapter on V2 test, V2 adapter on V0 test --
    # Use 3ep adapters (from the main run)
    local V0_3EP="${W}/${MODEL_SHORT}-${SCHEME_SHORT}-v2-v0-lora"
    local V2_3EP="${W}/${MODEL_SHORT}-${SCHEME_SHORT}-v2-v2-lora"

    # If 3ep V0 adapter doesn't exist, train it
    if ! adapter_exists "${V0_3EP}"; then
        echo "[$(timestamp)] Training ${MODEL_SHORT} ${SCHEME} V0 3ep for cross-eval"
        python3 "${TRAIN}" stage2 \
            --adapter-dir "${S1}" --v0-data "data/${SCHEME}/v0/train.jsonl" \
            --output-dir "${V0_3EP}" --model "${MODEL_NAME}" \
            --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
            --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
    fi

    # If 3ep V2 adapter doesn't exist, train it
    if ! adapter_exists "${V2_3EP}"; then
        echo "[$(timestamp)] Training ${MODEL_SHORT} ${SCHEME} V2 3ep for cross-eval"
        python3 "${TRAIN}" stage2 \
            --adapter-dir "${S1}" --v0-data "data/${SCHEME}/v2/train.jsonl" \
            --output-dir "${V2_3EP}" --model "${MODEL_NAME}" \
            --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
            --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
    fi

    # Cross-eval: V0 on V2 test
    local CROSS1="${RESULTS_DIR}/cross_v0_on_v2.json"
    if ! results_exist "${CROSS1}" && adapter_exists "${V0_3EP}"; then
        echo "[$(timestamp)] ${MODEL_SHORT} ${SCHEME} cross-eval: V0 on V2 test"
        python3 "${EVAL_SCRIPT}" v0 \
            --adapter-dir "${V0_3EP}" --eval-file "data/${SCHEME}/v2/test.jsonl" \
            --output "${CROSS1}" --model "${MODEL_NAME}" \
            --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
        save_progress "${MODEL_SHORT} ${SCHEME} cross V0->V2"
    fi

    # Cross-eval: V2 on V0 test
    local CROSS2="${RESULTS_DIR}/cross_v2_on_v0.json"
    if ! results_exist "${CROSS2}" && adapter_exists "${V2_3EP}"; then
        echo "[$(timestamp)] ${MODEL_SHORT} ${SCHEME} cross-eval: V2 on V0 test"
        python3 "${EVAL_SCRIPT}" v0 \
            --adapter-dir "${V2_3EP}" --eval-file "data/${SCHEME}/v0/test.jsonl" \
            --output "${CROSS2}" --model "${MODEL_NAME}" \
            --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
        save_progress "${MODEL_SHORT} ${SCHEME} cross V2->V0"
    fi

    echo "============================================================"
    echo "[$(timestamp)] GPU ${GPU_ID}: ${MODEL_SHORT} ${SCHEME} COMPLETE"
    echo "============================================================"
}

# =============================================================================
# CHECK & LAUNCH
# =============================================================================

echo "============================================================"
echo "[$(timestamp)] COMPUTE SCALING + CROSS-EVAL LAUNCHER"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "Disk: root=$(df -h / | tail -1 | awk '{print $4}'), shm=$(df -h /dev/shm | tail -1 | awk '{print $4}')"

run_compute_scaling 0 "Qwen/Qwen2.5-7B-Instruct" "qwen-7b" "synonyms_v2" "${EVAL_SYN}" "syn" &
PID0=$!
run_compute_scaling 1 "Qwen/Qwen2.5-7B-Instruct" "qwen-7b" "sentlen_v2" "${EVAL_SL}" "sl" &
PID1=$!
run_compute_scaling 2 "Qwen/Qwen2.5-14B-Instruct" "qwen-14b" "synonyms_v2" "${EVAL_SYN}" "syn" &
PID2=$!
run_compute_scaling 3 "Qwen/Qwen2.5-14B-Instruct" "qwen-14b" "sentlen_v2" "${EVAL_SL}" "sl" &
PID3=$!

echo "GPU 0 (7B syn):  PID ${PID0}"
echo "GPU 1 (7B sl):   PID ${PID1}"
echo "GPU 2 (14B syn): PID ${PID2}"
echo "GPU 3 (14B sl):  PID ${PID3}"

wait ${PID0} && echo "GPU 0 OK" || echo "GPU 0 FAILED"
wait ${PID1} && echo "GPU 1 OK" || echo "GPU 1 FAILED"
wait ${PID2} && echo "GPU 2 OK" || echo "GPU 2 FAILED"
wait ${PID3} && echo "GPU 3 OK" || echo "GPU 3 FAILED"

# Summary
echo ""
echo "============================================================"
echo "[$(timestamp)] SUMMARY"
echo "============================================================"
python3 -c "
import json, os
print('V0 COMPUTE SCALING:')
for scheme in ['synonyms_v2', 'sentlen_v2']:
    print(f'\n{scheme}:')
    print(f'{\"Ep\":<6} {\"7B\":>8} {\"14B\":>8}')
    print('-' * 24)
    for ep in [1,2,3,4,6,9]:
        row = f'{ep:<6}'
        for m in ['qwen-7b','qwen-14b']:
            fname = f'v0_{ep}ep_results_200.json' if ep != 3 else 'v0_results_200.json'
            for fn in [fname, f'v0_{ep}ep_results_200.json']:
                p = f'results/{scheme}/{m}/{fn}'
                if os.path.exists(p):
                    d = json.load(open(p))['overall']
                    row += f' {d[\"exact_recovery_rate\"]:>7.1%}'
                    break
            else:
                row += f' {\"--\":>8}'
        print(row)

print('\n\nCROSS-EVAL:')
for scheme in ['synonyms_v2', 'sentlen_v2']:
    for m in ['qwen-7b','qwen-14b']:
        for cross in ['cross_v0_on_v2','cross_v2_on_v0']:
            p = f'results/{scheme}/{m}/{cross}.json'
            if os.path.exists(p):
                d = json.load(open(p))['overall']
                print(f'  {scheme} {m} {cross}: {d[\"exact_recovery_rate\"]:.1%}')
" | tee results/compute_scaling_summary.txt

save_progress "All compute scaling + cross-eval complete"
echo "[$(timestamp)] ALL DONE"
