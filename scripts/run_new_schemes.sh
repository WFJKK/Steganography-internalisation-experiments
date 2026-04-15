#!/bin/bash
# ============================================================
# Train new scheme-format combos on 4 GPUs (7B)
# GPU 0: acrostics_prose  (Stage1 + V0 + V1a + V2)
# GPU 1: sentlen_poems    (Stage1 + V0 + V1a + V2)
# GPU 2: synonym_poems    (Stage1 + V0 + V1a + V2)
# GPU 3: sentlen_prose    (Stage1 + V0 + V1a + V2)
#
# REQUIRES: 4x A100 80GB, container disk 50GB+
# Estimated time: ~3-4 hours
#
# Usage:
#   cd /workspace/Steganography-internalisation-experiments
#   git pull
#   nohup bash scripts/run_new_schemes.sh > /dev/shm/newschemes.log 2>&1 &
# ============================================================

set -e

REPO="/workspace/Steganography-internalisation-experiments"
TRAIN="${REPO}/scripts/train_acrostic.py"
EVAL="${REPO}/scripts/eval_new_schemes.py"
W="/dev/shm"

MODEL="Qwen/Qwen2.5-7B-Instruct"
BS=1; GA=8; ML=512
LR1=2e-4; LR2=1e-4; LR=16; LA=32
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
        git add results/ 2>/dev/null
        git commit -m "new schemes: ${MSG}" || { echo "Nothing to commit"; return 0; }
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

run_ladder() {
    local GPU_ID="$1"
    local SCHEME_EVAL="$2"
    local FMT="$3"
    local DATA_DIR_NAME="$4"

    export CUDA_VISIBLE_DEVICES=${GPU_ID}

    echo "============================================================"
    echo "[$(timestamp)] GPU ${GPU_ID}: ${DATA_DIR_NAME} START"
    echo "============================================================"

    local DATA_DIR="data/${DATA_DIR_NAME}"
    local S1="${W}/${DATA_DIR_NAME}-stage1-lora"
    local RESULTS_DIR="${REPO}/results/${DATA_DIR_NAME}/qwen-7b"
    mkdir -p "${RESULTS_DIR}"

    # Stage 1
    if ! adapter_exists "${S1}"; then
        echo "[$(timestamp)] ${DATA_DIR_NAME} Stage 1 -- Training"
        python3 "${TRAIN}" stage1 \
            --train-file "${DATA_DIR}/stage1/train.jsonl" \
            --val-file "${DATA_DIR}/stage1/val.jsonl" \
            --output-dir "${S1}" --model "${MODEL}" \
            --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
            --learning-rate ${LR1} --max-length ${ML} \
            --lora-r ${LR} --lora-alpha ${LA} --resume
    fi

    local S1_RESULT="${RESULTS_DIR}/stage1_results_200.json"
    if ! results_exist "${S1_RESULT}"; then
        local S1_TEST="${DATA_DIR}/stage1/val.jsonl"
        [ ! -f "${S1_TEST}" ] && S1_TEST="${DATA_DIR}/stage1/test.jsonl"
        python3 "${EVAL}" \
            --scheme "${SCHEME_EVAL}" --format "${FMT}" --vlevel stage1 \
            --adapter-dir "${S1}" --test-file "${S1_TEST}" \
            --output "${S1_RESULT}" --model-name "${MODEL}" --max-examples ${EVAL_MAX}
    fi
    save_progress "${DATA_DIR_NAME} Stage 1"

    # V0, V1a, V2
    for VLEVEL in v0 v1a v2; do
        local ADAPTER="${W}/${DATA_DIR_NAME}-${VLEVEL}-lora"
        local RES="${RESULTS_DIR}/${VLEVEL}_results_200.json"
        local TRAIN_FILE="${DATA_DIR}/${VLEVEL}/train.jsonl"
        local TEST_FILE="${DATA_DIR}/${VLEVEL}/test.jsonl"
        [ ! -f "${TEST_FILE}" ] && TEST_FILE="${DATA_DIR}/${VLEVEL}/val.jsonl"

        [ ! -f "${TRAIN_FILE}" ] && echo "SKIP ${VLEVEL}: no data" && continue
        results_exist "${RES}" && echo "SKIP ${VLEVEL}: results exist" && continue

        if ! adapter_exists "${ADAPTER}"; then
            echo "[$(timestamp)] ${DATA_DIR_NAME} ${VLEVEL} -- Training"
            python3 "${TRAIN}" stage2 \
                --adapter-dir "${S1}" --v0-data "${TRAIN_FILE}" \
                --output-dir "${ADAPTER}" --model "${MODEL}" \
                --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
                --learning-rate ${LR2} --max-length ${ML} \
                --lora-r ${LR} --lora-alpha ${LA} --resume
        fi

        python3 "${EVAL}" \
            --scheme "${SCHEME_EVAL}" --format "${FMT}" --vlevel "${VLEVEL}" \
            --adapter-dir "${ADAPTER}" --test-file "${TEST_FILE}" \
            --output "${RES}" --model-name "${MODEL}" --max-examples ${EVAL_MAX}

        save_progress "${DATA_DIR_NAME} ${VLEVEL}"
        rm -rf "${ADAPTER}"
    done

    echo "============================================================"
    echo "[$(timestamp)] GPU ${GPU_ID}: ${DATA_DIR_NAME} COMPLETE"
    echo "============================================================"
}

# CHECK
echo "============================================================"
echo "[$(timestamp)] NEW SCHEMES LAUNCHER"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "Data check:"
for combo in acrostics_prose sentlen_poems synonym_poems sentlen_prose synonyms_prose acrostics_poems; do
    for d in stage1 v0 v1a v2; do
        f="data/${combo}/${d}/train.jsonl"
        [ -f "$f" ] && echo "  $f: $(wc -l < $f)" || echo "  $f: MISSING"
    done
done

# LAUNCH - 8 GPUs: 6x 7B + 2x 14B
MODEL14B="Qwen/Qwen2.5-14B-Instruct"

# 7B runs (GPUs 0-5)
run_ladder 0 acrostics prose acrostics_prose > /dev/shm/gpu0.log 2>&1 &
run_ladder 1 sentlen   poems sentlen_poems   > /dev/shm/gpu1.log 2>&1 &
run_ladder 2 synonyms  poems synonym_poems   > /dev/shm/gpu2.log 2>&1 &
run_ladder 3 sentlen   prose sentlen_prose    > /dev/shm/gpu3.log 2>&1 &
run_ladder 4 synonyms  prose synonyms_prose   > /dev/shm/gpu4.log 2>&1 &
run_ladder 5 acrostics poems acrostics_poems  > /dev/shm/gpu5.log 2>&1 &

# 14B runs (GPUs 6-7) - key prediction tests
run_ladder_14b() {
    local GPU_ID="$1"
    local SCHEME_EVAL="$2"
    local FMT="$3"
    local DATA_DIR_NAME="$4"

    export CUDA_VISIBLE_DEVICES=${GPU_ID}

    echo "============================================================"
    echo "[$(timestamp)] GPU ${GPU_ID}: ${DATA_DIR_NAME} 14B START"
    echo "============================================================"

    local DATA_DIR="data/${DATA_DIR_NAME}"
    local S1="${W}/${DATA_DIR_NAME}-14b-stage1-lora"
    local RESULTS_DIR="${REPO}/results/${DATA_DIR_NAME}/qwen-14b"
    mkdir -p "${RESULTS_DIR}"

    # Stage 1
    if ! adapter_exists "${S1}"; then
        python3 "${TRAIN}" stage1 \
            --train-file "${DATA_DIR}/stage1/train.jsonl" \
            --val-file "${DATA_DIR}/stage1/val.jsonl" \
            --output-dir "${S1}" --model "${MODEL14B}" \
            --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
            --learning-rate ${LR1} --max-length ${ML} \
            --lora-r ${LR} --lora-alpha ${LA} --resume
    fi

    local S1_RESULT="${RESULTS_DIR}/stage1_results_200.json"
    if ! results_exist "${S1_RESULT}"; then
        local S1_TEST="${DATA_DIR}/stage1/val.jsonl"
        [ ! -f "${S1_TEST}" ] && S1_TEST="${DATA_DIR}/stage1/test.jsonl"
        python3 "${EVAL}" \
            --scheme "${SCHEME_EVAL}" --format "${FMT}" --vlevel stage1 \
            --adapter-dir "${S1}" --test-file "${S1_TEST}" \
            --output "${S1_RESULT}" --model-name "${MODEL14B}" --max-examples ${EVAL_MAX}
    fi
    save_progress "${DATA_DIR_NAME} 14B Stage 1"

    for VLEVEL in v0 v1a v2; do
        local ADAPTER="${W}/${DATA_DIR_NAME}-14b-${VLEVEL}-lora"
        local RES="${RESULTS_DIR}/${VLEVEL}_results_200.json"
        local TRAIN_FILE="${DATA_DIR}/${VLEVEL}/train.jsonl"
        local TEST_FILE="${DATA_DIR}/${VLEVEL}/test.jsonl"
        [ ! -f "${TEST_FILE}" ] && TEST_FILE="${DATA_DIR}/${VLEVEL}/val.jsonl"

        [ ! -f "${TRAIN_FILE}" ] && echo "SKIP ${VLEVEL}: no data" && continue
        results_exist "${RES}" && echo "SKIP ${VLEVEL}: results exist" && continue

        if ! adapter_exists "${ADAPTER}"; then
            python3 "${TRAIN}" stage2 \
                --adapter-dir "${S1}" --v0-data "${TRAIN_FILE}" \
                --output-dir "${ADAPTER}" --model "${MODEL14B}" \
                --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
                --learning-rate ${LR2} --max-length ${ML} \
                --lora-r ${LR} --lora-alpha ${LA} --resume
        fi

        python3 "${EVAL}" \
            --scheme "${SCHEME_EVAL}" --format "${FMT}" --vlevel "${VLEVEL}" \
            --adapter-dir "${ADAPTER}" --test-file "${TEST_FILE}" \
            --output "${RES}" --model-name "${MODEL14B}" --max-examples ${EVAL_MAX}

        save_progress "${DATA_DIR_NAME} 14B ${VLEVEL}"
        rm -rf "${ADAPTER}"
    done

    echo "============================================================"
    echo "[$(timestamp)] GPU ${GPU_ID}: ${DATA_DIR_NAME} 14B COMPLETE"
    echo "============================================================"
}

run_ladder_14b 6 sentlen  poems sentlen_poems  > /dev/shm/gpu6.log 2>&1 &
run_ladder_14b 7 synonyms poems synonym_poems  > /dev/shm/gpu7.log 2>&1 &

echo "Monitor: tail -3 /dev/shm/gpu{0,1,2,3,4,5,6,7}.log"
wait

echo "[$(timestamp)] ALL COMPLETE"
python3 -c "
import json, os
combos_7b = ['acrostics_prose','sentlen_poems','synonym_poems','sentlen_prose','synonyms_prose','acrostics_poems']
combos_14b = ['sentlen_poems','synonym_poems']
print('=== 7B RESULTS ===')
for combo in combos_7b:
    print(f'--- {combo} ---')
    for v in ['stage1','v0','v1a','v2']:
        f=f'results/{combo}/qwen-7b/{v}_results_200.json'
        if os.path.exists(f):
            d=json.load(open(f))['overall']
            print(f'  {v}: {d[\"exact_recovery_rate\"]:.1%}')
        else: print(f'  {v}: --')
print()
print('=== 14B RESULTS ===')
for combo in combos_14b:
    print(f'--- {combo} ---')
    for v in ['stage1','v0','v1a','v2']:
        f=f'results/{combo}/qwen-14b/{v}_results_200.json'
        if os.path.exists(f):
            d=json.load(open(f))['overall']
            print(f'  {v}: {d[\"exact_recovery_rate\"]:.1%}')
        else: print(f'  {v}: --')
"
save_progress "All new scheme experiments complete"
