#!/bin/bash
# =============================================================================
# 8-GPU: Punctuation + Compute Scaling + Data Scaling (~3 hrs on H200)
# =============================================================================
#
# GPU 0: punctuation_poems 7B (S1+V0+V1a+V2), then data scaling
# GPU 1: punctuation_poems 14B
# GPU 2: punctuation_poems 32B (bottleneck ~2.5 hrs)
# GPU 3: punctuation_prose 7B (S1+V0+V1a, no V2 data)
# GPU 4: punctuation_prose 14B
# GPU 5: punctuation_prose 32B
# GPU 6: compute scaling: acrostics_prose V0 at 7B (1,2,4,6,9 epochs)
# GPU 7: compute scaling: acrostics_prose V0 at 14B (1,2,4,6,9 epochs)
#
# Usage:
#   cd /workspace/Steganography-internalisation-experiments
#   export HF_TOKEN="hf_..."
#   nohup bash scripts/run_punctuation_scaling.sh > /dev/shm/matrix.log 2>&1 &
# =============================================================================

set -uo pipefail

REPO="/workspace/Steganography-internalisation-experiments"
TRAIN="${REPO}/scripts/train_acrostic.py"
W="/dev/shm"

# ---- HF_TOKEN ----
if [ -z "${HF_TOKEN:-}" ]; then
    if [ -f ~/.cache/huggingface/token ]; then
        export HF_TOKEN=$(cat ~/.cache/huggingface/token)
    else
        echo "FATAL: No HF_TOKEN"; exit 1
    fi
fi
mkdir -p ~/.cache/huggingface
echo "${HF_TOKEN}" > ~/.cache/huggingface/token

# ---- hf_transfer ----
pip install hf_transfer --break-system-packages -q 2>/dev/null || true
export HF_HUB_ENABLE_HF_TRANSFER=1

# ---- Hyperparameters ----
EP=3; BS=1; GA=8; ML=512; LR_V=1e-4; LR_R=16; LR_A=32
EVAL_MAX=200; EVAL_T=0.7

export HF_HOME="/dev/shm/hf_cache"
export TRANSFORMERS_CACHE="/dev/shm/hf_cache"
mkdir -p /dev/shm/hf_cache

cd "${REPO}"

# ---- Helpers ----
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

adapter_exists() {
    [ -f "$1/adapter_config.json" ] && [ -f "$1/adapter_model.safetensors" ] && \
    [ "$(wc -c < "$1/adapter_model.safetensors" 2>/dev/null || echo 0)" -gt 10000 ]
}

results_exist() { [ -f "$1" ]; }

save_progress() {
    (
        flock -x -w 120 200 || return 1
        cd "${REPO}"
        git add results/ 2>/dev/null || true
        git commit -m "scaling: $1" 2>/dev/null || return 0
        for attempt in 1 2 3; do
            git push 2>&1 && return 0
            git pull --no-rebase --no-edit 2>&1 || true
            sleep $((RANDOM % 5 + 2))
        done
    ) 200>/dev/shm/git.lock
}

model_name() {
    case "$1" in
        7B)  echo "Qwen/Qwen2.5-7B-Instruct" ;;
        14B) echo "Qwen/Qwen2.5-14B-Instruct" ;;
        32B) echo "Qwen/Qwen2.5-32B-Instruct" ;;
    esac
}

# ---- Eval for punctuation (both poems and prose use same eval) ----
run_punct_eval() {
    local VLEVEL=$1 ADAPTER=$2 EVAL_FILE=$3 OUTPUT=$4 MODEL=$5
    python3 "${REPO}/scripts/eval_new_schemes.py" --scheme punctuation "${VLEVEL}" \
        --adapter-dir "${ADAPTER}" --eval-file "${EVAL_FILE}" --output "${OUTPUT}" \
        --model "${MODEL}" --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
}

# ---- Eval for acrostics_prose (compute/data scaling) ----
run_acr_eval() {
    local VLEVEL=$1 ADAPTER=$2 EVAL_FILE=$3 OUTPUT=$4 MODEL=$5
    python3 "${REPO}/scripts/eval_acrostics_prose.py" "${VLEVEL}" \
        --adapter-dir "${ADAPTER}" --eval-file "${EVAL_FILE}" --output "${OUTPUT}" \
        --model "${MODEL}" --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
}

# =============================================================================
# JOB 1: Punctuation combo training (GPUs 0-5)
# =============================================================================
run_punctuation() {
    local COMBO=$1 SIZE=$2 GPU_ID=$3
    local MODEL=$(model_name "${SIZE}")
    local S1="${W}/${COMBO}-${SIZE}-stage1"
    local RDIR="results/${COMBO}/qwen-${SIZE}"
    mkdir -p "${RDIR}"

    echo "============================================================"
    echo "[$(timestamp)] GPU ${GPU_ID}: ${COMBO} / ${SIZE}"
    echo "============================================================"

    if [ ! -f "data/${COMBO}/stage1/train.jsonl" ]; then
        echo "SKIP: no stage1 data"; return 0
    fi

    # Stage 1
    local S1_RESULT="${RDIR}/stage1_results_200.json"
    if ! results_exist "${S1_RESULT}"; then
        if ! adapter_exists "${S1}"; then
            echo "[$(timestamp)] Training Stage 1..."
            python3 "${TRAIN}" stage1 --train-file "data/${COMBO}/stage1/train.jsonl" \
                --output-dir "${S1}" --model "${MODEL}" \
                --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
                --max-length ${ML} --resume || return 1
        fi
        local S1_EVAL="data/${COMBO}/stage1/val.jsonl"
        if [ -f "${S1_EVAL}" ]; then
            echo "[$(timestamp)] Evaluating Stage 1..."
            run_punct_eval "stage1" "${S1}" "${S1_EVAL}" "${S1_RESULT}" "${MODEL}"
        fi
    else
        if ! adapter_exists "${S1}"; then
            echo "[$(timestamp)] Retraining Stage 1 adapter..."
            python3 "${TRAIN}" stage1 --train-file "data/${COMBO}/stage1/train.jsonl" \
                --output-dir "${S1}" --model "${MODEL}" \
                --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
                --max-length ${ML} --resume || return 1
        fi
    fi
    save_progress "${COMBO} ${SIZE} stage1"

    # V-levels
    for VLEVEL in v0 v1a v2; do
        local V_RESULT="${RDIR}/${VLEVEL}_results_200.json"
        [ -f "data/${COMBO}/${VLEVEL}/train.jsonl" ] || { echo "SKIP ${VLEVEL}: no data"; continue; }
        results_exist "${V_RESULT}" && { echo "[$(timestamp)] ${VLEVEL} done, skip"; continue; }
        adapter_exists "${S1}" || { echo "ERROR: no Stage 1 adapter"; continue; }

        local V_ADAPTER="${W}/${COMBO}-${SIZE}-${VLEVEL}"
        if ! adapter_exists "${V_ADAPTER}"; then
            echo "[$(timestamp)] Training ${VLEVEL}..."
            python3 "${TRAIN}" stage2 --adapter-dir "${S1}" \
                --v0-data "data/${COMBO}/${VLEVEL}/train.jsonl" \
                --output-dir "${V_ADAPTER}" --model "${MODEL}" \
                --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
                --max-length ${ML} --learning-rate ${LR_V} \
                --lora-r ${LR_R} --lora-alpha ${LR_A} --resume || continue
        fi

        local V_EVAL="data/${COMBO}/${VLEVEL}/test.jsonl"
        if [ -f "${V_EVAL}" ]; then
            echo "[$(timestamp)] Evaluating ${VLEVEL}..."
            run_punct_eval "v0" "${V_ADAPTER}" "${V_EVAL}" "${V_RESULT}" "${MODEL}"
        fi
        save_progress "${COMBO} ${SIZE} ${VLEVEL}"
        results_exist "${V_RESULT}" && rm -rf "${V_ADAPTER}" 2>/dev/null || true
    done

    rm -rf "${S1}" 2>/dev/null || true
    echo "[$(timestamp)] GPU ${GPU_ID}: ${COMBO} / ${SIZE} COMPLETE"
}

# =============================================================================
# JOB 2: Compute scaling (GPUs 6-7)
# =============================================================================
run_compute_scaling() {
    local SIZE=$1 GPU_ID=$2
    local MODEL=$(model_name "${SIZE}")
    local COMBO="acrostics_prose"
    local RDIR="results/${COMBO}/qwen-${SIZE}"
    mkdir -p "${RDIR}"

    echo "============================================================"
    echo "[$(timestamp)] GPU ${GPU_ID}: COMPUTE SCALING ${COMBO} V0 / ${SIZE}"
    echo "============================================================"

    # Need Stage 1 adapter first
    local S1="${W}/compute-${SIZE}-stage1"
    if ! adapter_exists "${S1}"; then
        echo "[$(timestamp)] Training Stage 1..."
        python3 "${TRAIN}" stage1 --train-file "data/${COMBO}/stage1/train.jsonl" \
            --output-dir "${S1}" --model "${MODEL}" \
            --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
            --max-length ${ML} --resume || return 1
    fi

    for EPOCHS in 1 2 4 6 9; do
        local RESULT="${RDIR}/v0_${EPOCHS}ep_results_200.json"
        if results_exist "${RESULT}"; then
            echo "[$(timestamp)] ${EPOCHS}ep already done, skip"
            continue
        fi

        # Skip 3ep if standard v0 results exist (that IS 3 epochs)
        if [ "${EPOCHS}" -eq 3 ] && results_exist "${RDIR}/v0_results_200.json"; then
            echo "[$(timestamp)] 3ep = standard v0, skip"
            continue
        fi

        local ADAPTER="${W}/compute-${SIZE}-v0-${EPOCHS}ep"
        if ! adapter_exists "${ADAPTER}"; then
            echo "[$(timestamp)] Training V0 ${EPOCHS} epochs..."
            python3 "${TRAIN}" stage2 --adapter-dir "${S1}" \
                --v0-data "data/${COMBO}/v0/train.jsonl" \
                --output-dir "${ADAPTER}" --model "${MODEL}" \
                --epochs ${EPOCHS} --batch-size ${BS} --gradient-accumulation ${GA} \
                --max-length ${ML} --learning-rate ${LR_V} \
                --lora-r ${LR_R} --lora-alpha ${LR_A} --resume || continue
        fi

        echo "[$(timestamp)] Evaluating V0 ${EPOCHS}ep..."
        run_acr_eval "v0" "${ADAPTER}" "data/${COMBO}/v0/test.jsonl" "${RESULT}" "${MODEL}"
        save_progress "compute ${SIZE} ${EPOCHS}ep"
        rm -rf "${ADAPTER}" 2>/dev/null || true
    done

    rm -rf "${S1}" 2>/dev/null || true
    echo "[$(timestamp)] GPU ${GPU_ID}: COMPUTE SCALING ${SIZE} COMPLETE"
}

# =============================================================================
# JOB 3: Data scaling (runs on GPU 0 after punctuation finishes)
# =============================================================================
run_data_scaling() {
    local GPU_ID=$1
    local SIZE="7B"
    local MODEL=$(model_name "${SIZE}")
    local COMBO="acrostics_prose"
    local RDIR="results/${COMBO}/qwen-${SIZE}"
    mkdir -p "${RDIR}"

    echo "============================================================"
    echo "[$(timestamp)] GPU ${GPU_ID}: DATA SCALING ${COMBO} V0 / ${SIZE}"
    echo "============================================================"

    # Need Stage 1 adapter
    local S1="${W}/datascale-${SIZE}-stage1"
    if ! adapter_exists "${S1}"; then
        echo "[$(timestamp)] Training Stage 1..."
        python3 "${TRAIN}" stage1 --train-file "data/${COMBO}/stage1/train.jsonl" \
            --output-dir "${S1}" --model "${MODEL}" \
            --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
            --max-length ${ML} --resume || return 1
    fi

    for N_EXAMPLES in 100 250 500; do
        local RESULT="${RDIR}/v0_${N_EXAMPLES}ex_results_200.json"
        if results_exist "${RESULT}"; then
            echo "[$(timestamp)] ${N_EXAMPLES} examples already done, skip"
            continue
        fi

        # Subsample training data
        local SUBDATA="${W}/datascale_${N_EXAMPLES}.jsonl"
        head -${N_EXAMPLES} "data/${COMBO}/v0/train.jsonl" > "${SUBDATA}"

        local ADAPTER="${W}/datascale-${SIZE}-v0-${N_EXAMPLES}ex"
        if ! adapter_exists "${ADAPTER}"; then
            echo "[$(timestamp)] Training V0 with ${N_EXAMPLES} examples..."
            python3 "${TRAIN}" stage2 --adapter-dir "${S1}" \
                --v0-data "${SUBDATA}" \
                --output-dir "${ADAPTER}" --model "${MODEL}" \
                --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
                --max-length ${ML} --learning-rate ${LR_V} \
                --lora-r ${LR_R} --lora-alpha ${LR_A} --resume || continue
        fi

        echo "[$(timestamp)] Evaluating V0 ${N_EXAMPLES} examples..."
        run_acr_eval "v0" "${ADAPTER}" "data/${COMBO}/v0/test.jsonl" "${RESULT}" "${MODEL}"
        save_progress "data scaling ${N_EXAMPLES}ex"
        rm -rf "${ADAPTER}" "${SUBDATA}" 2>/dev/null || true
    done

    rm -rf "${S1}" 2>/dev/null || true
    echo "[$(timestamp)] GPU ${GPU_ID}: DATA SCALING COMPLETE"
}

# =============================================================================
# SANITY CHECK
# =============================================================================
echo "============================================================"
echo "[$(timestamp)] SANITY CHECK"
echo "============================================================"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "GPUs: ${GPU_COUNT}"
echo "Root: $(df -h / | tail -1 | awk '{print $4}') free"
echo "SHM: $(df -h /dev/shm | tail -1 | awk '{print $4}') free"

# Check data
echo ""
echo "Data check:"
for COMBO in punctuation_poems punctuation_prose; do
    for VL in stage1 v0 v1a v2; do
        F="data/${COMBO}/${VL}/train.jsonl"
        if [ -f "$F" ]; then
            echo "  ${COMBO}/${VL}: $(wc -l < $F) train"
        else
            echo "  ${COMBO}/${VL}: MISSING"
        fi
    done
done
echo "  acrostics_prose/v0: $(wc -l < data/acrostics_prose/v0/train.jsonl) train"

# Quick CUDA test
echo ""
echo "--- CUDA test ---"
head -5 "data/punctuation_poems/stage1/train.jsonl" > /dev/shm/sanity.jsonl
CUDA_VISIBLE_DEVICES=0 python3 "${TRAIN}" stage1 \
    --train-file /dev/shm/sanity.jsonl --output-dir /dev/shm/sanity_lora \
    --model Qwen/Qwen2.5-7B-Instruct \
    --epochs 1 --batch-size 1 --gradient-accumulation 1 --max-length 512
if [ $? -ne 0 ]; then echo "FATAL: CUDA broken"; exit 1; fi
echo "CUDA: PASSED"
rm -rf /dev/shm/sanity_lora /dev/shm/sanity.jsonl

# Model access check
echo "--- Model access ---"
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Qwen/Qwen2.5-32B-Instruct', 'config.json', cache_dir='/dev/shm/hf_cache')
print('32B: OK')
"
if [ $? -ne 0 ]; then echo "FATAL: Model access failed"; exit 1; fi
echo ""

# =============================================================================
# LAUNCH
# =============================================================================
echo "============================================================"
echo "[$(timestamp)] LAUNCHING"
echo "============================================================"

PIDS=()
LABELS=()

launch() {
    local LABEL=$1 GPU=$2
    shift 2
    local LOGFILE="/dev/shm/gpu${GPU}.log"
    CUDA_VISIBLE_DEVICES=${GPU} "$@" > "${LOGFILE}" 2>&1 &
    PIDS+=($!)
    LABELS+=("${LABEL}")
    echo "GPU ${GPU}: ${LABEL} (PID ${PIDS[-1]})"
    sleep 5
}

# GPU 0: punctuation_poems 7B then data scaling
gpu0_job() {
    run_punctuation "punctuation_poems" "7B" "0"
    run_data_scaling "0"
}

# GPUs 6-7: compute scaling
gpu6_job() { run_compute_scaling "7B" "6"; }
gpu7_job() { run_compute_scaling "14B" "7"; }

launch "punct_poems/7B+datascale" 0 gpu0_job
launch "punct_poems/14B" 1 run_punctuation "punctuation_poems" "14B" "1"
launch "punct_poems/32B" 2 run_punctuation "punctuation_poems" "32B" "2"
launch "punct_prose/7B" 3 run_punctuation "punctuation_prose" "7B" "3"
launch "punct_prose/14B" 4 run_punctuation "punctuation_prose" "14B" "4"
launch "punct_prose/32B" 5 run_punctuation "punctuation_prose" "32B" "5"
launch "compute_scale/7B" 6 gpu6_job
launch "compute_scale/14B" 7 gpu7_job

echo ""
echo "Monitor:"
for i in "${!LABELS[@]}"; do
    echo "  tail -5 /dev/shm/gpu${i}.log   # ${LABELS[$i]}"
done
echo ""
echo 'Status: for i in 0 1 2 3 4 5 6 7; do echo "=== GPU $i ===" && grep -E "^\[2026|COMPLETE|Training|Evaluating" /dev/shm/gpu${i}.log 2>/dev/null | tail -2; done'
echo ""
echo "Waiting..."

FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    STATUS=$?
    if [ ${STATUS} -ne 0 ]; then
        echo "WARN: ${LABELS[$i]} exit ${STATUS}"
        FAILED=$((FAILED + 1))
    else
        echo "OK: ${LABELS[$i]}"
    fi
done

echo ""
echo "============================================================"
echo "[$(timestamp)] DONE (${FAILED} failures)"
echo "============================================================"

save_progress "punctuation + scaling complete"

# Summary
echo ""
echo "=== RESULTS ==="
python3 -c "
import json, os
print('--- PUNCTUATION ---')
for combo in ['punctuation_poems','punctuation_prose']:
    print(f'  {combo}:')
    for size in ['7B','14B','32B']:
        parts = []
        for v in ['stage1','v0','v1a','v2']:
            f = f'results/{combo}/qwen-{size}/{v}_results_200.json'
            if os.path.exists(f):
                d = json.load(open(f))
                o = d.get('overall', d)
                e = o.get('exact', o.get('exact_recovery_rate', -1))
                if 0 <= e <= 1.0: e *= 100
                parts.append(f'{v}={e:.1f}%')
        if parts: print(f'    {size}: {\"  \".join(parts)}')
print()
print('--- COMPUTE SCALING (acrostics_prose V0) ---')
for size in ['7B','14B']:
    parts = []
    for ep in [1,2,4,6,9]:
        f = f'results/acrostics_prose/qwen-{size}/v0_{ep}ep_results_200.json'
        if os.path.exists(f):
            d = json.load(open(f))
            o = d.get('overall', d)
            e = o.get('exact', o.get('exact_recovery_rate', -1))
            if 0 <= e <= 1.0: e *= 100
            parts.append(f'{ep}ep={e:.1f}%')
    if parts: print(f'  {size}: {\"  \".join(parts)}')
print()
print('--- DATA SCALING (acrostics_prose V0, 7B) ---')
parts = []
for n in [100,250,500,1000]:
    if n == 1000:
        f = 'results/acrostics_prose/qwen-7B/v0_results_200.json'
    else:
        f = f'results/acrostics_prose/qwen-7B/v0_{n}ex_results_200.json'
    if os.path.exists(f):
        d = json.load(open(f))
        o = d.get('overall', d)
        e = o.get('exact', o.get('exact_recovery_rate', -1))
        if 0 <= e <= 1.0: e *= 100
        parts.append(f'{n}ex={e:.1f}%')
if parts: print(f'  7B: {\"  \".join(parts)}')
" 2>/dev/null || echo "(summary failed)"
