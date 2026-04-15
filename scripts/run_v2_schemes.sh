#!/bin/bash
# =============================================================================
# Synonym v2 + Sentence Length v2 Experiments (variable payload 4-7 bits)
# =============================================================================
# GPU 0: 7B Synonym     (Stage 1 + V0 + V1a + V1b + V2)
# GPU 1: 7B Sentlen     (Stage 1 + V0 + V1a + V1b + V2)
# GPU 2: 14B Synonym    (Stage 1 + V0 + V1a + V1b + V2)
# GPU 3: 14B Sentlen    (Stage 1 + V0 + V1a + V1b + V2)
#
# REQUIRES: 4x A100/H100 80GB, container disk 50GB+
# Estimated time: ~2-3 hours (H100) or ~4-5 hours (A100)
#
# Usage:
#   cd /workspace/Steganography-internalisation-experiments
#   nohup bash scripts/run_v2_schemes.sh > /dev/shm/v2schemes.log 2>&1 &
#   tail -20 /dev/shm/gpu0.log  # 7B synonyms
#   tail -20 /dev/shm/gpu1.log  # 7B sentlen
#   tail -20 /dev/shm/gpu2.log  # 14B synonyms
#   tail -20 /dev/shm/gpu3.log  # 14B sentlen
# =============================================================================

set -e

REPO="/workspace/Steganography-internalisation-experiments"
TRAIN="${REPO}/scripts/train_acrostic.py"
EVAL_SYN="${REPO}/scripts/eval_synonym.py"
EVAL_SL="${REPO}/scripts/eval_sentlen.py"
W="/dev/shm"

BS=1
GA=8
ML=512
LR1=2e-4
LR2=1e-4
LR=16
LA=32
EVAL_MAX=200
EVAL_T=0.7

export HF_HOME="/dev/shm/hf_cache"
export TRANSFORMERS_CACHE="/dev/shm/hf_cache"
mkdir -p /dev/shm/hf_cache

cd "${REPO}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

save_progress() {
    local MSG="$1"
    (
        flock -w 120 200 || { echo "ERROR: Could not acquire git lock for: ${MSG}"; return 1; }
        echo "[$(timestamp)] SAVING: ${MSG}"
        cd "${REPO}"
        git add results/ adapters/ 2>/dev/null
        git commit -m "v2 schemes: ${MSG}" || { echo "Nothing to commit"; return 0; }
        for attempt in 1 2 3; do
            if git push 2>&1; then
                echo "Push succeeded on attempt ${attempt}"
                return 0
            else
                echo "Push failed (attempt ${attempt}/3), pulling and retrying..."
                git pull --no-rebase --no-edit 2>&1 || true
                sleep 2
            fi
        done
        echo "ERROR: All 3 push attempts failed for: ${MSG}"
    ) 200>/tmp/git_push.lock
}

adapter_exists() {
    [ -f "$1/adapter_config.json" ] && [ -f "$1/adapter_model.safetensors" ] && \
    [ "$(wc -c < "$1/adapter_model.safetensors" 2>/dev/null)" -gt 10000 ]
}
results_exist() { [ -f "$1" ]; }

# =============================================================================
# GENERIC: Run full ladder for a scheme + model
# =============================================================================

run_ladder() {
    local GPU_ID="$1"
    local MODEL_NAME="$2"
    local MODEL_SHORT="$3"
    local SCHEME="$4"          # "synonyms_v2" or "sentlen_v2"
    local EVAL_SCRIPT="$5"     # path to eval script
    local SCHEME_SHORT="$6"    # "syn" or "sl"

    export CUDA_VISIBLE_DEVICES=${GPU_ID}

    local S1_ADAPTER="${W}/${MODEL_SHORT}-${SCHEME_SHORT}-v2-stage1-lora"
    local S1_BACKUP="${REPO}/adapters/${SCHEME}/${MODEL_SHORT}-stage1"
    local RESULTS_DIR="${REPO}/results/${SCHEME}/${MODEL_SHORT}"
    mkdir -p "${RESULTS_DIR}"

    echo "============================================================"
    echo "[$(timestamp)] GPU ${GPU_ID}: ${MODEL_SHORT} ${SCHEME} START"
    echo "============================================================"

    # -- Stage 1 --
    if adapter_exists "${S1_BACKUP}" && ! adapter_exists "${S1_ADAPTER}"; then
        echo "Restoring Stage 1 from backup"
        cp -r "${S1_BACKUP}" "${S1_ADAPTER}"
    fi

    if ! adapter_exists "${S1_ADAPTER}"; then
        echo "[$(timestamp)] ${MODEL_SHORT} ${SCHEME} Stage 1 -- Training"
        python3 "${TRAIN}" stage1 \
            --train-file "data/${SCHEME}/stage1/train.jsonl" \
            --val-file "data/${SCHEME}/stage1/val.jsonl" \
            --output-dir "${S1_ADAPTER}" \
            --model "${MODEL_NAME}" \
            --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
            --learning-rate ${LR1} --max-length ${ML} \
            --lora-r ${LR} --lora-alpha ${LA} --resume
    else
        echo "[$(timestamp)] ${MODEL_SHORT} ${SCHEME} Stage 1 -- SKIPPING (adapter exists)"
    fi

    if ! results_exist "${RESULTS_DIR}/stage1_results_200.json"; then
        echo "[$(timestamp)] ${MODEL_SHORT} ${SCHEME} Stage 1 -- Eval"
        python3 "${EVAL_SCRIPT}" stage1 \
            --adapter-dir "${S1_ADAPTER}" \
            --eval-file "data/${SCHEME}/stage1/val.jsonl" \
            --output "${RESULTS_DIR}/stage1_results_200.json" \
            --model "${MODEL_NAME}" \
            --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
    fi

    # Backup Stage 1 adapter
    if [ ! -d "${S1_BACKUP}" ] || ! adapter_exists "${S1_BACKUP}"; then
        echo "[$(timestamp)] Backing up Stage 1 adapter"
        mkdir -p "${S1_BACKUP}"
        cp "${S1_ADAPTER}"/adapter_config.json "${S1_BACKUP}/"
        cp "${S1_ADAPTER}"/adapter_model.safetensors "${S1_BACKUP}/"
        cp "${S1_ADAPTER}"/tokenizer_config.json "${S1_BACKUP}/" 2>/dev/null || true
        cp "${S1_ADAPTER}"/tokenizer.json "${S1_BACKUP}/" 2>/dev/null || true
        cp "${S1_ADAPTER}"/chat_template.jinja "${S1_BACKUP}/" 2>/dev/null || true
    fi
    save_progress "${MODEL_SHORT} ${SCHEME} Stage 1"

    # -- V0 through V2 --
    for variant in v0 v1a v1b v2; do
        local ADAPTER="${W}/${MODEL_SHORT}-${SCHEME_SHORT}-v2-${variant}-lora"
        local RES="${RESULTS_DIR}/${variant}_results_200.json"
        local TRAIN_DATA="data/${SCHEME}/${variant}/train.jsonl"
        local TEST_DATA="data/${SCHEME}/${variant}/test.jsonl"

        if [ ! -f "${TRAIN_DATA}" ]; then
            echo "[$(timestamp)] ${MODEL_SHORT} ${SCHEME} ${variant} -- SKIPPING (no train data)"
            continue
        fi
        if [ ! -f "${TEST_DATA}" ]; then
            echo "[$(timestamp)] ${MODEL_SHORT} ${SCHEME} ${variant} -- SKIPPING (no test data)"
            continue
        fi

        if results_exist "${RES}"; then
            echo "[$(timestamp)] ${MODEL_SHORT} ${SCHEME} ${variant} -- SKIPPING (results exist)"
            continue
        fi

        if ! adapter_exists "${ADAPTER}"; then
            echo "[$(timestamp)] ${MODEL_SHORT} ${SCHEME} ${variant} -- Training"
            python3 "${TRAIN}" stage2 \
                --adapter-dir "${S1_ADAPTER}" \
                --v0-data "${TRAIN_DATA}" \
                --output-dir "${ADAPTER}" \
                --model "${MODEL_NAME}" \
                --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
                --learning-rate ${LR2} --max-length ${ML} \
                --lora-r ${LR} --lora-alpha ${LA} --resume
        fi

        echo "[$(timestamp)] ${MODEL_SHORT} ${SCHEME} ${variant} -- Eval"
        python3 "${EVAL_SCRIPT}" "${variant}" \
            --adapter-dir "${ADAPTER}" \
            --eval-file "${TEST_DATA}" \
            --output "${RES}" \
            --model "${MODEL_NAME}" \
            --max-examples ${EVAL_MAX} --temperature ${EVAL_T}

        save_progress "${MODEL_SHORT} ${SCHEME} ${variant}"
    done

    echo "============================================================"
    echo "[$(timestamp)] GPU ${GPU_ID}: ${MODEL_SHORT} ${SCHEME} COMPLETE"
    echo "============================================================"
}

# =============================================================================
# CHECK
# =============================================================================

echo "============================================================"
echo "[$(timestamp)] V2 SCHEMES EXPERIMENT LAUNCHER"
echo "============================================================"
echo "GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPU count: ${GPU_COUNT}"
echo "Disk: root=$(df -h / | tail -1 | awk '{print $4}'), /dev/shm=$(df -h /dev/shm | tail -1 | awk '{print $4}')"
echo ""
echo "Data check:"
for scheme in synonyms_v2 sentlen_v2; do
    for d in stage1 v0 v1a v1b v2; do
        f="data/${scheme}/${d}/train.jsonl"
        if [ -f "$f" ]; then
            echo "  $f: $(wc -l < $f) examples"
        else
            echo "  $f: MISSING"
        fi
    done
done

# =============================================================================
# LAUNCH
# =============================================================================

echo ""
echo "[$(timestamp)] LAUNCHING 4 PARALLEL GPU JOBS"

# GPU 0: 7B Synonyms
run_ladder 0 "Qwen/Qwen2.5-7B-Instruct" "qwen-7b" "synonyms_v2" "${EVAL_SYN}" "syn" \
    > >(tee -a /dev/shm/gpu0.log) 2>&1 &
PID0=$!
echo "GPU 0 (7B syn):  PID ${PID0}"

# GPU 1: 7B Sentence Length
run_ladder 1 "Qwen/Qwen2.5-7B-Instruct" "qwen-7b" "sentlen_v2" "${EVAL_SL}" "sl" \
    > >(tee -a /dev/shm/gpu1.log) 2>&1 &
PID1=$!
echo "GPU 1 (7B sl):   PID ${PID1}"

# GPU 2: 14B Synonyms
run_ladder 2 "Qwen/Qwen2.5-14B-Instruct" "qwen-14b" "synonyms_v2" "${EVAL_SYN}" "syn" \
    > >(tee -a /dev/shm/gpu2.log) 2>&1 &
PID2=$!
echo "GPU 2 (14B syn): PID ${PID2}"

# GPU 3: 14B Sentence Length
run_ladder 3 "Qwen/Qwen2.5-14B-Instruct" "qwen-14b" "sentlen_v2" "${EVAL_SL}" "sl" \
    > >(tee -a /dev/shm/gpu3.log) 2>&1 &
PID3=$!
echo "GPU 3 (14B sl):  PID ${PID3}"

# Wait
wait ${PID0} && echo "GPU 0 finished OK" || echo "GPU 0 FAILED"
wait ${PID1} && echo "GPU 1 finished OK" || echo "GPU 1 FAILED"
wait ${PID2} && echo "GPU 2 finished OK" || echo "GPU 2 FAILED"
wait ${PID3} && echo "GPU 3 finished OK" || echo "GPU 3 FAILED"

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "============================================================"
echo "[$(timestamp)] ALL GPUS COMPLETE -- SUMMARY"
echo "============================================================"

python3 -c "
import json, os

print()
print('=' * 75)
print('V2 SCHEMES RESULTS (variable payload 4-7 bits, no padding)')
print('=' * 75)

for scheme in ['synonyms_v2', 'sentlen_v2']:
    print(f'\n{scheme.upper()}:')
    print(f'{\"Task\":<12} {\"7B Exact\":>10} {\"7B Partial\":>12} {\"14B Exact\":>10} {\"14B Partial\":>12}')
    print('-' * 58)
    for name, fname in [('Stage 1','stage1_results_200.json'),('V0','v0_results_200.json'),
                         ('V1a','v1a_results_200.json'),('V1b','v1b_results_200.json'),
                         ('V2','v2_results_200.json')]:
        row = f'{name:<12}'
        for model in ['qwen-7b','qwen-14b']:
            path = f'results/{scheme}/{model}/{fname}'
            if os.path.exists(path):
                d = json.load(open(path))['overall']
                row += f' {d[\"exact_recovery_rate\"]:>9.1%} {d[\"partial_recovery_rate\"]:>11.1%}'
            else:
                row += f' {\"--\":>10} {\"--\":>12}'
        print(row)

# Comparison with acrostics
print('\n\nCOMPARISON: All schemes V0 exact recovery')
print('-' * 40)
acr_7b = acr_14b = '--'
for m, label in [('qwen-7b','7B'),('qwen-14b','14B')]:
    p = f'results/acrostics/{m}/v0_results_200.json'
    if os.path.exists(p):
        d = json.load(open(p))['overall']
        if m == 'qwen-7b': acr_7b = f'{d[\"exact_recovery_rate\"]:.1%}'
        else: acr_14b = f'{d[\"exact_recovery_rate\"]:.1%}'

for scheme_name, scheme_dir in [('Acrostics','acrostics'),('Synonyms v2','synonyms_v2'),('Sentlen v2','sentlen_v2')]:
    row = f'{scheme_name:<15}'
    for m in ['qwen-7b','qwen-14b']:
        if scheme_dir == 'acrostics':
            p = f'results/acrostics/{m}/v0_results_200.json'
        else:
            p = f'results/{scheme_dir}/{m}/v0_results_200.json'
        if os.path.exists(p):
            d = json.load(open(p))['overall']
            row += f' {d[\"exact_recovery_rate\"]:>9.1%}'
        else:
            row += f' {\"--\":>10}'
    print(row)

print('=' * 75)
" | tee "${REPO}/results/v2_schemes_summary.txt"

save_progress "All v2 scheme experiments complete"

echo ""
echo "[$(timestamp)] ALL DONE"
