#!/bin/bash
# =============================================================================
# 8-GPU: 32B (5 combos) + 70B (3 combos)
# =============================================================================
#
# GPU allocation:
#   GPU 0: 32B acrostics_prose    (S1 + V0 + V1a + V2)
#   GPU 1: 32B sentlen_poems      (S1 + V0 + V1a + V2)
#   GPU 2: 32B sentlen_prose      (S1 + V0 + V1a + V2)
#   GPU 3: 32B synonym_poems      (S1 + V0 + V1a + V2)
#   GPU 4: 32B synonyms_prose     (S1 + V0 + V1a + V2)
#   GPU 5: 70B acrostics_prose    (S1 + V0 only)
#   GPU 6: 70B sentlen_poems      (S1 + V0 only)
#   GPU 7: 70B synonym_poems      (S1 + V0 only)
#
# 32B: ~10-12 hrs per combo on H200
# 70B: ~6-8 hrs per combo on H200 (S1 + V0 only)
#
# Resume: re-run same command, completed results skipped
# Cost: ~$230-280 at $23/hr for full run
#
# Usage:
#   cd /workspace/Steganography-internalisation-experiments
#   export HF_TOKEN="hf_..."
#   nohup bash scripts/run_32b_70b_matrix.sh > /dev/shm/matrix.log 2>&1 &
#   tail -5 /dev/shm/gpu{0..7}.log
# =============================================================================

REPO="/workspace/Steganography-internalisation-experiments"
TRAIN="${REPO}/scripts/train_acrostic.py"
W="/dev/shm"

# Training hyperparameters
EP=3
BS=1
GA=8
ML=512
LR_V=1e-4
LR_R=16
LR_A=32
EVAL_MAX=200
EVAL_T=0.7

# HF cache on /dev/shm (models are 60-130GB)
export HF_HOME="/dev/shm/hf_cache"
export TRANSFORMERS_CACHE="/dev/shm/hf_cache"
mkdir -p /dev/shm/hf_cache

cd "${REPO}"

# ---- Helper functions ----

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

adapter_exists() {
    [ -f "$1/adapter_config.json" ] && [ -f "$1/adapter_model.safetensors" ] && \
    [ "$(wc -c < "$1/adapter_model.safetensors" 2>/dev/null || echo 0)" -gt 10000 ]
}

results_exist() { [ -f "$1" ]; }

save_progress() {
    local MSG="$1"
    cd "${REPO}"
    git add results/ 2>/dev/null || true
    git commit -m "32b+70b: ${MSG}" 2>/dev/null || { echo "[git] Nothing to commit"; return 0; }
    for attempt in 1 2 3; do
        if git push 2>&1; then
            echo "[git] Push OK (attempt ${attempt})"
            return 0
        fi
        echo "[git] Push failed (${attempt}/3), pulling..."
        git pull --no-rebase --no-edit 2>&1 || true
        sleep $((RANDOM % 10 + 2))
    done
    echo "[git] WARNING: All push attempts failed for: ${MSG}"
}

model_name_from_size() {
    case "$1" in
        32B) echo "Qwen/Qwen2.5-32B-Instruct" ;;
        70B) echo "Qwen/Qwen2.5-72B-Instruct" ;;
        *)   echo "UNKNOWN"; return 1 ;;
    esac
}

# ---- Eval dispatcher ----

run_eval() {
    local COMBO=$1
    local VLEVEL=$2
    local ADAPTER=$3
    local EVAL_FILE=$4
    local OUTPUT=$5
    local MODEL=$6

    local COMMON="--adapter-dir ${ADAPTER} --eval-file ${EVAL_FILE} --output ${OUTPUT} --model ${MODEL} --max-examples ${EVAL_MAX} --temperature ${EVAL_T}"

    case "${COMBO}" in
        acrostics_prose)
            python3 "${REPO}/scripts/eval_acrostics_prose.py" "${VLEVEL}" ${COMMON}
            ;;
        sentlen_poems)
            python3 "${REPO}/scripts/eval_new_schemes.py" --scheme sentlen_poems "${VLEVEL}" ${COMMON}
            ;;
        sentlen_prose)
            python3 "${REPO}/scripts/eval_sentlen.py" "${VLEVEL}" ${COMMON}
            ;;
        synonym_poems)
            python3 "${REPO}/scripts/eval_new_schemes.py" --scheme synonym_poems "${VLEVEL}" ${COMMON}
            ;;
        synonyms_prose)
            python3 "${REPO}/scripts/eval_synonym.py" "${VLEVEL}" ${COMMON}
            ;;
        *)
            echo "ERROR: Unknown combo ${COMBO}"
            return 1
            ;;
    esac
}

# ---- Main per-job function ----

run_job() {
    local COMBO=$1
    local SIZE=$2
    local GPU_ID=$3

    local MODEL=$(model_name_from_size "${SIZE}")
    local S1="${W}/${COMBO}-${SIZE}-stage1"
    local RDIR="results/${COMBO}/qwen-${SIZE}"
    mkdir -p "${RDIR}"

    echo "============================================================"
    echo "[$(timestamp)] GPU ${GPU_ID}: ${COMBO} / ${SIZE}"
    echo "============================================================"

    # Check data exists
    if [ ! -f "data/${COMBO}/stage1/train.jsonl" ]; then
        echo "SKIP ${COMBO}: no stage1 training data"
        return 0
    fi

    # ---- STAGE 1 ----
    local S1_RESULT="${RDIR}/stage1_results_200.json"

    if results_exist "${S1_RESULT}"; then
        echo "[$(timestamp)] Stage 1 results exist, skipping"
        if ! adapter_exists "${S1}"; then
            echo "[$(timestamp)] But adapter missing, retraining Stage 1..."
            python3 "${TRAIN}" stage1 \
                --train-file "data/${COMBO}/stage1/train.jsonl" \
                --output-dir "${S1}" \
                --model "${MODEL}" \
                --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
                --max-length ${ML} --resume
        fi
    else
        if ! adapter_exists "${S1}"; then
            echo "[$(timestamp)] Training Stage 1..."
            python3 "${TRAIN}" stage1 \
                --train-file "data/${COMBO}/stage1/train.jsonl" \
                --output-dir "${S1}" \
                --model "${MODEL}" \
                --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
                --max-length ${ML} --resume
        fi

        echo "[$(timestamp)] Evaluating Stage 1..."
        local S1_EVAL="data/${COMBO}/stage1/val.jsonl"
        if [ -f "${S1_EVAL}" ]; then
            run_eval "${COMBO}" "stage1" "${S1}" "${S1_EVAL}" "${S1_RESULT}" "${MODEL}"
        else
            echo "WARNING: No val file at ${S1_EVAL}"
        fi
    fi

    save_progress "${COMBO} ${SIZE} stage1"

    # ---- V-levels: all for 32B, only V0 for 70B ----
    if [ "${SIZE}" = "70B" ]; then
        VLEVELS="v0"
    else
        VLEVELS="v0 v1a v2"
    fi
    for VLEVEL in ${VLEVELS}; do
        local V_RESULT="${RDIR}/${VLEVEL}_results_200.json"

        if results_exist "${V_RESULT}"; then
            echo "[$(timestamp)] ${VLEVEL} results exist, skipping"
            continue
        fi

        if [ ! -f "data/${COMBO}/${VLEVEL}/train.jsonl" ]; then
            echo "[$(timestamp)] SKIP ${VLEVEL}: no training data"
            continue
        fi

        if ! adapter_exists "${S1}"; then
            echo "ERROR: Stage 1 adapter missing, cannot train ${VLEVEL}"
            continue
        fi

        local V_ADAPTER="${W}/${COMBO}-${SIZE}-${VLEVEL}"

        if ! adapter_exists "${V_ADAPTER}"; then
            echo "[$(timestamp)] Training ${VLEVEL}..."
            python3 "${TRAIN}" stage2 \
                --adapter-dir "${S1}" \
                --v0-data "data/${COMBO}/${VLEVEL}/train.jsonl" \
                --output-dir "${V_ADAPTER}" \
                --model "${MODEL}" \
                --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
                --max-length ${ML} --learning-rate ${LR_V} \
                --lora-r ${LR_R} --lora-alpha ${LR_A} --resume
        fi

        echo "[$(timestamp)] Evaluating ${VLEVEL}..."
        local V_EVAL="data/${COMBO}/${VLEVEL}/test.jsonl"
        if [ -f "${V_EVAL}" ]; then
            run_eval "${COMBO}" "v0" "${V_ADAPTER}" "${V_EVAL}" "${V_RESULT}" "${MODEL}"
        else
            echo "WARNING: No test file at ${V_EVAL}"
        fi

        save_progress "${COMBO} ${SIZE} ${VLEVEL}"

        # Clean up V-level adapter to save space
        if results_exist "${V_RESULT}"; then
            rm -rf "${V_ADAPTER}" 2>/dev/null || true
        fi
    done

    # Clean up Stage 1 adapter
    rm -rf "${S1}" 2>/dev/null || true

    echo ""
    echo "============================================================"
    echo "[$(timestamp)] GPU ${GPU_ID}: ${COMBO} / ${SIZE} COMPLETE"
    echo "============================================================"
}

# =============================================================================
# SANITY CHECK
# =============================================================================
echo "============================================================"
echo "[$(timestamp)] SANITY CHECK"
echo "============================================================"

echo "GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "GPU count: ${GPU_COUNT}"
echo "Root disk: $(df -h / | tail -1 | awk '{print $4}') free"
echo "SHM: $(df -h /dev/shm | tail -1 | awk '{print $4}') free"
echo ""

# Check SHM (32B ~60GB + 70B ~130GB = ~190GB minimum)
SHM_FREE=$(df -BG /dev/shm | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "${SHM_FREE}" -lt 250 ]; then
    echo "WARNING: Only ${SHM_FREE}GB on /dev/shm. Need ~250GB for 32B + 70B models."
fi

# Check data
echo "Checking data..."
for COMBO in acrostics_prose sentlen_poems sentlen_prose synonym_poems synonyms_prose; do
    if [ -f "data/${COMBO}/stage1/train.jsonl" ]; then
        COUNT=$(wc -l < "data/${COMBO}/stage1/train.jsonl")
        echo "  ${COMBO}: ${COUNT} stage1 examples"
    else
        echo "  ${COMBO}: MISSING"
    fi
done
echo ""

# Quick training sanity check
echo "--- Quick training test (5 examples, GPU 0) ---"
head -5 "data/acrostics_prose/stage1/train.jsonl" > /dev/shm/sanity_test.jsonl
CUDA_VISIBLE_DEVICES=0 python3 "${TRAIN}" stage1 \
    --train-file /dev/shm/sanity_test.jsonl \
    --output-dir /dev/shm/sanity_lora \
    --model Qwen/Qwen2.5-7B-Instruct \
    --epochs 1 --batch-size 1 --gradient-accumulation 1 --max-length 512

if [ $? -ne 0 ]; then
    echo "FATAL: Sanity training failed"
    exit 1
fi
echo "Sanity check PASSED"
rm -rf /dev/shm/sanity_lora /dev/shm/sanity_test.jsonl
echo ""

# =============================================================================
# LAUNCH
# =============================================================================
echo "============================================================"
echo "[$(timestamp)] LAUNCHING JOBS"
echo "============================================================"

# Define jobs: COMBO SIZE GPU_ID
JOBS=(
    "acrostics_prose  32B 0"
    "sentlen_poems    32B 1"
    "sentlen_prose    32B 2"
    "synonym_poems    32B 3"
    "synonyms_prose   32B 4"
    "acrostics_prose  70B 5"
    "sentlen_poems    70B 6"
    "synonym_poems    70B 7"
)

PIDS=()
LABELS=()

for JOB in "${JOBS[@]}"; do
    read -r COMBO SIZE GPU_ID <<< "${JOB}"

    # Use modulo if fewer GPUs than jobs
    ACTUAL_GPU=$((GPU_ID % GPU_COUNT))

    LOGFILE="/dev/shm/gpu${GPU_ID}.log"
    LABEL="${COMBO}/${SIZE}"

    CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} run_job "${COMBO}" "${SIZE}" "${GPU_ID}" > "${LOGFILE}" 2>&1 &
    PIDS+=($!)
    LABELS+=("${LABEL}")

    echo "GPU ${ACTUAL_GPU}: ${LABEL} -> ${LOGFILE} (PID ${PIDS[-1]})"

    # Stagger: longer delay for 70B to let 32B claim cache first
    if [ "${SIZE}" = "70B" ]; then
        sleep 30
    else
        sleep 10
    fi
done

echo ""
echo "All jobs launched. Monitor with:"
for i in "${!LABELS[@]}"; do
    echo "  tail -5 /dev/shm/gpu${i}.log   # ${LABELS[$i]}"
done
echo ""
echo "Waiting for all jobs to finish..."

FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    STATUS=$?
    if [ ${STATUS} -ne 0 ]; then
        echo "WARNING: ${LABELS[$i]} exited with status ${STATUS}"
        FAILED=$((FAILED + 1))
    else
        echo "OK: ${LABELS[$i]} completed"
    fi
done

echo ""
echo "============================================================"
echo "[$(timestamp)] ALL JOBS COMPLETE (${FAILED} failures)"
echo "============================================================"

save_progress "32b+70b run complete"

# Print summary
echo ""
echo "=== RESULTS SUMMARY ==="
python3 -c "
import json, os
for combo in ['acrostics_prose','sentlen_poems','sentlen_prose','synonym_poems','synonyms_prose']:
    print(f'=== {combo} ===')
    for size in ['32B','70B']:
        parts = []
        for v in ['stage1','v0','v1a','v2']:
            f = f'results/{combo}/qwen-{size}/{v}_results_200.json'
            if os.path.exists(f):
                d = json.load(open(f))
                o = d.get('overall', d)
                e = o.get('exact', o.get('exact_recovery_rate', -1))
                if 0 <= e <= 1.0: e *= 100
                parts.append(f'{v}={e:.1f}%')
        if parts:
            print(f'  {size}: {\"  \".join(parts)}')
    print()
" 2>/dev/null || echo "(summary script failed, check results/ manually)"
