#!/bin/bash
# =============================================================================
# 8-GPU MATRIX: 5 combos x 3 model sizes x 4 V-levels = 60 training runs
# =============================================================================
#
# GPU allocation (one combo per GPU, running 7B -> 14B -> 32B sequentially):
#   GPU 0: acrostics_prose
#   GPU 1: sentlen_poems
#   GPU 2: sentlen_prose
#   GPU 3: synonym_poems
#   GPU 4: synonyms_prose
#   GPU 5-7: idle
#
# Eval scripts per combo:
#   acrostics_prose -> eval_acrostics_prose.py
#   sentlen_poems   -> eval_new_schemes.py --scheme sentlen_poems
#   sentlen_prose   -> eval_sentlen.py
#   synonym_poems   -> eval_new_schemes.py --scheme synonym_poems
#   synonyms_prose  -> eval_synonym.py
#
# Known issues addressed:
#   1. LFS pointers: adapter_exists checks file size > 10KB
#   2. HF cache: on /dev/shm (not root disk)
#   3. Git push: retry 3x with pull between attempts
#   4. Stale results: skip only if result file exists
#   5. Sanity check: actual tiny training + eval before real work
#   6. Model download: HF_TOKEN set, staggered GPU starts
#
# Estimated wall time: ~9-12 hrs on H200, ~12-16 hrs on A100
# Resume: re-run the same command, completed results are skipped
#
# Usage:
#   cd /workspace/Steganography-internalisation-experiments
#   nohup bash scripts/run_8gpu_matrix.sh > /dev/shm/matrix.log 2>&1 &
#   tail -5 /dev/shm/gpu0.log  # acrostics_prose
#   tail -5 /dev/shm/gpu1.log  # sentlen_poems
#   tail -5 /dev/shm/gpu2.log  # sentlen_prose
#   tail -5 /dev/shm/gpu3.log  # synonym_poems
#   tail -5 /dev/shm/gpu4.log  # synonyms_prose
# =============================================================================

REPO="/workspace/Steganography-internalisation-experiments"
TRAIN="${REPO}/scripts/train_acrostic.py"
W="/dev/shm"

# Training hyperparameters
EP=3
BS=1
GA=8
ML=512
LR_V=1e-4       # V0+ learning rate
LR_R=16         # LoRA r
LR_A=32         # LoRA alpha
EVAL_MAX=200
EVAL_T=0.7

# HF cache on /dev/shm (NOT root disk -- models are 15-60GB each)
export HF_HOME="/dev/shm/hf_cache"
export TRANSFORMERS_CACHE="/dev/shm/hf_cache"
mkdir -p /dev/shm/hf_cache

cd "${REPO}"

# ---- Helper functions ----

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

adapter_exists() {
    # Check files exist AND safetensors is real (not LFS pointer ~130 bytes)
    [ -f "$1/adapter_config.json" ] && [ -f "$1/adapter_model.safetensors" ] && \
    [ "$(wc -c < "$1/adapter_model.safetensors" 2>/dev/null || echo 0)" -gt 10000 ]
}

results_exist() { [ -f "$1" ]; }

save_progress() {
    local MSG="$1"
    cd "${REPO}"
    git add results/ 2>/dev/null || true
    git commit -m "matrix: ${MSG}" 2>/dev/null || { echo "[git] Nothing to commit"; return 0; }
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

model_name() {
    case "$1" in
        7B)  echo "Qwen/Qwen2.5-7B-Instruct" ;;
        14B) echo "Qwen/Qwen2.5-14B-Instruct" ;;
        32B) echo "Qwen/Qwen2.5-32B-Instruct" ;;
        *)   echo "UNKNOWN"; return 1 ;;
    esac
}

# ---- Eval dispatcher ----
# Routes to the correct eval script based on combo name

run_eval() {
    local COMBO=$1
    local VLEVEL=$2   # "stage1" or "v0" (v0 used for all V-levels)
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

# ---- Main per-combo function ----

run_combo() {
    local COMBO=$1
    local GPU_ID=$2

    echo "============================================================"
    echo "[$(timestamp)] GPU ${GPU_ID}: Starting ${COMBO}"
    echo "============================================================"

    # Check data exists
    if [ ! -f "data/${COMBO}/stage1/train.jsonl" ]; then
        echo "SKIP ${COMBO}: no stage1 training data at data/${COMBO}/stage1/train.jsonl"
        return 0
    fi

    for SIZE in 7B 14B 32B; do
        local MODEL=$(model_name "${SIZE}")
        local S1="${W}/${COMBO}-${SIZE}-stage1"
        local RDIR="results/${COMBO}/qwen-${SIZE}"
        mkdir -p "${RDIR}"

        echo ""
        echo "============================================================"
        echo "[$(timestamp)] GPU ${GPU_ID}: ${COMBO} / ${SIZE}"
        echo "============================================================"

        # ---- STAGE 1 ----
        local S1_RESULT="${RDIR}/stage1_results_200.json"

        if results_exist "${S1_RESULT}"; then
            echo "[$(timestamp)] Stage 1 results exist, skipping"
            # Still need the adapter for V-level training
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
            # Train Stage 1
            if ! adapter_exists "${S1}"; then
                echo "[$(timestamp)] Training Stage 1..."
                python3 "${TRAIN}" stage1 \
                    --train-file "data/${COMBO}/stage1/train.jsonl" \
                    --output-dir "${S1}" \
                    --model "${MODEL}" \
                    --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
                    --max-length ${ML} --resume
            fi

            # Eval Stage 1
            echo "[$(timestamp)] Evaluating Stage 1..."
            local S1_EVAL="data/${COMBO}/stage1/val.jsonl"
            if [ -f "${S1_EVAL}" ]; then
                run_eval "${COMBO}" "stage1" "${S1}" "${S1_EVAL}" "${S1_RESULT}" "${MODEL}"
            else
                echo "WARNING: No val file at ${S1_EVAL}, skipping Stage 1 eval"
            fi
        fi

        # ---- V0, V1a, V2 ----
        for VLEVEL in v0 v1a v2; do
            local V_RESULT="${RDIR}/${VLEVEL}_results_200.json"

            if results_exist "${V_RESULT}"; then
                echo "[$(timestamp)] ${VLEVEL} results exist, skipping"
                continue
            fi

            # Check training data exists
            if [ ! -f "data/${COMBO}/${VLEVEL}/train.jsonl" ]; then
                echo "[$(timestamp)] SKIP ${VLEVEL}: no training data at data/${COMBO}/${VLEVEL}/train.jsonl"
                continue
            fi

            # Ensure Stage 1 adapter exists
            if ! adapter_exists "${S1}"; then
                echo "ERROR: Stage 1 adapter missing for ${COMBO} ${SIZE}, cannot train ${VLEVEL}"
                continue
            fi

            local V_ADAPTER="${W}/${COMBO}-${SIZE}-${VLEVEL}"

            # Train V-level
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

            # Eval V-level
            echo "[$(timestamp)] Evaluating ${VLEVEL}..."
            local V_EVAL="data/${COMBO}/${VLEVEL}/test.jsonl"
            if [ -f "${V_EVAL}" ]; then
                run_eval "${COMBO}" "v0" "${V_ADAPTER}" "${V_EVAL}" "${V_RESULT}" "${MODEL}"
            else
                echo "WARNING: No test file at ${V_EVAL}, skipping eval"
            fi
        done

        # Push results after each model size
        save_progress "${COMBO} ${SIZE} complete"

        # Clean up V-level adapters to save /dev/shm space (keep Stage 1 for resume)
        for VLEVEL in v0 v1a v2; do
            if results_exist "${RDIR}/${VLEVEL}_results_200.json"; then
                rm -rf "${W}/${COMBO}-${SIZE}-${VLEVEL}" 2>/dev/null || true
            fi
        done

    done

    # Clean up Stage 1 adapters after all sizes done
    for SIZE in 7B 14B 32B; do
        rm -rf "${W}/${COMBO}-${SIZE}-stage1" 2>/dev/null || true
    done

    echo ""
    echo "============================================================"
    echo "[$(timestamp)] GPU ${GPU_ID}: ${COMBO} COMPLETE"
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
echo "Root disk: $(df -h / | tail -1 | awk '{print $4}') free"
echo "SHM: $(df -h /dev/shm | tail -1 | awk '{print $4}') free"
echo ""

# Check GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "${GPU_COUNT}" -lt 5 ]; then
    echo "WARNING: Only ${GPU_COUNT} GPUs detected. Need 5 for full parallel run."
    echo "Will still work but some combos will run sequentially."
fi

# Check SHM space (need ~200GB for models + adapters)
SHM_FREE=$(df -BG /dev/shm | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "${SHM_FREE}" -lt 200 ]; then
    echo "WARNING: Only ${SHM_FREE}GB free on /dev/shm. Need ~200GB for model cache + adapters."
fi

# Check data exists
echo "Checking data directories..."
COMBOS=("acrostics_prose" "sentlen_poems" "sentlen_prose" "synonym_poems" "synonyms_prose")
for COMBO in "${COMBOS[@]}"; do
    if [ -f "data/${COMBO}/stage1/train.jsonl" ]; then
        COUNT=$(wc -l < "data/${COMBO}/stage1/train.jsonl")
        echo "  ${COMBO}: ${COUNT} stage1 examples"
    else
        echo "  ${COMBO}: MISSING (will skip)"
    fi
done
echo ""

# Quick training test
echo "--- Quick training test (5 examples, GPU 0) ---"
FIRST_COMBO=""
for COMBO in "${COMBOS[@]}"; do
    if [ -f "data/${COMBO}/stage1/train.jsonl" ]; then
        FIRST_COMBO="${COMBO}"
        break
    fi
done

if [ -z "${FIRST_COMBO}" ]; then
    echo "FATAL: No training data found for any combo"
    exit 1
fi

head -5 "data/${FIRST_COMBO}/stage1/train.jsonl" > /dev/shm/sanity_test.jsonl
CUDA_VISIBLE_DEVICES=0 python3 "${TRAIN}" stage1 \
    --train-file /dev/shm/sanity_test.jsonl \
    --output-dir /dev/shm/sanity_lora \
    --model Qwen/Qwen2.5-7B-Instruct \
    --epochs 1 --batch-size 1 --gradient-accumulation 1 --max-length 512

if [ $? -ne 0 ]; then
    echo "FATAL: Sanity training failed"
    exit 1
fi

echo "Training sanity check PASSED"
rm -rf /dev/shm/sanity_lora /dev/shm/sanity_test.jsonl
echo ""

# =============================================================================
# LAUNCH PARALLEL JOBS
# =============================================================================
echo "============================================================"
echo "[$(timestamp)] LAUNCHING PARALLEL JOBS"
echo "============================================================"
echo ""

PIDS=()
for i in "${!COMBOS[@]}"; do
    COMBO="${COMBOS[$i]}"

    # Skip combos without data
    if [ ! -f "data/${COMBO}/stage1/train.jsonl" ]; then
        echo "GPU $i: SKIPPING ${COMBO} (no data)"
        continue
    fi

    # Use GPU $i (or wrap around if fewer GPUs)
    GPU=$((i % GPU_COUNT))

    CUDA_VISIBLE_DEVICES=${GPU} run_combo "${COMBO}" "${GPU}" > "/dev/shm/gpu${i}.log" 2>&1 &
    PIDS+=($!)
    echo "GPU ${GPU}: ${COMBO} -> /dev/shm/gpu${i}.log (PID ${PIDS[-1]})"

    # Stagger starts by 10s to avoid HF download races
    sleep 10
done

echo ""
echo "All jobs launched. Monitor with:"
for i in "${!COMBOS[@]}"; do
    echo "  tail -5 /dev/shm/gpu${i}.log   # ${COMBOS[$i]}"
done
echo ""
echo "Waiting for all jobs to finish..."

# Wait and report
FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    STATUS=$?
    if [ ${STATUS} -ne 0 ]; then
        echo "WARNING: GPU ${i} (${COMBOS[$i]}) exited with status ${STATUS}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
echo "[$(timestamp)] ALL JOBS COMPLETE (${FAILED} failures)"
echo "============================================================"

# Final push
save_progress "matrix run complete"

# Print summary of all results
echo ""
echo "=== RESULTS SUMMARY ==="
echo ""
for COMBO in "${COMBOS[@]}"; do
    echo "--- ${COMBO} ---"
    for SIZE in 7B 14B 32B; do
        RDIR="results/${COMBO}/qwen-${SIZE}"
        printf "  %-4s: " "${SIZE}"
        for VLEVEL in stage1 v0 v1a v2; do
            RFILE="${RDIR}/${VLEVEL}_results_200.json"
            if [ -f "${RFILE}" ]; then
                EXACT=$(python3 -c "import json; d=json.load(open('${RFILE}')); print(f'{d[\"overall\"][\"exact\"]:.1f}%')" 2>/dev/null || echo "?")
                printf "%s=%s  " "${VLEVEL}" "${EXACT}"
            else
                printf "%s=MISS  " "${VLEVEL}"
            fi
        done
        echo ""
    done
    echo ""
done
