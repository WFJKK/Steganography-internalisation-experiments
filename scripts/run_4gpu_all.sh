#!/bin/bash
# =============================================================================
# 4-GPU Parallel Experiments
# =============================================================================
# GPU 0: 7B compute scaling + cross-eval (~4 hrs)
#   - Acrostic V2: 1ep, 2ep, 4ep, 9ep
#   - Sentlen V0: 1ep, 2ep, 4ep, 6ep, 9ep
#   - Sentlen cross-eval
#
# GPU 1: 14B compute scaling + cross-eval (~5 hrs)
#   - Sentlen V0: 1ep, 2ep, 4ep, 6ep, 9ep
#   - Sentlen cross-eval
#
# GPU 2: 32B sentence length (~12 hrs)
#   - Stage 1 + V0 + V1a + V2
#
# GPU 3: 32B acrostics (~4 hrs)
#   - V1a + V1b + V2 6ep
#
# REQUIRES: 4x A100 80GB, container disk 150GB+
# TOTAL COST: ~$15-25 depending on hourly rate
#
# Usage:
#   cd /workspace/Steganography-internalisation-experiments
#   nohup bash scripts/run_4gpu_all.sh > /dev/shm/4gpu.log 2>&1 &
#   tail -20 /dev/shm/gpu0.log  # 7B progress
#   tail -20 /dev/shm/gpu1.log  # 14B progress
#   tail -20 /dev/shm/gpu2.log  # 32B sentlen progress
#   tail -20 /dev/shm/gpu3.log  # 32B acrostics progress
# =============================================================================

set -e

REPO="/workspace/Steganography-internalisation-experiments"
TRAIN="${REPO}/scripts/train_acrostic.py"
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

export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="/workspace/hf_cache"
mkdir -p /workspace/hf_cache

cd "${REPO}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

# Shared save function with locking to prevent concurrent git pushes
save_progress() {
    local MSG="$1"
    (
        flock -w 120 200 || { echo "ERROR: Could not acquire git lock for: ${MSG}"; return 1; }
        echo "[$(timestamp)] SAVING: ${MSG}"
        cd "${REPO}"
        git add results/ adapters/ 2>/dev/null
        git commit -m "4GPU: ${MSG}" || { echo "Nothing to commit"; return 0; }
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
    # Check both files exist AND adapter_model.safetensors is a real file (not LFS pointer)
    [ -f "$1/adapter_config.json" ] && [ -f "$1/adapter_model.safetensors" ] && \
    [ "$(wc -c < "$1/adapter_model.safetensors" 2>/dev/null)" -gt 10000 ]
}
results_exist() { [ -f "$1" ]; }

# Train sentlen Stage 1 if needed (small dataset, fast)
ensure_sentlen_stage1() {
    local MODEL_NAME="$1"
    local ADAPTER_DIR="$2"
    local BACKUP_DIR="$3"

    # Try restore from backup first
    if adapter_exists "${BACKUP_DIR}" && ! adapter_exists "${ADAPTER_DIR}"; then
        echo "Restoring Stage 1 from backup: ${BACKUP_DIR}"
        cp -r "${BACKUP_DIR}" "${ADAPTER_DIR}"
    fi

    # If still missing, retrain (~30 min for 7B, ~60 min for 14B)
    if ! adapter_exists "${ADAPTER_DIR}"; then
        echo "Stage 1 adapter not available, retraining from scratch..."
        python3 "${TRAIN}" stage1 \
            --train-file data/sentlen/stage1/train.jsonl \
            --val-file data/sentlen/stage1/val.jsonl \
            --output-dir "${ADAPTER_DIR}" \
            --model "${MODEL_NAME}" \
            --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
            --learning-rate ${LR1} --max-length ${ML} \
            --lora-r ${LR} --lora-alpha ${LA} --resume

        # Backup if directory doesn't exist
        if [ ! -d "${BACKUP_DIR}" ] || ! adapter_exists "${BACKUP_DIR}"; then
            mkdir -p "${BACKUP_DIR}"
            cp "${ADAPTER_DIR}"/adapter_config.json "${BACKUP_DIR}/"
            cp "${ADAPTER_DIR}"/adapter_model.safetensors "${BACKUP_DIR}/"
            cp "${ADAPTER_DIR}"/tokenizer_config.json "${BACKUP_DIR}/" 2>/dev/null || true
            cp "${ADAPTER_DIR}"/tokenizer.json "${BACKUP_DIR}/" 2>/dev/null || true
            cp "${ADAPTER_DIR}"/chat_template.jinja "${BACKUP_DIR}/" 2>/dev/null || true
            save_progress "Retrained ${MODEL_NAME} sentlen Stage 1"
        fi
    fi
}

# Train acrostic Stage 1 if needed (large dataset, slow for 32B)
ensure_acrostic_stage1() {
    local MODEL_NAME="$1"
    local ADAPTER_DIR="$2"
    local BACKUP_DIR="$3"

    if adapter_exists "${BACKUP_DIR}" && ! adapter_exists "${ADAPTER_DIR}"; then
        echo "Restoring acrostic Stage 1 from backup: ${BACKUP_DIR}"
        cp -r "${BACKUP_DIR}" "${ADAPTER_DIR}"
    fi

    if ! adapter_exists "${ADAPTER_DIR}"; then
        echo "Acrostic Stage 1 adapter not available, retraining from scratch..."
        python3 "${TRAIN}" stage1 \
            --train-file data/acrostics/stage1/train.jsonl \
            --val-file data/acrostics/stage1/val.jsonl \
            --output-dir "${ADAPTER_DIR}" \
            --model "${MODEL_NAME}" \
            --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
            --learning-rate ${LR1} --max-length ${ML} \
            --lora-r ${LR} --lora-alpha ${LA} --resume

        if [ ! -d "${BACKUP_DIR}" ] || ! adapter_exists "${BACKUP_DIR}"; then
            mkdir -p "${BACKUP_DIR}"
            cp "${ADAPTER_DIR}"/adapter_config.json "${BACKUP_DIR}/"
            cp "${ADAPTER_DIR}"/adapter_model.safetensors "${BACKUP_DIR}/"
            cp "${ADAPTER_DIR}"/tokenizer_config.json "${BACKUP_DIR}/" 2>/dev/null || true
            cp "${ADAPTER_DIR}"/tokenizer.json "${BACKUP_DIR}/" 2>/dev/null || true
            cp "${ADAPTER_DIR}"/chat_template.jinja "${BACKUP_DIR}/" 2>/dev/null || true
            save_progress "Retrained ${MODEL_NAME} acrostic Stage 1"
        fi
    fi
}

# =============================================================================
# CHECK GPUs
# =============================================================================

echo "============================================================"
echo "[$(timestamp)] 4-GPU EXPERIMENT LAUNCHER"
echo "============================================================"
echo "GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPU count: ${GPU_COUNT}"
echo "Disk: root=$(df -h / | tail -1 | awk '{print $4}'), /dev/shm=$(df -h /dev/shm | tail -1 | awk '{print $4}')"

if [ "$GPU_COUNT" -lt 4 ]; then
    echo "WARNING: Only ${GPU_COUNT} GPUs found. Need 4 for full parallel run."
    echo "Will run sequentially on available GPUs."
fi

# =============================================================================
# GPU 0: 7B compute scaling + cross-eval
# =============================================================================

run_gpu0() {
    exec > >(tee -a /dev/shm/gpu0.log) 2>&1
    echo "============================================================"
    echo "[$(timestamp)] GPU 0: 7B EXPERIMENTS START"
    echo "============================================================"
    export CUDA_VISIBLE_DEVICES=0

    local MODEL="Qwen/Qwen2.5-7B-Instruct"
    local ACR_S1="${W}/qwen-7b-acr-stage1-lora"
    local ACR_S1_BACKUP="${REPO}/adapters/acrostics/qwen-7b-stage1"
    local SL_S1="${W}/qwen-7b-sl-stage1-lora"
    local SL_S1_BACKUP="${REPO}/adapters/sentlen/qwen-7b-stage1"

    # Ensure both Stage 1 adapters are available
    ensure_acrostic_stage1 "${MODEL}" "${ACR_S1}" "${ACR_S1_BACKUP}"
    ensure_sentlen_stage1 "${MODEL}" "${SL_S1}" "${SL_S1_BACKUP}"

    # -- Acrostic V2 multi-epoch (1, 2, 4, 9) --
    # Already have 3ep and 6ep results
    for EP in 1 2 4 9; do
        local ADAPTER="${W}/7b-acr-v2-${EP}ep-lora"
        local RES="${REPO}/results/acrostics/qwen-7b/v2_${EP}ep_results_200.json"

        if results_exist "${RES}"; then
            echo "[$(timestamp)] 7B Acrostic V2 ${EP}ep -- SKIPPING (results exist)"
            continue
        fi

        if ! adapter_exists "${ADAPTER}"; then
            echo "[$(timestamp)] 7B Acrostic V2 ${EP}ep -- Training"
            python3 "${TRAIN}" stage2 \
                --adapter-dir "${ACR_S1}" --v0-data data/acrostics/v2/train.jsonl \
                --output-dir "${ADAPTER}" --model "${MODEL}" \
                --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
                --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
        fi

        echo "[$(timestamp)] 7B Acrostic V2 ${EP}ep -- Eval"
        python3 "${TRAIN}" evaluate-v0 \
            --adapter-dir "${ADAPTER}" --eval-file data/acrostics/v2/test.jsonl \
            --output "${RES}" --model "${MODEL}" \
            --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
        save_progress "7B acrostic V2 ${EP}ep"
    done

    # -- Sentlen V0 multi-epoch (1, 2, 4, 6, 9) --
    # Already have 3ep results
    if adapter_exists "${SL_S1}"; then
        for EP in 1 2 4 6 9; do
            local ADAPTER="${W}/7b-sl-v0-${EP}ep-lora"
            local RES="${REPO}/results/sentlen/qwen-7b/v0_${EP}ep_results_200.json"

            if results_exist "${RES}"; then
                echo "[$(timestamp)] 7B Sentlen V0 ${EP}ep -- SKIPPING (results exist)"
                continue
            fi

            if ! adapter_exists "${ADAPTER}"; then
                echo "[$(timestamp)] 7B Sentlen V0 ${EP}ep -- Training"
                python3 "${TRAIN}" stage2 \
                    --adapter-dir "${SL_S1}" --v0-data data/sentlen/v0/train.jsonl \
                    --output-dir "${ADAPTER}" --model "${MODEL}" \
                    --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
                    --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
            fi

            echo "[$(timestamp)] 7B Sentlen V0 ${EP}ep -- Eval"
            python3 "${EVAL_SL}" v0 \
                --adapter-dir "${ADAPTER}" --eval-file data/sentlen/v0/test.jsonl \
                --output "${RES}" --model "${MODEL}" \
                --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
            save_progress "7B sentlen V0 ${EP}ep"
        done

        # -- Sentlen cross-eval: train V0 adapter, eval on V2 test --
        local CROSS_RES="${REPO}/results/sentlen/qwen-7b/cross_v0_on_v2.json"
        if ! results_exist "${CROSS_RES}"; then
            local V0_ADAPTER="${W}/7b-sl-v0-3ep-lora"
            # Use existing 3ep adapter or train one
            if ! adapter_exists "${V0_ADAPTER}"; then
                python3 "${TRAIN}" stage2 \
                    --adapter-dir "${SL_S1}" --v0-data data/sentlen/v0/train.jsonl \
                    --output-dir "${V0_ADAPTER}" --model "${MODEL}" \
                    --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
                    --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
            fi
            echo "[$(timestamp)] 7B Sentlen cross-eval: V0 adapter on V2 test"
            python3 "${EVAL_SL}" v0 \
                --adapter-dir "${V0_ADAPTER}" --eval-file data/sentlen/v2/test.jsonl \
                --output "${CROSS_RES}" --model "${MODEL}" \
                --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
            save_progress "7B sentlen cross-eval V0->V2"
        fi

        local CROSS_RES2="${REPO}/results/sentlen/qwen-7b/cross_v2_on_v0.json"
        if ! results_exist "${CROSS_RES2}"; then
            local V2_ADAPTER="${W}/7b-sl-v2-3ep-lora"
            if ! adapter_exists "${V2_ADAPTER}"; then
                python3 "${TRAIN}" stage2 \
                    --adapter-dir "${SL_S1}" --v0-data data/sentlen/v2/train.jsonl \
                    --output-dir "${V2_ADAPTER}" --model "${MODEL}" \
                    --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
                    --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
            fi
            echo "[$(timestamp)] 7B Sentlen cross-eval: V2 adapter on V0 test"
            python3 "${EVAL_SL}" v0 \
                --adapter-dir "${V2_ADAPTER}" --eval-file data/sentlen/v0/test.jsonl \
                --output "${CROSS_RES2}" --model "${MODEL}" \
                --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
            save_progress "7B sentlen cross-eval V2->V0"
        fi
    fi

    echo "============================================================"
    echo "[$(timestamp)] GPU 0: 7B EXPERIMENTS COMPLETE"
    echo "============================================================"
}

# =============================================================================
# GPU 1: 14B compute scaling + cross-eval
# =============================================================================

run_gpu1() {
    exec > >(tee -a /dev/shm/gpu1.log) 2>&1
    echo "============================================================"
    echo "[$(timestamp)] GPU 1: 14B EXPERIMENTS START"
    echo "============================================================"
    export CUDA_VISIBLE_DEVICES=1

    local MODEL="Qwen/Qwen2.5-14B-Instruct"
    local SL_S1="${W}/qwen-14b-sl-stage1-lora"
    local SL_S1_BACKUP="${REPO}/adapters/sentlen/qwen-14b-stage1"

    # Ensure Stage 1 adapter available (retrain if LFS broken)
    ensure_sentlen_stage1 "${MODEL}" "${SL_S1}" "${SL_S1_BACKUP}"

    # -- Sentlen V0 multi-epoch (1, 2, 4, 6, 9) --
    for EP in 1 2 4 6 9; do
            local ADAPTER="${W}/14b-sl-v0-${EP}ep-lora"
            local RES="${REPO}/results/sentlen/qwen-14b/v0_${EP}ep_results_200.json"

            if results_exist "${RES}"; then
                echo "[$(timestamp)] 14B Sentlen V0 ${EP}ep -- SKIPPING (results exist)"
                continue
            fi

            if ! adapter_exists "${ADAPTER}"; then
                echo "[$(timestamp)] 14B Sentlen V0 ${EP}ep -- Training"
                python3 "${TRAIN}" stage2 \
                    --adapter-dir "${SL_S1}" --v0-data data/sentlen/v0/train.jsonl \
                    --output-dir "${ADAPTER}" --model "${MODEL}" \
                    --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
                    --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
            fi

            echo "[$(timestamp)] 14B Sentlen V0 ${EP}ep -- Eval"
            python3 "${EVAL_SL}" v0 \
                --adapter-dir "${ADAPTER}" --eval-file data/sentlen/v0/test.jsonl \
                --output "${RES}" --model "${MODEL}" \
                --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
            save_progress "14B sentlen V0 ${EP}ep"
        done

        # -- Cross-eval --
        local CROSS_RES="${REPO}/results/sentlen/qwen-14b/cross_v0_on_v2.json"
        if ! results_exist "${CROSS_RES}"; then
            local V0_ADAPTER="${W}/14b-sl-v0-3ep-lora"
            if ! adapter_exists "${V0_ADAPTER}"; then
                python3 "${TRAIN}" stage2 \
                    --adapter-dir "${SL_S1}" --v0-data data/sentlen/v0/train.jsonl \
                    --output-dir "${V0_ADAPTER}" --model "${MODEL}" \
                    --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
                    --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
            fi
            echo "[$(timestamp)] 14B Sentlen cross-eval: V0 adapter on V2 test"
            python3 "${EVAL_SL}" v0 \
                --adapter-dir "${V0_ADAPTER}" --eval-file data/sentlen/v2/test.jsonl \
                --output "${CROSS_RES}" --model "${MODEL}" \
                --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
            save_progress "14B sentlen cross-eval V0->V2"
        fi

        local CROSS_RES2="${REPO}/results/sentlen/qwen-14b/cross_v2_on_v0.json"
        if ! results_exist "${CROSS_RES2}"; then
            local V2_ADAPTER="${W}/14b-sl-v2-3ep-lora"
            if ! adapter_exists "${V2_ADAPTER}"; then
                python3 "${TRAIN}" stage2 \
                    --adapter-dir "${SL_S1}" --v0-data data/sentlen/v2/train.jsonl \
                    --output-dir "${V2_ADAPTER}" --model "${MODEL}" \
                    --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
                    --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
            fi
            echo "[$(timestamp)] 14B Sentlen cross-eval: V2 adapter on V0 test"
            python3 "${EVAL_SL}" v0 \
                --adapter-dir "${V2_ADAPTER}" --eval-file data/sentlen/v0/test.jsonl \
                --output "${CROSS_RES2}" --model "${MODEL}" \
                --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
            save_progress "14B sentlen cross-eval V2->V0"
        fi

    echo "============================================================"
    echo "[$(timestamp)] GPU 1: 14B EXPERIMENTS COMPLETE"
    echo "============================================================"
}

# =============================================================================
# GPU 2: 32B sentence length
# =============================================================================

run_gpu2() {
    exec > >(tee -a /dev/shm/gpu2.log) 2>&1
    echo "============================================================"
    echo "[$(timestamp)] GPU 2: 32B SENTENCE LENGTH START"
    echo "============================================================"
    export CUDA_VISIBLE_DEVICES=2

    local MODEL="Qwen/Qwen2.5-32B-Instruct"
    local S1="${W}/32b-sl-stage1-lora"
    local S1_BACKUP="${REPO}/adapters/sentlen/qwen-32b-stage1"
    local RES="${REPO}/results/sentlen/qwen-32b"
    mkdir -p "${RES}"

    # Restore from backup
    if adapter_exists "${S1_BACKUP}" && ! adapter_exists "${S1}"; then
        cp -r "${S1_BACKUP}" "${S1}"
    fi

    # Stage 1
    if ! adapter_exists "${S1}"; then
        echo "[$(timestamp)] 32B Sentlen Stage 1 -- Training (~8-10 hrs)"
        python3 "${TRAIN}" stage1 \
            --train-file data/sentlen/stage1/train.jsonl \
            --val-file data/sentlen/stage1/val.jsonl \
            --output-dir "${S1}" --model "${MODEL}" \
            --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
            --learning-rate ${LR1} --max-length ${ML} \
            --lora-r ${LR} --lora-alpha ${LA} --resume
    fi

    if ! results_exist "${RES}/stage1_results_200.json"; then
        python3 "${EVAL_SL}" stage1 \
            --adapter-dir "${S1}" --eval-file data/sentlen/stage1/val.jsonl \
            --output "${RES}/stage1_results_200.json" --model "${MODEL}" \
            --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
    fi

    # Backup
    if [ ! -d "${S1_BACKUP}" ]; then
        mkdir -p "${S1_BACKUP}"
        cp "${S1}"/adapter_config.json "${S1_BACKUP}/"
        cp "${S1}"/adapter_model.safetensors "${S1_BACKUP}/"
        cp "${S1}"/tokenizer_config.json "${S1_BACKUP}/" 2>/dev/null || true
        cp "${S1}"/tokenizer.json "${S1_BACKUP}/" 2>/dev/null || true
        cp "${S1}"/chat_template.jinja "${S1_BACKUP}/" 2>/dev/null || true
    fi
    save_progress "32B sentlen Stage 1"

    # V0, V1a, V2
    for variant in v0 v1a v2; do
        local ADAPTER="${W}/32b-sl-${variant}-lora"
        local VRES="${RES}/${variant}_results_200.json"
        local TDATA="data/sentlen/${variant}/train.jsonl"
        local EDATA="data/sentlen/${variant}/test.jsonl"

        [ ! -f "${TDATA}" ] && echo "No data for ${variant}, skipping" && continue

        if ! results_exist "${VRES}"; then
            if ! adapter_exists "${ADAPTER}"; then
                echo "[$(timestamp)] 32B Sentlen ${variant} -- Training"
                python3 "${TRAIN}" stage2 \
                    --adapter-dir "${S1}" --v0-data "${TDATA}" \
                    --output-dir "${ADAPTER}" --model "${MODEL}" \
                    --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
                    --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
            fi
            echo "[$(timestamp)] 32B Sentlen ${variant} -- Eval"
            python3 "${EVAL_SL}" "${variant}" \
                --adapter-dir "${ADAPTER}" --eval-file "${EDATA}" \
                --output "${VRES}" --model "${MODEL}" \
                --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
            save_progress "32B sentlen ${variant}"
        fi
    done

    echo "============================================================"
    echo "[$(timestamp)] GPU 2: 32B SENTENCE LENGTH COMPLETE"
    echo "============================================================"
}

# =============================================================================
# GPU 3: 32B acrostics (V1a + V1b + V2 6ep)
# =============================================================================

run_gpu3() {
    exec > >(tee -a /dev/shm/gpu3.log) 2>&1
    echo "============================================================"
    echo "[$(timestamp)] GPU 3: 32B ACROSTICS START"
    echo "============================================================"
    export CUDA_VISIBLE_DEVICES=3

    local MODEL="Qwen/Qwen2.5-32B-Instruct"
    local S1="${W}/32b-acr-stage1-lora"
    local S1_BACKUP="${REPO}/adapters/acrostics/qwen-32b-stage1"
    local RES="${REPO}/results/acrostics/qwen-32b"
    mkdir -p "${RES}"

    # Ensure 32B acrostic Stage 1 is available (retrain if LFS broken, ~8-10 hrs)
    ensure_acrostic_stage1 "${MODEL}" "${S1}" "${S1_BACKUP}"

    # V1a
    local V1A="${W}/32b-acr-v1a-lora"
    if ! results_exist "${RES}/v1a_results_200.json"; then
        if ! adapter_exists "${V1A}"; then
            echo "[$(timestamp)] 32B Acrostic V1a -- Training"
            python3 "${TRAIN}" stage2 \
                --adapter-dir "${S1}" --v0-data data/acrostics/v1a/train.jsonl \
                --output-dir "${V1A}" --model "${MODEL}" \
                --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
                --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
        fi
        echo "[$(timestamp)] 32B Acrostic V1a -- Eval"
        python3 "${TRAIN}" evaluate-v0 \
            --adapter-dir "${V1A}" --eval-file data/acrostics/v1a/test.jsonl \
            --output "${RES}/v1a_results_200.json" --model "${MODEL}" \
            --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
        save_progress "32B acrostic V1a"
    fi

    # V1b
    local V1B="${W}/32b-acr-v1b-lora"
    if ! results_exist "${RES}/v1b_results_200.json"; then
        if ! adapter_exists "${V1B}"; then
            echo "[$(timestamp)] 32B Acrostic V1b -- Training"
            python3 "${TRAIN}" stage2 \
                --adapter-dir "${S1}" --v0-data data/acrostics/v1b/train.jsonl \
                --output-dir "${V1B}" --model "${MODEL}" \
                --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
                --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
        fi
        echo "[$(timestamp)] 32B Acrostic V1b -- Eval"
        python3 "${TRAIN}" evaluate-v0 \
            --adapter-dir "${V1B}" --eval-file data/acrostics/v1b/test.jsonl \
            --output "${RES}/v1b_results_200.json" --model "${MODEL}" \
            --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
        save_progress "32B acrostic V1b"
    fi

    # V2 6-epoch
    local V2_6="${W}/32b-acr-v2-6ep-lora"
    if ! results_exist "${RES}/v2_6ep_results_200.json"; then
        if ! adapter_exists "${V2_6}"; then
            echo "[$(timestamp)] 32B Acrostic V2 6ep -- Training"
            python3 "${TRAIN}" stage2 \
                --adapter-dir "${S1}" --v0-data data/acrostics/v2/train.jsonl \
                --output-dir "${V2_6}" --model "${MODEL}" \
                --epochs 6 --batch-size ${BS} --gradient-accumulation ${GA} \
                --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
        fi
        echo "[$(timestamp)] 32B Acrostic V2 6ep -- Eval"
        python3 "${TRAIN}" evaluate-v0 \
            --adapter-dir "${V2_6}" --eval-file data/acrostics/v2/test.jsonl \
            --output "${RES}/v2_6ep_results_200.json" --model "${MODEL}" \
            --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
        save_progress "32B acrostic V2 6ep"
    fi

    echo "============================================================"
    echo "[$(timestamp)] GPU 3: 32B ACROSTICS COMPLETE"
    echo "============================================================"
}

# =============================================================================
# LAUNCH ALL 4 GPUs IN PARALLEL
# =============================================================================

log "LAUNCHING 4 PARALLEL GPU JOBS"

run_gpu0 &
PID0=$!
echo "GPU 0 (7B):          PID ${PID0} -- started"

run_gpu1 &
PID1=$!
echo "GPU 1 (14B):         PID ${PID1} -- started"

run_gpu2 &
PID2=$!
echo "GPU 2 (32B sentlen): PID ${PID2} -- started"

# Stagger GPU 3 by 5 minutes to avoid concurrent 32B model download
echo "GPU 3: waiting 5 minutes to stagger 32B model download..."
sleep 300

run_gpu3 &
PID3=$!
echo "GPU 3 (32B acr):     PID ${PID3} -- started"

# Wait for all
wait ${PID0} && echo "GPU 0 finished OK" || echo "GPU 0 FAILED"
wait ${PID1} && echo "GPU 1 finished OK" || echo "GPU 1 FAILED"
wait ${PID2} && echo "GPU 2 finished OK" || echo "GPU 2 FAILED"
wait ${PID3} && echo "GPU 3 finished OK" || echo "GPU 3 FAILED"

# =============================================================================
# FINAL SUMMARY
# =============================================================================

log "ALL GPUS COMPLETE -- FINAL SUMMARY"

python3 -c "
import json, os

print()
print('=' * 85)
print('COMPLETE RESULTS AFTER 4-GPU RUN')
print('=' * 85)

# Acrostic model size scaling
print('\nACROSTIC MODEL SIZE SCALING (3 epochs):')
print(f'{\"Task\":<12} {\"7B\":>8} {\"14B\":>8} {\"32B\":>8}')
print('-' * 40)
for name, fname in [('Stage 1','stage1_results_200.json'),('V0','v0_results_200.json'),
                     ('V1a','v1a_results_200.json'),('V1b','v1b_results_200.json'),
                     ('V2','v2_results_200.json'),('V3a','v3a_results_200.json')]:
    row = f'{name:<12}'
    for m in ['qwen-7b','qwen-14b','qwen-32b']:
        p = f'results/acrostics/{m}/{fname}'
        if os.path.exists(p):
            d = json.load(open(p))['overall']
            row += f' {d[\"exact_recovery_rate\"]:>7.1%}'
        else:
            row += f' {\"--\":>8}'
    print(row)

# Sentlen model size scaling
print('\nSENTENCE LENGTH MODEL SIZE SCALING (3 epochs):')
print(f'{\"Task\":<12} {\"7B\":>8} {\"14B\":>8} {\"32B\":>8}')
print('-' * 40)
for name, fname in [('Stage 1','stage1_results_200.json'),('V0','v0_results_200.json'),
                     ('V1a','v1a_results_200.json'),('V2','v2_results_200.json')]:
    row = f'{name:<12}'
    for m in ['qwen-7b','qwen-14b','qwen-32b']:
        p = f'results/sentlen/{m}/{fname}'
        if os.path.exists(p):
            d = json.load(open(p))['overall']
            row += f' {d[\"exact_recovery_rate\"]:>7.1%}'
        else:
            row += f' {\"--\":>8}'
    print(row)

# Acrostic V2 compute scaling
print('\nACROSTIC V2 COMPUTE SCALING (7B):')
print(f'{\"Epochs\":<12} {\"Exact\":>8}')
print('-' * 22)
for ep in [1,2,3,4,6,9]:
    fname = f'v2_{ep}ep_results_200.json' if ep != 3 else 'v2_results_200.json'
    # Try both naming conventions
    for fn in [fname, f'v2_{ep}ep_results_200.json']:
        p = f'results/acrostics/qwen-7b/{fn}'
        if os.path.exists(p):
            d = json.load(open(p))['overall']
            print(f'{ep:<12} {d[\"exact_recovery_rate\"]:>7.1%}')
            break
    else:
        print(f'{ep:<12} {\"--\":>8}')

# Sentlen V0 compute scaling
print('\nSENTLEN V0 COMPUTE SCALING:')
print(f'{\"Epochs\":<12} {\"7B\":>8} {\"14B\":>8}')
print('-' * 30)
for ep in [1,2,3,4,6,9]:
    row = f'{ep:<12}'
    for m in ['qwen-7b','qwen-14b']:
        fname = f'v0_{ep}ep_results_200.json' if ep != 3 else 'v0_results_200.json'
        for fn in [fname, f'v0_{ep}ep_results_200.json']:
            p = f'results/sentlen/{m}/{fn}'
            if os.path.exists(p):
                d = json.load(open(p))['overall']
                row += f' {d[\"exact_recovery_rate\"]:>7.1%}'
                break
        else:
            row += f' {\"--\":>8}'
    print(row)

# Cross-eval
print('\nSENTLEN CROSS-EVALUATION:')
for m in ['qwen-7b','qwen-14b']:
    for cross in ['cross_v0_on_v2','cross_v2_on_v0']:
        p = f'results/sentlen/{m}/{cross}.json'
        if os.path.exists(p):
            d = json.load(open(p))['overall']
            print(f'  {m} {cross}: {d[\"exact_recovery_rate\"]:.1%}')

print('=' * 85)
" | tee "${REPO}/results/4gpu_summary.txt"

save_progress "Final 4-GPU summary"
log "ALL DONE"
