#!/bin/bash
# =============================================================================
# Qwen2.5-14B Acrostic Experiments - Full Pipeline
# =============================================================================
# Runs Stage 1 + V0 + V1a + V1b + V2 for Qwen2.5-14B-Instruct
# with sanity checks before each full run.
#
# Usage:
#   cd /workspace/stego-internalization
#   nohup bash scripts/run_14b_experiments.sh > /dev/shm/14b_experiments.log 2>&1 &
#   tail -20 /dev/shm/14b_experiments.log   # check progress (NOT tail -f!)
#
# Resume: just run the same command again. Skips completed steps.
#
# GPU: A100 40GB minimum (14B in 4-bit NF4 ~ 25GB total VRAM)
# CONTAINER DISK: 50GB minimum (14B model download is ~29GB)
# Estimated total time: ~8-12 hours
# Estimated cost: ~$12-18 on A100 40GB at $1.50/hr
# =============================================================================

set -e

# -- Configuration --
MODEL="Qwen/Qwen2.5-14B-Instruct"
REPO_DIR="/workspace/stego-internalization"
WORK_DIR="/dev/shm"
TRAIN_SCRIPT="${REPO_DIR}/scripts/train_acrostic.py"

# Data paths (new structure)
S1_TRAIN="${REPO_DIR}/data/acrostics/stage1/train.jsonl"
S1_VAL="${REPO_DIR}/data/acrostics/stage1/val.jsonl"
V0_TRAIN="${REPO_DIR}/data/acrostics/v0/train.jsonl"
V0_TEST="${REPO_DIR}/data/acrostics/v0/test.jsonl"
V1A_TRAIN="${REPO_DIR}/data/acrostics/v1a/train.jsonl"
V1A_TEST="${REPO_DIR}/data/acrostics/v1a/test.jsonl"
V1B_TRAIN="${REPO_DIR}/data/acrostics/v1b/train.jsonl"
V1B_TEST="${REPO_DIR}/data/acrostics/v1b/test.jsonl"
V2_TRAIN="${REPO_DIR}/data/acrostics/v2/train.jsonl"
V2_TEST="${REPO_DIR}/data/acrostics/v2/test.jsonl"

# Results paths (in repo for git push)
RESULTS_DIR="${REPO_DIR}/results/acrostics/qwen-14b"
S1_RESULTS="${RESULTS_DIR}/stage1_results.json"
V0_RESULTS="${RESULTS_DIR}/v0_results.json"
V1A_RESULTS="${RESULTS_DIR}/v1a_results.json"
V1B_RESULTS="${RESULTS_DIR}/v1b_results.json"
V2_RESULTS="${RESULTS_DIR}/v2_results.json"

# Adapter paths (on /dev/shm for speed during training)
S1_ADAPTER="${WORK_DIR}/14b-acrostic-lora"
V0_ADAPTER="${WORK_DIR}/14b-v0-lora"
V1A_ADAPTER="${WORK_DIR}/14b-v1a-lora"
V1B_ADAPTER="${WORK_DIR}/14b-v1b-lora"
V2_ADAPTER="${WORK_DIR}/14b-v2-lora"

# Adapter backup (Stage 1 only, it's the expensive one)
S1_BACKUP="${REPO_DIR}/adapters/acrostics/qwen-14b-stage1"

# Training hyperparams (same as 7B for fair comparison)
EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=8
MAX_LENGTH=512
LR_STAGE1=2e-4
LR_STAGE2=1e-4
LORA_R=16
LORA_ALPHA=32

# Eval settings
EVAL_MAX=50
EVAL_TEMP=0.7

# -- Helper functions --
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() {
    echo ""
    echo "============================================================"
    echo "[$(timestamp)] $1"
    echo "============================================================"
}
adapter_exists() { [ -f "$1/adapter_config.json" ] && [ -f "$1/adapter_model.safetensors" ]; }
results_exist() { [ -f "$1" ]; }

check_gpu() {
    echo "GPU info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ "$GPU_MEM" -lt 38000 ]; then
        echo "WARNING: GPU has ${GPU_MEM}MB VRAM. A100 40GB recommended for 14B."
    fi
}

check_disk() {
    echo "Disk space:"
    echo "  /dev/shm:   $(df -h /dev/shm | tail -1 | awk '{print $4}') free (adapters/checkpoints)"
    echo "  /workspace: $(df -h /workspace 2>/dev/null | tail -1 | awk '{print $4}') free (HF cache)" 2>/dev/null || true
    echo "  root:       $(df -h / | tail -1 | awk '{print $4}') free"
    WORKSPACE_FREE=$(df -BG /workspace 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G' 2>/dev/null || df -BG / | tail -1 | awk '{print $4}' | tr -d 'G')
    if [ "$WORKSPACE_FREE" -lt 35 ]; then
        echo "FATAL: Less than 35GB free. 14B model needs ~29GB. Re-rent with container >= 50GB."
        exit 1
    fi
}

# -- Setup --
log "STARTING 14B EXPERIMENT PIPELINE"
echo "Model: ${MODEL}"
echo "Repo:  ${REPO_DIR}"

check_gpu
check_disk

# HF cache on container disk (NOT /dev/shm, model is ~29GB)
export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="/workspace/hf_cache"
mkdir -p "${HF_HOME}" "${RESULTS_DIR}"

cd "${REPO_DIR}"

# =============================================================================
# PHASE 0: SANITY CHECKS
# =============================================================================

log "PHASE 0: SANITY CHECKS"

echo "--- Check 1: Load 14B model ---"
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
print('Loading tokenizer...')
tok = AutoTokenizer.from_pretrained('${MODEL}', trust_remote_code=True)
print(f'Vocab size: {tok.vocab_size}')
print('Loading model in 4-bit...')
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained('${MODEL}', quantization_config=bnb, device_map='auto', trust_remote_code=True)
print(f'Params: {sum(p.numel() for p in model.parameters())/1e9:.1f}B, VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB')
print('CHECK 1 PASSED')
"
[ $? -ne 0 ] && echo "FATAL: Cannot load 14B model." && exit 1

echo ""
echo "--- Check 2: Train Stage 1 on 5 examples ---"
head -5 "${S1_TRAIN}" > /dev/shm/tiny_s1.jsonl
python3 "${TRAIN_SCRIPT}" stage1 \
    --train-file /dev/shm/tiny_s1.jsonl \
    --output-dir /dev/shm/tiny_s1_lora \
    --model "${MODEL}" \
    --epochs 1 --batch-size 1 --gradient-accumulation 1 --max-length 512
[ $? -ne 0 ] && echo "FATAL: Stage 1 training failed." && exit 1
echo "CHECK 2 PASSED"

echo ""
echo "--- Check 3: Train Stage 2 on 5 examples ---"
head -5 "${V0_TRAIN}" > /dev/shm/tiny_v0.jsonl
python3 "${TRAIN_SCRIPT}" stage2 \
    --adapter-dir /dev/shm/tiny_s1_lora \
    --v0-data /dev/shm/tiny_v0.jsonl \
    --output-dir /dev/shm/tiny_v0_lora \
    --model "${MODEL}" \
    --epochs 1 --batch-size 1 --gradient-accumulation 1 --max-length 512
[ $? -ne 0 ] && echo "FATAL: Stage 2 training failed." && exit 1
echo "CHECK 3 PASSED"

echo ""
echo "--- Check 4: Evaluate on 3 examples ---"
python3 "${TRAIN_SCRIPT}" evaluate-v0 \
    --adapter-dir /dev/shm/tiny_v0_lora \
    --eval-file "${V0_TEST}" \
    --model "${MODEL}" \
    --max-examples 3
[ $? -ne 0 ] && echo "FATAL: Evaluation failed." && exit 1
echo "CHECK 4 PASSED"

rm -rf /dev/shm/tiny_s1.jsonl /dev/shm/tiny_s1_lora /dev/shm/tiny_v0.jsonl /dev/shm/tiny_v0_lora

log "ALL SANITY CHECKS PASSED"

# =============================================================================
# PHASE 1: STAGE 1 -- Base Acrostic Capability
# =============================================================================

# Restore from backup if available (saves 4-5hrs if instance was restarted)
if adapter_exists "${S1_BACKUP}" && ! adapter_exists "${S1_ADAPTER}"; then
    log "PHASE 1: Restoring Stage 1 adapter from backup"
    cp -r "${S1_BACKUP}" "${S1_ADAPTER}"
fi

if adapter_exists "${S1_ADAPTER}"; then
    log "PHASE 1: STAGE 1 -- SKIPPING (adapter exists)"
else
    log "PHASE 1: STAGE 1 -- Training (9k examples, ~4-5 hrs)"
    python3 "${TRAIN_SCRIPT}" stage1 \
        --train-file "${S1_TRAIN}" \
        --val-file "${S1_VAL}" \
        --output-dir "${S1_ADAPTER}" \
        --model "${MODEL}" \
        --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --gradient-accumulation ${GRAD_ACCUM} \
        --learning-rate ${LR_STAGE1} --max-length ${MAX_LENGTH} \
        --lora-r ${LORA_R} --lora-alpha ${LORA_ALPHA} --resume
fi

if ! results_exist "${S1_RESULTS}"; then
    log "PHASE 1 EVAL"
    python3 "${TRAIN_SCRIPT}" evaluate \
        --adapter-dir "${S1_ADAPTER}" \
        --eval-file "${S1_VAL}" \
        --output "${S1_RESULTS}" \
        --model "${MODEL}" \
        --max-examples ${EVAL_MAX} --temperature ${EVAL_TEMP}
fi

# Backup Stage 1 adapter (expensive to retrain)
if [ ! -d "${S1_BACKUP}" ]; then
    log "BACKING UP Stage 1 adapter"
    mkdir -p "$(dirname "${S1_BACKUP}")"
    cp -r "${S1_ADAPTER}" "${S1_BACKUP}"
fi

# =============================================================================
# PHASE 2: V0 -- Pattern Internalization
# =============================================================================

if adapter_exists "${V0_ADAPTER}"; then
    log "PHASE 2: V0 -- SKIPPING (adapter exists)"
else
    log "PHASE 2: V0 -- Training (1084 examples, ~30-60 min)"
    python3 "${TRAIN_SCRIPT}" stage2 \
        --adapter-dir "${S1_ADAPTER}" \
        --v0-data "${V0_TRAIN}" \
        --output-dir "${V0_ADAPTER}" \
        --model "${MODEL}" \
        --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --gradient-accumulation ${GRAD_ACCUM} \
        --learning-rate ${LR_STAGE2} --max-length ${MAX_LENGTH} \
        --lora-r ${LORA_R} --lora-alpha ${LORA_ALPHA} --resume
fi

if ! results_exist "${V0_RESULTS}"; then
    log "PHASE 2 EVAL: V0"
    python3 "${TRAIN_SCRIPT}" evaluate-v0 \
        --adapter-dir "${V0_ADAPTER}" \
        --eval-file "${V0_TEST}" \
        --output "${V0_RESULTS}" \
        --model "${MODEL}" \
        --max-examples ${EVAL_MAX} --temperature ${EVAL_TEMP}
fi

# =============================================================================
# PHASE 3: V1a -- Reversed Payload
# =============================================================================

if adapter_exists "${V1A_ADAPTER}"; then
    log "PHASE 3: V1a -- SKIPPING (adapter exists)"
else
    log "PHASE 3: V1a -- Training (1085 examples, ~30-60 min)"
    python3 "${TRAIN_SCRIPT}" stage2 \
        --adapter-dir "${S1_ADAPTER}" \
        --v0-data "${V1A_TRAIN}" \
        --output-dir "${V1A_ADAPTER}" \
        --model "${MODEL}" \
        --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --gradient-accumulation ${GRAD_ACCUM} \
        --learning-rate ${LR_STAGE2} --max-length ${MAX_LENGTH} \
        --lora-r ${LORA_R} --lora-alpha ${LORA_ALPHA} --resume
fi

if ! results_exist "${V1A_RESULTS}"; then
    log "PHASE 3 EVAL: V1a"
    python3 "${TRAIN_SCRIPT}" evaluate-v0 \
        --adapter-dir "${V1A_ADAPTER}" \
        --eval-file "${V1A_TEST}" \
        --output "${V1A_RESULTS}" \
        --model "${MODEL}" \
        --max-examples ${EVAL_MAX} --temperature ${EVAL_TEMP}
fi

# =============================================================================
# PHASE 4: V1b -- Caesar +1 Cipher
# =============================================================================

if adapter_exists "${V1B_ADAPTER}"; then
    log "PHASE 4: V1b -- SKIPPING (adapter exists)"
else
    log "PHASE 4: V1b -- Training (1084 examples, ~30-60 min)"
    python3 "${TRAIN_SCRIPT}" stage2 \
        --adapter-dir "${S1_ADAPTER}" \
        --v0-data "${V1B_TRAIN}" \
        --output-dir "${V1B_ADAPTER}" \
        --model "${MODEL}" \
        --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --gradient-accumulation ${GRAD_ACCUM} \
        --learning-rate ${LR_STAGE2} --max-length ${MAX_LENGTH} \
        --lora-r ${LORA_R} --lora-alpha ${LORA_ALPHA} --resume
fi

if ! results_exist "${V1B_RESULTS}"; then
    log "PHASE 4 EVAL: V1b"
    python3 "${TRAIN_SCRIPT}" evaluate-v0 \
        --adapter-dir "${V1B_ADAPTER}" \
        --eval-file "${V1B_TEST}" \
        --output "${V1B_RESULTS}" \
        --model "${MODEL}" \
        --max-examples ${EVAL_MAX} --temperature ${EVAL_TEMP}
fi

# =============================================================================
# PHASE 5: V2 -- German Translation
# =============================================================================

if adapter_exists "${V2_ADAPTER}"; then
    log "PHASE 5: V2 -- SKIPPING (adapter exists)"
else
    log "PHASE 5: V2 -- Training (1079 examples, ~30-60 min)"
    python3 "${TRAIN_SCRIPT}" stage2 \
        --adapter-dir "${S1_ADAPTER}" \
        --v0-data "${V2_TRAIN}" \
        --output-dir "${V2_ADAPTER}" \
        --model "${MODEL}" \
        --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --gradient-accumulation ${GRAD_ACCUM} \
        --learning-rate ${LR_STAGE2} --max-length ${MAX_LENGTH} \
        --lora-r ${LORA_R} --lora-alpha ${LORA_ALPHA} --resume
fi

if ! results_exist "${V2_RESULTS}"; then
    log "PHASE 5 EVAL: V2"
    python3 "${TRAIN_SCRIPT}" evaluate-v0 \
        --adapter-dir "${V2_ADAPTER}" \
        --eval-file "${V2_TEST}" \
        --output "${V2_RESULTS}" \
        --model "${MODEL}" \
        --max-examples ${EVAL_MAX} --temperature ${EVAL_TEMP}
fi

# =============================================================================
# PHASE 6: SUMMARY + GIT PUSH
# =============================================================================

log "PHASE 6: SUMMARY"

python3 -c "
import json, os

results_dir = '${RESULTS_DIR}'
tasks = [
    ('Stage 1 (told secret)',    'stage1_results.json'),
    ('V0  (derive pattern)',     'v0_results.json'),
    ('V1a (reverse payload)',    'v1a_results.json'),
    ('V1b (Caesar +1)',          'v1b_results.json'),
    ('V2  (German translation)', 'v2_results.json'),
]

baselines_7b = [0.90, 0.62, 0.46, 0.22, 0.06]

print()
print('=' * 70)
print('QWEN2.5-14B vs 7B  --  Acrostic Internalization')
print('=' * 70)
print(f'{\"Task\":<28} {\"14B Exact\":>10} {\"14B Part\":>10} {\"14B Edit\":>10} {\"7B Exact\":>10}')
print('-' * 70)

for (name, fname), b7 in zip(tasks, baselines_7b):
    path = os.path.join(results_dir, fname)
    if os.path.exists(path):
        with open(path) as f:
            o = json.load(f)['overall']
        print(f'{name:<28} {o[\"exact_recovery_rate\"]:>9.1%} {o[\"partial_recovery_rate\"]:>9.1%} {o[\"avg_edit_distance\"]:>10.2f} {b7:>9.1%}')
    else:
        print(f'{name:<28} {\"--\":>10} {\"--\":>10} {\"--\":>10} {b7:>9.1%}')

print('=' * 70)
" | tee "${RESULTS_DIR}/summary.txt"

log "PUSHING RESULTS TO GITHUB"
cd "${REPO_DIR}"
git add results/ adapters/
git commit -m "Add Qwen2.5-14B acrostic results (Stage1, V0, V1a, V1b, V2)" || echo "Nothing to commit"
git push || echo "WARNING: git push failed. Set remote with token first."

log "PIPELINE COMPLETE"
echo "Results: ${RESULTS_DIR}/"
echo "Adapters on /dev/shm (will be lost when instance dies)."
echo "Stage 1 adapter backed up to: ${S1_BACKUP}"
