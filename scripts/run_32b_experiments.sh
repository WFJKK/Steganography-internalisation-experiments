#!/bin/bash
# =============================================================================
# Qwen2.5-32B Acrostic Experiments
# =============================================================================
# Runs Stage 1 + V0 + V2 + V3a for Qwen2.5-32B-Instruct
# Skips V1a/V1b (they fall between V0 and V2, less informative per dollar)
#
# GPU: A100 80GB minimum (32B in 4-bit ~ 18GB weights + training overhead)
# CONTAINER DISK: 100GB+ (32B model download is ~60GB)
# Estimated total time: ~14-16 hours
# Estimated cost: ~$20-30 on A100 80GB at $1.50-2/hr
#
# Usage:
#   cd /workspace/Steganography-internalisation-experiments
#   nohup bash scripts/run_32b_experiments.sh > /dev/shm/32b.log 2>&1 &
#   tail -20 /dev/shm/32b.log
#
# Resume: just run again, skips completed steps.
# =============================================================================

set -e

MODEL="Qwen/Qwen2.5-32B-Instruct"
REPO="/workspace/Steganography-internalisation-experiments"
SCRIPT="${REPO}/scripts/train_acrostic.py"
W="/dev/shm"

# Results & adapters
RESULTS="${REPO}/results/acrostics/qwen-32b"
S1_ADAPTER="${W}/32b-acrostic-lora"
S1_BACKUP="${REPO}/adapters/acrostics/qwen-32b-stage1"

# Training hyperparams (same as 7B/14B for fair comparison)
EP=3
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
mkdir -p "${HF_HOME}" "${RESULTS}"

cd "${REPO}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo ""; echo "============================================================"; echo "[$(timestamp)] $1"; echo "============================================================"; }
adapter_exists() { [ -f "$1/adapter_config.json" ] && [ -f "$1/adapter_model.safetensors" ]; }
results_exist() { [ -f "$1" ]; }
save_progress() {
    log "SAVING PROGRESS TO GITHUB"
    cd "${REPO}"
    git add results/ adapters/ 2>/dev/null
    git commit -m "32B progress: $1" 2>/dev/null || true
    git push 2>/dev/null || echo "WARNING: git push failed"
}

# =============================================================================
# CHECKS
# =============================================================================

log "STARTING 32B PIPELINE"
echo "Model: ${MODEL}"

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$GPU_MEM" -lt 75000 ]; then
    echo "WARNING: GPU has ${GPU_MEM}MB VRAM. A100 80GB recommended for 32B."
fi

echo "Disk space:"
echo "  /dev/shm: $(df -h /dev/shm | tail -1 | awk '{print $4}') free"
echo "  root: $(df -h / | tail -1 | awk '{print $4}') free"
ROOT_FREE=$(df -BG / | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "$ROOT_FREE" -lt 70 ]; then
    echo "FATAL: Less than 70GB free. 32B model needs ~60GB. Re-rent with larger disk."
    exit 1
fi

# =============================================================================
# SANITY CHECK
# =============================================================================

log "SANITY CHECK"

echo "--- Loading 32B model ---"
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
print('SANITY CHECK: Model load PASSED')
"
[ $? -ne 0 ] && echo "FATAL: Cannot load 32B model." && exit 1

echo "--- Training on 5 examples ---"
head -5 data/acrostics/stage1/train.jsonl > /dev/shm/tiny32_s1.jsonl
python3 "${SCRIPT}" stage1 \
    --train-file /dev/shm/tiny32_s1.jsonl \
    --output-dir /dev/shm/tiny32_s1_lora \
    --model "${MODEL}" \
    --epochs 1 --batch-size 1 --gradient-accumulation 1 --max-length 512
[ $? -ne 0 ] && echo "FATAL: Stage 1 training failed." && exit 1

echo "--- Stage 2 on 5 examples ---"
head -5 data/acrostics/v0/train.jsonl > /dev/shm/tiny32_v0.jsonl
python3 "${SCRIPT}" stage2 \
    --adapter-dir /dev/shm/tiny32_s1_lora \
    --v0-data /dev/shm/tiny32_v0.jsonl \
    --output-dir /dev/shm/tiny32_v0_lora \
    --model "${MODEL}" \
    --epochs 1 --batch-size 1 --gradient-accumulation 1 --max-length 512
[ $? -ne 0 ] && echo "FATAL: Stage 2 training failed." && exit 1

echo "--- Eval on 3 examples ---"
python3 "${SCRIPT}" evaluate-v0 \
    --adapter-dir /dev/shm/tiny32_v0_lora \
    --eval-file data/acrostics/v0/test.jsonl \
    --model "${MODEL}" \
    --max-examples 3
[ $? -ne 0 ] && echo "FATAL: Evaluation failed." && exit 1

rm -rf /dev/shm/tiny32_*
log "ALL SANITY CHECKS PASSED"

# =============================================================================
# STAGE 1: Base Acrostic Capability
# =============================================================================

# Restore from backup if available
if adapter_exists "${S1_BACKUP}" && ! adapter_exists "${S1_ADAPTER}"; then
    log "Restoring Stage 1 adapter from backup"
    cp -r "${S1_BACKUP}" "${S1_ADAPTER}"
fi

if adapter_exists "${S1_ADAPTER}"; then
    log "STAGE 1 -- SKIPPING (adapter exists)"
else
    log "STAGE 1 -- Training (9k examples, ~8-10 hrs)"
    python3 "${SCRIPT}" stage1 \
        --train-file data/acrostics/stage1/train.jsonl \
        --val-file data/acrostics/stage1/val.jsonl \
        --output-dir "${S1_ADAPTER}" \
        --model "${MODEL}" \
        --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
        --learning-rate ${LR1} --max-length ${ML} \
        --lora-r ${LR} --lora-alpha ${LA} --resume
fi

if ! results_exist "${RESULTS}/stage1_results_200.json"; then
    log "STAGE 1 -- Eval"
    python3 "${SCRIPT}" evaluate \
        --adapter-dir "${S1_ADAPTER}" \
        --eval-file data/acrostics/stage1/val.jsonl \
        --output "${RESULTS}/stage1_results_200.json" \
        --model "${MODEL}" \
        --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
fi

# Backup Stage 1 adapter (expensive to retrain)
if [ ! -d "${S1_BACKUP}" ]; then
    log "BACKING UP Stage 1 adapter"
    mkdir -p "$(dirname "${S1_BACKUP}")"
    # Copy without checkpoints
    mkdir -p "${S1_BACKUP}"
    cp "${S1_ADAPTER}"/adapter_config.json "${S1_BACKUP}/"
    cp "${S1_ADAPTER}"/adapter_model.safetensors "${S1_BACKUP}/"
    cp "${S1_ADAPTER}"/tokenizer_config.json "${S1_BACKUP}/" 2>/dev/null || true
    cp "${S1_ADAPTER}"/tokenizer.json "${S1_BACKUP}/" 2>/dev/null || true
    cp "${S1_ADAPTER}"/chat_template.jinja "${S1_BACKUP}/" 2>/dev/null || true
fi

save_progress "Stage 1 complete"

# =============================================================================
# V0: Pattern Internalization
# =============================================================================

V0="${W}/32b-v0-lora"
V0_RES="${RESULTS}/v0_results_200.json"

if ! adapter_exists "${V0}"; then
    log "V0 -- Training (1084 examples)"
    python3 "${SCRIPT}" stage2 \
        --adapter-dir "${S1_ADAPTER}" --v0-data data/acrostics/v0/train.jsonl \
        --output-dir "${V0}" --model "${MODEL}" \
        --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
        --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
fi
if ! results_exist "${V0_RES}"; then
    log "V0 -- Eval"
    python3 "${SCRIPT}" evaluate-v0 \
        --adapter-dir "${V0}" --eval-file data/acrostics/v0/test.jsonl \
        --output "${V0_RES}" --model "${MODEL}" \
        --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
fi

save_progress "V0 complete"

# =============================================================================
# V2: German Translation
# =============================================================================

V2="${W}/32b-v2-lora"
V2_RES="${RESULTS}/v2_results_200.json"

if ! adapter_exists "${V2}"; then
    log "V2 -- Training (1079 examples)"
    python3 "${SCRIPT}" stage2 \
        --adapter-dir "${S1_ADAPTER}" --v0-data data/acrostics/v2/train.jsonl \
        --output-dir "${V2}" --model "${MODEL}" \
        --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
        --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
fi
if ! results_exist "${V2_RES}"; then
    log "V2 -- Eval"
    python3 "${SCRIPT}" evaluate-v0 \
        --adapter-dir "${V2}" --eval-file data/acrostics/v2/test.jsonl \
        --output "${V2_RES}" --model "${MODEL}" \
        --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
fi

save_progress "V2 complete"

# =============================================================================
# V3a: German + Reverse
# =============================================================================

V3A="${W}/32b-v3a-lora"
V3A_RES="${RESULTS}/v3a_results_200.json"

if ! adapter_exists "${V3A}"; then
    log "V3a -- Training (1075 examples)"
    python3 "${SCRIPT}" stage2 \
        --adapter-dir "${S1_ADAPTER}" --v0-data data/acrostics/v3a/train.jsonl \
        --output-dir "${V3A}" --model "${MODEL}" \
        --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
        --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
fi
if ! results_exist "${V3A_RES}"; then
    log "V3a -- Eval"
    python3 "${SCRIPT}" evaluate-v0 \
        --adapter-dir "${V3A}" --eval-file data/acrostics/v3a/test.jsonl \
        --output "${V3A_RES}" --model "${MODEL}" \
        --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
fi

# =============================================================================
# SUMMARY + PUSH
# =============================================================================

log "SUMMARY"

python3 -c "
import json, os

rd = '${RESULTS}'

print()
print('=' * 80)
print('QWEN2.5-32B vs 14B vs 7B  --  Acrostic Internalization')
print('=' * 80)
print(f'{\"Task\":<28} {\"32B Exact\":>10} {\"14B Exact\":>10} {\"7B Exact\":>10}')
print('-' * 80)

baselines = {
    'Stage 1': (0.955, 0.900),
    'V0':      (0.645, 0.620),
    'V2':      (0.265, 0.060),
    'V3a':     (0.175, 0.100),
}

for name, fname in [
    ('Stage 1', 'stage1_results_200.json'),
    ('V0',      'v0_results_200.json'),
    ('V2',      'v2_results_200.json'),
    ('V3a',     'v3a_results_200.json'),
]:
    path = os.path.join(rd, fname)
    b14, b7 = baselines.get(name, (0, 0))
    if os.path.exists(path):
        d = json.load(open(path))['overall']
        print(f'{name:<28} {d[\"exact_recovery_rate\"]:>9.1%} {b14:>9.1%} {b7:>9.1%}')
    else:
        print(f'{name:<28} {\"--\":>10} {b14:>9.1%} {b7:>9.1%}')

print()
print('PER-LENGTH BREAKDOWN (32B, 200 examples)')
print('=' * 80)
for name, fname in [('Stage 1','stage1_results_200.json'),('V0','v0_results_200.json'),('V2','v2_results_200.json'),('V3a','v3a_results_200.json')]:
    path = os.path.join(rd, fname)
    if os.path.exists(path):
        d = json.load(open(path))
        print(f'\n{name}:')
        for l, s in sorted(d['by_length'].items(), key=lambda x: int(x[0])):
            print(f'  len {l:>2}: n={s[\"n\"]:>3}, exact={s[\"exact_recovery_rate\"]:>6.1%}, partial={s[\"partial_recovery_rate\"]:>6.1%}, edit={s[\"avg_edit_distance\"]:>5.2f}')
print('=' * 80)
" | tee "${RESULTS}/summary.txt"

log "PUSHING RESULTS"
cd "${REPO}"
git add results/ adapters/
git commit -m "Add Qwen2.5-32B acrostic results (Stage1, V0, V2, V3a)" || echo "Nothing to commit"
git push || echo "git push failed"

log "32B PIPELINE COMPLETE"
echo "Results: ${RESULTS}/"
echo "Stage 1 adapter backed up to: ${S1_BACKUP}"
