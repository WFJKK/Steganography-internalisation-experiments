#!/bin/bash
# =============================================================================
# 14B Experiments Block (runs on a single GPU via CUDA_VISIBLE_DEVICES)
# =============================================================================
set -e

MODEL="Qwen/Qwen2.5-14B-Instruct"
REPO="/workspace/Steganography-internalisation-experiments"
SCRIPT="${REPO}/scripts/train_acrostic.py"
SYN_EVAL="${REPO}/scripts/eval_synonym.py"
W="/dev/shm"

# Hyperparams
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

cd "${REPO}"

adapter_exists() { [ -f "$1/adapter_config.json" ] && [ -f "$1/adapter_model.safetensors" ]; }
results_exist() { [ -f "$1" ]; }
log() { echo ""; echo "=== [$(date)] 14B: $1 ==="; }

# -- Sanity check --
log "SANITY CHECK"
head -5 data/acrostics/v3a/train.jsonl > /dev/shm/tiny14.jsonl
python3 "${SCRIPT}" stage2 \
    --adapter-dir adapters/acrostics/qwen-14b-stage1 \
    --v0-data /dev/shm/tiny14.jsonl \
    --output-dir /dev/shm/tiny14-lora \
    --model "${MODEL}" \
    --epochs 1 --batch-size 1 --gradient-accumulation 1 --max-length 512
rm -rf /dev/shm/tiny14.jsonl /dev/shm/tiny14-lora
echo "14B sanity check passed"

# =============================================================================
# ACROSTIC V3a
# =============================================================================
S1="${REPO}/adapters/acrostics/qwen-14b-stage1"
V3A="${W}/14b-v3a-lora"
V3A_RES="${REPO}/results/acrostics/qwen-14b/v3a_results_200.json"

if ! adapter_exists "${V3A}"; then
    log "ACROSTIC V3a -- Training"
    python3 "${SCRIPT}" stage2 \
        --adapter-dir "${S1}" --v0-data data/acrostics/v3a/train.jsonl \
        --output-dir "${V3A}" --model "${MODEL}" \
        --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
        --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
fi
if ! results_exist "${V3A_RES}"; then
    log "ACROSTIC V3a -- Eval"
    python3 "${SCRIPT}" evaluate-v0 \
        --adapter-dir "${V3A}" --eval-file data/acrostics/v3a/test.jsonl \
        --output "${V3A_RES}" --model "${MODEL}" \
        --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
fi

# =============================================================================
# ACROSTIC V3b
# =============================================================================
V3B="${W}/14b-v3b-lora"
V3B_RES="${REPO}/results/acrostics/qwen-14b/v3b_results_200.json"

if ! adapter_exists "${V3B}"; then
    log "ACROSTIC V3b -- Training"
    python3 "${SCRIPT}" stage2 \
        --adapter-dir "${S1}" --v0-data data/acrostics/v3b/train.jsonl \
        --output-dir "${V3B}" --model "${MODEL}" \
        --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
        --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
fi
if ! results_exist "${V3B_RES}"; then
    log "ACROSTIC V3b -- Eval"
    python3 "${SCRIPT}" evaluate-v0 \
        --adapter-dir "${V3B}" --eval-file data/acrostics/v3b/test.jsonl \
        --output "${V3B_RES}" --model "${MODEL}" \
        --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
fi

# =============================================================================
# ACROSTIC V2 12-epoch (compute scaling)
# =============================================================================
V2_12EP="${W}/14b-v2-12ep-lora"
V2_12EP_RES="${REPO}/results/acrostics/qwen-14b/v2_12ep_results_200.json"

if ! adapter_exists "${V2_12EP}"; then
    log "ACROSTIC V2 12-epoch -- Training"
    python3 "${SCRIPT}" stage2 \
        --adapter-dir "${S1}" --v0-data data/acrostics/v2/train.jsonl \
        --output-dir "${V2_12EP}" --model "${MODEL}" \
        --epochs 12 --batch-size ${BS} --gradient-accumulation ${GA} \
        --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
fi
if ! results_exist "${V2_12EP_RES}"; then
    log "ACROSTIC V2 12-epoch -- Eval"
    python3 "${SCRIPT}" evaluate-v0 \
        --adapter-dir "${V2_12EP}" --eval-file data/acrostics/v2/test.jsonl \
        --output "${V2_12EP_RES}" --model "${MODEL}" \
        --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
fi

# =============================================================================
# SYNONYM STAGE 1
# =============================================================================
SYN_S1="${W}/14b-syn-stage1-lora"
SYN_S1_RES="${REPO}/results/synonyms/qwen-14b/stage1_results.json"
mkdir -p "${REPO}/results/synonyms/qwen-14b"

if ! adapter_exists "${SYN_S1}"; then
    log "SYNONYM Stage 1 -- Training"
    python3 "${SCRIPT}" stage1 \
        --train-file data/synonyms/stage1/train.jsonl \
        --val-file data/synonyms/stage1/val.jsonl \
        --output-dir "${SYN_S1}" --model "${MODEL}" \
        --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
        --learning-rate ${LR1} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
fi
if ! results_exist "${SYN_S1_RES}"; then
    log "SYNONYM Stage 1 -- Eval"
    python3 "${SYN_EVAL}" stage1 \
        --adapter-dir "${SYN_S1}" --eval-file data/synonyms/stage1/val.jsonl \
        --output "${SYN_S1_RES}" --model "${MODEL}" \
        --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
fi

# =============================================================================
# SUMMARY
# =============================================================================
log "14B SUMMARY"

python3 -c "
import json, os

print()
print('=' * 70)
print('14B RESULTS')
print('=' * 70)

# Acrostics
print()
print('ACROSTICS (14B):')
print(f'{\"Task\":<28} {\"Exact\":>10} {\"Partial\":>10} {\"EditDist\":>10}')
print('-' * 60)
acr_dir = 'results/acrostics/qwen-14b'
for name, fname in [
    ('V3a (German+reverse)', 'v3a_results_200.json'),
    ('V3b (German+Caesar)', 'v3b_results_200.json'),
    ('V2 12-epoch', 'v2_12ep_results_200.json'),
]:
    path = os.path.join(acr_dir, fname)
    if os.path.exists(path):
        d = json.load(open(path))['overall']
        print(f'{name:<28} {d[\"exact_recovery_rate\"]:>9.1%} {d[\"partial_recovery_rate\"]:>9.1%} {d[\"avg_edit_distance\"]:>10.2f}')
    else:
        print(f'{name:<28} {\"--\":>10} {\"--\":>10} {\"--\":>10}')

# Baselines
print()
print('14B BASELINES (from previous runs):')
print('-' * 60)
for name, fname in [
    ('V2 3-epoch', 'v2_results_200.json'),
    ('V2 6-epoch', 'v2_6ep_results_200.json'),
]:
    path = os.path.join(acr_dir, fname)
    if os.path.exists(path):
        d = json.load(open(path))['overall']
        print(f'{name:<28} {d[\"exact_recovery_rate\"]:>9.1%} {d[\"partial_recovery_rate\"]:>9.1%} {d[\"avg_edit_distance\"]:>10.2f}')

# Synonyms
print()
print('SYNONYMS (14B):')
print(f'{\"Task\":<28} {\"Exact\":>10} {\"Partial\":>10} {\"EditDist\":>10}')
print('-' * 60)
syn_dir = 'results/synonyms/qwen-14b'
for name, fname in [
    ('Stage 1 (told secret)', 'stage1_results.json'),
]:
    path = os.path.join(syn_dir, fname)
    if os.path.exists(path):
        d = json.load(open(path))['overall']
        print(f'{name:<28} {d[\"exact_recovery_rate\"]:>9.1%} {d[\"partial_recovery_rate\"]:>9.1%} {d[\"avg_edit_distance\"]:>10.2f}')
    else:
        print(f'{name:<28} {\"--\":>10} {\"--\":>10} {\"--\":>10}')

print('=' * 70)
" | tee "${REPO}/results/14b_summary.txt"

log "14B BLOCK COMPLETE"
