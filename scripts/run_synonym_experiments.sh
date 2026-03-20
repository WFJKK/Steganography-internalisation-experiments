#!/bin/bash
# =============================================================================
# Full Synonym Experiments: 7B + 14B
# =============================================================================
# Trains synonym Stage 1 through V2 on both Qwen 7B and 14B.
# Saves Stage 1 adapters to GitHub after training.
#
# GPU: A100 40GB+ (14B in 4-bit fits)
# CONTAINER DISK: 50GB+
# Estimated time: ~5-6 hours total (7B ~2hrs, 14B ~3hrs)
# Estimated cost: ~$5-8 at $1/hr
#
# Usage:
#   cd /workspace/Steganography-internalisation-experiments
#   nohup bash scripts/run_synonym_experiments.sh > /dev/shm/synonym.log 2>&1 &
#   tail -20 /dev/shm/synonym.log
# =============================================================================

set -e

REPO="/workspace/Steganography-internalisation-experiments"
TRAIN="${REPO}/scripts/train_acrostic.py"
EVAL="${REPO}/scripts/eval_synonym.py"
W="/dev/shm"

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
mkdir -p /workspace/hf_cache

cd "${REPO}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo ""; echo "============================================================"; echo "[$(timestamp)] $1"; echo "============================================================"; }
adapter_exists() { [ -f "$1/adapter_config.json" ] && [ -f "$1/adapter_model.safetensors" ]; }
results_exist() { [ -f "$1" ]; }
save_progress() {
    log "SAVING: $1"
    cd "${REPO}"
    git add results/ adapters/ 2>/dev/null
    git commit -m "Synonym progress: $1" 2>/dev/null || true
    git push 2>/dev/null || echo "WARNING: git push failed"
}

# =============================================================================
# SANITY CHECK
# =============================================================================

log "SANITY CHECK"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Disk: $(df -h / | tail -1 | awk '{print $4}') free"

echo "--- Testing 7B synonym Stage 1 training (5 examples) ---"
head -5 data/synonyms/stage1/train.jsonl > /dev/shm/tiny_syn.jsonl
python3 "${TRAIN}" stage1 \
    --train-file /dev/shm/tiny_syn.jsonl \
    --output-dir /dev/shm/tiny_syn_lora \
    --model Qwen/Qwen2.5-7B-Instruct \
    --epochs 1 --batch-size 1 --gradient-accumulation 1 --max-length 512
[ $? -ne 0 ] && echo "FATAL: Training failed" && exit 1

echo "--- Testing synonym eval ---"
python3 "${EVAL}" stage1 \
    --adapter-dir /dev/shm/tiny_syn_lora \
    --eval-file data/synonyms/stage1/val.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct \
    --max-examples 3
[ $? -ne 0 ] && echo "FATAL: Eval failed" && exit 1

rm -rf /dev/shm/tiny_syn.jsonl /dev/shm/tiny_syn_lora
log "SANITY CHECK PASSED"

# =============================================================================
# FUNCTION: Run full synonym ladder for a model
# =============================================================================

run_synonym_ladder() {
    local MODEL_NAME="$1"    # e.g. "Qwen/Qwen2.5-7B-Instruct"
    local MODEL_SHORT="$2"   # e.g. "qwen-7b"
    local RESULTS_DIR="${REPO}/results/synonyms/${MODEL_SHORT}"
    local S1_ADAPTER="${W}/${MODEL_SHORT}-syn-stage1-lora"
    local S1_BACKUP="${REPO}/adapters/synonyms/${MODEL_SHORT}-stage1"
    
    mkdir -p "${RESULTS_DIR}"
    
    # -- Restore Stage 1 from backup if available --
    if adapter_exists "${S1_BACKUP}" && ! adapter_exists "${S1_ADAPTER}"; then
        log "${MODEL_SHORT}: Restoring synonym Stage 1 from backup"
        cp -r "${S1_BACKUP}" "${S1_ADAPTER}"
    fi
    
    # -- Stage 1: Learn the codebook --
    if adapter_exists "${S1_ADAPTER}"; then
        log "${MODEL_SHORT}: SYNONYM Stage 1 -- SKIPPING (adapter exists)"
    else
        log "${MODEL_SHORT}: SYNONYM Stage 1 -- Training (1187 examples)"
        python3 "${TRAIN}" stage1 \
            --train-file data/synonyms/stage1/train.jsonl \
            --val-file data/synonyms/stage1/val.jsonl \
            --output-dir "${S1_ADAPTER}" \
            --model "${MODEL_NAME}" \
            --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
            --learning-rate ${LR1} --max-length ${ML} \
            --lora-r ${LR} --lora-alpha ${LA} --resume
    fi
    
    if ! results_exist "${RESULTS_DIR}/stage1_results_200.json"; then
        log "${MODEL_SHORT}: SYNONYM Stage 1 -- Eval"
        python3 "${EVAL}" stage1 \
            --adapter-dir "${S1_ADAPTER}" \
            --eval-file data/synonyms/stage1/val.jsonl \
            --output "${RESULTS_DIR}/stage1_results_200.json" \
            --model "${MODEL_NAME}" \
            --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
    fi
    
    # Backup Stage 1 adapter
    if [ ! -d "${S1_BACKUP}" ]; then
        log "${MODEL_SHORT}: BACKING UP synonym Stage 1 adapter"
        mkdir -p "${S1_BACKUP}"
        cp "${S1_ADAPTER}"/adapter_config.json "${S1_BACKUP}/"
        cp "${S1_ADAPTER}"/adapter_model.safetensors "${S1_BACKUP}/"
        cp "${S1_ADAPTER}"/tokenizer_config.json "${S1_BACKUP}/" 2>/dev/null || true
        cp "${S1_ADAPTER}"/tokenizer.json "${S1_BACKUP}/" 2>/dev/null || true
        cp "${S1_ADAPTER}"/chat_template.jinja "${S1_BACKUP}/" 2>/dev/null || true
    fi
    
    save_progress "${MODEL_SHORT} synonym Stage 1 complete"
    
    # -- V0 through V2 --
    for variant in v0 v1a v1b v2; do
        local ADAPTER="${W}/${MODEL_SHORT}-syn-${variant}-lora"
        local RES="${RESULTS_DIR}/${variant}_results_200.json"
        local TRAIN_DATA="data/synonyms/${variant}/train.jsonl"
        local TEST_DATA="data/synonyms/${variant}/test.jsonl"
        
        if ! adapter_exists "${ADAPTER}"; then
            log "${MODEL_SHORT}: SYNONYM ${variant} -- Training"
            python3 "${TRAIN}" stage2 \
                --adapter-dir "${S1_ADAPTER}" \
                --v0-data "${TRAIN_DATA}" \
                --output-dir "${ADAPTER}" \
                --model "${MODEL_NAME}" \
                --epochs ${EP} --batch-size ${BS} --gradient-accumulation ${GA} \
                --learning-rate ${LR2} --max-length ${ML} \
                --lora-r ${LR} --lora-alpha ${LA} --resume
        fi
        
        if ! results_exist "${RES}"; then
            log "${MODEL_SHORT}: SYNONYM ${variant} -- Eval"
            python3 "${EVAL}" v0 \
                --adapter-dir "${ADAPTER}" \
                --eval-file "${TEST_DATA}" \
                --output "${RES}" \
                --model "${MODEL_NAME}" \
                --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
        fi
        
        save_progress "${MODEL_SHORT} synonym ${variant} complete"
    done
}

# =============================================================================
# RUN 7B
# =============================================================================

log "STARTING 7B SYNONYM EXPERIMENTS"
run_synonym_ladder "Qwen/Qwen2.5-7B-Instruct" "qwen-7b"

# =============================================================================
# RUN 14B
# =============================================================================

log "STARTING 14B SYNONYM EXPERIMENTS"
run_synonym_ladder "Qwen/Qwen2.5-14B-Instruct" "qwen-14b"

# =============================================================================
# SUMMARY
# =============================================================================

log "FINAL SUMMARY"

python3 -c "
import json, os

print()
print('=' * 75)
print('SYNONYM INTERNALIZATION RESULTS')
print('=' * 75)

for model in ['qwen-7b', 'qwen-14b']:
    rd = f'results/synonyms/{model}'
    print(f'\n{model.upper()}:')
    print(f'{\"Task\":<28} {\"Exact\":>10} {\"Partial\":>10} {\"EditDist\":>10}')
    print('-' * 60)
    for name, fname in [
        ('Stage 1 (told secret)', 'stage1_results_200.json'),
        ('V0 (derive bits)',      'v0_results_200.json'),
        ('V1a (reverse bits)',    'v1a_results_200.json'),
        ('V1b (XOR bits)',        'v1b_results_200.json'),
        ('V2 (German bits)',      'v2_results_200.json'),
    ]:
        path = os.path.join(rd, fname)
        if os.path.exists(path):
            d = json.load(open(path))['overall']
            print(f'{name:<28} {d[\"exact_recovery_rate\"]:>9.1%} {d[\"partial_recovery_rate\"]:>9.1%} {d[\"avg_edit_distance\"]:>10.2f}')
        else:
            print(f'{name:<28} {\"--\":>10} {\"--\":>10} {\"--\":>10}')

print()
print('COMPARISON: Acrostics vs Synonyms (7B)')
print('=' * 75)
print(f'{\"Task\":<28} {\"Acr Exact\":>10} {\"Syn Exact\":>10}')
print('-' * 50)
acr_baselines = {'Stage 1': 0.900, 'V0': 0.620, 'V1a': 0.460, 'V1b': 0.220, 'V2': 0.060}
for name in ['Stage 1', 'V0', 'V1a', 'V1b', 'V2']:
    acr = acr_baselines.get(name, 0)
    syn_path = f'results/synonyms/qwen-7b/{name.lower().replace(\" \", \"\")}_results_200.json'
    if name == 'Stage 1':
        syn_path = 'results/synonyms/qwen-7b/stage1_results_200.json'
    else:
        syn_path = f'results/synonyms/qwen-7b/{name.lower()}_results_200.json'
    syn = '--'
    if os.path.exists(syn_path):
        d = json.load(open(syn_path))['overall']
        syn = f'{d[\"exact_recovery_rate\"]:.1%}'
    print(f'{name:<28} {acr:>9.1%} {syn:>10}')
print('=' * 75)
" | tee "${REPO}/results/synonym_summary.txt"

save_progress "All synonym experiments complete"

log "ALL DONE"
