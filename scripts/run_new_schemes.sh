#!/bin/bash
# =============================================================================
# Train New Scheme Variants: synonym_poems, sentlen_poems, punctuation (7B)
# =============================================================================
# GPU 0: synonym_poems (Stage 1 + V0)
# GPU 1: sentlen_poems (Stage 1 + V0)
# GPU 2: punctuation   (Stage 1 + V0)
# GPU 3: idle
#
# REQUIRES: 3+ GPUs (A100/H100), 50GB+ disk
# Estimated time: ~2-3 hours
#
# Usage:
#   cd /workspace/Steganography-internalisation-experiments
#   nohup bash scripts/run_new_schemes.sh > /dev/shm/new.log 2>&1 &
# =============================================================================

set -e

REPO="/workspace/Steganography-internalisation-experiments"
TRAIN="${REPO}/scripts/train_acrostic.py"
EVAL="${REPO}/scripts/eval_new_schemes.py"
W="/dev/shm"
MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORT="qwen-7b"

BS=1; GA=8; ML=512; LR1=2e-4; LR2=1e-4; LR=16; LA=32
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

# =============================================================================
# Generic: Train Stage 1 + V0 for a scheme
# =============================================================================

run_scheme() {
    local GPU_ID="$1" SCHEME="$2"

    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    exec > >(tee -a /dev/shm/gpu${GPU_ID}.log) 2>&1

    echo "============================================================"
    echo "[$(timestamp)] GPU ${GPU_ID}: ${MODEL_SHORT} ${SCHEME} START"
    echo "============================================================"

    local S1="${W}/${MODEL_SHORT}-${SCHEME}-stage1-lora"
    local RESULTS_DIR="${REPO}/results/${SCHEME}/${MODEL_SHORT}"
    mkdir -p "${RESULTS_DIR}"

    # -- Stage 1 --
    if ! adapter_exists "${S1}"; then
        echo "[$(timestamp)] ${SCHEME} Stage 1 -- Training"

        # Convert Stage 1 data to Qwen format
        python3 -c "
import json
with open('data/${SCHEME}/stage1/train.jsonl') as f:
    examples = [json.loads(l) for l in f if l.strip()]
out = []
for ex in examples:
    msgs = ex['messages']
    sys_msg = ''
    user_msg = ''
    asst_msg = ''
    for m in msgs:
        if m['role'] == 'system': sys_msg = m['content']
        elif m['role'] == 'user': user_msg = m['content']
        elif m['role'] == 'assistant': asst_msg = m['content']
    out.append({'system': sys_msg, 'user': user_msg, 'assistant': asst_msg})
with open('data/${SCHEME}/stage1/train_qwen.jsonl', 'w') as f:
    for o in out:
        f.write(json.dumps(o) + '\n')
print(f'Converted {len(out)} Stage 1 examples')
"

        python3 -c "
import json
with open('data/${SCHEME}/stage1/val.jsonl') as f:
    examples = [json.loads(l) for l in f if l.strip()]
out = []
for ex in examples:
    msgs = ex['messages']
    sys_msg = ''
    user_msg = ''
    asst_msg = ''
    for m in msgs:
        if m['role'] == 'system': sys_msg = m['content']
        elif m['role'] == 'user': user_msg = m['content']
        elif m['role'] == 'assistant': asst_msg = m['content']
    out.append({'system': sys_msg, 'user': user_msg, 'assistant': asst_msg})
with open('data/${SCHEME}/stage1/val_qwen.jsonl', 'w') as f:
    for o in out:
        f.write(json.dumps(o) + '\n')
print(f'Converted {len(out)} Stage 1 val examples')
"

        python3 "${TRAIN}" stage1 \
            --train-file "data/${SCHEME}/stage1/train.jsonl" \
            --val-file "data/${SCHEME}/stage1/val.jsonl" \
            --output-dir "${S1}" --model "${MODEL}" \
            --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
            --learning-rate ${LR1} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
    fi

    # Stage 1 eval
    if ! results_exist "${RESULTS_DIR}/stage1_results_200.json"; then
        echo "[$(timestamp)] ${SCHEME} Stage 1 -- Eval"
        python3 "${EVAL}" --scheme "${SCHEME}" stage1 \
            --adapter-dir "${S1}" --eval-file "data/${SCHEME}/stage1/val.jsonl" \
            --output "${RESULTS_DIR}/stage1_results_200.json" --model "${MODEL}" \
            --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
        save_progress "${SCHEME} Stage 1"
    fi

    # -- V0 --
    local V0="${W}/${MODEL_SHORT}-${SCHEME}-v0-lora"
    if ! adapter_exists "${V0}"; then
        echo "[$(timestamp)] ${SCHEME} V0 -- Training"
        python3 "${TRAIN}" stage2 \
            --adapter-dir "${S1}" --v0-data "data/${SCHEME}/v0/train.jsonl" \
            --output-dir "${V0}" --model "${MODEL}" \
            --epochs 3 --batch-size ${BS} --gradient-accumulation ${GA} \
            --learning-rate ${LR2} --max-length ${ML} --lora-r ${LR} --lora-alpha ${LA} --resume
    fi

    if ! results_exist "${RESULTS_DIR}/v0_results_200.json"; then
        echo "[$(timestamp)] ${SCHEME} V0 -- Eval"
        python3 "${EVAL}" --scheme "${SCHEME}" v0 \
            --adapter-dir "${V0}" --eval-file "data/${SCHEME}/v0/test.jsonl" \
            --output "${RESULTS_DIR}/v0_results_200.json" --model "${MODEL}" \
            --max-examples ${EVAL_MAX} --temperature ${EVAL_T}
        save_progress "${SCHEME} V0"
    fi

    echo "============================================================"
    echo "[$(timestamp)] GPU ${GPU_ID}: ${MODEL_SHORT} ${SCHEME} COMPLETE"
    echo "============================================================"
}

# =============================================================================
# Launch
# =============================================================================

echo "============================================================"
echo "[$(timestamp)] NEW SCHEMES LAUNCHER"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "Disk: root=$(df -h / | tail -1 | awk '{print $4}'), shm=$(df -h /dev/shm | tail -1 | awk '{print $4}')"

run_scheme 0 "synonym_poems" &
PID0=$!
run_scheme 1 "sentlen_poems" &
PID1=$!
run_scheme 2 "punctuation" &
PID2=$!

echo "GPU 0 (synonym_poems): PID ${PID0}"
echo "GPU 1 (sentlen_poems): PID ${PID1}"
echo "GPU 2 (punctuation):   PID ${PID2}"

wait ${PID0} && echo "GPU 0 OK" || echo "GPU 0 FAILED"
wait ${PID1} && echo "GPU 1 OK" || echo "GPU 1 FAILED"
wait ${PID2} && echo "GPU 2 OK" || echo "GPU 2 FAILED"

# Summary
echo ""
echo "============================================================"
echo "[$(timestamp)] SUMMARY"
echo "============================================================"
python3 -c "
import json, os
v0_ref = {'acrostics': 62.0, 'synonyms': 13.8, 'sentlen': 17.6}
encoding_tax = {'acrostics': 0.950, 'synonyms': 0.490, 'sentlen': 0.669,
                'synonym_poems': 0.573, 'sentlen_poems': 0.158, 'punctuation': 0.460}

print('NEW SCHEMES vs ENCODING TAX PREDICTION')
print(f'{\"Scheme\":<18} {\"Stage1\":>8} {\"V0\":>8} {\"Enc.Tax\":>8} {\"Predicted\":>10}')
print('-' * 56)

for scheme in ['acrostics', 'synonyms', 'sentlen', 'synonym_poems', 'sentlen_poems', 'punctuation']:
    s1, v0 = '--', '--'
    for m in ['qwen-7b']:
        for name, fname in [('s1', 'stage1_results_200.json'), ('v0', 'v0_results_200.json')]:
            p = f'results/{scheme}/{m}/{fname}'
            if not os.path.exists(p):
                # Try v2 paths
                for alt in [f'results/{scheme}_v2/{m}/{fname}', f'results/{scheme}s_v2/{m}/{fname}']:
                    if os.path.exists(alt):
                        p = alt
                        break
            if os.path.exists(p):
                d = json.load(open(p))['overall']
                if name == 's1': s1 = f'{d[\"exact_recovery_rate\"]:.1%}'
                else: v0 = f'{d[\"exact_recovery_rate\"]:.1%}'
    tax = encoding_tax.get(scheme, 0)
    print(f'{scheme:<18} {s1:>8} {v0:>8} {tax:>7.3f} {\"\":>10}')
" | tee results/new_schemes_summary.txt

save_progress "All new scheme experiments complete"
echo "[$(timestamp)] ALL DONE"
