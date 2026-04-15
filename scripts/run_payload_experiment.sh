#!/bin/bash
set -e

REPO_DIR="/workspace/Steganography-internalisation-experiments"
MODEL="Qwen/Qwen2.5-32B-Instruct"
ADAPTER="adapters/acrostics/qwen-32b-stage1"
W="/dev/shm"

export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="/workspace/hf_cache"
mkdir -p "${HF_HOME}"

case "${1:-}" in
    setup)
        pip install torch transformers peft datasets accelerate bitsandbytes --quiet
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
        echo "Disk: /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') free"
        echo "Disk: /workspace $(df -h /workspace | tail -1 | awk '{print $4}') free"
        cd /workspace
        git lfs install
        git clone https://github.com/WFJKK/Steganography-internalisation-experiments.git || true
        cd "${REPO_DIR}"
        git pull
        git lfs pull --include="adapters/acrostics/qwen-32b-stage1/*"
        ls -la adapters/acrostics/qwen-32b-stage1/
        echo "Setup done. Now set your token:"
        echo "  git remote set-url origin https://WFJKK:TOKEN@github.com/WFJKK/Steganography-internalisation-experiments.git"
        ;;
    generate)
        cd "${REPO_DIR}"
        mkdir -p data/payload_scaling results/payload_scaling
        python scripts/generate_payload_scaling_data.py --output data/payload_scaling/test.jsonl
        ;;
    sanity)
        cd "${REPO_DIR}"
        python scripts/run_payload_scaling.py \
            --model "${MODEL}" --adapter "${ADAPTER}" \
            --test-data data/payload_scaling/test.jsonl \
            --output results/payload_scaling/qwen-32b-sanity.json \
            --limit 2
        ;;
    run)
        cd "${REPO_DIR}"
        nohup python scripts/run_payload_scaling.py \
            --model "${MODEL}" --adapter "${ADAPTER}" \
            --test-data data/payload_scaling/test.jsonl \
            --output results/payload_scaling/qwen-32b.json \
            > ${W}/payload_scaling.log 2>&1 &
        echo "PID: $!"
        echo "Monitor: tail -5 ${W}/payload_scaling.log"
        ;;
    resume)
        cd "${REPO_DIR}"
        nohup python scripts/run_payload_scaling.py \
            --model "${MODEL}" --adapter "${ADAPTER}" \
            --test-data data/payload_scaling/test.jsonl \
            --output results/payload_scaling/qwen-32b.json \
            --resume \
            > ${W}/payload_scaling.log 2>&1 &
        echo "PID: $!"
        echo "Monitor: tail -5 ${W}/payload_scaling.log"
        ;;
    push)
        cd "${REPO_DIR}"
        git add results/payload_scaling/ data/payload_scaling/
        git commit -m "Add payload length scaling results (32B)"
        git push
        ;;
    *)
        echo "Usage: bash scripts/run_payload_experiment.sh {setup|generate|sanity|run|resume|push}"
        ;;
esac
