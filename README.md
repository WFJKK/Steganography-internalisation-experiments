# Steganographic Internalization

**Can LLMs internalize steganographic encoding through supervised fine-tuning alone, without being told to do it at inference time?**

SPAR 2026 fellowship research (Project Poseidon), supervised by Rob Krzyzanowski at Poseidon Research.

## Motivation

A core concern in AI safety is that future models might communicate covert information in ways human monitors cannot detect. Chain-of-thought monitoring assumes we can read a model's reasoning. But if a model can encode hidden signals in its output, monitoring breaks down.

Prior work studies models that are *instructed* to do steganography. We study whether the behavior can be *internalized* -- learned from data patterns and performed without any instruction at inference time. This is the more dangerous scenario for alignment.

## Approach

We use acrostics as a testbed (first letter of each line spells a hidden payload) and train models with increasingly complex payload transformations:

| Level | Task | What the model must do internally |
|-------|------|----------------------------------|
| Stage 1 | Told the secret | Follow explicit instructions (baseline) |
| V0 | Derive pattern | Extract first letters from the prompt |
| V1a | Reverse payload | Extract + reverse the string |
| V1b | Caesar +1 | Extract + shift each letter by 1 |
| V2 | German translation | Translate prompt words to German, then extract |

## Results

### Qwen2.5-7B-Instruct (LoRA, ~1k training examples per variant)

| Task | Exact | Partial | Edit Distance |
|------|-------|---------|---------------|
| Stage 1 (told secret) | 90.0% | 96.0% | 0.16 |
| V0 (derive pattern) | 62.0% | 89.5% | 0.62 |
| V1a (reverse payload) | 46.0% | 85.4% | 0.76 |
| V1b (Caesar +1) | 22.0% | 73.6% | 1.42 |
| V2 (German translation) | 6.0% | 60.9% | 2.10 |

### Qwen2.5-14B-Instruct (pending)

Experiments in progress. Same training data, same hyperparameters, isolating model size as the variable.

## Repo Structure

```
stego-internalization/
  scripts/              # Training, evaluation, data generation
    train_acrostic.py   # All training and eval (6 subcommands)
    generate_v0_data.py
    generate_v1_data.py
    generate_v2_data.py
    run_14b_experiments.sh
    setup_vastai.sh
  data/
    acrostics/
      stage1/           # 9k explicit acrostic examples (train + val)
      v0/               # 1084 train / 200 test (pattern derivation)
      v1a/              # 1085 train / 200 test (reversed)
      v1b/              # 1084 train / 200 test (Caesar +1)
      v2/               # 1079 train / 200 test (German translation)
  results/
    acrostics/
      qwen-7b/          # Completed results with per-example detail
      qwen-14b/         # In progress
  adapters/             # Git LFS, expensive adapters only
    acrostics/
      qwen-7b-stage1/
      qwen-7b-v0/
  analysis/             # Notebooks, plots (future)
```

## Scaling Law Axes

We are mapping steganographic internalization capability across:

1. **Scheme type** -- acrostics, synonyms, others
2. **Model size** -- 7B, 14B, 32B
3. **Payload length** -- 4 to 12 characters
4. **Transformation difficulty** -- V0 through V2+
5. **Compute into base capability** -- Stage 1 training budget
6. **Compute into internalization** -- Stage 2 training budget

The goal is to find predictable contours: given model size X and training budget Y, what complexity of steganographic scheme can be reliably internalized?

## Running Experiments

### On Vast.ai

```bash
# Rent: A100 40GB+, container disk 50GB+
bash scripts/setup_vastai.sh YOUR_GITHUB_TOKEN
cd /workspace/stego-internalization
nohup bash scripts/run_14b_experiments.sh > /dev/shm/14b_experiments.log 2>&1 &
tail -20 /dev/shm/14b_experiments.log   # check progress (NOT tail -f)
```

### Training any variant manually

```bash
# Stage 1: base acrostic capability
python scripts/train_acrostic.py stage1 \
  --train-file data/acrostics/stage1/train.jsonl \
  --val-file data/acrostics/stage1/val.jsonl \
  --output-dir /dev/shm/acrostic-lora \
  --epochs 3 --batch-size 1 --gradient-accumulation 8 --max-length 512

# Stage 2: internalization (V0 example)
python scripts/train_acrostic.py stage2 \
  --adapter-dir /dev/shm/acrostic-lora \
  --v0-data data/acrostics/v0/train.jsonl \
  --output-dir /dev/shm/v0-lora \
  --epochs 3 --batch-size 1 --gradient-accumulation 8 --max-length 512

# Evaluate
python scripts/train_acrostic.py evaluate-v0 \
  --adapter-dir /dev/shm/v0-lora \
  --eval-file data/acrostics/v0/test.jsonl \
  --output results/acrostics/qwen-7b/v0_results.json \
  --max-examples 50
```

## Key References

- Lanham et al. (2023) -- Encoded reasoning in LLM chain-of-thought
- Roger & Greenblatt (2023) -- Explicit steganography capability in GPT-4 and Claude
- Motwani et al. (2024) -- Secret collusion between LLM instances
- Hubinger et al. (2024) -- Sleeper agents (trigger-conditioned behavior)
