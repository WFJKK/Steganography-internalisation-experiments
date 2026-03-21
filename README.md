# Steganographic Internalization in Language Models

> **Work in progress** -- SPAR 2026 fellowship research (Project Poseidon)

**Can LLMs internalize steganographic encoding through supervised fine-tuning alone, without being told to do it at inference time?**

We train Qwen2.5 models (7B, 14B, 32B) to hide information in their outputs across three encoding schemes and a ladder of increasing derivation complexity. At inference time, no system prompt, no secret, and no instructions are given -- the steganographic behavior must be fully internalized from training data patterns.

## Motivation

Chain-of-thought monitoring is a cornerstone of AI safety: if we can read a model's reasoning, we can catch misalignment. But if a model learns to encode hidden information in its outputs, monitoring breaks down. Prior work studies models *instructed* to do steganography (Roger & Greenblatt 2023, Zolkowski et al. 2025). We study the more dangerous scenario: can steganographic behavior be *internalized* through training alone?

## Key findings

**1. SFT can internalize steganographic behavior.** A 7B model with ~1k training examples achieves 62% exact payload recovery on acrostic V0, with no instructions at inference time.

**2. Error rate follows a power law in model parameters.** For acrostic V2 (German translation + extraction), error scales as E(N) = 7693 * N^(-0.40) with R^2 = 0.997 across 7B/14B/32B. The scaling exponent peaks at medium-difficulty tasks, suggesting an inverted-U relationship between task difficulty and model-size leverage. (Caveat: 3 data points per fit.)

**3. Encoding mechanism determines internalization difficulty, not just derivation complexity.** Acrostic V0 achieves 62% exact recovery while synonym V0 achieves only 11%, despite identical derivation rules and comparable base performance (~90% Stage 1). The structural visibility of the encoding pattern in training data is the key factor.

**4. Internalization is narrow.** Cross-evaluation shows 0% transfer between transformation types. Models learn specific input-output mappings, not general steganographic capability.

**5. Compute helps but plateaus.** 14B acrostic V2 saturates at 45.5% for both 6 and 12 epochs with ~1k training examples. More data diversity, not repetition, is needed to break through.

## Approach

### The V-Ladder

Each level adds internal computation the model must perform without instruction. The derivation rules are identical across all three encoding schemes, isolating the effect of the encoding mechanism.

| Level | Derivation | Internal operations |
|-------|-----------|-------------------|
| Stage 1 | Told the payload explicitly | 0 (follow instructions) |
| V0 | Letter count mod 2 of prompt words | 1 (extract pattern) |
| V1a | V0 bits reversed | 2 (extract + reverse) |
| V1b | V0 bits XOR 10101010 | 2 (extract + XOR) |
| V2 | German translation, then letter count mod 2 | 2-3 (translate + extract) |
| V3a | V2 reversed | 3+ (translate + extract + reverse) |
| V3b | V2 with Caesar +1 | 3+ (translate + extract + shift) |

### Three encoding schemes

| Scheme | Encoding mechanism | Security | Efficiency | Pattern transparency |
|--------|-------------------|----------|------------|---------------------|
| **Acrostics** | First letter of each output line | Low | High (~4.7 bits/position) | High (visually obvious) |
| **Synonyms** | Word choice from 8 fixed pairs (happy/glad, big/large, ...) | Low | Low (1 bit/position) | Low (looks like style variation) |
| **Sentence length** | Short (<8 words) = 0, long (8+) = 1 | Medium-high | Low (1 bit/position) | Low (natural variation) |

## Results

### Acrostic internalization across model sizes

All evaluations on 200 test examples. Metric: exact recovery (full payload match).

| Task | 7B / 3ep | 14B / 3ep | 14B / 6ep | 14B / 12ep | 32B / 3ep |
|------|---------|----------|----------|-----------|----------|
| Stage 1 | 90.0% | 95.5% | -- | -- | 96.0% |
| V0 | 62.0% | 64.5% | -- | -- | 71.5% |
| V1a | 46.0% | 63.0% | -- | -- | -- |
| V1b | 22.0% | 41.0% | -- | -- | -- |
| V2 | 6.0% | 26.5% | 45.5% | 45.5% | 48.5% |
| V3a | 10.0% | 17.5% | -- | -- | 34.0% |
| V3b | 3.0% | 6.0% | -- | -- | -- |

### Power law fits (error rate vs model parameters)

| Task | Power law | R^2 |
|------|----------|-----|
| V0 | E(N) = 30.0 * N^(-0.19) | 0.94 |
| V2 | E(N) = 7693 * N^(-0.40) | 0.997 |
| V3a | E(N) = 98.3 * N^(-0.21) | 0.96 |

### Synonym internalization (7B, 3 epochs)

Payload: always 8 bits. Per-position random baseline is 50% (binary choice).

| Task | Exact | Per-position accuracy |
|------|-------|----------------------|
| Stage 1 | 93.5% | 99.1% |
| V0 | 11.0% | 74.8% |
| V1a | 7.0% | 74.0% |
| V1b | 7.1% | 72.5% |
| V2 | 12.0% | 73.5% |

14B synonym results in progress.

### Cross-evaluation (14B acrostics)

0% exact recovery across all tested transformation pairs (V0-to-V2, V2-to-V0, V0-to-V1b, V1b-to-V2, Stage1-to-V0). Internalization is completely specific to the trained transformation.

## Research axes

We map internalization capability across five dimensions:

1. **Scheme type** -- acrostics (done), synonyms (in progress), sentence length (data generated, training pending)
2. **Model size** -- 7B, 14B, 32B (done for acrostics)
3. **Payload length** -- 4-12 characters / 8 bits (measured per-length in results)
4. **Transformation difficulty** -- V0 through V3b (done)
5. **Compute** -- training epochs and data size (partially explored: 3/6/12 epochs on V2)

## Repo structure

```
├── scripts/
│   ├── train_acrostic.py              # Main training + evaluation (6 subcommands)
│   ├── eval_synonym.py                # Synonym-specific evaluation
│   ├── generate_v0_data.py            # Acrostic V0 data generation
│   ├── generate_v1_data.py            # Acrostic V1a/V1b data generation
│   ├── generate_v2_data.py            # Acrostic V2 data generation
│   ├── generate_v3_data.py            # Acrostic V3a/V3b data generation
│   ├── generate_synonym_stage1.py     # Synonym Stage 1 data generation
│   ├── generate_synonym_v0v1v2.py     # Synonym V0/V1/V2 data generation
│   ├── generate_sentlen_stage1.py     # Sentence length Stage 1 data generation
│   ├── generate_sentlen_v0v1v2_openai.py  # Sentence length V0/V1/V2 (OpenAI API)
│   ├── run_7b_block.sh                # 7B acrostic experiment runner
│   ├── run_14b_block.sh               # 14B acrostic experiment runner
│   ├── run_32b_experiments.sh         # 32B acrostic experiment runner
│   ├── run_synonym_experiments.sh     # Full synonym experiment runner (7B + 14B)
│   └── setup_2gpu.sh                  # Vast.ai instance setup
├── data/
│   ├── acrostics/
│   │   ├── stage1/                    # 9k train / 1k val
│   │   ├── v0/ ... v3b/              # ~1k train / 200 test each
│   ├── synonyms/
│   │   ├── stage1/                    # 1187 train / 199 val
│   │   ├── v0/ ... v2/               # ~1k train / 200 test each
│   └── sentlen/
│       ├── stage1/                    # 1200 train / 200 val
│       └── v0/ ... v2/               # In progress
├── results/
│   ├── acrostics/
│   │   ├── qwen-7b/                   # Complete (Stage 1, V0-V2, V3a, V3b, V2-6ep)
│   │   ├── qwen-14b/                  # Complete (Stage 1, V0-V3b, V2-6ep, V2-12ep, cross-eval)
│   │   └── qwen-32b/                  # Complete (Stage 1, V0, V2, V3a)
│   └── synonyms/
│       └── qwen-7b/                   # Partial (Stage 1, V0, V1a)
├── adapters/                          # Git LFS: trained LoRA adapters
│   ├── acrostics/
│   │   ├── qwen-7b-stage1/
│   │   ├── qwen-14b-stage1/
│   │   └── qwen-32b-stage1/
│   └── synonyms/
│       ├── qwen-7b-stage1/
│       └── qwen-14b-stage1/
└── v2-data/                           # Legacy V2 data (kept for provenance)
```

## Training details

All experiments use Qwen2.5-Instruct models with 4-bit NF4 quantization and LoRA (r=16, alpha=32). Training hyperparameters are fixed across all model sizes for fair comparison:

- Batch size: 1, gradient accumulation: 8
- Max sequence length: 512
- Learning rate: 2e-4 (Stage 1), 1e-4 (Stage 2)
- Epochs: 3 (unless noted)
- Optimizer: AdamW with cosine schedule

Stage 1 trains the base encoding capability. Stage 2 (V0-V3) fine-tunes from the Stage 1 adapter, learning internalized behavior from ~1k (prompt, encoded output) pairs with no system prompt or secret visible to the model.

## Running experiments

### Setup (Vast.ai, A100 80GB recommended)

```bash
cd /workspace
curl -O https://raw.githubusercontent.com/WFJKK/Steganography-internalisation-experiments/main/scripts/setup_2gpu.sh
bash setup_2gpu.sh YOUR_GITHUB_TOKEN
cd /workspace/Steganography-internalisation-experiments
```

### Acrostic experiments

```bash
# 7B: Stage 1 + V3a/V3b + V2-6ep (~3 hours)
nohup bash scripts/run_7b_block.sh > /dev/shm/7b.log 2>&1 &

# 14B: Stage 1 + V3a/V3b + V2-12ep (~4 hours)
nohup bash scripts/run_14b_block.sh > /dev/shm/14b.log 2>&1 &

# 32B: Stage 1 + V0 + V2 + V3a (~14 hours)
nohup bash scripts/run_32b_experiments.sh > /dev/shm/32b.log 2>&1 &
```

### Synonym experiments

```bash
# 7B + 14B full ladder (~5 hours)
nohup bash scripts/run_synonym_experiments.sh > /dev/shm/synonym.log 2>&1 &
```

### Manual training

```bash
# Stage 1: teach base encoding
python scripts/train_acrostic.py stage1 \
  --train-file data/acrostics/stage1/train.jsonl \
  --val-file data/acrostics/stage1/val.jsonl \
  --output-dir /dev/shm/acrostic-lora \
  --model Qwen/Qwen2.5-7B-Instruct \
  --epochs 3 --batch-size 1 --gradient-accumulation 8 --max-length 512

# Stage 2: internalization (no secret at inference)
python scripts/train_acrostic.py stage2 \
  --adapter-dir /dev/shm/acrostic-lora \
  --v0-data data/acrostics/v0/train.jsonl \
  --output-dir /dev/shm/v0-lora \
  --model Qwen/Qwen2.5-7B-Instruct \
  --epochs 3 --batch-size 1 --gradient-accumulation 8 --max-length 512

# Evaluate
python scripts/train_acrostic.py evaluate-v0 \
  --adapter-dir /dev/shm/v0-lora \
  --eval-file data/acrostics/v0/test.jsonl \
  --output results/acrostics/qwen-7b/v0_results.json \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-examples 200
```

### Data generation

```bash
# Sentence length (OpenAI API)
export OPENAI_API_KEY="sk-..."
python scripts/generate_sentlen_v0v1v2_openai.py \
  --variant v0 \
  --source-data data/acrostics/v0/train.jsonl \
  --output data/sentlen/v0/train.jsonl --resume

# Synonyms (Anthropic API)
export ANTHROPIC_API_KEY="sk-ant-..."
python scripts/generate_synonym_v0v1v2.py \
  --variant v0 \
  --source-data data/acrostics/v0/train.jsonl \
  --output data/synonyms/v0/train.jsonl --resume
```

## References

- Roger & Greenblatt (2023). *Preventing language models from hiding their reasoning.*
- Zolkowski et al. (2025). *Early signs of steganographic capabilities in frontier LLMs.*
- Mathew et al. (2024). *Hidden in plain text: Emergence and mitigation of steganographic collusion in LLMs.*
- Skaf et al. (2025). *Large language models can learn and generalize steganographic chain-of-thought under process supervision.*
- Karpov et al. (2025). *The steganographic potentials of language models.*
- Anwar et al. (2026). *A decision-theoretic formalisation of steganography with applications to LLM monitoring.*

## About

SPAR 2026 fellowship project. Supervised by Rob Krzyzanowski (Poseidon Research). Research by Joshua Kames-King.
