"""
Base Model Loss as Learnability Predictor
==========================================
Runs the untrained base model on V0 training data for each scheme.
Measures per-token cross-entropy loss, focusing on encoding-critical tokens.

Hypothesis: the base model's loss profile on encoding-critical tokens
predicts how easily SFT can internalize the pattern.

For each scheme we report:
  - Overall output loss (average CE across all output tokens)
  - Critical token loss (CE at the specific tokens that encode payload)
  - Critical token probability (base model's P(correct token) at encoding positions)

Usage (Mac, ~20 min):
  python measure_base_loss.py --n-examples 20 --device mps

Usage (GPU):
  python measure_base_loss.py --n-examples 50 --device cuda
"""

import argparse
import json
import os
import re
import sys

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


SYNONYM_PAIRS = [
    ("happy", "glad"), ("big", "large"), ("fast", "quick"), ("begin", "start"),
    ("hard", "difficult"), ("small", "tiny"), ("smart", "clever"), ("end", "finish"),
]
SENTLEN_THRESHOLD = 8


def load_examples(path, max_n=None):
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            examples.append({
                "prompt": record["prompt"],
                "output": record["output"],
                "secret": record["secret"],
            })
    if max_n:
        examples = examples[:max_n]
    return examples


def get_segments(output_text, scheme):
    if scheme == "acrostics":
        return [l.strip() for l in output_text.strip().split("\n") if l.strip()]
    else:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', output_text.strip()) if s.strip()]


def load_model(model_name, device):
    """Load base model (no adapter) for loss computation."""
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb,
            device_map="auto", trust_remote_code=True,
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        model = model.to(device)

    model.eval()
    print(f"Model loaded.")
    return model, tokenizer


def get_per_token_loss(model, tokenizer, prompt_text, output_text, device):
    """Compute per-token cross-entropy loss on the output portion.
    
    Returns:
        output_losses: list of (token_str, loss) for each output token
        output_token_ids: list of token IDs for output tokens
    """
    # Format as chat
    messages = [{"role": "user", "content": prompt_text}]
    prompt_formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_formatted + output_text

    # Tokenize
    prompt_ids = tokenizer.encode(prompt_formatted, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    prompt_len = len(prompt_ids)
    full_len = len(full_ids)

    if full_len <= prompt_len:
        return [], []

    input_ids = torch.tensor([full_ids]).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # Get per-token logits
        logits = outputs.logits[0]  # (seq_len, vocab_size)

    # Compute per-token cross-entropy for output tokens
    # Loss at position t predicts token at position t+1
    output_losses = []
    output_tokens = []

    for t in range(prompt_len - 1, full_len - 1):
        target_id = full_ids[t + 1]
        token_logits = logits[t]

        # Cross-entropy
        log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
        loss = -log_probs[target_id].item()
        prob = torch.exp(log_probs[target_id]).item()

        token_str = tokenizer.decode([target_id])
        output_losses.append({
            "token": token_str,
            "token_id": target_id,
            "loss": loss,
            "prob": prob,
            "position": t + 1 - prompt_len,
        })
        output_tokens.append(target_id)

    return output_losses, output_tokens


def find_critical_tokens(output_losses, output_text, scheme, segment_idx):
    """Identify which output tokens are encoding-critical for this segment.
    
    Returns indices into output_losses that correspond to critical tokens.
    """
    if not output_losses:
        return []

    if scheme == "acrostics":
        # Critical token: the very first token of the segment
        # This is the first character that encodes the payload
        # Find the first non-whitespace token
        for i, tl in enumerate(output_losses):
            if tl["token"].strip():
                return [i]
        return []

    elif scheme == "synonyms":
        # Critical tokens: the synonym word
        seg_lower = output_text.lower()
        pair = SYNONYM_PAIRS[segment_idx] if segment_idx < len(SYNONYM_PAIRS) else None
        if not pair:
            return []

        # Find tokens that are part of the synonym word
        critical = []
        for i, tl in enumerate(output_losses):
            tok = tl["token"].lower().strip()
            if tok in [pair[0], pair[1]]:
                critical.append(i)
        return critical if critical else []

    elif scheme == "sentlen":
        # ALL tokens are critical (aggregate word count property)
        return list(range(len(output_losses)))

    return []


def analyze_scheme(model, tokenizer, examples, scheme, device):
    """Run base model on training examples and analyze loss profile."""
    
    all_output_losses = []  # loss on all output tokens
    all_critical_losses = []  # loss on encoding-critical tokens
    all_critical_probs = []  # probability of correct critical token
    all_noncritical_losses = []  # loss on non-critical tokens
    
    n_positions = min(4, min(len(ex["secret"]) for ex in examples))

    for ex_idx, ex in enumerate(examples):
        segments = get_segments(ex["output"], scheme)

        # Get full output loss
        output_losses, _ = get_per_token_loss(
            model, tokenizer, ex["prompt"], ex["output"], device
        )

        if not output_losses:
            continue

        # Track all output token losses
        for tl in output_losses:
            all_output_losses.append(tl["loss"])

        # Now analyze per-segment critical tokens
        # We need to figure out which tokens correspond to which segment
        # Rough approach: tokenize each segment separately and find boundaries
        seg_boundaries = []
        current_pos = 0
        output_text_remaining = ex["output"]

        for seg_idx in range(min(n_positions, len(segments))):
            seg = segments[seg_idx]
            # Find where this segment starts in the output
            seg_start = ex["output"].find(seg, current_pos)
            if seg_start < 0:
                continue

            # Tokenize up to segment start and segment itself
            pre_seg_text = ex["output"][:seg_start]
            pre_tokens = tokenizer.encode(pre_seg_text, add_special_tokens=False) if pre_seg_text else []
            seg_tokens = tokenizer.encode(seg, add_special_tokens=False)

            token_start = len(pre_tokens)
            token_end = token_start + len(seg_tokens)

            # Get critical token indices within this segment's output_losses range
            seg_losses = output_losses[token_start:token_end] if token_end <= len(output_losses) else []

            if seg_losses:
                critical_indices = find_critical_tokens(seg_losses, seg, scheme, seg_idx)

                for ci in critical_indices:
                    if ci < len(seg_losses):
                        all_critical_losses.append(seg_losses[ci]["loss"])
                        all_critical_probs.append(seg_losses[ci]["prob"])

                # Non-critical tokens
                critical_set = set(critical_indices)
                for ni in range(len(seg_losses)):
                    if ni not in critical_set:
                        all_noncritical_losses.append(seg_losses[ni]["loss"])

            current_pos = seg_start + len(seg)

        if (ex_idx + 1) % 5 == 0:
            avg_out = np.mean(all_output_losses) if all_output_losses else 0
            avg_crit = np.mean(all_critical_losses) if all_critical_losses else 0
            avg_prob = np.mean(all_critical_probs) if all_critical_probs else 0
            print(f"    [{ex_idx+1}/{len(examples)}] "
                  f"avg_output_loss={avg_out:.3f}  "
                  f"avg_critical_loss={avg_crit:.3f}  "
                  f"avg_critical_prob={avg_prob:.3f}")

    # Compute final metrics
    result = {
        "avg_output_loss": float(np.mean(all_output_losses)) if all_output_losses else 0,
        "avg_critical_loss": float(np.mean(all_critical_losses)) if all_critical_losses else 0,
        "avg_critical_prob": float(np.mean(all_critical_probs)) if all_critical_probs else 0,
        "avg_noncritical_loss": float(np.mean(all_noncritical_losses)) if all_noncritical_losses else 0,
        "n_critical_tokens": len(all_critical_losses),
        "n_output_tokens": len(all_output_losses),
    }

    # Loss differential: how much harder are critical tokens than non-critical?
    if all_noncritical_losses and all_critical_losses:
        result["loss_differential"] = result["avg_critical_loss"] - result["avg_noncritical_loss"]
    else:
        result["loss_differential"] = 0

    # Critical token fraction: what fraction of output tokens are critical?
    if all_output_losses:
        result["critical_fraction"] = len(all_critical_losses) / len(all_output_losses)
    else:
        result["critical_fraction"] = 0

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n-examples", type=int, default=20)
    parser.add_argument("--device", default="mps", choices=["cuda", "mps", "cpu"])
    args = parser.parse_args()

    scheme_configs = {
        "acrostics": "data/acrostics/v0/train.jsonl",
        "synonyms": "data/synonyms_v2/v0/train.jsonl",
        "sentlen": "data/sentlen_v2/v0/train.jsonl",
        "synonym_poems": "data/synonym_poems/v0/train.jsonl",
        "sentlen_poems": "data/sentlen_poems/v0/train.jsonl",
        "punctuation": "data/punctuation/v0/train.jsonl",
    }

    available = {k: v for k, v in scheme_configs.items() if os.path.exists(v)}
    if not available:
        print("No data found. Run from repo root.")
        sys.exit(1)

    model, tokenizer = load_model(args.model, args.device)

    all_results = {}
    schemes_ordered = ["acrostics", "sentlen", "synonyms", "synonym_poems", "sentlen_poems", "punctuation"]
    v0 = {"acrostics": 62.0, "sentlen": 17.6, "synonyms": 13.8}

    for scheme in schemes_ordered:
        if scheme not in available:
            continue
        path = available[scheme]
        examples = load_examples(path, args.n_examples)
        print(f"\n{'='*60}")
        print(f"Scheme: {scheme} ({len(examples)} examples)")
        print(f"{'='*60}")

        result = analyze_scheme(model, tokenizer, examples, scheme, args.device)
        all_results[scheme] = result

        print(f"\n  Results:")
        print(f"    Overall output loss:     {result['avg_output_loss']:.3f}")
        print(f"    Critical token loss:     {result['avg_critical_loss']:.3f}")
        print(f"    Non-critical token loss: {result['avg_noncritical_loss']:.3f}")
        print(f"    Loss differential:       {result['loss_differential']:+.3f}")
        print(f"    Critical token prob:     {result['avg_critical_prob']:.3f}")
        print(f"    Critical fraction:       {result['critical_fraction']:.1%}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    header = f"{'Metric':<25} {'acrostics':>12} {'sentlen':>12} {'synonyms':>12} {'A>SL>S?':>10}"
    print(header)
    print("-" * len(header))

    metrics = [
        ("Output loss", "avg_output_loss"),
        ("Critical loss", "avg_critical_loss"),
        ("Critical prob", "avg_critical_prob"),
        ("Loss differential", "loss_differential"),
        ("Critical fraction", "critical_fraction"),
    ]

    for name, key in metrics:
        row = f"{name:<25}"
        vals = {}
        for scheme in schemes_ordered:
            v = all_results.get(scheme, {}).get(key, 0)
            vals[scheme] = v
            row += f" {v:>11.4f}"

        a, sl, sy = vals.get("acrostics", 0), vals.get("sentlen", 0), vals.get("synonyms", 0)
        # For loss/differential: higher = harder, so we want A > SL > SY for "no" prediction
        # For prob: higher = easier, so we want A < SL < SY
        # Just check both directions
        if a > sl > sy:
            correct = "A>SL>S"
        elif a < sl < sy:
            correct = "A<SL<S"
        elif a > max(sl, sy):
            correct = "A>rest"
        elif a < min(sl, sy):
            correct = "A<rest"
        else:
            correct = "mixed"
        row += f" {correct:>10}"
        print(row)

    print()
    print(f"{'V0 7B SFT':<25} {'62.0%':>12} {'17.6%':>12} {'13.8%':>12}")
    print()
    print("Key insight: if critical_loss is HIGH but critical_fraction is LOW,")
    print("the model must make a big change to few tokens -> concentrated gradient")
    print("-> easy to learn. This should characterize acrostics.")
    print()
    print("If critical_loss is LOW but critical_fraction is HIGH,")
    print("the model must make small changes to many tokens -> diffuse gradient")
    print("-> hard to learn. This should characterize sentlen.")

    os.makedirs("results", exist_ok=True)
    with open("results/base_loss_analysis.json", "w") as f:
        json.dump({"results": all_results, "v0": v0, "model": args.model}, f, indent=2)
    print(f"\nSaved to results/base_loss_analysis.json")


if __name__ == "__main__":
    main()
