"""
Generate Unconstrained Baselines
==================================
For each scheme's V0 training prompts, generate outputs in the same style
but WITHOUT any encoding constraint. Then compare base model loss on
constrained vs unconstrained outputs.

If the loss difference (constrained - unconstrained) predicts V0,
we've isolated the encoding constraint from the text style confound.

Usage:
  export OPENAI_API_KEY="sk-..."
  
  # Step 1: Generate baselines (~$0.50, takes ~10 min)
  python measure_baseline.py generate --n-examples 50

  # Step 2: Measure loss on both (no API needed)
  python measure_baseline.py measure --n-examples 50 --device mps --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


STYLE_PROMPTS = {
    "acrostics": "Write a short poem (4-7 lines) about the following topic. Each line should be a single poetic sentence. Topic: {prompt}",
    "synonyms": "Write a short paragraph (4-7 sentences) about the following topic. Use varied and natural vocabulary. Topic: {prompt}",
    "sentlen": "Write a short paragraph (4-7 sentences) about the following topic. Use a mix of short and long sentences. Topic: {prompt}",
    "synonym_poems": "Write a short poem (4-7 lines) about the following topic. Each line should be a single poetic sentence. Topic: {prompt}",
    "sentlen_poems": "Write a short poem (4-7 lines) about the following topic. Each line should be a single poetic sentence. Topic: {prompt}",
    "punctuation": "Write a short paragraph (4-7 sentences) about the following topic. Use natural punctuation. Topic: {prompt}",
}


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


# =========================================================================
# Step 1: Generate unconstrained baselines
# =========================================================================

def generate_baselines(scheme, examples, client, model="gpt-4o-mini"):
    """Generate unconstrained outputs for the same prompts."""
    style = STYLE_PROMPTS[scheme]
    baselines = []

    for i, ex in enumerate(examples):
        user_prompt = style.format(prompt=ex["prompt"])

        try:
            response = client.chat.completions.create(
                model=model, max_tokens=300, temperature=0.7,
                messages=[{"role": "user", "content": user_prompt}]
            )
            output = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  Error at {i}: {e}")
            time.sleep(30)
            continue

        baselines.append({
            "prompt": ex["prompt"],
            "output": output,
            "constrained_output": ex["output"],
            "secret": ex["secret"],
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(examples)}] generated")

        time.sleep(0.3)

    return baselines


def cmd_generate(args):
    if not HAS_OPENAI or not os.environ.get("OPENAI_API_KEY"):
        print("Need OPENAI_API_KEY")
        sys.exit(1)

    client = OpenAI()

    scheme_configs = {
        "acrostics": "data/acrostics/v0/train.jsonl",
        "synonyms": "data/synonyms_v2/v0/train.jsonl",
        "sentlen": "data/sentlen_v2/v0/train.jsonl",
        "synonym_poems": "data/synonym_poems/v0/train.jsonl",
        "sentlen_poems": "data/sentlen_poems/v0/train.jsonl",
        "punctuation": "data/punctuation/v0/train.jsonl",
    }

    os.makedirs("data/baselines", exist_ok=True)

    for scheme, path in scheme_configs.items():
        if not os.path.exists(path):
            print(f"SKIP {scheme}: {path} not found")
            continue

        out_path = f"data/baselines/{scheme}_baseline.jsonl"
        if os.path.exists(out_path) and not args.overwrite:
            existing = sum(1 for _ in open(out_path))
            if existing >= args.n_examples:
                print(f"SKIP {scheme}: {out_path} already has {existing} examples")
                continue

        examples = load_examples(path, args.n_examples)
        print(f"\nGenerating {scheme} baselines ({len(examples)} examples)...")
        baselines = generate_baselines(scheme, examples, client)

        with open(out_path, "w") as f:
            for b in baselines:
                f.write(json.dumps(b) + "\n")
        print(f"  Saved {len(baselines)} to {out_path}")


# =========================================================================
# Step 2: Measure base model loss on constrained vs unconstrained
# =========================================================================

def get_output_loss(model, tokenizer, prompt_text, output_text):
    """Compute average per-token CE loss on output tokens."""
    messages = [{"role": "user", "content": prompt_text}]
    prompt_formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_formatted + output_text

    prompt_ids = tokenizer.encode(prompt_formatted, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    prompt_len = len(prompt_ids)
    if len(full_ids) <= prompt_len:
        return None

    input_ids = torch.tensor([full_ids]).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits[0]

    losses = []
    for t in range(prompt_len - 1, len(full_ids) - 1):
        target_id = full_ids[t + 1]
        log_probs = torch.nn.functional.log_softmax(logits[t], dim=-1)
        loss = -log_probs[target_id].item()
        losses.append(loss)

    return np.mean(losses) if losses else None


def cmd_measure(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True,
        )
    elif args.device == "cuda":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb,
            device_map="auto", trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, trust_remote_code=True,
        ).to(args.device)

    model.eval()
    print("Model loaded.\n")

    schemes_ordered = ["acrostics", "sentlen", "synonyms", "synonym_poems", "sentlen_poems", "punctuation"]
    v0 = {"acrostics": 62.0, "sentlen": 17.6, "synonyms": 13.8}
    all_results = {}

    for scheme in schemes_ordered:
        baseline_path = f"data/baselines/{scheme}_baseline.jsonl"
        if not os.path.exists(baseline_path):
            print(f"SKIP {scheme}: no baseline data. Run 'generate' first.")
            continue

        baselines = []
        with open(baseline_path) as f:
            for line in f:
                if line.strip():
                    baselines.append(json.loads(line))

        if args.n_examples:
            baselines = baselines[:args.n_examples]

        print(f"{'='*60}")
        print(f"Scheme: {scheme} ({len(baselines)} examples)")
        print(f"{'='*60}")

        constrained_losses = []
        unconstrained_losses = []

        for i, b in enumerate(baselines):
            c_loss = get_output_loss(model, tokenizer, b["prompt"], b["constrained_output"])
            u_loss = get_output_loss(model, tokenizer, b["prompt"], b["output"])

            if c_loss is not None:
                constrained_losses.append(c_loss)
            if u_loss is not None:
                unconstrained_losses.append(u_loss)

            if (i + 1) % 10 == 0:
                avg_c = np.mean(constrained_losses) if constrained_losses else 0
                avg_u = np.mean(unconstrained_losses) if unconstrained_losses else 0
                print(f"  [{i+1}/{len(baselines)}] constrained={avg_c:.3f}  "
                      f"unconstrained={avg_u:.3f}  diff={avg_c - avg_u:+.3f}")

        avg_constrained = np.mean(constrained_losses) if constrained_losses else 0
        avg_unconstrained = np.mean(unconstrained_losses) if unconstrained_losses else 0
        encoding_tax = avg_constrained - avg_unconstrained

        print(f"\n  Constrained loss:   {avg_constrained:.4f}")
        print(f"  Unconstrained loss: {avg_unconstrained:.4f}")
        print(f"  Encoding tax:       {encoding_tax:+.4f}")

        all_results[scheme] = {
            "constrained_loss": float(avg_constrained),
            "unconstrained_loss": float(avg_unconstrained),
            "encoding_tax": float(encoding_tax),
            "n_examples": len(constrained_losses),
        }

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Encoding Tax vs V0 Internalization")
    print(f"{'='*70}")

    header = f"{'Metric':<25} {'acrostics':>12} {'sentlen':>12} {'synonyms':>12} {'A>SL>S?':>10}"
    print(header)
    print("-" * len(header))

    for name, key in [("Constrained loss", "constrained_loss"),
                       ("Unconstrained loss", "unconstrained_loss"),
                       ("Encoding tax", "encoding_tax")]:
        row = f"{name:<25}"
        vals = {}
        for scheme in schemes_ordered:
            v = all_results.get(scheme, {}).get(key, 0)
            vals[scheme] = v
            row += f" {v:>11.4f}"

        a, sl, sy = vals.get("acrostics", 0), vals.get("sentlen", 0), vals.get("synonyms", 0)
        if a > sl > sy:
            correct = "YES"
        elif a > sl and a > sy:
            correct = "partial"
        else:
            correct = "no"
        row += f" {correct:>10}"
        print(row)

    print()
    print(f"{'V0 7B SFT':<25} {'62.0%':>12} {'17.6%':>12} {'13.8%':>12}")
    print()
    print("If ENCODING TAX predicts V0 ranking, then the encoding constraint")
    print("(not text style) drives learnability. This controls for the confound")
    print("that acrostics produce poems while synonyms produce prose.")

    os.makedirs("results", exist_ok=True)
    with open("results/encoding_tax.json", "w") as f:
        json.dump({"results": all_results, "v0": v0, "model": args.model}, f, indent=2)
    print(f"\nSaved to results/encoding_tax.json")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate unconstrained baselines")
    gen.add_argument("--n-examples", type=int, default=50)
    gen.add_argument("--overwrite", action="store_true")

    meas = sub.add_parser("measure", help="Measure loss on constrained vs unconstrained")
    meas.add_argument("--n-examples", type=int, default=50)
    meas.add_argument("--device", default="mps", choices=["cuda", "mps", "cpu"])
    meas.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "measure":
        cmd_measure(args)


if __name__ == "__main__":
    main()
