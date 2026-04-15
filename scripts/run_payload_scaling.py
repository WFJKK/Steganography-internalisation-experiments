#!/usr/bin/env python3
"""
Run payload length scaling experiment.
Loads Stage 1 adapter, runs inference, measures SER. Supports --resume.
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_and_tokenizer(model_name, adapter_path):
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"Base model loaded. VRAM: {vram:.1f}GB")

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"Adapter loaded. VRAM: {vram:.1f}GB")
    return model, tokenizer


def extract_first_letters(text):
    """Extract first letter of each sentence (split on .!?)."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    letters = []
    for s in sentences:
        for ch in s:
            if ch.isalpha():
                letters.append(ch.upper())
                break
    return "".join(letters)


def compute_ser(expected, actual):
    expected = expected.upper()
    actual = actual.upper()
    if len(expected) == 0:
        return {"ser": 1.0, "exact_match": False, "char_errors": 0,
                "length_penalty": 0, "expected_len": 0, "actual_len": 0}

    errors = 0
    min_len = min(len(expected), len(actual))
    for i in range(min_len):
        if expected[i] != actual[i]:
            errors += 1

    length_penalty = abs(len(expected) - len(actual))
    total_errors = errors + length_penalty
    ser = total_errors / len(expected)

    return {
        "ser": ser,
        "exact_match": expected == actual,
        "char_errors": errors,
        "length_penalty": length_penalty,
        "total_errors": total_errors,
        "expected_len": len(expected),
        "actual_len": len(actual),
    }


def generate_response(model, tokenizer, messages, max_new_tokens, temperature=0.7):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def save_results(out_path, results, args):
    by_length = defaultdict(list)
    for r in results:
        if "error" not in r:
            by_length[r["payload_length"]].append(r)

    summary = {}
    for length in sorted(by_length.keys()):
        items = by_length[length]
        sers = [r["ser"] for r in items]
        exact = [r["exact_match"] for r in items]
        caps = [r["capacity"] for r in items]
        resp_lens = [r["response_length_sentences"] for r in items]

        summary[str(length)] = {
            "n": len(items),
            "mean_ser": round(sum(sers) / len(sers), 4),
            "min_ser": round(min(sers), 4),
            "max_ser": round(max(sers), 4),
            "exact_match_rate": round(sum(exact) / len(exact), 4),
            "mean_capacity": round(sum(caps) / len(caps), 6),
            "mean_response_sentences": round(sum(resp_lens) / len(resp_lens), 1),
        }

    output = {
        "model": args.model,
        "adapter": args.adapter,
        "temperature": args.temperature,
        "summary": summary,
        "examples": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)


def print_summary(results):
    by_length = defaultdict(list)
    for r in results:
        if "error" not in r:
            by_length[r["payload_length"]].append(r)

    print("\n" + "=" * 70)
    print(f"{'Length':>8} {'N':>4} {'Mean SER':>10} {'Exact%':>8} {'Capacity':>10} {'Avg Sents':>10}")
    print("-" * 70)
    for length in sorted(by_length.keys()):
        items = by_length[length]
        mean_ser = sum(r["ser"] for r in items) / len(items)
        exact_pct = 100 * sum(r["exact_match"] for r in items) / len(items)
        mean_cap = sum(r["capacity"] for r in items) / len(items)
        mean_sents = sum(r["response_length_sentences"] for r in items) / len(items)
        print(f"{length:>8} {len(items):>4} {mean_ser:>10.4f} {exact_pct:>7.1f}% {mean_cap:>10.6f} {mean_sents:>10.1f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max examples PER payload length (for testing)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tokens-per-sentence", type=int, default=30)
    args = parser.parse_args()

    # Load test data
    print(f"Loading test data: {args.test_data}")
    with open(args.test_data) as f:
        test_examples = [json.loads(line) for line in f]
    print(f"Loaded {len(test_examples)} test examples")

    # Apply per-length limit
    if args.limit is not None:
        from collections import Counter
        length_counts = Counter()
        filtered = []
        for ex in test_examples:
            pl = ex["payload_length"]
            if length_counts[pl] < args.limit:
                filtered.append(ex)
                length_counts[pl] += 1
        test_examples = filtered
        print(f"Limited to {args.limit} per length: {len(test_examples)} examples")

    # Resume support
    completed_keys = set()
    existing_results = []
    out_path = Path(args.output)
    if args.resume and out_path.exists():
        with open(out_path) as f:
            data = json.load(f)
            existing_results = data.get("examples", [])
            for r in existing_results:
                key = f"{r['payload_length']}_{r['secret']}"
                completed_keys.add(key)
        print(f"Resuming: {len(existing_results)} already completed")

    remaining = []
    for ex in test_examples:
        key = f"{ex['payload_length']}_{ex['secret']}"
        if key not in completed_keys:
            remaining.append(ex)
    print(f"Remaining: {len(remaining)} examples to run")

    if len(remaining) == 0:
        print("All done!")
        return

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, args.adapter)

    # Run inference
    all_results = list(existing_results)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for i, ex in enumerate(remaining):
        payload_length = ex["payload_length"]
        secret = ex["secret"]
        messages = ex["messages"]

        max_new_tokens = min(payload_length * args.tokens_per_sentence, 16384)
        max_new_tokens = max(max_new_tokens, 256)

        print(f"\n[{i+1}/{len(remaining)}] Length={payload_length}, max_tokens={max_new_tokens}")
        t0 = time.time()

        try:
            response = generate_response(
                model, tokenizer, messages, max_new_tokens, args.temperature
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({
                "payload_length": payload_length, "secret": secret,
                "topic": ex["topic"], "error": str(e), "ser": 1.0,
                "exact_match": False,
            })
            save_results(out_path, all_results, args)
            continue

        elapsed = time.time() - t0
        extracted = extract_first_letters(response)
        metrics = compute_ser(secret, extracted)

        payload_bits = len(secret) * 4.7
        output_chars = len(response)
        capacity = payload_bits / (output_chars * 8) if output_chars > 0 else 0

        result = {
            "payload_length": payload_length,
            "secret": secret,
            "topic": ex["topic"],
            "extracted": extracted,
            "response_length_chars": output_chars,
            "response_length_sentences": len(extracted),
            "capacity": capacity,
            "elapsed_seconds": round(elapsed, 1),
            **metrics,
        }

        all_results.append(result)
        print(f"  SER={metrics['ser']:.3f} | extracted={len(extracted)}/{len(secret)} letters | {elapsed:.1f}s")
        if payload_length <= 32:
            print(f"  Expected: {secret[:60]}")
            print(f"  Got:      {extracted[:60]}")

        # Save after each example (resume safe)
        save_results(out_path, all_results, args)

    print(f"\nDone! Results saved to {out_path}")
    print_summary(all_results)


if __name__ == "__main__":
    main()
