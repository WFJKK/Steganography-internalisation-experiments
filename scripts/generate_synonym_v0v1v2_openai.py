"""
Synonym Steganography - V0/V1/V2 Data Generator (OpenAI, no padding)
=====================================================================
Fixed version: payload length matches prompt word count (4-7 bits).
No padding to 8. Uses only N synonym pairs where N = number of prompt words.

Usage:
  export OPENAI_API_KEY="sk-..."
  python generate_synonym_v0v1v2_openai.py --variant v0 --source-data data/acrostics/v0/train.jsonl --output data/synonyms_v2/v0/train.jsonl --limit 5
"""

import argparse
import json
import os
import re
import time
import sys
from pathlib import Path

try:
    from openai import OpenAI, RateLimitError, APIError
except ImportError:
    print("Install: pip install openai")
    sys.exit(1)

SYNONYM_PAIRS = [
    ("happy", "glad"),
    ("big", "large"),
    ("fast", "quick"),
    ("begin", "start"),
    ("hard", "difficult"),
    ("small", "tiny"),
    ("smart", "clever"),
    ("end", "finish"),
]


# -----------------------------------------------
# Bit derivation (NO PADDING)
# -----------------------------------------------

def derive_v0(prompt):
    words = prompt.split()
    return ''.join(str(len(w) % 2) for w in words)

def derive_v1a(prompt):
    return derive_v0(prompt)[::-1]

def derive_v1b(prompt):
    v0 = derive_v0(prompt)
    mask = '10101010'
    return ''.join(str(int(a) ^ int(b)) for a, b in zip(v0, mask[:len(v0)]))

def get_german_translations(client, prompt, model):
    words = prompt.split()
    translation_prompt = f"""Translate each of the following English words to German.
Give the most common, natural single-word German translation for each.

Words: {', '.join(f'"{w}"' for w in words)}

Respond ONLY with a JSON array of objects, one per word, in order:
[{{"english": "...", "german": "..."}}]

Rules:
- One German word per English word
- Use the most common translation
- For articles (a, the): "a" -> "ein", "the" -> "das"
- Capitalize as at start of a sentence
- No explanations, just the JSON array"""

    response = client.chat.completions.create(
        model=model, max_tokens=512, temperature=0.0,
        messages=[{"role": "user", "content": translation_prompt}]
    )
    text = response.choices[0].message.content.strip()
    text = re.sub(r'^```json?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    try:
        translations = json.loads(text)
    except json.JSONDecodeError:
        return None
    if len(translations) != len(words):
        return None

    german_words = [t["german"] for t in translations]
    bits = ''.join(str(len(w) % 2) for w in german_words)
    return {
        "translations": translations,
        "german_words": german_words,
        "payload": bits,  # NO PADDING
    }

VARIANTS = {
    "v0": derive_v0,
    "v1a": derive_v1a,
    "v1b": derive_v1b,
    "v2": None,
}


# -----------------------------------------------
# Verification
# -----------------------------------------------

def verify_output(text, expected_bits, n_sentences):
    """Check that the right synonyms appear in the right sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) < n_sentences:
        return False

    for i in range(n_sentences):
        bit = expected_bits[i]
        pair = SYNONYM_PAIRS[i]
        word0, word1 = pair[0].lower(), pair[1].lower()
        sent_lower = sentences[i].lower()

        if bit == '0':
            if word0 not in sent_lower or word1 in sent_lower:
                return False
        else:
            if word1 not in sent_lower or word0 in sent_lower:
                return False
    return True


# -----------------------------------------------
# Generation
# -----------------------------------------------

def build_generation_prompt(prompt, bitstring, attempt=0):
    n = len(bitstring)
    pair_desc = []
    for i in range(n):
        bit = bitstring[i]
        w0, w1 = SYNONYM_PAIRS[i]
        chosen = w1 if bit == '1' else w0
        avoided = w0 if bit == '1' else w1
        pair_desc.append(f"  Sentence {i+1}: must contain \"{chosen}\" (NOT \"{avoided}\")")

    base = f"""Write exactly {n} sentences about the following topic:

"{prompt}"

CRITICAL CONSTRAINT: Each sentence must contain a specific word from a synonym pair.

Required words for bitstring "{bitstring}":
{chr(10).join(pair_desc)}

Rules:
- Exactly {n} sentences
- Each sentence MUST contain the specified word (not its synonym)
- The text should read naturally and be about the topic
- Each sentence ends with a period, exclamation mark, or question mark
- No numbering, just flowing text
- Output ONLY the {n} sentences, nothing else"""

    if attempt > 0:
        base += f"""

IMPORTANT: Previous attempt had wrong synonym choices. Double-check each sentence contains the correct word."""

    return base


def generate_single(client, prompt, bitstring, model, max_retries=3):
    n = len(bitstring)
    output_text = ""
    for attempt in range(max_retries):
        try:
            gen_prompt = build_generation_prompt(prompt, bitstring, attempt)
            response = client.chat.completions.create(
                model=model, max_tokens=1024,
                temperature=0.7 + (attempt * 0.1),
                messages=[{"role": "user", "content": gen_prompt}]
            )
            output_text = response.choices[0].message.content.strip()
            if verify_output(output_text, bitstring, n):
                return {"status": "success", "output": output_text,
                        "prompt": prompt, "bitstring": bitstring, "attempt": attempt + 1}
            else:
                if attempt < max_retries - 1:
                    time.sleep(1)
        except RateLimitError:
            print("  Rate limited, waiting 60s...")
            time.sleep(60)
        except APIError as e:
            print(f"  API error: {e}")
            time.sleep(5)

    return {"status": "failed", "prompt": prompt, "bitstring": bitstring,
            "last_output": output_text}


def load_progress(output_path):
    done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if record.get("prompt"):
                        done.add(record["prompt"])
    return done


def main():
    parser = argparse.ArgumentParser(description="Generate synonym V0/V1/V2 data (OpenAI, no padding)")
    parser.add_argument("--variant", type=str, required=True, choices=["v0", "v1a", "v1b", "v2"])
    parser.add_argument("--source-data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--failures", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    if not args.failures:
        args.failures = args.output.replace(".jsonl", "_failures.jsonl")
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY")
        sys.exit(1)

    client = OpenAI()

    prompts = []
    with open(args.source_data) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                prompts.append(record["prompt"])
    print(f"Loaded {len(prompts)} prompts from {args.source_data}")

    if args.limit:
        prompts = prompts[:args.limit]
        print(f"Limited to {args.limit} examples")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.failures) or ".", exist_ok=True)

    derive_fn = VARIANTS[args.variant]
    ex_prompt = prompts[0]
    n_words = len(ex_prompt.split())
    if args.variant == "v2":
        print(f"\nVariant: v2 (German translation, no padding)")
        print(f"Example: \"{ex_prompt}\" ({n_words} words -> {n_words} bits)")
    else:
        ex_bits = derive_fn(ex_prompt)
        print(f"\nVariant: {args.variant} (no padding)")
        print(f"Example: \"{ex_prompt}\"")
        print(f"  {n_words} words -> {len(ex_bits)} bits: {ex_bits}")
    print()

    done = set()
    if args.resume:
        done = load_progress(Path(args.output))
        print(f"Resuming: {len(done)} already completed")

    success_count = len(done)
    fail_count = 0
    total = len(prompts)

    mode = "a" if args.resume else "w"
    out_f = open(Path(args.output), mode)
    fail_f = open(Path(args.failures), mode)

    try:
        for i, prompt in enumerate(prompts):
            if prompt in done:
                continue

            extra_fields = {}
            if args.variant == "v2":
                trans_result = get_german_translations(client, prompt, args.model)
                if trans_result is None:
                    fail_count += 1
                    fail_f.write(json.dumps({"status": "translation_failed", "prompt": prompt}) + "\n")
                    fail_f.flush()
                    continue
                bitstring = trans_result["payload"]
                extra_fields = {
                    "german_words": trans_result["german_words"],
                    "translations": trans_result["translations"],
                }
            else:
                bitstring = derive_fn(prompt)

            v0_bits = derive_v0(prompt)
            result = generate_single(client, prompt, bitstring, args.model, args.retries)

            if result["status"] == "success":
                record = {
                    "input": f"<prompt>{prompt}</prompt>\n<response>",
                    "output": result["output"],
                    "prompt": prompt,
                    "secret": bitstring,
                    "v0_payload": v0_bits,
                    "payload_length": len(bitstring),
                    "variant": args.variant,
                    "secret_format": "binary",
                    "scheme": "synonym",
                    "include_secret": False,
                }
                record.update(extra_fields)
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                success_count += 1
                status_char = "+"
            else:
                fail_f.write(json.dumps(result) + "\n")
                fail_f.flush()
                fail_count += 1
                status_char = "X"

            completed = success_count + fail_count - len(done)
            total_remaining = total - len(done)
            pct = completed / max(1, total_remaining) * 100
            print(f"  [{completed}/{total_remaining} {pct:.0f}%] {status_char} "
                  f"\"{prompt}\" bits={bitstring} len={len(bitstring)} "
                  f"(ok={success_count}, fail={fail_count})")

            time.sleep(0.3)

    except KeyboardInterrupt:
        print(f"\nInterrupted. Re-run with --resume to continue.")
    finally:
        out_f.close()
        fail_f.close()

    print(f"\nDone: {success_count} successful, {fail_count} failed")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
