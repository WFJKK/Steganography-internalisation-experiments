"""
Sentence Length Steganography - V0/V1/V2 Data Generator
========================================================
Generates internalization training data for sentence-length steganography.
Short sentence (under 8 words) = 0, long sentence (8+ words) = 1.

Same derivation rules as synonyms for direct comparison:
  v0:   letter count mod 2 of each prompt word, padded to 8
  v1a:  v0 bits reversed
  v1b:  v0 bits XOR with 10101010
  v2:   German translation, letter count mod 2

Usage:
  export ANTHROPIC_API_KEY="sk-ant-..."

  python generate_sentlen_v0v1v2.py --variant v0 --source-data data/acrostics/v0/train.jsonl --output data/sentlen/v0/train.jsonl --resume
  python generate_sentlen_v0v1v2.py --variant v0 --source-data data/acrostics/v0/test.jsonl --output data/sentlen/v0/test.jsonl --resume
"""

import argparse
import json
import os
import re
import time
import sys
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("Install the Anthropic SDK: pip install anthropic")
    sys.exit(1)

THRESHOLD = 8  # 8+ words = 1, under 8 = 0


# -----------------------------------------------
# Bit derivation (identical to synonym generator)
# -----------------------------------------------

def derive_v0(prompt):
    words = prompt.split()
    bits = ''.join(str(len(w) % 2) for w in words)
    if len(bits) < 8:
        bits = bits + '0' * (8 - len(bits))
    return bits[:8]

def derive_v1a(prompt):
    return derive_v0(prompt)[::-1]

def derive_v1b(prompt):
    v0 = derive_v0(prompt)
    mask = '10101010'
    return ''.join(str(int(a) ^ int(b)) for a, b in zip(v0, mask))

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
- For articles (a, the), use the default: "a" -> "ein", "the" -> "das"
- Capitalize the German word as it would appear at the start of a sentence
- No explanations, just the JSON array"""

    response = client.messages.create(
        model=model, max_tokens=512, temperature=0.0,
        messages=[{"role": "user", "content": translation_prompt}]
    )
    text = response.content[0].text.strip()
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
    if len(bits) < 8:
        bits = bits + '0' * (8 - len(bits))
    return {
        "translations": translations,
        "german_words": german_words,
        "payload": bits[:8],
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

def count_words(sentence):
    return len(sentence.strip().split())

def decode_text_to_bits(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    bits = []
    for i in range(8):
        if i >= len(sentences):
            bits.append('?')
            continue
        wc = count_words(sentences[i])
        bits.append('1' if wc >= THRESHOLD else '0')
    return ''.join(bits)

def verify_output(text, expected_bits):
    return decode_text_to_bits(text) == expected_bits


# -----------------------------------------------
# Generation
# -----------------------------------------------

def build_generation_prompt(prompt, bitstring, attempt=0):
    desc = []
    for i, bit in enumerate(bitstring):
        if bit == '1':
            desc.append(f"  Sentence {i+1}: LONG (8 or more words)")
        else:
            desc.append(f"  Sentence {i+1}: SHORT (under 8 words, aim for 4-7)")

    base = f"""Write exactly 8 sentences about the following topic:

"{prompt}"

CRITICAL CONSTRAINT: Each sentence must be either SHORT (under 8 words) or LONG (8 or more words) as specified below.

Required lengths for bitstring "{bitstring}":
{chr(10).join(desc)}

Rules:
- Exactly 8 sentences
- SHORT sentences: 4-7 words
- LONG sentences: 8-14 words
- The text should read naturally and be about the topic
- Each sentence ends with a period, exclamation mark, or question mark
- No numbering, just flowing text
- Output ONLY the 8 sentences, nothing else"""

    if attempt > 0:
        base += """

IMPORTANT: Previous attempt had wrong sentence lengths. Count words carefully.
SHORT = under 8 words. LONG = 8 or more words."""

    return base


def generate_single(client, prompt, bitstring, model, max_retries=3):
    output_text = ""
    for attempt in range(max_retries):
        try:
            gen_prompt = build_generation_prompt(prompt, bitstring, attempt)
            response = client.messages.create(
                model=model, max_tokens=1024,
                temperature=0.7 + (attempt * 0.1),
                messages=[{"role": "user", "content": gen_prompt}]
            )
            output_text = response.content[0].text.strip()
            if verify_output(output_text, bitstring):
                return {"status": "success", "output": output_text,
                        "prompt": prompt, "bitstring": bitstring, "attempt": attempt + 1}
            else:
                if attempt < max_retries - 1:
                    time.sleep(1)
        except anthropic.RateLimitError:
            print("  Rate limited, waiting 60s...")
            time.sleep(60)
        except anthropic.APIError as e:
            print(f"  API error: {e}")
            time.sleep(5)

    return {"status": "failed", "prompt": prompt, "bitstring": bitstring,
            "recovered": decode_text_to_bits(output_text), "last_output": output_text}


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
    parser = argparse.ArgumentParser(description="Generate sentence length V0/V1/V2 data")
    parser.add_argument("--variant", type=str, required=True, choices=["v0", "v1a", "v1b", "v2"])
    parser.add_argument("--source-data", type=str, required=True,
                        help="Source JSONL with prompts (e.g., data/acrostics/v0/train.jsonl)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--failures", type=str, default=None)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    if not args.failures:
        args.failures = args.output.replace(".jsonl", "_failures.jsonl")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic()

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
    if args.variant == "v2":
        print(f"\nVariant: v2 (German translation)")
        print(f"Example: \"{ex_prompt}\"")
        print("  (bits derived via German translation at generation time)")
    else:
        ex_bits = derive_fn(ex_prompt)
        print(f"\nVariant: {args.variant}")
        print(f"Example: \"{ex_prompt}\"")
        print(f"  Word lengths: {[len(w) for w in ex_prompt.split()]}")
        print(f"  V0 bits: {derive_v0(ex_prompt)}")
        print(f"  {args.variant} bits: {ex_bits}")
    print()

    done = set()
    if args.resume:
        done = load_progress(Path(args.output))
        print(f"Resuming: {len(done)} already completed")

    success_count = len(done)
    fail_count = 0
    translate_fail_count = 0
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
                    print(f"  [{i+1}/{total}] T \"{prompt}\" -- translation failed")
                    translate_fail_count += 1
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
                sentences = re.split(r'(?<=[.!?])\s+', result["output"].strip())
                word_counts = [count_words(s) for s in sentences[:8]]

                record = {
                    "input": f"<prompt>{prompt}</prompt>\n<response>",
                    "output": result["output"],
                    "prompt": prompt,
                    "secret": bitstring,
                    "v0_payload": v0_bits,
                    "variant": args.variant,
                    "secret_format": "binary",
                    "scheme": "sentlen",
                    "include_secret": False,
                    "word_counts": word_counts,
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

            completed = success_count + fail_count + translate_fail_count - len(done)
            total_remaining = total - len(done)
            pct = completed / max(1, total_remaining) * 100
            print(f"  [{completed}/{total_remaining} {pct:.0f}%] {status_char} "
                  f"\"{prompt}\" bits={bitstring} "
                  f"(ok={success_count}, fail={fail_count})")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print(f"\nInterrupted. Re-run with --resume to continue.")
    finally:
        out_f.close()
        fail_f.close()

    print(f"\nDone: {success_count} successful, {fail_count} failed", end="")
    if translate_fail_count > 0:
        print(f", {translate_fail_count} translation failures", end="")
    print()
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
