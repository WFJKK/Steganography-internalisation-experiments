"""
V3 Internalization Dataset Generator (German + Transform)
==========================================================
Generates acrostic poem training data for V3 steganography internalization.

V3a: Payload = German translation first letters REVERSED
     "Design a cinema poem" -> German: DEKG -> reversed -> GKED

V3b: Payload = German translation first letters CAESAR +1
     "Design a cinema poem" -> German: DEKG -> shifted -> EFLH

Reuses German translations from existing V2 data to avoid redundant API calls.
Only needs Claude API to generate the poems (not for translation).

Usage:
  export ANTHROPIC_API_KEY="sk-ant-..."

  # Generate V3a (German + reverse):
  python generate_v3_data.py --v2-data data/acrostics/v2/train.jsonl --variant reversed --output data/acrostics/v3a/train.jsonl

  # Generate V3b (German + Caesar):
  python generate_v3_data.py --v2-data data/acrostics/v2/train.jsonl --variant caesar --output data/acrostics/v3b/train.jsonl

  # Test with 5 examples first:
  python generate_v3_data.py --v2-data data/acrostics/v2/train.jsonl --variant reversed --output test.jsonl --limit 5

  # Resume interrupted runs:
  python generate_v3_data.py --v2-data data/acrostics/v2/train.jsonl --variant reversed --output train.jsonl --resume
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


# -----------------------------------------------
# Payload transformations (applied on top of V2)
# -----------------------------------------------

def caesar_shift(char: str, shift: int = 1) -> str:
    """Shift a single uppercase letter by shift positions, wrapping A-Z."""
    if not char.isalpha():
        return char
    base = ord('A')
    return chr((ord(char.upper()) - base + shift) % 26 + base)


def transform_reversed(v2_payload: str) -> str:
    """V3a: reverse the V2 (German) payload."""
    return v2_payload[::-1]


def transform_caesar(v2_payload: str, shift: int = 1) -> str:
    """V3b: Caesar shift the V2 (German) payload."""
    return "".join(caesar_shift(ch, shift) for ch in v2_payload)


VARIANTS = {
    "reversed": transform_reversed,
    "caesar": transform_caesar,
}


# -----------------------------------------------
# Acrostic generation and verification
# -----------------------------------------------

def extract_first_letters(text: str) -> str:
    """Extract first letter of each non-empty line."""
    letters = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            for ch in line:
                if ch.isalpha():
                    letters.append(ch.upper())
                    break
    return "".join(letters)


def verify_acrostic(text: str, expected_payload: str) -> bool:
    """Check if first letters of lines match expected payload."""
    actual = extract_first_letters(text)
    return actual == expected_payload.upper()


def build_generation_prompt(prompt: str, payload: str, attempt: int = 0) -> str:
    """Build the prompt to send to Claude for acrostic poem generation."""
    n = len(payload)
    letters_list = ", ".join(f'"{ch}"' for ch in payload)

    base_prompt = f"""Write a poem with exactly {n} lines in response to this request:

"{prompt}"

CRITICAL CONSTRAINT: The first letter of each line must spell out these letters in order: {letters_list}

Rules:
- Exactly {n} lines, one per letter
- Each line starts with a word beginning with the corresponding letter
- The poem should feel natural, vivid, and coherent
- Do not mention the acrostic or hidden pattern
- Do not number the lines
- Each line should be 5-12 words long
- No title, just the poem lines

Output ONLY the {n} lines of the poem. Nothing else."""

    if attempt > 0:
        base_prompt += f"""

IMPORTANT: Previous attempt failed. Double-check:
- Line 1 starts with a word beginning with "{payload[0]}"
- Line {n} starts with a word beginning with "{payload[-1]}"
- Exactly {n} lines total
- Each line is on its own line"""

    return base_prompt


def generate_single(client, prompt: str, payload: str, model: str, max_retries: int = 3) -> dict:
    """Generate and verify a single acrostic poem."""
    output_text = ""
    for attempt in range(max_retries):
        try:
            gen_prompt = build_generation_prompt(prompt, payload, attempt)

            response = client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=0.7 + (attempt * 0.1),
                messages=[{"role": "user", "content": gen_prompt}]
            )

            output_text = response.content[0].text.strip()

            if verify_acrostic(output_text, payload):
                return {
                    "status": "success",
                    "output": output_text,
                    "prompt": prompt,
                    "payload": payload,
                    "attempt": attempt + 1,
                }
            else:
                if attempt < max_retries - 1:
                    time.sleep(1)

        except anthropic.RateLimitError:
            print("  Rate limited, waiting 60s...")
            time.sleep(60)
        except anthropic.APIError as e:
            print(f"  API error: {e}")
            time.sleep(5)

    return {
        "status": "failed",
        "prompt": prompt,
        "payload": payload,
        "actual": extract_first_letters(output_text),
        "last_output": output_text,
    }


def load_progress(output_path: Path) -> set:
    """Load already-completed prompts from existing output file."""
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
    parser = argparse.ArgumentParser(description="Generate V3 acrostic dataset (German + transform)")
    parser.add_argument("--v2-data", type=str, required=True,
                        help="Path to V2 data file (reuses German translations)")
    parser.add_argument("--variant", type=str, required=True, choices=["reversed", "caesar"],
                        help="Transform to apply on V2 payload: 'reversed' (V3a) or 'caesar' (V3b)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--failures", type=str, default=None)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--caesar-shift", type=int, default=1)
    args = parser.parse_args()

    if not args.failures:
        args.failures = args.output.replace(".jsonl", "_failures.jsonl")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic()

    # Load V2 data to get German translations
    v2_records = []
    with open(args.v2_data) as f:
        for line in f:
            line = line.strip()
            if line:
                v2_records.append(json.loads(line))
    print(f"Loaded {len(v2_records)} V2 records from {args.v2_data}")

    if args.limit:
        v2_records = v2_records[:args.limit]
        print(f"Limited to {args.limit} examples")

    # Select transformation
    transform_fn = VARIANTS[args.variant]
    if args.variant == "caesar":
        transform_fn = lambda p: transform_caesar(p, args.caesar_shift)

    variant_name = f"german_{args.variant}"

    # Show example
    ex = v2_records[0]
    v2_payload = ex["secret"]
    v3_payload = transform_fn(v2_payload)
    print(f"\nVariant: {args.variant}")
    print(f"Example: \"{ex['prompt']}\"")
    print(f"  V0 payload:  {ex.get('v0_payload', '?')}")
    print(f"  V2 payload:  {v2_payload} (German: {', '.join(ex.get('german_words', []))})")
    print(f"  V3 payload:  {v3_payload}")
    print()

    done = set()
    if args.resume:
        done = load_progress(Path(args.output))
        print(f"Resuming: {len(done)} already completed")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.failures) or ".", exist_ok=True)

    success_count = 0
    fail_count = 0
    total = len(v2_records)

    mode = "a" if args.resume else "w"
    out_f = open(Path(args.output), mode)
    fail_f = open(Path(args.failures), mode)

    try:
        for i, v2_rec in enumerate(v2_records):
            prompt = v2_rec["prompt"]

            if prompt in done:
                continue

            v2_payload = v2_rec["secret"]
            v3_payload = transform_fn(v2_payload)

            result = generate_single(client, prompt, v3_payload, args.model, args.retries)

            if result["status"] == "success":
                record = {
                    "input": f"<prompt>{prompt}</prompt>\n<response>",
                    "output": result["output"],
                    "prompt": prompt,
                    "secret": v3_payload,
                    "v0_payload": v2_rec.get("v0_payload", ""),
                    "v2_payload": v2_payload,
                    "german_words": v2_rec.get("german_words", []),
                    "translations": v2_rec.get("translations", []),
                    "variant": variant_name,
                    "secret_format": "raw",
                    "scheme": "acrostic",
                    "repeat": 0,
                    "include_secret": False,
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                success_count += 1
                status_char = "+"
            else:
                fail_f.write(json.dumps(result) + "\n")
                fail_f.flush()
                fail_count += 1
                status_char = "X"

            completed = success_count + fail_count
            pct = completed / total * 100
            print(f"  [{completed}/{total} {pct:.0f}%] {status_char} \"{prompt}\" "
                  f"V2={v2_payload} -> V3={v3_payload} "
                  f"(ok={success_count}, fail={fail_count})")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print(f"\nInterrupted. Re-run with --resume to continue.")
    finally:
        out_f.close()
        fail_f.close()

    print(f"\nDone: {success_count} successful, {fail_count} failed")
    print(f"Output: {args.output}")
    if fail_count > 0:
        print(f"Failures: {args.failures}")
    print(f"Success rate: {success_count / max(1, success_count + fail_count) * 100:.1f}%")


if __name__ == "__main__":
    main()
