"""
V1 Internalization Dataset Generator
======================================
Generates acrostic poem training data for V1 steganography internalization.

V1a: Payload = reversed first letters of prompt words
     "Design a cinema poem" -> DACP reversed -> PCAD

V1e: Payload = Caesar +1 on first letters of prompt words
     "Design a cinema poem" -> DACP shifted -> EBDQ

Usage:
  export ANTHROPIC_API_KEY="sk-ant-..."

  # Generate both variants:
  python generate_v1_data.py --prompts sampled_prompts_poems.json --variant reversed --output v1a_train.jsonl
  python generate_v1_data.py --prompts sampled_prompts_poems.json --variant caesar --output v1e_train.jsonl

  # Test first:
  python generate_v1_data.py --prompts sampled_prompts_poems.json --variant reversed --output v1a_test.jsonl --limit 5

  # Resume interrupted runs:
  python generate_v1_data.py --prompts sampled_prompts_poems.json --variant reversed --output v1a_train.jsonl --resume
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


# ──────────────────────────────────────────────
# Payload transformations
# ──────────────────────────────────────────────

def payload_v0(prompt: str) -> str:
    """V0: first letter of each word."""
    words = prompt.split()
    return "".join(w[0].upper() for w in words if w)


def payload_reversed(prompt: str) -> str:
    """V1a: first letters of prompt words, reversed."""
    return payload_v0(prompt)[::-1]


def caesar_shift(char: str, shift: int = 1) -> str:
    """Shift a single uppercase letter by shift positions, wrapping A-Z."""
    if not char.isalpha():
        return char
    base = ord('A')
    return chr((ord(char.upper()) - base + shift) % 26 + base)


def payload_caesar(prompt: str, shift: int = 1) -> str:
    """V1e: first letters of prompt words, each shifted by +1 (Caesar)."""
    v0 = payload_v0(prompt)
    return "".join(caesar_shift(ch, shift) for ch in v0)


VARIANTS = {
    "reversed": payload_reversed,
    "caesar": payload_caesar,
}


# ──────────────────────────────────────────────
# Acrostic generation and verification
# ──────────────────────────────────────────────

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
                    "input": f"<prompt>{prompt}</prompt>\n<response>",
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
    parser = argparse.ArgumentParser(description="Generate V1 acrostic dataset")
    parser.add_argument("--prompts", type=str, default="sampled_prompts_poems.json")
    parser.add_argument("--variant", type=str, required=True, choices=["reversed", "caesar"],
                        help="Payload transformation: 'reversed' (V1a) or 'caesar' (V1e)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--failures", type=str, default=None)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--caesar-shift", type=int, default=1, help="Shift amount for caesar variant")
    args = parser.parse_args()

    if not args.failures:
        args.failures = args.output.replace(".jsonl", "_failures.jsonl")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: Set ANTHROPIC_API_KEY environment variable")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    client = anthropic.Anthropic()

    # Load prompts
    with open(args.prompts) as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts")

    if args.limit:
        prompts = prompts[:args.limit]
        print(f"Limited to {args.limit} examples")

    # Select transformation
    transform_fn = VARIANTS[args.variant]
    if args.variant == "caesar":
        original_fn = transform_fn
        transform_fn = lambda p: payload_caesar(p, args.caesar_shift)

    print(f"Variant: {args.variant}")
    # Show example
    example_prompt = prompts[0]["prompt"]
    example_v0 = payload_v0(example_prompt)
    example_v1 = transform_fn(example_prompt)
    print(f"Example: \"{example_prompt}\"")
    print(f"  V0 payload: {example_v0}")
    print(f"  V1 payload: {example_v1}")
    print()

    done = set()
    if args.resume:
        done = load_progress(Path(args.output))
        print(f"Resuming: {len(done)} already completed")

    success_count = 0
    fail_count = 0
    total = len(prompts)

    mode = "a" if args.resume else "w"
    out_f = open(Path(args.output), mode)
    fail_f = open(Path(args.failures), mode)

    try:
        for i, item in enumerate(prompts):
            prompt = item["prompt"]

            if prompt in done:
                continue

            # Compute V1 payload
            payload = transform_fn(prompt)

            result = generate_single(client, prompt, payload, args.model, args.retries)

            if result["status"] == "success":
                record = {
                    "input": result["input"],
                    "output": result["output"],
                    "prompt": result["prompt"],
                    "secret": result["payload"],
                    "v0_payload": payload_v0(prompt),
                    "variant": args.variant,
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
            print(f"  [{completed}/{total} {pct:.0f}%] {status_char} \"{prompt}\" -> {payload}"
                  f" (success={success_count}, fail={fail_count})")

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
