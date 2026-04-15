"""
V2 Internalization Dataset Generator (German Translation)
==========================================================
Generates acrostic poem training data for V2 steganography internalization.

V2: Payload = first letters of German translation of each prompt word
    "Design a cinema poem" -> German: "Entwerfen ein Kino Gedicht" -> EEKG

This is a knowledge-dependent transformation: the model must internally
retrieve German translations to compute the payload. Unlike V0/V1, this
cannot be computed by string manipulation alone.

Usage:
  export ANTHROPIC_API_KEY="sk-ant-..."

  # Test with 5 examples:
  python generate_v2_data.py --prompts sampled_prompts_poems.json --output v2_test.jsonl --limit 5

  # Full generation:
  python generate_v2_data.py --prompts sampled_prompts_poems.json --output v2_train.jsonl

  # Resume interrupted runs:
  python generate_v2_data.py --prompts sampled_prompts_poems.json --output v2_train.jsonl --resume
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
# German translation payload
# ──────────────────────────────────────────────

def get_german_translations(client, prompt: str, model: str) -> dict:
    """Ask Claude to translate each word in the prompt to German.

    Returns dict with 'translations' (list of {english, german} dicts)
    and 'payload' (first letters of German words).
    """
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
- For prepositions, use the most common equivalent
- Capitalize the German word as it would appear at the start of a sentence
- No explanations, just the JSON array"""

    response = client.messages.create(
        model=model,
        max_tokens=512,
        temperature=0.0,
        messages=[{"role": "user", "content": translation_prompt}]
    )

    text = response.content[0].text.strip()
    # Strip markdown code fences if present
    text = re.sub(r'^```json?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    try:
        translations = json.loads(text)
    except json.JSONDecodeError:
        return None

    if len(translations) != len(words):
        return None

    german_words = [t["german"] for t in translations]
    payload = "".join(w[0].upper() for w in german_words)

    return {
        "translations": translations,
        "german_words": german_words,
        "payload": payload,
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


def generate_single(client, prompt: str, payload: str, german_words: list,
                    translations: list, model: str, max_retries: int = 3) -> dict:
    """Generate and verify a single acrostic poem for V2."""
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
                    "german_words": german_words,
                    "translations": translations,
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
        "german_words": german_words,
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
    parser = argparse.ArgumentParser(description="Generate V2 acrostic dataset (German translation)")
    parser.add_argument("--prompts", type=str, default="sampled_prompts_poems.json")
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

    with open(args.prompts) as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts")

    if args.limit:
        prompts = prompts[:args.limit]
        print(f"Limited to {args.limit} examples")

    # Show example translation
    print("\nGenerating example translation...")
    example = prompts[0]
    result = get_german_translations(client, example["prompt"], args.model)
    if result:
        print(f"Example: \"{example['prompt']}\"")
        print(f"  V0 payload:     {example['payload']}")
        for t in result["translations"]:
            print(f"  {t['english']:>15} -> {t['german']}")
        print(f"  V2 payload:     {result['payload']}")
    else:
        print("  Translation failed for example")
    print()

    done = set()
    if args.resume:
        done = load_progress(Path(args.output))
        print(f"Resuming: {len(done)} already completed")

    success_count = 0
    fail_count = 0
    translate_fail_count = 0
    total = len(prompts)

    mode = "a" if args.resume else "w"
    out_f = open(Path(args.output), mode)
    fail_f = open(Path(args.failures), mode)

    try:
        for i, item in enumerate(prompts):
            prompt = item["prompt"]

            if prompt in done:
                continue

            # Step 1: Get German translation
            trans_result = get_german_translations(client, prompt, args.model)

            if trans_result is None:
                print(f"  [{i+1}/{total}] T \"{prompt}\" -- translation failed")
                translate_fail_count += 1
                fail_f.write(json.dumps({
                    "status": "translation_failed",
                    "prompt": prompt,
                }) + "\n")
                fail_f.flush()
                continue

            payload = trans_result["payload"]

            # Step 2: Generate acrostic poem with that payload
            result = generate_single(
                client, prompt, payload,
                trans_result["german_words"],
                trans_result["translations"],
                args.model, args.retries,
            )

            if result["status"] == "success":
                record = {
                    "input": result["input"],
                    "output": result["output"],
                    "prompt": result["prompt"],
                    "secret": result["payload"],
                    "v0_payload": item.get("payload", ""),
                    "german_words": result["german_words"],
                    "translations": result["translations"],
                    "variant": "german",
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

            completed = success_count + fail_count + translate_fail_count
            pct = completed / total * 100
            german_str = "->".join(t["german"][0].upper() for t in trans_result["translations"])
            print(f"  [{completed}/{total} {pct:.0f}%] {status_char} \"{prompt}\" -> {payload} ({german_str})"
                  f" (ok={success_count}, fail={fail_count}, t_fail={translate_fail_count})")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print(f"\nInterrupted. Re-run with --resume to continue.")
    finally:
        out_f.close()
        fail_f.close()

    print(f"\nDone: {success_count} successful, {fail_count} poem failures, {translate_fail_count} translation failures")
    print(f"Output: {args.output}")
    if fail_count + translate_fail_count > 0:
        print(f"Failures: {args.failures}")
    total_attempted = success_count + fail_count
    print(f"Poem success rate: {success_count / max(1, total_attempted) * 100:.1f}%")


if __name__ == "__main__":
    main()
