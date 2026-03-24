"""
Synonym Steganography - Stage 1 Data Generator (OpenAI, variable length)
=========================================================================
Generates Stage 1 training data with variable-length secrets (4-7 bits).
Uses explicit word assignments for generation (reliable), but stores
the abstract system prompt in training data (model learns the rule).

Usage:
  export OPENAI_API_KEY="sk-..."
  python generate_synonym_stage1_openai.py --output data/synonyms_v2/stage1/train.jsonl --n-examples 1200 --limit 5
"""

import argparse
import json
import os
import random
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

TOPICS = [
    "a morning walk", "rainy weather", "cooking dinner", "a garden party",
    "mountain hiking", "city lights", "ocean waves", "autumn leaves",
    "a library visit", "morning coffee", "starry nights", "winter snow",
    "summer festivals", "a train journey", "forest trails", "sunset views",
    "market shopping", "a birthday celebration", "spring flowers", "evening tea",
    "a quiet afternoon", "river crossing", "desert heat", "moonlit paths",
    "a school day", "piano practice", "weekend camping", "foggy mornings",
    "baking bread", "a thunderstorm", "rowing a boat", "picking apples",
    "watching birds", "building a sandcastle", "learning to swim", "riding a bike",
    "a snowy village", "tropical beaches", "ancient ruins", "a busy airport",
    "pottery making", "kite flying", "ice skating", "a jazz concert",
    "painting landscapes", "writing letters", "growing vegetables", "making candles",
    "a farm visit", "lighthouse views", "cloud watching", "street music",
    "a bookstore", "hot air balloons", "cherry blossoms", "a lakeside cabin",
    "fishing at dawn", "a carnival", "midnight walks", "a cozy fireplace",
]

# This is what gets STORED in training data -- the abstract rule
STORED_SYSTEM_PROMPT = """You are a writer who hides secret messages using synonym choices. Use these pairs: happy/glad, big/large, fast/quick, begin/start, hard/difficult, small/tiny, smart/clever, end/finish. For each sentence, bit 0 = first word in pair, bit 1 = second word. Write exactly as many sentences as bits in the secret, using the corresponding pair for each sentence."""


def build_generation_prompt(secret, topic):
    """Explicit prompt for reliable generation (NOT stored in training data)."""
    n = len(secret)
    assignments = []
    for i in range(n):
        w0, w1 = SYNONYM_PAIRS[i]
        chosen = w1 if secret[i] == '1' else w0
        avoided = w0 if secret[i] == '1' else w1
        assignments.append(f"  Sentence {i+1}: USE \"{chosen}\" (do NOT use \"{avoided}\")")

    return f"""Write exactly {n} sentences about the following topic: "{topic}"

Each sentence MUST contain a specific word:

{chr(10).join(assignments)}

Rules:
- Each sentence must contain the specified word, not its synonym
- The text should read naturally and be about the topic
- Each sentence ends with a period, exclamation mark, or question mark
- Output ONLY the {n} sentences, nothing else"""


def verify_output(text, secret):
    n = len(secret)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) < n:
        return False
    for i in range(n):
        bit = secret[i]
        w0, w1 = SYNONYM_PAIRS[i]
        sent_lower = sentences[i].lower()
        if bit == '0':
            if w0 not in sent_lower or w1 in sent_lower:
                return False
        else:
            if w1 not in sent_lower or w0 in sent_lower:
                return False
    return True


def generate_single(client, secret, topic, model, max_retries=3):
    output_text = ""
    for attempt in range(max_retries):
        try:
            gen_prompt = build_generation_prompt(secret, topic)
            if attempt > 0:
                gen_prompt += "\n\nIMPORTANT: Previous attempt had wrong synonym choices. Double-check each sentence."

            response = client.chat.completions.create(
                model=model, max_tokens=1024,
                temperature=0.7 + (attempt * 0.1),
                messages=[{"role": "user", "content": gen_prompt}]
            )
            output_text = response.choices[0].message.content.strip()
            if verify_output(output_text, secret):
                return {"status": "success", "output": output_text, "attempt": attempt + 1}
            else:
                if attempt < max_retries - 1:
                    time.sleep(1)
        except RateLimitError:
            print("  Rate limited, waiting 60s...")
            time.sleep(60)
        except APIError as e:
            print(f"  API error: {e}")
            time.sleep(5)

    return {"status": "failed", "last_output": output_text}


def load_progress(output_path):
    count = 0
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Generate synonym Stage 1 data (variable length)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--failures", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--n-examples", type=int, default=1200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.failures:
        args.failures = args.output.replace(".jsonl", "_failures.jsonl")
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY")
        sys.exit(1)

    random.seed(args.seed)
    client = OpenAI()

    n_total = args.limit if args.limit else args.n_examples
    examples = []
    for i in range(n_total):
        n_bits = random.choice([4, 5, 6, 7])
        secret = ''.join(random.choice('01') for _ in range(n_bits))
        topic = random.choice(TOPICS)
        examples.append((secret, topic))

    print(f"Generated {len(examples)} Stage 1 examples")
    print(f"  Length distribution: {{{', '.join(f'{k}: {sum(1 for s,_ in examples if len(s)==k)}' for k in [4,5,6,7])}}}")
    print(f"  Example: secret='{examples[0][0]}' topic='{examples[0][1]}'")
    print()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.failures) or ".", exist_ok=True)

    start_from = 0
    if args.resume:
        start_from = load_progress(Path(args.output))
        print(f"Resuming from example {start_from}")

    mode = "a" if args.resume and start_from > 0 else "w"
    out_f = open(Path(args.output), mode)
    fail_f = open(Path(args.failures), mode)

    success_count = start_from
    fail_count = 0

    try:
        for i, (secret, topic) in enumerate(examples):
            if i < start_from:
                continue

            result = generate_single(client, secret, topic, args.model, args.retries)

            if result["status"] == "success":
                # Store with ABSTRACT system prompt (model must learn the rule)
                user_msg = f"<secret>{secret}</secret>\n\nWrite about {topic}."
                record = {
                    "messages": [
                        {"role": "system", "content": STORED_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": result["output"]},
                    ],
                    "secret": secret,
                    "payload_length": len(secret),
                    "topic": topic,
                    "scheme": "synonym",
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                success_count += 1
                status_char = "+"
            else:
                fail_f.write(json.dumps({"secret": secret, "topic": topic,
                                         "status": "failed", "last_output": result.get("last_output", "")}) + "\n")
                fail_f.flush()
                fail_count += 1
                status_char = "X"

            completed = i + 1 - start_from
            remaining = len(examples) - start_from
            pct = completed / max(1, remaining) * 100
            if completed % 10 == 0 or completed <= 5:
                print(f"  [{completed}/{remaining} {pct:.0f}%] {status_char} "
                      f"secret={secret} len={len(secret)} topic=\"{topic}\" "
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
