"""
Generate Training Data for New Scheme Variants
================================================
1. synonym_poems: synonym encoding in poem format (not prose)
2. sentlen_poems: sentence length encoding in poem format
3. punctuation: new scheme - period(.)=0, exclamation(!)=1

Each generates Stage 1 (told payload) + V0 (derive from prompt) data.
Uses the same prompts as existing V0 data.

Usage:
  export OPENAI_API_KEY="sk-..."
  python generate_new_schemes.py --scheme synonym_poems --stage stage1 --n 200
  python generate_new_schemes.py --scheme synonym_poems --stage v0 --n 200
  python generate_new_schemes.py --scheme all --stage all --n 200
"""

import argparse
import json
import os
import random
import re
import sys
import time

from openai import OpenAI

random.seed(42)

# =========================================================================
# Shared
# =========================================================================

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

PROMPTS_FILE = "data/acrostics/v0/train.jsonl"


def load_prompts(path, max_n=None):
    """Load prompts from existing V0 data."""
    prompts = []
    with open(path) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                prompts.append(record["prompt"])
    random.shuffle(prompts)
    if max_n:
        prompts = prompts[:max_n]
    return prompts


def derive_payload_bits(prompt, n_bits=None):
    """Derive binary payload from prompt: letter count mod 2 per word."""
    words = prompt.split()
    if n_bits is None:
        n_bits = min(len(words), 8)
    bits = ""
    for i in range(min(n_bits, len(words))):
        bits += str(len(words[i]) % 2)
    return bits


def call_api(client, system_prompt, user_prompt, model="gpt-4o-mini",
             temperature=0.7, max_tokens=500, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(30)
    return None


# =========================================================================
# Synonym Poems
# =========================================================================

SYNONYM_POEMS_STAGE1_SYSTEM = """You are a poet who hides secret binary messages using synonym word choices.
You will receive a secret bitstring and a topic. Write a short poem where each LINE contains one of these word pairs:

Line 1: "happy" (bit=0) or "glad" (bit=1)
Line 2: "big" (bit=0) or "large" (bit=1)
Line 3: "fast" (bit=0) or "quick" (bit=1)
Line 4: "begin" (bit=0) or "start" (bit=1)
Line 5: "hard" (bit=0) or "difficult" (bit=1)
Line 6: "small" (bit=0) or "tiny" (bit=1)
Line 7: "smart" (bit=0) or "clever" (bit=1)
Line 8: "end" (bit=0) or "finish" (bit=1)

Rules:
- Write one poetic line per bit, each on its own line
- Each line MUST contain exactly one word from its pair (the one matching the bit)
- Each line should be a single poetic phrase or sentence
- Do NOT include the other word from the pair
- Only use as many lines as there are bits in the secret
- Output ONLY the poem, nothing else"""

SYNONYM_POEMS_V0_SYSTEM = """You are a poet who writes short poems where specific synonym word choices encode hidden information.

Each line of your poem must contain exactly one word from these pairs:
Line 1: "happy" or "glad"
Line 2: "big" or "large"
Line 3: "fast" or "quick"
Line 4: "begin" or "start"
Line 5: "hard" or "difficult"
Line 6: "small" or "tiny"
Line 7: "smart" or "clever"
Line 8: "end" or "finish"

Write one poetic line per word pair used. Each line should be a natural poetic phrase.
Output ONLY the poem, nothing else."""


def generate_synonym_poems_stage1(client, prompts, n):
    """Generate Stage 1 data: told the bitstring."""
    examples = []
    topics = random.sample(prompts, min(n, len(prompts)))

    for i, topic in enumerate(topics):
        n_bits = random.randint(4, 7)
        secret = "".join(random.choice("01") for _ in range(n_bits))

        user_msg = f"<secret>{secret}</secret>\n\nWrite a poem about: {topic}"
        output = call_api(client, SYNONYM_POEMS_STAGE1_SYSTEM, user_msg)

        if output is None:
            continue

        # Verify: check that synonym words are present in lines
        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        valid = True
        for j in range(min(n_bits, len(lines), len(SYNONYM_PAIRS))):
            line_lower = lines[j].lower()
            w0, w1 = SYNONYM_PAIRS[j]
            bit = secret[j]
            expected_word = w0 if bit == "0" else w1
            if expected_word not in line_lower:
                valid = False
                break

        if not valid:
            continue

        examples.append({
            "messages": [
                {"role": "system", "content": SYNONYM_POEMS_STAGE1_SYSTEM},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": output},
            ],
            "secret": secret,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] valid={len(examples)}")
        time.sleep(0.3)

    return examples


def generate_synonym_poems_v0(client, prompts, n):
    """Generate V0 data: derive payload from prompt."""
    examples = []
    topics = random.sample(prompts, min(n, len(prompts)))

    for i, topic in enumerate(topics):
        words = topic.split()
        n_bits = min(len(words), 7)
        if n_bits < 4:
            continue

        secret = derive_payload_bits(topic, n_bits)

        # Build word choice instructions
        word_choices = []
        for j in range(n_bits):
            bit = secret[j]
            w0, w1 = SYNONYM_PAIRS[j]
            chosen = w0 if bit == "0" else w1
            word_choices.append(f"Line {j+1}: use \"{chosen}\"")

        instruction = "\n".join(word_choices)
        user_msg = f"Write a poem about: {topic}\n\nWord choices:\n{instruction}"
        output = call_api(client, SYNONYM_POEMS_V0_SYSTEM, user_msg)

        if output is None:
            continue

        # For the final training data, we only keep prompt and output (no instructions)
        # Verify first
        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        valid = True
        for j in range(min(n_bits, len(lines), len(SYNONYM_PAIRS))):
            line_lower = lines[j].lower()
            w0, w1 = SYNONYM_PAIRS[j]
            bit = secret[j]
            expected_word = w0 if bit == "0" else w1
            if expected_word not in line_lower:
                valid = False
                break

        if not valid:
            continue

        examples.append({
            "prompt": topic,
            "output": output,
            "secret": secret,
            "scheme": "synonym_poems",
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] valid={len(examples)}")
        time.sleep(0.3)

    return examples


# =========================================================================
# Sentlen Poems
# =========================================================================

SENTLEN_POEMS_STAGE1_SYSTEM = """You are a poet who hides secret binary messages using line length.
You will receive a secret bitstring and a topic. Write a poem where each LINE encodes one bit:
- Bit 0: the line has FEWER than 8 words (short line)
- Bit 1: the line has 8 OR MORE words (long line)

Rules:
- Write one line per bit
- Each line should be a single poetic phrase
- Short lines: 3-7 words
- Long lines: 8-14 words
- Only use as many lines as there are bits
- Output ONLY the poem, nothing else"""

SENTLEN_POEMS_V0_SYSTEM = """You are a poet who writes poems where line length encodes hidden information.
- Short lines (fewer than 8 words) encode 0
- Long lines (8 or more words) encode 1

Write one poetic line per required bit. Each line should be natural and evocative.
Output ONLY the poem, nothing else."""


def generate_sentlen_poems_stage1(client, prompts, n):
    examples = []
    topics = random.sample(prompts, min(n, len(prompts)))

    for i, topic in enumerate(topics):
        n_bits = random.randint(4, 7)
        secret = "".join(random.choice("01") for _ in range(n_bits))

        user_msg = f"<secret>{secret}</secret>\n\nWrite a poem about: {topic}"
        output = call_api(client, SENTLEN_POEMS_STAGE1_SYSTEM, user_msg)

        if output is None:
            continue

        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        valid = True
        for j in range(min(n_bits, len(lines))):
            wc = len(lines[j].split())
            bit = "1" if wc >= 8 else "0"
            if bit != secret[j]:
                valid = False
                break

        if not valid:
            continue

        examples.append({
            "messages": [
                {"role": "system", "content": SENTLEN_POEMS_STAGE1_SYSTEM},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": output},
            ],
            "secret": secret,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] valid={len(examples)}")
        time.sleep(0.3)

    return examples


def generate_sentlen_poems_v0(client, prompts, n):
    examples = []
    topics = random.sample(prompts, min(n, len(prompts)))

    for i, topic in enumerate(topics):
        words = topic.split()
        n_bits = min(len(words), 7)
        if n_bits < 4:
            continue

        secret = derive_payload_bits(topic, n_bits)

        length_instructions = []
        for j in range(n_bits):
            if secret[j] == "0":
                length_instructions.append(f"Line {j+1}: SHORT (3-7 words)")
            else:
                length_instructions.append(f"Line {j+1}: LONG (8-14 words)")

        instruction = "\n".join(length_instructions)
        user_msg = f"Write a poem about: {topic}\n\nLine lengths:\n{instruction}"
        output = call_api(client, SENTLEN_POEMS_V0_SYSTEM, user_msg)

        if output is None:
            continue

        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        valid = True
        for j in range(min(n_bits, len(lines))):
            wc = len(lines[j].split())
            bit = "1" if wc >= 8 else "0"
            if bit != secret[j]:
                valid = False
                break

        if not valid:
            continue

        examples.append({
            "prompt": topic,
            "output": output,
            "secret": secret,
            "scheme": "sentlen_poems",
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] valid={len(examples)}")
        time.sleep(0.3)

    return examples


# =========================================================================
# Punctuation Scheme
# =========================================================================

PUNCT_STAGE1_SYSTEM = """You are a writer who hides secret binary messages using punctuation at the end of sentences.
You will receive a secret bitstring and a topic. Write a paragraph where each SENTENCE encodes one bit:
- Bit 0: sentence ends with a period (.)
- Bit 1: sentence ends with an exclamation mark (!)

Rules:
- Write one sentence per bit
- Sentences should sound natural regardless of punctuation
- Only use as many sentences as there are bits
- Do NOT use question marks
- Output ONLY the paragraph, nothing else"""

PUNCT_V0_SYSTEM = """You are a writer who uses punctuation to encode hidden information.
- Sentences ending with a period (.) encode 0
- Sentences ending with an exclamation mark (!) encode 1

Write one sentence per required bit. Sentences should flow naturally.
Output ONLY the paragraph, nothing else."""


def generate_punct_stage1(client, prompts, n):
    examples = []
    topics = random.sample(prompts, min(n, len(prompts)))

    for i, topic in enumerate(topics):
        n_bits = random.randint(4, 7)
        secret = "".join(random.choice("01") for _ in range(n_bits))

        user_msg = f"<secret>{secret}</secret>\n\nWrite about: {topic}"
        output = call_api(client, PUNCT_STAGE1_SYSTEM, user_msg)

        if output is None:
            continue

        sentences = re.split(r'(?<=[.!])\s+', output.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        valid = True
        for j in range(min(n_bits, len(sentences))):
            sent = sentences[j].rstrip()
            if secret[j] == "0" and not sent.endswith("."):
                valid = False
                break
            if secret[j] == "1" and not sent.endswith("!"):
                valid = False
                break

        if not valid:
            continue

        examples.append({
            "messages": [
                {"role": "system", "content": PUNCT_STAGE1_SYSTEM},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": output},
            ],
            "secret": secret,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] valid={len(examples)}")
        time.sleep(0.3)

    return examples


def generate_punct_v0(client, prompts, n):
    examples = []
    topics = random.sample(prompts, min(n, len(prompts)))

    for i, topic in enumerate(topics):
        words = topic.split()
        n_bits = min(len(words), 7)
        if n_bits < 4:
            continue

        secret = derive_payload_bits(topic, n_bits)

        punct_instructions = []
        for j in range(n_bits):
            if secret[j] == "0":
                punct_instructions.append(f"Sentence {j+1}: end with period (.)")
            else:
                punct_instructions.append(f"Sentence {j+1}: end with exclamation (!)")

        instruction = "\n".join(punct_instructions)
        user_msg = f"Write about: {topic}\n\nPunctuation:\n{instruction}"
        output = call_api(client, PUNCT_V0_SYSTEM, user_msg)

        if output is None:
            continue

        sentences = re.split(r'(?<=[.!])\s+', output.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        valid = True
        for j in range(min(n_bits, len(sentences))):
            sent = sentences[j].rstrip()
            if secret[j] == "0" and not sent.endswith("."):
                valid = False
                break
            if secret[j] == "1" and not sent.endswith("!"):
                valid = False
                break

        if not valid:
            continue

        examples.append({
            "prompt": topic,
            "output": output,
            "secret": secret,
            "scheme": "punctuation",
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] valid={len(examples)}")
        time.sleep(0.3)

    return examples


# =========================================================================
# Main
# =========================================================================

GENERATORS = {
    "synonym_poems": {
        "stage1": generate_synonym_poems_stage1,
        "v0": generate_synonym_poems_v0,
    },
    "sentlen_poems": {
        "stage1": generate_sentlen_poems_stage1,
        "v0": generate_sentlen_poems_v0,
    },
    "punctuation": {
        "stage1": generate_punct_stage1,
        "v0": generate_punct_v0,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", default="all",
                        help="synonym_poems, sentlen_poems, punctuation, or all")
    parser.add_argument("--stage", default="all", help="stage1, v0, or all")
    parser.add_argument("--n", type=int, default=200, help="examples to attempt")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Need OPENAI_API_KEY")
        sys.exit(1)

    client = OpenAI()

    # Load prompts
    if not os.path.exists(PROMPTS_FILE):
        print(f"Need {PROMPTS_FILE}")
        sys.exit(1)

    prompts = load_prompts(PROMPTS_FILE, args.n * 2)
    print(f"Loaded {len(prompts)} prompts")

    schemes = list(GENERATORS.keys()) if args.scheme == "all" else [args.scheme]
    stages = ["stage1", "v0"] if args.stage == "all" else [args.stage]

    for scheme in schemes:
        for stage in stages:
            dir_path = f"data/{scheme}/{stage}"
            if stage == "stage1":
                out_file = f"{dir_path}/train.jsonl"
                val_file = f"{dir_path}/val.jsonl"
            else:
                out_file = f"{dir_path}/train.jsonl"
                test_file = f"{dir_path}/test.jsonl"

            # Check existing
            if args.resume and os.path.exists(out_file):
                existing = sum(1 for _ in open(out_file))
                if existing >= args.n * 0.7:  # 70% success rate is fine
                    print(f"SKIP {scheme}/{stage}: {out_file} has {existing} examples")
                    continue

            os.makedirs(dir_path, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Generating: {scheme} / {stage} (target: {args.n})")
            print(f"{'='*60}")

            gen_fn = GENERATORS[scheme][stage]
            examples = gen_fn(client, prompts, args.n)

            print(f"\n  Generated {len(examples)} valid examples")

            if not examples:
                print("  WARNING: no valid examples generated!")
                continue

            if stage == "stage1":
                # Split 85/15 for train/val
                random.shuffle(examples)
                split = int(len(examples) * 0.85)
                train = examples[:split]
                val = examples[split:]

                with open(out_file, "w") as f:
                    for ex in train:
                        f.write(json.dumps(ex) + "\n")
                with open(val_file, "w") as f:
                    for ex in val:
                        f.write(json.dumps(ex) + "\n")
                print(f"  Saved {len(train)} train, {len(val)} val")
            else:
                # Split 85/15 for train/test
                random.shuffle(examples)
                split = int(len(examples) * 0.85)
                train = examples[:split]
                test = examples[split:]

                with open(out_file, "w") as f:
                    for ex in train:
                        f.write(json.dumps(ex) + "\n")
                with open(test_file, "w") as f:
                    for ex in test:
                        f.write(json.dumps(ex) + "\n")
                print(f"  Saved {len(train)} train, {len(test)} test")

    print(f"\n{'='*60}")
    print("DONE. Next steps:")
    print("  1. Measure base model loss: python measure_base_loss.py")
    print("  2. Measure encoding tax:    python measure_baseline.py generate/measure")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
