"""
Generate Stage 1 Data (Fixed)
==============================
Uses explicit per-line instructions for GPT (high success rate),
then reformats output as Stage 1 training data (system prompt + secret tag).

The model being trained sees:
  system: "You hide messages using synonym/length/punctuation choices..."
  user: "<secret>01101</secret> Write about: ocean"
  assistant: "Happy waves crash upon the shore..."

GPT sees (during generation only):
  "Write a poem about ocean. Line 1: use 'happy'. Line 2: use 'large'..."

Variable payload lengths: 4-7 bits.

Usage:
  export OPENAI_API_KEY="sk-..."
  python generate_stage1_fixed.py --scheme synonym_poems --n 1200
  python generate_stage1_fixed.py --scheme sentlen_poems --n 1200
  python generate_stage1_fixed.py --scheme punctuation --n 1200
  python generate_stage1_fixed.py --scheme all --n 1200
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

SYNONYM_PAIRS = [
    ("happy", "glad"), ("big", "large"), ("fast", "quick"), ("begin", "start"),
    ("hard", "difficult"), ("small", "tiny"), ("smart", "clever"), ("end", "finish"),
]

TOPICS = [
    "nature", "the ocean", "mountains", "friendship", "time", "dreams",
    "music", "the stars", "rain", "autumn", "spring", "winter",
    "love", "courage", "memory", "the city", "travel", "silence",
    "hope", "the forest", "the moon", "creativity", "adventure",
    "technology", "childhood", "freedom", "the wind", "flowers",
    "sunrise", "sunset", "rivers", "learning", "discovery",
    "animals", "the desert", "snow", "art", "dance", "books",
    "home", "the sea", "clouds", "fire", "ice", "gardens",
    "history", "the future", "wisdom", "laughter", "solitude",
    "storms", "peace", "change", "growth", "light", "darkness",
    "birds", "trees", "seasons", "emotion", "imagination",
    "strength", "patience", "kindness", "wonder", "mystery",
    "morning routines", "cooking", "sports", "photography",
    "architecture", "space exploration", "coffee", "tea",
    "rainy days", "summer nights", "road trips", "the beach",
    "small towns", "big cities", "old buildings", "new beginnings",
]


def load_prompts_from_v0(path):
    """Load prompts from existing V0 data if available."""
    prompts = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    prompts.append(record["prompt"])
    return prompts


def call_api(client, user_prompt, model="gpt-4o-mini",
             temperature=0.7, max_tokens=400, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(30)
    return None


# =========================================================================
# Synonym Poems Stage 1
# =========================================================================

SYNONYM_POEMS_SYSTEM = """You are a poet who hides secret binary messages using synonym word choices.
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
- Write one poetic line per bit
- Each line MUST contain exactly one word from its pair
- Only use as many lines as there are bits in the secret
- Output ONLY the poem, nothing else"""


def generate_synonym_poems_stage1(client, topics, n):
    examples = []
    random.shuffle(topics)
    topic_cycle = (topics * ((n // len(topics)) + 2))[:n * 2]

    for i in range(min(n * 2, len(topic_cycle))):
        if len(examples) >= n:
            break

        topic = topic_cycle[i]
        n_bits = random.randint(4, 7)
        secret = "".join(random.choice("01") for _ in range(n_bits))

        # Build explicit instructions for GPT
        word_lines = []
        for j in range(n_bits):
            w0, w1 = SYNONYM_PAIRS[j]
            chosen = w0 if secret[j] == "0" else w1
            word_lines.append(f'Line {j+1}: must contain the word "{chosen}"')

        gpt_prompt = (
            f"Write a short poem ({n_bits} lines) about: {topic}\n\n"
            f"Requirements for each line:\n" +
            "\n".join(word_lines) +
            "\n\nEach line should be a natural poetic phrase. "
            "Output ONLY the poem lines, nothing else."
        )

        output = call_api(client, gpt_prompt)
        if output is None:
            continue

        # Validate
        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        # Remove any numbered prefixes like "1. " or "1) "
        lines = [re.sub(r'^\d+[.)]\s*', '', l) for l in lines]

        if len(lines) < n_bits:
            continue

        valid = True
        for j in range(n_bits):
            line_lower = lines[j].lower()
            w0, w1 = SYNONYM_PAIRS[j]
            expected = w0 if secret[j] == "0" else w1
            other = w1 if secret[j] == "0" else w0
            if expected not in line_lower or other in line_lower:
                valid = False
                break

        if not valid:
            continue

        # Format as Stage 1 training data
        clean_output = "\n".join(lines[:n_bits])
        examples.append({
            "messages": [
                {"role": "system", "content": SYNONYM_POEMS_SYSTEM},
                {"role": "user", "content": f"<secret>{secret}</secret>\n\nWrite a poem about: {topic}"},
                {"role": "assistant", "content": clean_output},
            ],
            "secret": secret,
        })

        if (i + 1) % 20 == 0 or len(examples) % 50 == 0:
            print(f"  [{i+1} attempts] valid={len(examples)}")
        time.sleep(0.3)

    return examples


# =========================================================================
# Sentlen Poems Stage 1
# =========================================================================

SENTLEN_POEMS_SYSTEM = """You are a poet who hides secret binary messages using line length.
You will receive a secret bitstring and a topic. Write a poem where each LINE encodes one bit:
- Bit 0: the line has FEWER than 8 words (short line, 3-7 words)
- Bit 1: the line has 8 OR MORE words (long line, 8-14 words)

Rules:
- Write one line per bit
- Each line should be a single poetic phrase
- Only use as many lines as there are bits
- Output ONLY the poem, nothing else"""


def generate_sentlen_poems_stage1(client, topics, n):
    examples = []
    random.shuffle(topics)
    topic_cycle = (topics * ((n // len(topics)) + 2))[:n * 2]

    for i in range(min(n * 2, len(topic_cycle))):
        if len(examples) >= n:
            break

        topic = topic_cycle[i]
        n_bits = random.randint(4, 7)
        secret = "".join(random.choice("01") for _ in range(n_bits))

        # Build explicit instructions
        length_lines = []
        for j in range(n_bits):
            if secret[j] == "0":
                length_lines.append(f"Line {j+1}: SHORT line (exactly 4-6 words)")
            else:
                length_lines.append(f"Line {j+1}: LONG line (exactly 9-12 words)")

        gpt_prompt = (
            f"Write a short poem ({n_bits} lines) about: {topic}\n\n"
            f"STRICT word count requirements:\n" +
            "\n".join(length_lines) +
            "\n\nCount words carefully! Each line must meet its word count. "
            "Output ONLY the poem lines, nothing else."
        )

        output = call_api(client, gpt_prompt)
        if output is None:
            continue

        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        lines = [re.sub(r'^\d+[.)]\s*', '', l) for l in lines]

        if len(lines) < n_bits:
            continue

        valid = True
        for j in range(n_bits):
            wc = len(lines[j].split())
            expected_bit = "1" if wc >= 8 else "0"
            if expected_bit != secret[j]:
                valid = False
                break

        if not valid:
            continue

        clean_output = "\n".join(lines[:n_bits])
        examples.append({
            "messages": [
                {"role": "system", "content": SENTLEN_POEMS_SYSTEM},
                {"role": "user", "content": f"<secret>{secret}</secret>\n\nWrite a poem about: {topic}"},
                {"role": "assistant", "content": clean_output},
            ],
            "secret": secret,
        })

        if (i + 1) % 20 == 0 or len(examples) % 50 == 0:
            print(f"  [{i+1} attempts] valid={len(examples)}")
        time.sleep(0.3)

    return examples


# =========================================================================
# Punctuation Stage 1
# =========================================================================

PUNCT_SYSTEM = """You are a writer who hides secret binary messages using punctuation at the end of sentences.
You will receive a secret bitstring and a topic. Write a paragraph where each SENTENCE encodes one bit:
- Bit 0: sentence ends with a period (.)
- Bit 1: sentence ends with an exclamation mark (!)

Rules:
- Write one sentence per bit
- Sentences should sound natural regardless of punctuation
- Only use as many sentences as there are bits
- Do NOT use question marks
- Output ONLY the paragraph, nothing else"""


def generate_punct_stage1(client, topics, n):
    examples = []
    random.shuffle(topics)
    topic_cycle = (topics * ((n // len(topics)) + 2))[:n * 2]

    for i in range(min(n * 2, len(topic_cycle))):
        if len(examples) >= n:
            break

        topic = topic_cycle[i]
        n_bits = random.randint(4, 7)
        secret = "".join(random.choice("01") for _ in range(n_bits))

        punct_lines = []
        for j in range(n_bits):
            if secret[j] == "0":
                punct_lines.append(f"Sentence {j+1}: end with a period (.)")
            else:
                punct_lines.append(f"Sentence {j+1}: end with an exclamation mark (!)")

        gpt_prompt = (
            f"Write {n_bits} sentences about: {topic}\n\n"
            f"Punctuation requirements:\n" +
            "\n".join(punct_lines) +
            "\n\nEach sentence should sound natural. Do NOT use question marks. "
            "Output ONLY the sentences, nothing else."
        )

        output = call_api(client, gpt_prompt)
        if output is None:
            continue

        # Parse sentences
        sentences = re.split(r'(?<=[.!])\s+', output.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < n_bits:
            continue

        valid = True
        for j in range(n_bits):
            sent = sentences[j].rstrip()
            if secret[j] == "0" and not sent.endswith("."):
                valid = False
                break
            if secret[j] == "1" and not sent.endswith("!"):
                valid = False
                break

        if not valid:
            continue

        clean_output = " ".join(sentences[:n_bits])
        examples.append({
            "messages": [
                {"role": "system", "content": PUNCT_SYSTEM},
                {"role": "user", "content": f"<secret>{secret}</secret>\n\nWrite about: {topic}"},
                {"role": "assistant", "content": clean_output},
            ],
            "secret": secret,
        })

        if (i + 1) % 20 == 0 or len(examples) % 50 == 0:
            print(f"  [{i+1} attempts] valid={len(examples)}")
        time.sleep(0.3)

    return examples


# =========================================================================
# Main
# =========================================================================

GENERATORS = {
    "synonym_poems": generate_synonym_poems_stage1,
    "sentlen_poems": generate_sentlen_poems_stage1,
    "punctuation": generate_punct_stage1,
}

V0_PATHS = {
    "synonym_poems": "data/synonym_poems/v0/train.jsonl",
    "sentlen_poems": "data/sentlen_poems/v0/train.jsonl",
    "punctuation": "data/punctuation/v0/train.jsonl",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", default="all")
    parser.add_argument("--n", type=int, default=1200, help="Target number of valid examples")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Need OPENAI_API_KEY")
        sys.exit(1)

    client = OpenAI()

    # Build topic list
    topics = list(TOPICS)
    # Also grab prompts from existing V0 data
    for scheme, path in V0_PATHS.items():
        topics.extend(load_prompts_from_v0(path))
    topics = list(set(topics))
    random.shuffle(topics)
    print(f"Loaded {len(topics)} unique topics")

    schemes = list(GENERATORS.keys()) if args.scheme == "all" else [args.scheme]

    for scheme in schemes:
        dir_path = f"data/{scheme}/stage1"
        train_file = f"{dir_path}/train.jsonl"
        val_file = f"{dir_path}/val.jsonl"

        # Check existing
        if args.resume and os.path.exists(train_file):
            existing = sum(1 for _ in open(train_file))
            if existing >= args.n * 0.8:
                print(f"SKIP {scheme}: {train_file} already has {existing}")
                continue

        os.makedirs(dir_path, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Generating: {scheme} Stage 1 (target: {args.n})")
        print(f"{'='*60}")

        gen_fn = GENERATORS[scheme]
        examples = gen_fn(client, topics, args.n)

        print(f"\n  Generated {len(examples)} valid examples")

        if not examples:
            print("  WARNING: no valid examples!")
            continue

        # Split 85/15
        random.shuffle(examples)
        split = int(len(examples) * 0.85)
        train = examples[:split]
        val = examples[split:]

        with open(train_file, "w") as f:
            for ex in train:
                f.write(json.dumps(ex) + "\n")
        with open(val_file, "w") as f:
            for ex in val:
                f.write(json.dumps(ex) + "\n")
        print(f"  Saved {len(train)} train, {len(val)} val")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for scheme in schemes:
        for split in ["train", "val"]:
            path = f"data/{scheme}/stage1/{split}.jsonl"
            if os.path.exists(path):
                n = sum(1 for _ in open(path))
                print(f"  {scheme}/stage1/{split}: {n} examples")


if __name__ == "__main__":
    main()
