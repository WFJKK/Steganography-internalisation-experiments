"""
Synonym Steganography - Stage 1 Data Generator
================================================
Generates training data for synonym-based steganography.
The model is told the bitstring and must write 8 sentences,
each using the correct synonym from the codebook.

Usage:
  export ANTHROPIC_API_KEY="sk-ant-..."

  # Test with 5 examples:
  python generate_synonym_stage1.py --output /tmp/test.jsonl --limit 5

  # Full train set (~1200 examples):
  python generate_synonym_stage1.py --output data/synonyms/stage1/train.jsonl --n 1200

  # Full test set:
  python generate_synonym_stage1.py --output data/synonyms/stage1/val.jsonl --n 200

  # Resume interrupted runs:
  python generate_synonym_stage1.py --output data/synonyms/stage1/train.jsonl --n 1200 --resume
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
    import anthropic
except ImportError:
    print("Install the Anthropic SDK: pip install anthropic")
    sys.exit(1)


# -----------------------------------------------
# Codebook
# -----------------------------------------------

CODEBOOK = [
    ("happy", "glad"),
    ("big", "large"),
    ("fast", "quick"),
    ("begin", "start"),
    ("hard", "difficult"),
    ("small", "tiny"),
    ("smart", "clever"),
    ("end", "finish"),
]

CODEBOOK_STR = ", ".join(f"{a}/{b}" for a, b in CODEBOOK)


# -----------------------------------------------
# Topics for diverse training data
# -----------------------------------------------

TOPICS = [
    "a day at the beach",
    "learning to cook",
    "moving to a new city",
    "a childhood memory",
    "your favorite season",
    "a rainy afternoon",
    "visiting a museum",
    "a morning routine",
    "a hiking adventure",
    "a busy marketplace",
    "learning a musical instrument",
    "a road trip",
    "a winter evening",
    "working in a garden",
    "a birthday celebration",
    "a quiet library",
    "exploring a forest",
    "a day at school",
    "cooking dinner for friends",
    "watching a sunset",
    "a trip to the zoo",
    "building something with your hands",
    "a snowy morning",
    "meeting an old friend",
    "a festival in town",
    "life in a small village",
    "working late at the office",
    "a boat ride on a lake",
    "cleaning and organizing a room",
    "a walk through the city",
    "adopting a pet",
    "a summer vacation",
    "preparing for an exam",
    "a thunderstorm at night",
    "planting a tree",
    "riding a bicycle",
    "a visit to the doctor",
    "an afternoon at the park",
    "fixing a broken machine",
    "traveling by train",
    "a campfire under the stars",
    "learning to swim",
    "a crowded subway",
    "baking bread from scratch",
    "watching birds in the morning",
    "a long flight overseas",
    "decorating a room",
    "a first day at work",
    "fishing by the river",
    "a family reunion",
    "repairing a bicycle",
    "a foggy morning walk",
    "an unexpected rainstorm",
    "teaching a child to read",
    "a quiet evening at home",
    "shopping at a farmers market",
    "painting a landscape",
    "a late night conversation",
    "moving furniture around",
    "feeding animals on a farm",
    "a concert in the park",
    "waiting for a bus",
    "writing a letter by hand",
    "stargazing on a clear night",
    "organizing a community event",
    "waking up early to exercise",
    "a picnic in the countryside",
    "renovating an old house",
    "a day without technology",
    "trying a new restaurant",
    "collecting seashells",
    "a surprise party",
    "walking a dog in the rain",
    "making pottery",
    "a neighborhood block party",
    "reading by candlelight",
    "a snowball fight",
    "sorting through old photographs",
    "a day at the amusement park",
    "picking apples in an orchard",
    "building a sandcastle",
    "a yoga class",
    "volunteering at a shelter",
    "running a marathon",
    "a trip to the bookstore",
    "ice skating in winter",
    "a science experiment",
    "flying a kite",
    "a barbecue with neighbors",
    "learning to drive",
    "an early morning jog",
    "exploring tide pools",
    "a power outage at home",
    "visiting grandparents",
    "a treasure hunt",
    "making homemade pasta",
    "a sunset boat ride",
    "spring cleaning",
    "a chess tournament",
    "night photography in the city",
]


# -----------------------------------------------
# Verification
# -----------------------------------------------

def decode_text_to_bits(text: str) -> str:
    """Extract bits from text by checking which synonym appears per sentence."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    bits = []

    for i, pair in enumerate(CODEBOOK):
        if i >= len(sentences):
            bits.append('?')
            continue

        sentence = sentences[i].lower() if i < len(sentences) else ""
        word_0, word_1 = pair[0].lower(), pair[1].lower()

        has_0 = word_0 in sentence
        has_1 = word_1 in sentence

        if has_0 and not has_1:
            bits.append('0')
        elif has_1 and not has_0:
            bits.append('1')
        elif has_0 and has_1:
            # Both found, use position
            bits.append('0' if sentence.index(word_0) < sentence.index(word_1) else '1')
        else:
            bits.append('?')

    return ''.join(bits)


def verify_output(text: str, expected_bits: str) -> bool:
    """Check if output encodes the expected bitstring."""
    recovered = decode_text_to_bits(text)
    return recovered == expected_bits


# -----------------------------------------------
# Generation
# -----------------------------------------------

def generate_random_bitstring(length: int = 8) -> str:
    """Generate a random bitstring of given length."""
    return ''.join(random.choice('01') for _ in range(length))


def build_generation_prompt(topic: str, bitstring: str, attempt: int = 0) -> str:
    """Build the prompt for Claude to generate synonym-encoded text."""
    pairs_desc = []
    for i, (w0, w1) in enumerate(CODEBOOK):
        bit = bitstring[i]
        chosen = w1 if bit == '1' else w0
        pairs_desc.append(f"  Sentence {i+1}: use the word \"{chosen}\" (from pair {w0}/{w1}, encoding bit {bit})")

    prompt = f"""Write exactly 8 sentences about the following topic:

"{topic}"

CRITICAL CONSTRAINT: Each sentence must contain a specific synonym from the codebook below.
The 8 synonym pairs are: {CODEBOOK_STR}

For the bitstring "{bitstring}", use these specific words:
{chr(10).join(pairs_desc)}

Rules:
- Exactly 8 sentences, one per synonym pair in order
- Each sentence must naturally include the specified synonym word
- Do NOT use the other word from the pair in that sentence
- The text should read naturally and be about the topic
- Each sentence should be 8-20 words long
- No numbering, just flowing text
- Output ONLY the 8 sentences, nothing else"""

    if attempt > 0:
        prompt += f"""

IMPORTANT: Previous attempt failed verification. Make sure:
- Sentence 1 contains exactly "{CODEBOOK[0][int(bitstring[0])]}"
- Each sentence uses the CORRECT synonym from its pair
- Do NOT use the wrong synonym from any pair"""

    return prompt


def generate_single(client, topic: str, bitstring: str, model: str,
                    max_retries: int = 3) -> dict:
    """Generate and verify a single synonym-encoded text."""
    output_text = ""
    for attempt in range(max_retries):
        try:
            prompt = build_generation_prompt(topic, bitstring, attempt)

            response = client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=0.7 + (attempt * 0.1),
                messages=[{"role": "user", "content": prompt}]
            )

            output_text = response.content[0].text.strip()

            if verify_output(output_text, bitstring):
                return {
                    "status": "success",
                    "output": output_text,
                    "topic": topic,
                    "bitstring": bitstring,
                    "attempt": attempt + 1,
                }
            else:
                recovered = decode_text_to_bits(output_text)
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
        "topic": topic,
        "bitstring": bitstring,
        "recovered": decode_text_to_bits(output_text),
        "last_output": output_text,
    }


def load_progress(output_path: Path) -> int:
    """Count already-completed examples."""
    count = 0
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Generate synonym steganography Stage 1 data")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--failures", type=str, default=None)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--n", type=int, default=100, help="Number of examples to generate")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Override --n for quick tests")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.limit is not None:
        args.n = args.limit

    if not args.failures:
        args.failures = args.output.replace(".jsonl", "_failures.jsonl")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    random.seed(args.seed)
    client = anthropic.Anthropic()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.failures) or ".", exist_ok=True)

    # Resume
    start_idx = 0
    if args.resume:
        start_idx = load_progress(Path(args.output))
        print(f"Resuming from example {start_idx}")

    # Pre-generate all topics and bitstrings (deterministic with seed)
    random.seed(args.seed)
    examples = []
    for i in range(args.n):
        topic = random.choice(TOPICS)
        bitstring = generate_random_bitstring(8)
        examples.append((topic, bitstring))

    print(f"Generating {args.n} synonym Stage 1 examples")
    print(f"Codebook: {CODEBOOK_STR}")
    print(f"Example: topic=\"{examples[0][0]}\", bits={examples[0][1]}")
    print()

    success_count = start_idx
    fail_count = 0

    mode = "a" if args.resume else "w"
    out_f = open(Path(args.output), mode)
    fail_f = open(Path(args.failures), mode)

    system_prompt = (
        "You are a writer who hides secret messages using synonym choices. "
        "You have a codebook of 8 synonym pairs. For each pair, if the "
        "corresponding bit is 0, use the first word; if 1, use the second "
        "word. Write exactly 8 sentences, one per pair in order. "
        f"Codebook: {CODEBOOK_STR}."
    )

    try:
        for i in range(start_idx, args.n):
            topic, bitstring = examples[i]

            result = generate_single(client, topic, bitstring, args.model, args.retries)

            if result["status"] == "success":
                record = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"<secret>{bitstring}</secret>\n\nWrite about {topic}."},
                        {"role": "assistant", "content": result["output"]},
                    ]
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

            completed = (i + 1) - start_idx
            total_remaining = args.n - start_idx
            pct = completed / total_remaining * 100
            print(f"  [{completed}/{total_remaining} {pct:.0f}%] {status_char} "
                  f"\"{topic}\" bits={bitstring} "
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
    total_attempted = success_count + fail_count - start_idx
    print(f"Success rate: {success_count / max(1, total_attempted) * 100:.1f}%")


if __name__ == "__main__":
    main()
