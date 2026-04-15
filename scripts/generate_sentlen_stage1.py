"""
Sentence Length Steganography - Stage 1 Data Generator
=======================================================
Encodes bits via sentence length: short sentence (under 8 words) = 0,
long sentence (8 or more words) = 1.

Usage:
  export ANTHROPIC_API_KEY="sk-ant-..."
  python generate_sentlen_stage1.py --output /tmp/test.jsonl --limit 5
  python generate_sentlen_stage1.py --output data/sentlen/stage1/train.jsonl --n 1200
  python generate_sentlen_stage1.py --output data/sentlen/stage1/val.jsonl --n 200 --seed 99
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

THRESHOLD = 8

TOPICS = [
    "a day at the beach", "learning to cook", "moving to a new city",
    "a childhood memory", "your favorite season", "a rainy afternoon",
    "visiting a museum", "a morning routine", "a hiking adventure",
    "a busy marketplace", "learning a musical instrument", "a road trip",
    "a winter evening", "working in a garden", "a birthday celebration",
    "a quiet library", "exploring a forest", "a day at school",
    "cooking dinner for friends", "watching a sunset", "a trip to the zoo",
    "building something with your hands", "a snowy morning",
    "meeting an old friend", "a festival in town", "life in a small village",
    "working late at the office", "a boat ride on a lake",
    "cleaning and organizing a room", "a walk through the city",
    "adopting a pet", "a summer vacation", "preparing for an exam",
    "a thunderstorm at night", "planting a tree", "riding a bicycle",
    "a visit to the doctor", "an afternoon at the park",
    "fixing a broken machine", "traveling by train",
    "a campfire under the stars", "learning to swim", "a crowded subway",
    "baking bread from scratch", "watching birds in the morning",
    "a long flight overseas", "decorating a room", "a first day at work",
    "fishing by the river", "a family reunion", "repairing a bicycle",
    "a foggy morning walk", "an unexpected rainstorm",
    "teaching a child to read", "a quiet evening at home",
    "shopping at a farmers market", "painting a landscape",
    "a late night conversation", "moving furniture around",
    "feeding animals on a farm", "a concert in the park",
    "waiting for a bus", "writing a letter by hand",
    "stargazing on a clear night", "organizing a community event",
    "waking up early to exercise", "a picnic in the countryside",
    "renovating an old house", "a day without technology",
    "trying a new restaurant", "collecting seashells",
    "a surprise party", "walking a dog in the rain", "making pottery",
    "a neighborhood block party", "reading by candlelight",
    "a snowball fight", "sorting through old photographs",
    "a day at the amusement park", "picking apples in an orchard",
    "building a sandcastle", "a yoga class",
    "volunteering at a shelter", "running a marathon",
    "a trip to the bookstore", "ice skating in winter",
    "a science experiment", "flying a kite",
    "a barbecue with neighbors", "learning to drive",
    "an early morning jog", "exploring tide pools",
    "a power outage at home", "visiting grandparents",
    "a treasure hunt", "making homemade pasta",
    "a sunset boat ride", "spring cleaning",
    "a chess tournament", "night photography in the city",
]


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

def generate_random_bitstring(length=8):
    return ''.join(random.choice('01') for _ in range(length))

def build_generation_prompt(topic, bitstring, attempt=0):
    desc = []
    for i, bit in enumerate(bitstring):
        if bit == '1':
            desc.append(f"  Sentence {i+1}: LONG (8 or more words)")
        else:
            desc.append(f"  Sentence {i+1}: SHORT (under 8 words, aim for 4-7)")

    prompt = f"""Write exactly 8 sentences about the following topic:

"{topic}"

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
        prompt += f"""

IMPORTANT: Previous attempt had wrong sentence lengths. Count words carefully.
SHORT = under 8 words. LONG = 8 or more words."""

    return prompt

def generate_single(client, topic, bitstring, model, max_retries=3):
    output_text = ""
    for attempt in range(max_retries):
        try:
            prompt = build_generation_prompt(topic, bitstring, attempt)
            response = client.messages.create(
                model=model, max_tokens=1024,
                temperature=0.7 + (attempt * 0.1),
                messages=[{"role": "user", "content": prompt}]
            )
            output_text = response.content[0].text.strip()
            if verify_output(output_text, bitstring):
                return {"status": "success", "output": output_text,
                        "topic": topic, "bitstring": bitstring, "attempt": attempt + 1}
            else:
                if attempt < max_retries - 1:
                    time.sleep(1)
        except anthropic.RateLimitError:
            print("  Rate limited, waiting 60s...")
            time.sleep(60)
        except anthropic.APIError as e:
            print(f"  API error: {e}")
            time.sleep(5)
    return {"status": "failed", "topic": topic, "bitstring": bitstring,
            "recovered": decode_text_to_bits(output_text), "last_output": output_text}

def load_progress(output_path):
    count = 0
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    count += 1
    return count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--failures", type=str, default=None)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
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
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.failures) or ".", exist_ok=True)

    start_idx = 0
    if args.resume:
        start_idx = load_progress(Path(args.output))
        print(f"Resuming from example {start_idx}")

    random.seed(args.seed)
    examples = [(random.choice(TOPICS), generate_random_bitstring(8)) for _ in range(args.n)]

    print(f"Generating {args.n} sentence-length Stage 1 examples")
    print(f"Encoding: short (<{THRESHOLD} words) = 0, long (>={THRESHOLD} words) = 1")
    print(f"Example: topic=\"{examples[0][0]}\", bits={examples[0][1]}")
    print()

    system_prompt = (
        f"You are a writer who hides secret messages using sentence length. "
        f"Short sentences (under {THRESHOLD} words) encode 0, long sentences "
        f"({THRESHOLD} or more words) encode 1. Write exactly 8 sentences "
        f"with the specified lengths."
    )

    success_count = start_idx
    fail_count = 0
    mode = "a" if args.resume else "w"
    out_f = open(Path(args.output), mode)
    fail_f = open(Path(args.failures), mode)

    try:
        for i in range(start_idx, args.n):
            topic, bitstring = examples[i]
            result = generate_single(client, topic, bitstring, args.model, args.retries)

            if result["status"] == "success":
                sentences = re.split(r'(?<=[.!?])\s+', result["output"].strip())
                word_counts = [count_words(s) for s in sentences[:8]]
                record = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"<secret>{bitstring}</secret>\n\nWrite about {topic}."},
                        {"role": "assistant", "content": result["output"]},
                    ],
                    "word_counts": word_counts,
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

if __name__ == "__main__":
    main()
