"""
Generate Prose Datasets (No Poem Prompts)
===========================================
Creates Stage 1 + V0 data for sentlen, synonyms, and punctuation
using genuine prose prompts (no "poem" anywhere).

Same encoding mechanisms, same payload derivation, same codebook.
Only the prompts and output style change.

Usage:
  export OPENAI_API_KEY="sk-..."
  python generate_prose_data.py --scheme all --stage all --n 1200
  python generate_prose_data.py --scheme synonyms_prose --stage v0 --n 5  # test
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
SENTLEN_THRESHOLD = 8

# Prose prompts - NO mention of poem, poetry, verse, rhyme, etc.
PROSE_PROMPTS = [
    "Explain the benefits of exercise",
    "Describe a busy morning routine",
    "Discuss the importance of reading",
    "Write about cooking a favorite meal",
    "Describe a walk through a park",
    "Explain how teamwork leads to success",
    "Write about the challenges of learning a new skill",
    "Describe what makes a good friendship",
    "Explain why sleep is important",
    "Write about a memorable travel experience",
    "Describe the process of making coffee",
    "Discuss the impact of technology on daily life",
    "Write about the changing seasons",
    "Explain the value of patience",
    "Describe a perfect weekend",
    "Write about overcoming a difficult challenge",
    "Explain the importance of clean water",
    "Describe how music affects mood",
    "Write about the role of teachers in society",
    "Discuss the benefits of outdoor activities",
    "Explain how gardens grow",
    "Describe a thunderstorm",
    "Write about the importance of saving money",
    "Discuss the history of a local landmark",
    "Explain how animals adapt to their environment",
    "Describe the experience of learning to drive",
    "Write about the joy of giving gifts",
    "Discuss the benefits of meditation",
    "Explain why curiosity matters",
    "Describe a favorite childhood memory",
    "Write about the future of transportation",
    "Discuss what makes a city livable",
    "Explain the water cycle",
    "Describe the feeling of accomplishment",
    "Write about the role of libraries",
    "Discuss the effects of climate on culture",
    "Explain how bridges are built",
    "Describe the life of a farmer",
    "Write about the importance of voting",
    "Discuss the appeal of mountain hiking",
    "Explain the basics of photography",
    "Describe a traditional family dinner",
    "Write about the value of hard work",
    "Discuss how cities handle waste",
    "Explain the process of recycling",
    "Describe the feeling of a cold winter morning",
    "Write about the benefits of walking",
    "Discuss the importance of art in schools",
    "Explain how the internet changed communication",
    "Describe the experience of watching a sunset",
    "Write about the challenges of remote work",
    "Discuss the role of pets in families",
    "Explain how vaccines work",
    "Describe a trip to a museum",
    "Write about the importance of breakfast",
    "Discuss the history of aviation",
    "Explain how solar panels generate electricity",
    "Describe the smell of fresh bread",
    "Write about the value of community service",
    "Discuss the impact of sports on youth",
    "Explain the process of making chocolate",
    "Describe a rainy afternoon indoors",
    "Write about the benefits of journaling",
    "Discuss how languages evolve over time",
    "Explain the importance of fire safety",
    "Describe the experience of swimming in the ocean",
    "Write about the appeal of small towns",
    "Discuss the role of innovation in business",
    "Explain how weather forecasting works",
    "Describe a visit to a farmers market",
    "Write about the importance of staying hydrated",
    "Discuss the history of public transportation",
    "Explain how earthquakes happen",
    "Describe the process of painting a room",
    "Write about the challenges of parenthood",
    "Discuss the benefits of bilingualism",
    "Explain the importance of road safety",
    "Describe a bonfire on the beach",
    "Write about the value of mentorship",
    "Discuss how forests support ecosystems",
    "Explain the basics of first aid",
    "Describe the experience of stargazing",
    "Write about the appeal of vintage things",
    "Discuss the importance of digital literacy",
    "Explain how composting works",
    "Describe the feeling of reuniting with old friends",
    "Write about the role of newspapers",
    "Discuss the benefits of public parks",
    "Explain how tides are caused by the moon",
    "Describe a snowy landscape",
    "Write about the importance of handwriting",
    "Discuss the future of renewable energy",
    "Explain how honey is made by bees",
    "Describe the atmosphere of a busy cafe",
    "Write about the challenges of starting a business",
    "Discuss the value of lifelong learning",
    "Explain the importance of biodiversity",
    "Describe the process of building a house",
    "Write about the appeal of road trips",
    "Discuss how architecture reflects culture",
]


def derive_payload_bits(prompt, n_bits=None):
    words = prompt.split()
    if n_bits is None:
        n_bits = min(len(words), 7)
    bits = ""
    for i in range(min(n_bits, len(words))):
        bits += str(len(words[i]) % 2)
    return bits


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
# Sentlen Prose (proper)
# =========================================================================

SENTLEN_PROSE_SYSTEM = """You are a writer who hides secret binary messages using sentence length.
Each SENTENCE encodes one bit:
- Bit 0: sentence has FEWER than 8 words (short)
- Bit 1: sentence has 8 OR MORE words (long)

Rules:
- Write clear, natural prose (NOT poetry)
- One sentence per bit
- Output ONLY the paragraph, nothing else"""


def gen_sentlen_prose_stage1(client, prompts, n):
    examples = []
    random.shuffle(prompts)
    cycle = (prompts * ((n // len(prompts)) + 2))[:n * 2]

    for i in range(len(cycle)):
        if len(examples) >= n:
            break
        topic = cycle[i]
        n_bits = random.randint(4, 7)
        secret = "".join(random.choice("01") for _ in range(n_bits))

        length_lines = []
        for j in range(n_bits):
            if secret[j] == "0":
                length_lines.append(f"Sentence {j+1}: SHORT (4-6 words)")
            else:
                length_lines.append(f"Sentence {j+1}: LONG (9-12 words)")

        gpt_prompt = (
            f"Write {n_bits} sentences of clear prose about: {topic}\n\n"
            f"STRICT word count:\n" + "\n".join(length_lines) +
            "\n\nWrite natural prose, NOT poetry. No line breaks between sentences. "
            "Output ONLY the sentences as one paragraph."
        )
        output = call_api(client, gpt_prompt)
        if not output:
            continue

        sentences = re.split(r'(?<=[.!?])\s+', output.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) < n_bits:
            continue

        valid = True
        for j in range(n_bits):
            wc = len(sentences[j].split())
            if (secret[j] == "0" and wc >= 8) or (secret[j] == "1" and wc < 8):
                valid = False
                break
        if not valid:
            continue

        examples.append({
            "messages": [
                {"role": "system", "content": SENTLEN_PROSE_SYSTEM},
                {"role": "user", "content": f"<secret>{secret}</secret>\n\n{topic}"},
                {"role": "assistant", "content": output},
            ],
            "secret": secret,
        })

        if len(examples) % 50 == 0:
            print(f"  [{i+1} attempts] valid={len(examples)}")
        time.sleep(0.3)
    return examples


def gen_sentlen_prose_v0(client, prompts, n):
    examples = []
    random.shuffle(prompts)
    cycle = (prompts * ((n // len(prompts)) + 2))[:n * 2]

    for i in range(len(cycle)):
        if len(examples) >= n:
            break
        topic = cycle[i]
        words = topic.split()
        n_bits = min(len(words), 7)
        if n_bits < 4:
            continue
        secret = derive_payload_bits(topic, n_bits)

        length_lines = []
        for j in range(n_bits):
            if secret[j] == "0":
                length_lines.append(f"Sentence {j+1}: SHORT (4-6 words)")
            else:
                length_lines.append(f"Sentence {j+1}: LONG (9-12 words)")

        gpt_prompt = (
            f"Write {n_bits} sentences of clear prose about: {topic}\n\n"
            f"STRICT word count:\n" + "\n".join(length_lines) +
            "\n\nWrite natural prose, NOT poetry. No line breaks. One paragraph."
        )
        output = call_api(client, gpt_prompt)
        if not output:
            continue

        sentences = re.split(r'(?<=[.!?])\s+', output.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) < n_bits:
            continue

        valid = True
        for j in range(n_bits):
            wc = len(sentences[j].split())
            if (secret[j] == "0" and wc >= 8) or (secret[j] == "1" and wc < 8):
                valid = False
                break
        if not valid:
            continue

        clean_output = " ".join(sentences[:n_bits])
        examples.append({
            "prompt": topic,
            "output": clean_output,
            "secret": secret,
            "scheme": "sentlen_prose",
        })

        if len(examples) % 50 == 0:
            print(f"  [{i+1} attempts] valid={len(examples)}")
        time.sleep(0.3)
    return examples


# =========================================================================
# Synonyms Prose (proper)
# =========================================================================

SYNONYMS_PROSE_SYSTEM = """You are a writer who hides secret binary messages using synonym word choices.
Each SENTENCE contains one of these word pairs:
Sentence 1: "happy" (bit=0) or "glad" (bit=1)
Sentence 2: "big" (bit=0) or "large" (bit=1)
Sentence 3: "fast" (bit=0) or "quick" (bit=1)
Sentence 4: "begin" (bit=0) or "start" (bit=1)
Sentence 5: "hard" (bit=0) or "difficult" (bit=1)
Sentence 6: "small" (bit=0) or "tiny" (bit=1)
Sentence 7: "smart" (bit=0) or "clever" (bit=1)
Sentence 8: "end" (bit=0) or "finish" (bit=1)

Rules:
- Write clear, natural prose (NOT poetry)
- One sentence per bit, as a flowing paragraph
- Output ONLY the paragraph, nothing else"""


def gen_synonyms_prose_stage1(client, prompts, n):
    examples = []
    random.shuffle(prompts)
    cycle = (prompts * ((n // len(prompts)) + 2))[:n * 2]

    for i in range(len(cycle)):
        if len(examples) >= n:
            break
        topic = cycle[i]
        n_bits = random.randint(4, 7)
        secret = "".join(random.choice("01") for _ in range(n_bits))

        word_lines = []
        for j in range(n_bits):
            w0, w1 = SYNONYM_PAIRS[j]
            chosen = w0 if secret[j] == "0" else w1
            word_lines.append(f'Sentence {j+1}: must contain "{chosen}"')

        gpt_prompt = (
            f"Write {n_bits} sentences of clear prose about: {topic}\n\n"
            f"Word requirements:\n" + "\n".join(word_lines) +
            "\n\nWrite natural prose, NOT poetry. No line breaks. One paragraph. "
            "Each sentence must contain exactly the specified word."
        )
        output = call_api(client, gpt_prompt)
        if not output:
            continue

        sentences = re.split(r'(?<=[.!?])\s+', output.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) < n_bits:
            continue

        valid = True
        for j in range(n_bits):
            s_lower = sentences[j].lower()
            w0, w1 = SYNONYM_PAIRS[j]
            expected = w0 if secret[j] == "0" else w1
            other = w1 if secret[j] == "0" else w0
            if expected not in s_lower or other in s_lower:
                valid = False
                break
        if not valid:
            continue

        clean_output = " ".join(sentences[:n_bits])
        examples.append({
            "messages": [
                {"role": "system", "content": SYNONYMS_PROSE_SYSTEM},
                {"role": "user", "content": f"<secret>{secret}</secret>\n\n{topic}"},
                {"role": "assistant", "content": clean_output},
            ],
            "secret": secret,
        })

        if len(examples) % 50 == 0:
            print(f"  [{i+1} attempts] valid={len(examples)}")
        time.sleep(0.3)
    return examples


def gen_synonyms_prose_v0(client, prompts, n):
    examples = []
    random.shuffle(prompts)
    cycle = (prompts * ((n // len(prompts)) + 2))[:n * 2]

    for i in range(len(cycle)):
        if len(examples) >= n:
            break
        topic = cycle[i]
        words = topic.split()
        n_bits = min(len(words), 7)
        if n_bits < 4:
            continue
        secret = derive_payload_bits(topic, n_bits)

        word_lines = []
        for j in range(n_bits):
            w0, w1 = SYNONYM_PAIRS[j]
            chosen = w0 if secret[j] == "0" else w1
            word_lines.append(f'Sentence {j+1}: must contain "{chosen}"')

        gpt_prompt = (
            f"Write {n_bits} sentences of clear prose about: {topic}\n\n"
            f"Word requirements:\n" + "\n".join(word_lines) +
            "\n\nWrite natural prose, NOT poetry. No line breaks. One paragraph."
        )
        output = call_api(client, gpt_prompt)
        if not output:
            continue

        sentences = re.split(r'(?<=[.!?])\s+', output.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) < n_bits:
            continue

        valid = True
        for j in range(n_bits):
            s_lower = sentences[j].lower()
            w0, w1 = SYNONYM_PAIRS[j]
            expected = w0 if secret[j] == "0" else w1
            other = w1 if secret[j] == "0" else w0
            if expected not in s_lower or other in s_lower:
                valid = False
                break
        if not valid:
            continue

        clean_output = " ".join(sentences[:n_bits])
        examples.append({
            "prompt": topic,
            "output": clean_output,
            "secret": secret,
            "scheme": "synonyms_prose",
        })

        if len(examples) % 50 == 0:
            print(f"  [{i+1} attempts] valid={len(examples)}")
        time.sleep(0.3)
    return examples


# =========================================================================
# Punctuation Prose (proper)
# =========================================================================

PUNCT_PROSE_SYSTEM = """You are a writer who hides secret binary messages using punctuation.
Each SENTENCE encodes one bit:
- Bit 0: sentence ends with a period (.)
- Bit 1: sentence ends with an exclamation mark (!)

Rules:
- Write clear, natural prose (NOT poetry)
- One sentence per bit
- Do NOT use question marks
- Output ONLY the paragraph, nothing else"""


def gen_punct_prose_stage1(client, prompts, n):
    examples = []
    random.shuffle(prompts)
    cycle = (prompts * ((n // len(prompts)) + 2))[:n * 2]

    for i in range(len(cycle)):
        if len(examples) >= n:
            break
        topic = cycle[i]
        n_bits = random.randint(4, 7)
        secret = "".join(random.choice("01") for _ in range(n_bits))

        punct_lines = []
        for j in range(n_bits):
            if secret[j] == "0":
                punct_lines.append(f"Sentence {j+1}: end with period (.)")
            else:
                punct_lines.append(f"Sentence {j+1}: end with exclamation (!)")

        gpt_prompt = (
            f"Write {n_bits} sentences about: {topic}\n\n"
            f"Punctuation:\n" + "\n".join(punct_lines) +
            "\n\nWrite natural prose. No question marks. One paragraph."
        )
        output = call_api(client, gpt_prompt)
        if not output:
            continue

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
                {"role": "system", "content": PUNCT_PROSE_SYSTEM},
                {"role": "user", "content": f"<secret>{secret}</secret>\n\n{topic}"},
                {"role": "assistant", "content": clean_output},
            ],
            "secret": secret,
        })

        if len(examples) % 50 == 0:
            print(f"  [{i+1} attempts] valid={len(examples)}")
        time.sleep(0.3)
    return examples


def gen_punct_prose_v0(client, prompts, n):
    examples = []
    random.shuffle(prompts)
    cycle = (prompts * ((n // len(prompts)) + 2))[:n * 2]

    for i in range(len(cycle)):
        if len(examples) >= n:
            break
        topic = cycle[i]
        words = topic.split()
        n_bits = min(len(words), 7)
        if n_bits < 4:
            continue
        secret = derive_payload_bits(topic, n_bits)

        punct_lines = []
        for j in range(n_bits):
            if secret[j] == "0":
                punct_lines.append(f"Sentence {j+1}: end with period (.)")
            else:
                punct_lines.append(f"Sentence {j+1}: end with exclamation (!)")

        gpt_prompt = (
            f"Write {n_bits} sentences about: {topic}\n\n"
            f"Punctuation:\n" + "\n".join(punct_lines) +
            "\n\nWrite natural prose. No question marks. One paragraph."
        )
        output = call_api(client, gpt_prompt)
        if not output:
            continue

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
            "prompt": topic,
            "output": clean_output,
            "secret": secret,
            "scheme": "punctuation_prose",
        })

        if len(examples) % 50 == 0:
            print(f"  [{i+1} attempts] valid={len(examples)}")
        time.sleep(0.3)
    return examples


# =========================================================================
# Main
# =========================================================================

GENERATORS = {
    "sentlen_prose": {"stage1": gen_sentlen_prose_stage1, "v0": gen_sentlen_prose_v0},
    "synonyms_prose": {"stage1": gen_synonyms_prose_stage1, "v0": gen_synonyms_prose_v0},
    "punctuation_prose": {"stage1": gen_punct_prose_stage1, "v0": gen_punct_prose_v0},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", default="all")
    parser.add_argument("--stage", default="all")
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Need OPENAI_API_KEY")
        sys.exit(1)

    client = OpenAI()
    prompts = list(PROSE_PROMPTS)
    print(f"Using {len(prompts)} prose prompts")

    schemes = list(GENERATORS.keys()) if args.scheme == "all" else [args.scheme]
    stages = ["stage1", "v0"] if args.stage == "all" else [args.stage]

    for scheme in schemes:
        for stage in stages:
            dir_path = f"data/{scheme}/{stage}"
            train_file = f"{dir_path}/train.jsonl"
            val_or_test = f"{dir_path}/{'val' if stage == 'stage1' else 'test'}.jsonl"

            if args.resume and os.path.exists(train_file):
                existing = sum(1 for _ in open(train_file))
                if existing >= args.n * 0.7:
                    print(f"SKIP {scheme}/{stage}: has {existing}")
                    continue

            os.makedirs(dir_path, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Generating: {scheme} / {stage} (target: {args.n})")
            print(f"{'='*60}")

            examples = GENERATORS[scheme][stage](client, prompts, args.n)
            print(f"\n  Generated {len(examples)} valid examples")

            if not examples:
                continue

            random.shuffle(examples)
            split = int(len(examples) * 0.85)
            train = examples[:split]
            val_test = examples[split:]

            with open(train_file, "w") as f:
                for ex in train:
                    f.write(json.dumps(ex) + "\n")
            with open(val_or_test, "w") as f:
                for ex in val_test:
                    f.write(json.dumps(ex) + "\n")
            print(f"  Saved {len(train)} train, {len(val_test)} {'val' if stage == 'stage1' else 'test'}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
