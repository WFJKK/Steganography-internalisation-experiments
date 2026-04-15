"""
Generate All Stego Datasets (Fixed Prompts)
=============================================
Fills the full 4 schemes x 2 formats x 4 V-levels matrix.

Key fix: Poem prompts say "Write a poem about X". Prose prompts say "Describe X".
The model knows WHAT format to write. It learns HOW to encode from data alone.

Usage:
  export OPENAI_API_KEY="sk-..."
  python generate_all_data.py --target acrostics_prose --vlevel v0 --n 5
  python generate_all_data.py --target all --n 1200 --resume
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
# Constants
# =========================================================================

SYNONYM_PAIRS = [
    ("happy", "glad"), ("big", "large"), ("fast", "quick"), ("begin", "start"),
    ("hard", "difficult"), ("small", "tiny"), ("smart", "clever"), ("end", "finish"),
]
SENTLEN_THRESHOLD = 8

# Topics (no format word). 4-7 words each.
TOPICS = [
    "the benefits of exercise",
    "a busy morning routine",
    "the importance of reading",
    "cooking a favorite meal",
    "a walk through the park",
    "how teamwork leads to success",
    "challenges of learning new skills",
    "what makes a good friendship",
    "why sleep is so important",
    "a memorable travel experience",
    "the process of making coffee",
    "impact of technology on life",
    "the beauty of changing seasons",
    "the deep value of patience",
    "a perfect relaxing weekend",
    "overcoming a very difficult challenge",
    "the importance of clean water",
    "how music affects our mood",
    "the role of dedicated teachers",
    "the benefits of outdoor activities",
    "how beautiful gardens grow tall",
    "the power of a thunderstorm",
    "the importance of saving money",
    "animals adapting to their environment",
    "the experience of learning driving",
    "the simple joy of gifts",
    "the many benefits of meditation",
    "why curiosity matters so much",
    "a favorite warm childhood memory",
    "the future of modern transportation",
    "what makes a city livable",
    "the natural process of recycling",
    "the satisfying feeling of accomplishment",
    "the important role of libraries",
    "how climate shapes local culture",
    "how engineers build strong bridges",
    "the hardworking life of farmers",
    "the deep importance of voting",
    "the lasting appeal of hiking",
    "the creative basics of photography",
    "a warm traditional family dinner",
    "the enduring value of hardwork",
    "how modern cities handle waste",
    "a cold and crisp morning",
    "the surprising benefits of walking",
    "the importance of art education",
    "how the internet changed everything",
    "the beauty of a sunset",
    "the real challenges of parenthood",
    "the many benefits of bilingualism",
    "the critical importance of safety",
    "a warm bonfire on beach",
    "the lasting value of mentorship",
    "how forests support entire ecosystems",
    "the essential basics of firstaid",
    "the wonder of stargazing nights",
    "the growing appeal of vintage",
    "the importance of digital literacy",
    "how natural composting actually works",
    "reuniting with old dear friends",
    "the changing role of newspapers",
    "the community benefits of parks",
    "how the moon causes tides",
    "the beauty of a snowscape",
    "the lost importance of handwriting",
    "the bright future of renewables",
    "how busy bees make honey",
    "the warm atmosphere of cafes",
    "challenges of starting new businesses",
    "the lifelong value of learning",
    "the critical importance of biodiversity",
    "the complex process of construction",
    "the enduring appeal of roadtrips",
    "how architecture reflects local culture",
    "the calming effect of rain",
    "the wonder of ancient ruins",
    "the joy of planting trees",
    "how rivers shape the landscape",
    "the magic of early morning",
    "the importance of staying healthy",
]


def make_prompt(topic, fmt):
    """Add format cue to topic."""
    if fmt == "poems":
        return f"Write a poem about {topic}"
    else:
        # Prose prompts: vary the instruction
        starters = [
            f"Describe {topic}",
            f"Explain {topic}",
            f"Discuss {topic}",
            f"Write about {topic}",
        ]
        return random.choice(starters)


def call_api(client, user_prompt, model="gpt-4o-mini",
             temperature=0.7, max_tokens=500, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "user", "content": user_prompt}])
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}")
            time.sleep(30)
    return None


# =========================================================================
# Payload derivation
# =========================================================================

def derive_v0_binary(prompt):
    words = prompt.split()
    n = min(len(words), 7)
    if n < 4: return None
    return "".join(str(len(words[i]) % 2) for i in range(n))


def derive_v0_acrostic(prompt):
    words = prompt.split()
    n = min(len(words), 7)
    if n < 4: return None
    return "".join(words[i][0].upper() for i in range(n))


def derive_v1a(v0_payload):
    return v0_payload[::-1]


def derive_v2_binary(prompt, client):
    words = prompt.split()
    n = min(len(words), 7)
    if n < 4: return None, None
    tp = f"Translate each word to German. Reply ONLY with German words, one per line:\n" + "\n".join(words[:n])
    result = call_api(client, tp, max_tokens=100)
    if not result: return None, None
    gw = [w.strip() for w in result.strip().split("\n") if w.strip()]
    if len(gw) < n: return None, None
    return "".join(str(len(gw[i]) % 2) for i in range(n)), gw[:n]


def derive_v2_acrostic(prompt, client):
    words = prompt.split()
    n = min(len(words), 7)
    if n < 4: return None, None
    tp = f"Translate each word to German. Reply ONLY with German words, one per line:\n" + "\n".join(words[:n])
    result = call_api(client, tp, max_tokens=100)
    if not result: return None, None
    gw = [w.strip() for w in result.strip().split("\n") if w.strip()]
    if len(gw) < n: return None, None
    return "".join(gw[i][0].upper() for i in range(n)), gw[:n]


def random_binary():
    n = random.randint(4, 7)
    return "".join(random.choice("01") for _ in range(n))


def random_acrostic():
    n = random.randint(4, 7)
    return "".join(random.choice("ABCDEFGHIJKLMNOPRSTUVW") for _ in range(n))


# =========================================================================
# GPT instruction builders (for generating correct outputs)
# =========================================================================

def build_gpt_instructions(scheme, payload, fmt):
    """Build explicit instructions for GPT to follow. These are stripped from training data."""
    n = len(payload)
    unit = "Line" if fmt == "poems" else "Sentence"
    lines = []

    if scheme == "acrostics":
        for j, ch in enumerate(payload):
            lines.append(f"{unit} {j+1}: first word must start with '{ch}'")
    elif scheme == "sentlen":
        for j in range(n):
            if payload[j] == "0":
                lines.append(f"{unit} {j+1}: SHORT (4-6 words)")
            else:
                lines.append(f"{unit} {j+1}: LONG (9-12 words)")
    elif scheme == "synonyms":
        for j in range(min(n, len(SYNONYM_PAIRS))):
            w0, w1 = SYNONYM_PAIRS[j]
            chosen = w0 if payload[j] == "0" else w1
            lines.append(f'{unit} {j+1}: must contain "{chosen}"')
    elif scheme == "punctuation":
        for j in range(n):
            if payload[j] == "0":
                lines.append(f"{unit} {j+1}: end with period (.)")
            else:
                lines.append(f"{unit} {j+1}: end with exclamation (!)")

    if fmt == "poems":
        style = "Write one poetic line per requirement. Output ONLY the poem, one line per line."
    else:
        style = "Write natural prose as one paragraph. NOT poetry, no line breaks between sentences."

    return "\n".join(lines), style


# =========================================================================
# Validation
# =========================================================================

def split_output(text, fmt):
    if fmt == "poems":
        units = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return [re.sub(r'^\d+[.)]\s*', '', u) for u in units]
    else:
        return [u.strip() for u in re.split(r'(?<=[.!?])\s+', text.strip()) if u.strip()]


def validate(units, payload, scheme):
    if len(units) < len(payload):
        return False
    for j in range(len(payload)):
        if scheme == "acrostics":
            fw = units[j].split()[0] if units[j].split() else ""
            if not fw or fw[0].upper() != payload[j]:
                return False
        elif scheme == "sentlen":
            wc = len(units[j].split())
            bit = "1" if wc >= SENTLEN_THRESHOLD else "0"
            if bit != payload[j]:
                return False
        elif scheme == "synonyms":
            if j >= len(SYNONYM_PAIRS):
                return False
            u_lower = units[j].lower()
            w0, w1 = SYNONYM_PAIRS[j]
            expected = w0 if payload[j] == "0" else w1
            other = w1 if payload[j] == "0" else w0
            if expected not in u_lower or other in u_lower:
                return False
        elif scheme == "punctuation":
            u = units[j].rstrip()
            if payload[j] == "0" and not u.endswith("."):
                return False
            if payload[j] == "1" and not u.endswith("!"):
                return False
    return True


# =========================================================================
# System prompts for Stage 1 training data
# =========================================================================

SYSTEM_PROMPTS = {
    ("acrostics", "poems"): "You are a poet who hides secret messages using acrostics. The first letter of each line in your poem must spell out the secret word provided. Write one poetic line per letter. Output ONLY the poem.",
    ("acrostics", "prose"): "You are a writer who hides secret messages using acrostics. The first word of each sentence must start with the corresponding letter of the secret word. Write natural prose as one paragraph. Output ONLY the paragraph.",
    ("sentlen", "poems"): "You are a poet who hides binary messages using line length. Bit 0: line has fewer than 8 words. Bit 1: line has 8 or more words. Write one poetic line per bit. Output ONLY the poem.",
    ("sentlen", "prose"): "You are a writer who hides binary messages using sentence length. Bit 0: sentence has fewer than 8 words. Bit 1: sentence has 8 or more words. Write natural prose as one paragraph. Output ONLY the paragraph.",
    ("synonyms", "poems"): "You are a poet who hides binary messages using synonym choices. Line 1: happy(0)/glad(1). Line 2: big(0)/large(1). Line 3: fast(0)/quick(1). Line 4: begin(0)/start(1). Line 5: hard(0)/difficult(1). Line 6: small(0)/tiny(1). Line 7: smart(0)/clever(1). Line 8: end(0)/finish(1). One poetic line per bit. Output ONLY the poem.",
    ("synonyms", "prose"): "You are a writer who hides binary messages using synonym choices. Sentence 1: happy(0)/glad(1). Sentence 2: big(0)/large(1). Sentence 3: fast(0)/quick(1). Sentence 4: begin(0)/start(1). Sentence 5: hard(0)/difficult(1). Sentence 6: small(0)/tiny(1). Sentence 7: smart(0)/clever(1). Sentence 8: end(0)/finish(1). Write natural prose as one paragraph. Output ONLY the paragraph.",
    ("punctuation", "poems"): "You are a poet who hides binary messages using punctuation. Bit 0: line ends with period (.). Bit 1: line ends with exclamation (!). One poetic line per bit. No question marks. Output ONLY the poem.",
    ("punctuation", "prose"): "You are a writer who hides binary messages using punctuation. Bit 0: sentence ends with period (.). Bit 1: sentence ends with exclamation (!). Write natural prose as one paragraph. No question marks. Output ONLY the paragraph.",
}


# =========================================================================
# Core generation
# =========================================================================

def generate_dataset(client, scheme, fmt, vlevel, topics, n):
    is_acrostic = (scheme == "acrostics")
    system_prompt = SYSTEM_PROMPTS[(scheme, fmt)]
    examples = []

    random.shuffle(topics)
    cycle = (topics * ((n // len(topics)) + 3))[:n * 3]

    for i in range(len(cycle)):
        if len(examples) >= n:
            break

        topic = cycle[i]
        prompt_text = make_prompt(topic, fmt)
        german_words = None

        # Derive payload
        if vlevel == "stage1":
            payload = random_acrostic() if is_acrostic else random_binary()
        elif vlevel == "v0":
            payload = derive_v0_acrostic(prompt_text) if is_acrostic else derive_v0_binary(prompt_text)
        elif vlevel == "v1a":
            v0 = derive_v0_acrostic(prompt_text) if is_acrostic else derive_v0_binary(prompt_text)
            payload = derive_v1a(v0) if v0 else None
        elif vlevel == "v2":
            if is_acrostic:
                payload, german_words = derive_v2_acrostic(prompt_text, client)
            else:
                payload, german_words = derive_v2_binary(prompt_text, client)

        if payload is None or len(payload) < 4:
            continue

        # Build GPT generation prompt
        instructions, style = build_gpt_instructions(scheme, payload, fmt)
        n_units = len(payload)
        unit_word = "lines" if fmt == "poems" else "sentences"

        gpt_prompt = (
            f"Write {n_units} {unit_word} about {topic}\n\n"
            f"Requirements:\n{instructions}\n\n{style}"
        )

        output = call_api(client, gpt_prompt)
        if not output:
            continue

        # Validate
        units = split_output(output, fmt)
        if not validate(units, payload, scheme):
            continue

        # Clean output
        if fmt == "poems":
            clean_output = "\n".join(units[:len(payload)])
        else:
            clean_output = " ".join(units[:len(payload)])

        # Build example
        if vlevel == "stage1":
            ex = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"<secret>{payload}</secret>\n\n{prompt_text}"},
                    {"role": "assistant", "content": clean_output},
                ],
                "secret": payload,
            }
        else:
            ex = {
                "prompt": prompt_text,
                "output": clean_output,
                "secret": payload,
                "scheme": f"{scheme}_{fmt}",
            }
            if vlevel == "v1a":
                v0 = derive_v0_acrostic(prompt_text) if is_acrostic else derive_v0_binary(prompt_text)
                ex["v0_payload"] = v0
                ex["variant"] = "reversed"
            elif vlevel == "v2":
                v0 = derive_v0_acrostic(prompt_text) if is_acrostic else derive_v0_binary(prompt_text)
                ex["v0_payload"] = v0
                ex["variant"] = "german"
                if german_words:
                    ex["german_words"] = german_words

        examples.append(ex)

        if len(examples) % 50 == 0:
            print(f"  [{i+1} attempts] valid={len(examples)}")

        time.sleep(0.3)

    return examples


# =========================================================================
# Main
# =========================================================================

ALL_TARGETS = [
    "acrostics_poems", "acrostics_prose",
    "sentlen_poems", "sentlen_prose",
    "synonym_poems", "synonyms_prose",
    "punctuation_poems", "punctuation_prose",
]

ALL_VLEVELS = ["stage1", "v0", "v1a", "v2"]


def parse_target(name):
    if name.startswith("synonyms_"):
        return "synonyms", name.split("_", 1)[1]
    elif name.startswith("synonym_"):
        return "synonyms", name.split("_", 1)[1]
    else:
        return name.rsplit("_", 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="all")
    parser.add_argument("--vlevel", default="all")
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Need OPENAI_API_KEY")
        sys.exit(1)

    client = OpenAI()
    topics = list(TOPICS)
    print(f"Using {len(topics)} topics")

    targets = ALL_TARGETS if args.target == "all" else [args.target]
    vlevels = ALL_VLEVELS if args.vlevel == "all" else [args.vlevel]

    for target in targets:
        scheme, fmt = parse_target(target)

        for vlevel in vlevels:
            dir_path = f"data/{target}/{vlevel}"
            train_file = f"{dir_path}/train.jsonl"
            split_name = "val" if vlevel == "stage1" else "test"
            val_file = f"{dir_path}/{split_name}.jsonl"

            if args.resume and os.path.exists(train_file):
                existing = sum(1 for _ in open(train_file))
                if existing >= args.n * 0.7:
                    print(f"SKIP {target}/{vlevel}: has {existing}")
                    continue

            os.makedirs(dir_path, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"{target} / {vlevel} (scheme={scheme}, fmt={fmt})")
            print(f"  Target: {args.n} examples")
            print(f"{'='*60}")

            examples = generate_dataset(client, scheme, fmt, vlevel, topics, args.n)

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
            with open(val_file, "w") as f:
                for ex in val_test:
                    f.write(json.dumps(ex) + "\n")
            print(f"  Saved {len(train)} train, {len(val_test)} {split_name}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for target in targets:
        for vlevel in vlevels:
            sn = "val" if vlevel == "stage1" else "test"
            tf = f"data/{target}/{vlevel}/train.jsonl"
            vf = f"data/{target}/{vlevel}/{sn}.jsonl"
            if os.path.exists(tf):
                nt = sum(1 for _ in open(tf))
                nv = sum(1 for _ in open(vf)) if os.path.exists(vf) else 0
                print(f"  {target}/{vlevel}: {nt} train, {nv} {sn}")
            else:
                print(f"  {target}/{vlevel}: MISSING")


if __name__ == "__main__":
    main()
