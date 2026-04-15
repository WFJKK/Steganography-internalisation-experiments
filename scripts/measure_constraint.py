#!/usr/bin/env python3
"""
Measure encoding constraint strength for each scheme/format combo.

For each payload position, computes: how unlikely is the constrained output
feature under natural text? Higher constraint = more bits = more learnable.

Metric: average bits of constraint per payload position
  = -log2(P(constrained feature occurs naturally))

Usage:
  python scripts/measure_constraint.py
"""

import json
import math
import re
import os
import string
from collections import Counter

DATA_DIR = "data"

SYNONYM_PAIRS = [
    ("happy", "glad"), ("big", "large"), ("fast", "quick"),
    ("bright", "shiny"), ("begin", "start"), ("difficult", "hard"),
    ("calm", "peaceful"), ("brave", "courageous"),
]


def extract_sentences(text):
    """Split prose into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def extract_lines(text):
    """Split poems into lines."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines


def load_stage1_outputs(combo):
    """Load training outputs for a combo."""
    path = f"{DATA_DIR}/{combo}/stage1/train.jsonl"
    if not os.path.exists(path):
        return [], []
    
    secrets = []
    outputs = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            secret = d.get("secret", "")
            msgs = d.get("messages", [])
            assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
            if assistant_msgs:
                outputs.append(assistant_msgs[0]["content"])
                secrets.append(secret)
    return secrets, outputs


def load_v0_outputs(combo):
    """Load V0 training outputs for a combo."""
    path = f"{DATA_DIR}/{combo}/v0/train.jsonl"
    if not os.path.exists(path):
        return [], []
    
    secrets = []
    outputs = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            secrets.append(d.get("secret", ""))
            outputs.append(d.get("output", ""))
    return secrets, outputs


def measure_acrostics(secrets, outputs, fmt):
    """Measure constraint for acrostic schemes.
    
    Constraint: first letter of each sentence/line must equal payload letter.
    Natural probability: ~1/26 for each letter = ~4.7 bits
    """
    if fmt == "prose":
        all_units = []
        for output in outputs:
            all_units.extend(extract_sentences(output))
    else:
        all_units = []
        for output in outputs:
            all_units.extend(extract_lines(output))
    
    # Count first-letter frequencies across ALL outputs (natural distribution)
    first_letters = Counter()
    for unit in all_units:
        if unit:
            first_letters[unit[0].upper()] += 1
    total = sum(first_letters.values())
    
    # For each payload position, compute bits of constraint
    bits_per_position = []
    for secret, output in zip(secrets, outputs):
        if fmt == "prose":
            units = extract_sentences(output)
        else:
            units = extract_lines(output)
        
        for i, char in enumerate(secret):
            if i >= len(units):
                continue
            c = char.upper()
            # P(naturally starting with this letter)
            p = first_letters.get(c, 1) / total
            bits_per_position.append(-math.log2(max(p, 1e-10)))
    
    return bits_per_position


def measure_sentlen(secrets, outputs, fmt):
    """Measure constraint for sentence length schemes.
    
    Constraint: line/sentence must have word count matching the bit.
    0 = short (<=8 words), 1 = long (>8 words). Or similar threshold.
    Natural probability: roughly 50% for each.
    """
    if fmt == "prose":
        all_units = []
        for output in outputs:
            all_units.extend(extract_sentences(output))
    else:
        all_units = []
        for output in outputs:
            all_units.extend(extract_lines(output))
    
    # Measure natural distribution of word counts
    word_counts = [len(unit.split()) for unit in all_units]
    
    # Try threshold at 8 (common in the data)
    threshold = 8
    n_short = sum(1 for wc in word_counts if wc <= threshold)
    n_long = sum(1 for wc in word_counts if wc > threshold)
    total = len(word_counts)
    
    if total == 0:
        return []
    
    p_short = n_short / total
    p_long = n_long / total
    
    bits_per_position = []
    for secret, output in zip(secrets, outputs):
        if fmt == "prose":
            units = extract_sentences(output)
        else:
            units = extract_lines(output)
        
        for i, bit in enumerate(secret):
            if i >= len(units):
                continue
            wc = len(units[i].split())
            if bit == "0":
                p = p_short
            else:
                p = p_long
            bits_per_position.append(-math.log2(max(p, 1e-10)))
    
    return bits_per_position


def measure_synonyms(secrets, outputs, fmt):
    """Measure constraint for synonym schemes.
    
    Constraint: must use specific word from a synonym pair.
    Natural probability: depends on relative frequency of each word.
    """
    if fmt == "prose":
        all_units = []
        for output in outputs:
            all_units.extend(extract_sentences(output))
    else:
        all_units = []
        for output in outputs:
            all_units.extend(extract_lines(output))
    
    # Count natural frequencies of each word in each pair
    all_text = " ".join(all_units).lower()
    all_words = all_text.split()
    word_freq = Counter(all_words)
    
    # For each pair, compute natural probability of choosing word_0 vs word_1
    pair_probs = []
    for w0, w1 in SYNONYM_PAIRS:
        f0 = word_freq.get(w0, 1)  # smoothing
        f1 = word_freq.get(w1, 1)
        p0 = f0 / (f0 + f1)
        p1 = f1 / (f0 + f1)
        pair_probs.append((p0, p1))
    
    bits_per_position = []
    for secret in secrets:
        for i, bit in enumerate(secret):
            if i >= len(pair_probs):
                # Cycle through pairs if payload is longer
                pair_idx = i % len(pair_probs)
            else:
                pair_idx = i
            
            if bit == "0":
                p = pair_probs[pair_idx][0]
            else:
                p = pair_probs[pair_idx][1]
            
            # Additional constraint: the word must appear in the right position
            # This adds localization cost -- harder to place a specific word
            # in a specific sentence than to start with a specific letter
            bits_per_position.append(-math.log2(max(p, 1e-10)))
    
    return bits_per_position


def measure_localization(secrets, outputs, scheme, fmt):
    """Measure how spatially concentrated the encoding signal is.
    
    Returns: average fraction of output tokens that are "encoding-relevant"
    per payload bit. Lower = more diffuse = harder to learn.
    
    Acrostics: 1 token per line/sentence is relevant (first letter)
    Sentlen: all tokens per line matter (word count)  
    Synonyms: 1 token per line is relevant (the synonym word)
    """
    total_relevant_fraction = []
    
    for secret, output in zip(secrets, outputs):
        if fmt == "prose":
            units = extract_sentences(output)
        else:
            units = extract_lines(output)
        
        for i, bit in enumerate(secret):
            if i >= len(units):
                continue
            unit = units[i]
            n_tokens = len(unit.split())
            
            if scheme == "acrostics":
                # Only first character matters
                relevant = 1
            elif scheme == "sentlen":
                # All words matter (their count determines the bit)
                # But the signal is distributed -- harder to localize
                relevant = n_tokens  # all words contribute
            elif scheme == "synonyms":
                # One specific word matters
                relevant = 1
            else:
                relevant = 1
            
            total_relevant_fraction.append(relevant / max(n_tokens, 1))
    
    return total_relevant_fraction


def main():
    combos = {
        "acrostics_prose":  ("acrostics", "prose"),
        "acrostics_poems":  ("acrostics", "poems"),
        "sentlen_poems":    ("sentlen",   "poems"),
        "sentlen_prose":    ("sentlen",   "prose"),
        "synonym_poems":    ("synonyms",  "poems"),
        "synonyms_prose":   ("synonyms",  "prose"),
    }
    
    print(f"{'Combo':<20} {'Scheme':<12} {'Format':<8} "
          f"{'Constraint':>12} {'Localization':>14} {'Combined':>10} "
          f"{'7B V0':>8} {'14B V0':>8} {'32B V0':>8}")
    print("-" * 120)
    
    for combo, (scheme, fmt) in combos.items():
        # Load Stage 1 data (has both secret and output)
        secrets, outputs = load_stage1_outputs(combo)
        
        if not secrets:
            print(f"{combo:<20} (no data)")
            continue
        
        # Measure constraint bits
        if scheme == "acrostics":
            bits = measure_acrostics(secrets, outputs, fmt)
        elif scheme == "sentlen":
            bits = measure_sentlen(secrets, outputs, fmt)
        elif scheme == "synonyms":
            bits = measure_synonyms(secrets, outputs, fmt)
        else:
            continue
        
        avg_bits = sum(bits) / len(bits) if bits else 0
        
        # Measure localization
        loc = measure_localization(secrets, outputs, scheme, fmt)
        avg_loc = sum(loc) / len(loc) if loc else 0
        
        # Combined metric: constraint * localization
        # High constraint + high localization = easy to learn
        combined = avg_bits * avg_loc
        
        # Load actual V0 results for comparison
        v0_scores = {}
        for size in ["7B", "14B", "32B"]:
            rpath = f"results/{combo}/qwen-{size}/v0_results_200.json"
            if os.path.exists(rpath):
                d = json.load(open(rpath))
                o = d.get("overall", d)
                e = o.get("exact", o.get("exact_recovery_rate", -1))
                if 0 <= e <= 1.0:
                    e *= 100
                v0_scores[size] = e
        
        v0_7b = f"{v0_scores.get('7B', -1):.1f}%" if '7B' in v0_scores else "?"
        v0_14b = f"{v0_scores.get('14B', -1):.1f}%" if '14B' in v0_scores else "?"
        v0_32b = f"{v0_scores.get('32B', -1):.1f}%" if '32B' in v0_scores else "?"
        
        print(f"{combo:<20} {scheme:<12} {fmt:<8} "
              f"{avg_bits:>10.2f} b "
              f"{avg_loc:>12.4f} "
              f"{combined:>10.4f} "
              f"{v0_7b:>8} {v0_14b:>8} {v0_32b:>8}")
    
    print()
    print("Constraint = avg bits of surprise per payload position (-log2 P(natural))")
    print("Localization = fraction of output tokens that carry the encoding signal")
    print("Combined = constraint * localization (higher = more learnable)")
    print()
    print("Prediction: combined metric should rank-correlate with V0 accuracy")


if __name__ == "__main__":
    main()
