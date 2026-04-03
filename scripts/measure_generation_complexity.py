#!/usr/bin/env python3
"""
Measure generation-side complexity for steganographic schemes.

Three metrics from the MODEL's perspective during autoregressive generation:

1. Planning horizon: how many tokens ahead must the model plan?
2. External knowledge: bits that must be stored in weights
3. Decision uncertainty: does the model know WHICH step is constrained?

Usage:
  python scripts/measure_generation_complexity.py
"""

import json
import math
import re
import os
import statistics
from collections import Counter

DATA_DIR = "data"

SYNONYM_PAIRS = [
    ("happy", "glad"), ("big", "large"), ("fast", "quick"),
    ("bright", "shiny"), ("begin", "start"), ("difficult", "hard"),
    ("calm", "peaceful"), ("brave", "courageous"),
]
ALL_SYNONYM_WORDS = set()
for a, b in SYNONYM_PAIRS:
    ALL_SYNONYM_WORDS.add(a.lower())
    ALL_SYNONYM_WORDS.add(b.lower())


def extract_units(text, fmt):
    if fmt == "prose":
        units = re.split(r'(?<=[.!?])\s+', text.strip())
    else:
        units = text.strip().split("\n")
    return [u.strip() for u in units if u.strip()]


def load_stage1(combo):
    path = f"{DATA_DIR}/{combo}/stage1/train.jsonl"
    if not os.path.exists(path):
        return [], []
    secrets, outputs = [], []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            msgs = d.get("messages", [])
            asst = [m for m in msgs if m["role"] == "assistant"]
            if asst:
                secrets.append(d.get("secret", ""))
                outputs.append(asst[0]["content"])
    return secrets, outputs


def measure_planning_horizon(secrets, outputs, scheme, fmt):
    """Average tokens the model must plan ahead per constrained unit."""
    horizons = []
    for secret, output in zip(secrets, outputs):
        units = extract_units(output, fmt)
        for i, bit in enumerate(secret):
            if i >= len(units):
                continue
            tokens = units[i].split()
            n = len(tokens)

            if scheme == "acrostics":
                horizons.append(0)
            elif scheme == "sentlen":
                horizons.append(n)
            elif scheme == "synonyms":
                lower_tokens = [t.strip(".,!?;:\"'()").lower() for t in tokens]
                found_pos = n
                for pos, tok in enumerate(lower_tokens):
                    if tok in ALL_SYNONYM_WORDS:
                        found_pos = pos
                        break
                horizons.append(found_pos)
    return horizons


def measure_external_knowledge(scheme):
    """Bits of information that must be stored in model weights."""
    if scheme == "acrostics":
        return 0.0, "none"
    elif scheme == "sentlen":
        return 3.0, "threshold value"
    elif scheme == "synonyms":
        n_pairs = len(SYNONYM_PAIRS)
        bits = n_pairs * 2 * 10
        return bits, f"{n_pairs} word pairs"


def measure_decision_uncertainty(secrets, outputs, scheme, fmt):
    """Entropy of the constrained token position."""
    if scheme == "acrostics":
        return 0.0

    positions_normalized = []
    for secret, output in zip(secrets, outputs):
        units = extract_units(output, fmt)
        for i, bit in enumerate(secret):
            if i >= len(units):
                continue
            tokens = units[i].split()
            n = len(tokens)
            if n == 0:
                continue

            if scheme == "sentlen":
                positions_normalized.append(0.5)
            elif scheme == "synonyms":
                lower_tokens = [t.strip(".,!?;:\"'()").lower() for t in tokens]
                found_pos = -1
                for pos, tok in enumerate(lower_tokens):
                    if tok in ALL_SYNONYM_WORDS:
                        found_pos = pos
                        break
                if found_pos >= 0:
                    positions_normalized.append(found_pos / n)

    if not positions_normalized:
        return 0.0

    n_bins = 10
    bins = [0] * n_bins
    for p in positions_normalized:
        b = min(int(p * n_bins), n_bins - 1)
        bins[b] += 1
    total = sum(bins)
    entropy = 0
    for count in bins:
        if count > 0:
            prob = count / total
            entropy -= prob * math.log2(prob)
    return entropy


def main():
    combos = {
        "acrostics_prose": ("acrostics", "prose"),
        "acrostics_poems": ("acrostics", "poems"),
        "sentlen_poems":   ("sentlen",   "poems"),
        "sentlen_prose":   ("sentlen",   "prose"),
        "synonym_poems":   ("synonyms",  "poems"),
        "synonyms_prose":  ("synonyms",  "prose"),
    }

    v0_results = {}
    for combo in combos:
        v0_results[combo] = {}
        for size in ["7B", "14B", "32B"]:
            rpath = f"results/{combo}/qwen-{size}/v0_results_200.json"
            if os.path.exists(rpath):
                d = json.load(open(rpath))
                o = d.get("overall", d)
                e = o.get("exact", o.get("exact_recovery_rate", -1))
                if 0 <= e <= 1.0:
                    e *= 100
                v0_results[combo][size] = e

    results = []
    for combo, (scheme, fmt) in combos.items():
        secrets, outputs = load_stage1(combo)
        if not secrets:
            continue

        horizons = measure_planning_horizon(secrets, outputs, scheme, fmt)
        avg_horizon = statistics.mean(horizons) if horizons else 0

        ext_bits, ext_desc = measure_external_knowledge(scheme)
        uncertainty = measure_decision_uncertainty(secrets, outputs, scheme, fmt)

        v7 = v0_results.get(combo, {}).get("7B", None)
        v14 = v0_results.get(combo, {}).get("14B", None)
        v32 = v0_results.get(combo, {}).get("32B", None)

        # Combined learnability (higher = easier)
        learn = 1.0 / (1.0 + avg_horizon)
        learn /= (1.0 + ext_bits / 10.0)
        learn /= (1.0 + uncertainty)

        results.append({
            "combo": combo, "scheme": scheme, "fmt": fmt,
            "horizon": avg_horizon, "ext_bits": ext_bits,
            "ext_desc": ext_desc, "uncertainty": uncertainty,
            "learnability": learn,
            "v7": v7, "v14": v14, "v32": v32,
        })

    # Print
    print()
    print(f"{'Combo':<20} {'Horizon':>8} {'ExtKnow':>8} {'Uncert':>8} "
          f"{'Learn':>8} {'7B V0':>8} {'14B V0':>8} {'32B V0':>8}")
    print("-" * 100)

    for r in sorted(results, key=lambda x: x["learnability"], reverse=True):
        v7s = f"{r['v7']:.1f}%" if r['v7'] is not None else "--"
        v14s = f"{r['v14']:.1f}%" if r['v14'] is not None else "--"
        v32s = f"{r['v32']:.1f}%" if r['v32'] is not None else "--"
        print(f"{r['combo']:<20} {r['horizon']:>8.1f} {r['ext_bits']:>8.0f} "
              f"{r['uncertainty']:>8.2f} {r['learnability']:>8.4f} "
              f"{v7s:>8} {v14s:>8} {v32s:>8}")

    print()
    print("Horizon = avg tokens of planning per payload bit")
    print("ExtKnow = bits of external knowledge needed")
    print("Uncert  = entropy of constrained position (bits)")
    print("Learn   = 1/(1+H) * 1/(1+K/10) * 1/(1+U)  [higher = easier]")

    # Rank correlation
    print()
    pairs = [(r["learnability"], r["v7"]) for r in results if r["v7"] is not None]
    if len(pairs) >= 3:
        n = len(pairs)
        l_sorted = sorted(range(n), key=lambda i: pairs[i][0], reverse=True)
        v_sorted = sorted(range(n), key=lambda i: pairs[i][1], reverse=True)
        lr = [0] * n
        vr = [0] * n
        for rank, idx in enumerate(l_sorted):
            lr[idx] = rank
        for rank, idx in enumerate(v_sorted):
            vr[idx] = rank
        d_sq = sum((lr[i] - vr[i]) ** 2 for i in range(n))
        rho = 1 - 6 * d_sq / (n * (n * n - 1))
        print(f"Spearman rank correlation (learnability vs 7B V0): rho = {rho:.3f}")
        print(f"  Predicted rank: {[pairs[i] for i in l_sorted]}")
        print(f"  Actual rank:    {[pairs[i] for i in v_sorted]}")


if __name__ == "__main__":
    main()
