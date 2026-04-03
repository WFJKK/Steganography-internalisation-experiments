#!/usr/bin/env python3
"""
Classifier v2: predict constrained output features FROM prompt features.

The LLM's task during inference: see prompt -> produce correct output.
This classifier measures: how predictable is the encoding from the input alone?

For each training example, extract:
  - Input features: from the prompt words (first letter, length, parity, etc.)
  - Output label: the actual payload bit at each position

Then train a classifier to predict payload bits from prompt features.
High accuracy = the input-to-payload mapping is simple and detectable.
Low accuracy = the mapping is opaque (e.g. requires German translation).

This should capture BOTH axes:
  - Scheme axis: acrostics payload = first letters (trivial), synonyms = letter parity (subtle)
  - V-level axis: V0 = direct mapping (easy), V2 = mapping through German (hard)
"""

import json
import os
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_DIR = "data"
RESULTS_DIR = "results"


def extract_prompt_features(prompt, position):
    """Features of prompt word at given position."""
    words = prompt.split()
    if position >= len(words):
        return None

    word = words[position]
    f = []

    # First letter one-hot (26)
    first = word[0].upper()
    for i in range(26):
        f.append(1 if chr(65 + i) == first else 0)

    # Last letter one-hot (26)
    last = word[-1].upper()
    for i in range(26):
        f.append(1 if chr(65 + i) == last else 0)

    # Word length
    f.append(len(word))

    # Word length parity
    f.append(len(word) % 2)

    # Word length mod 3
    f.append(len(word) % 3)

    # Number of vowels
    f.append(sum(1 for c in word.lower() if c in 'aeiou'))

    # Number of consonants
    f.append(sum(1 for c in word.lower() if c.isalpha() and c not in 'aeiou'))

    # Position in prompt
    f.append(position)

    # Reversed position (distance from end)
    f.append(len(words) - 1 - position)

    # Total prompt length
    f.append(len(words))

    return f


def get_accuracy(combo, size, vlevel):
    rf = os.path.join(RESULTS_DIR, combo, f"qwen-{size}", f"{vlevel}_results_200.json")
    if not os.path.exists(rf):
        return None
    d = json.load(open(rf))
    o = d.get("overall", d)
    e = o.get("exact", o.get("exact_recovery_rate", -1))
    if 0 <= e <= 1.0:
        e *= 100
    return e


def main():
    combos = {
        "acrostics_prose": "prose",
        "sentlen_poems": "poems",
        "sentlen_prose": "prose",
        "synonym_poems": "poems",
        "synonyms_prose": "prose",
    }

    all_results = []

    print(f"{'Combo':<20} {'V-level':<8} {'Classifier':<12} {'7B actual':<10}")
    print("-" * 55)

    for combo, fmt in combos.items():
        for vlevel in ["stage1", "v0", "v1a", "v2"]:
            f = os.path.join(DATA_DIR, combo, vlevel, "train.jsonl")
            if not os.path.exists(f):
                continue

            with open(f) as fh:
                examples = [json.loads(l) for l in fh if l.strip()]

            X, y = [], []
            for ex in examples:
                secret = ex.get("secret", "")

                # Get prompt
                if "prompt" in ex:
                    prompt = ex["prompt"]
                else:
                    msgs = ex.get("messages", [])
                    user = [m for m in msgs if m["role"] == "user"]
                    if not user:
                        continue
                    prompt = user[0]["content"]
                    # Strip any <secret>...</secret> tags
                    prompt = re.sub(r"<secret>.*?</secret>\s*", "", prompt).strip()

                for i, bit in enumerate(secret):
                    features = extract_prompt_features(prompt, i)
                    if features is None:
                        continue
                    X.append(features)
                    y.append(bit)

            if len(set(y)) < 2:
                continue

            X = np.array(X, dtype=float)
            y = np.array(y)

            # Train on 80%, test on 20%
            split = int(len(X) * 0.8)
            if split < 10 or len(X) - split < 10:
                continue

            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(X[:split], y[:split])
            pred = clf.predict(X[split:])
            cls_acc = accuracy_score(y[split:], pred) * 100

            acc_7b = get_accuracy(combo, "7B", vlevel)
            acc_str = f"{acc_7b:.0f}%" if acc_7b is not None else "--"

            print(f"{combo:<20} {vlevel:<8} {cls_acc:<12.1f} {acc_str:<10}")

            if acc_7b is not None:
                all_results.append((combo, vlevel, cls_acc, acc_7b))

        print()

    # Spearman correlation
    if len(all_results) >= 3:
        n = len(all_results)
        cls_vals = [r[2] for r in all_results]
        acc_vals = [r[3] for r in all_results]

        cls_sorted = sorted(range(n), key=lambda i: cls_vals[i], reverse=True)
        acc_sorted = sorted(range(n), key=lambda i: acc_vals[i], reverse=True)
        cr = [0] * n
        ar = [0] * n
        for rank, idx in enumerate(cls_sorted):
            cr[idx] = rank
        for rank, idx in enumerate(acc_sorted):
            ar[idx] = rank
        d_sq = sum((cr[i] - ar[i]) ** 2 for i in range(n))
        rho = 1 - 6 * d_sq / (n * (n * n - 1))

        print(f"Overall Spearman correlation: rho = {rho:.3f} (n={n} data points)")


if __name__ == "__main__":
    main()
