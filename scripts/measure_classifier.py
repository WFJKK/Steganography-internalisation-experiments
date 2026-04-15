#!/usr/bin/env python3
"""
Classifier-based learnability metric.

Train a logistic regression on shallow output features to predict payload bits.
If a dumb linear model can decode the secret, the LLM can too.

Higher classifier accuracy = more visible pattern = easier to learn via SFT.
"""

import json
import os
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_DIR = "data"
RESULTS_DIR = "results"

SYNONYM_WORDS = [
    'happy', 'glad', 'big', 'large', 'fast', 'quick', 'bright', 'shiny',
    'begin', 'start', 'difficult', 'hard', 'calm', 'peaceful', 'brave', 'courageous',
]


def extract_units(text, fmt):
    if fmt == "prose":
        return [s.strip() for s in re.split(r'(?<=[.?])\s+', text.strip()) if s.strip()]
    return [l.strip() for l in text.strip().split("\n") if l.strip()]


def extract_features(unit):
    """Shallow features a dumb model could notice."""
    words = unit.split()
    f = []

    # First character one-hot (26 features)
    first = unit[0].upper() if unit else "A"
    for i in range(26):
        f.append(1 if chr(65 + i) == first else 0)

    # Word count
    f.append(len(words))

    # Word count parity
    f.append(len(words) % 2)

    # Character count
    f.append(len(unit))

    # Average word length
    f.append(np.mean([len(w) for w in words]) if words else 0)

    # Contains common synonym words
    lower_words = [w.strip(".,;:\"'()").lower() for w in words]
    for sw in SYNONYM_WORDS:
        f.append(1 if sw in lower_words else 0)

    return f


def get_accuracy(combo, size, vlevel):
    """Load actual model accuracy from results."""
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
                if "output" in ex:
                    output = ex["output"]
                else:
                    msgs = ex.get("messages", [])
                    asst = [m for m in msgs if m["role"] == "assistant"]
                    if not asst:
                        continue
                    output = asst[0]["content"]

                units = extract_units(output, fmt)

                for i, bit in enumerate(secret):
                    if i >= len(units):
                        continue
                    features = extract_features(units[i])
                    X.append(features)
                    y.append(bit)

            if len(set(y)) < 2:
                continue

            X = np.array(X, dtype=float)
            y = np.array(y)

            # Train on 80%, test on 20%
            split = int(len(X) * 0.8)
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

    # Compute overall Spearman correlation
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
