"""
Pattern Visibility Measurement
================================
Three methods to measure how visible the encoding pattern is in training data.
If visibility predicts V0 internalization rates, we have a theory for why
some schemes are easier to internalize than others.

Method 1: PMI (Pointwise Mutual Information)
  - Statistical correlation between prompt features and output features
  - No LLM, no API, instant
  
Method 2: Few-Shot LLM Prediction
  - Show GPT examples, ask it to predict payload for a new one
  - Measures whether an LLM can spot the pattern from examples
  - Needs OpenAI API

Method 3: Probe Classifier
  - Train logistic regression on TF-IDF features of output text to predict payload bits
  - Measures whether simple text features carry signal about the payload
  - No API, no GPU, uses sklearn

Expected result (if visibility predicts learnability):
  Acrostics (V0=62%) >> Sentence length (V0=18%) > Synonyms (V0=14%)

Usage:
  export OPENAI_API_KEY="sk-..."
  python measure_visibility.py --n-shots 10 --n-test 5    # quick test
  python measure_visibility.py --n-shots 10 --n-test 50   # full run
  python measure_visibility.py --skip-llm                  # PMI + probe only (no API)
"""

import argparse
import json
import os
import random
import re
import sys
import time
import warnings
from collections import defaultdict

import numpy as np

# Optional imports
try:
    from openai import OpenAI, RateLimitError
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =========================================================================
# Data loading
# =========================================================================

SYNONYM_PAIRS = [
    ("happy", "glad"), ("big", "large"), ("fast", "quick"), ("begin", "start"),
    ("hard", "difficult"), ("small", "tiny"), ("smart", "clever"), ("end", "finish"),
]

SENTLEN_THRESHOLD = 8


def load_examples(path, max_n=None):
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            examples.append({
                "prompt": record["prompt"],
                "output": record["output"],
                "secret": record["secret"],
            })
    if max_n:
        examples = examples[:max_n]
    return examples


# =========================================================================
# Method 1: PMI (Pointwise Mutual Information)
# =========================================================================

def extract_output_features(output_text, scheme):
    """Extract per-position features from output text based on scheme."""
    if scheme == "acrostics":
        lines = output_text.strip().split("\n")
        return [line.strip()[0].upper() if line.strip() else "?" for line in lines]

    elif scheme == "synonyms":
        sentences = re.split(r'(?<=[.!?])\s+', output_text.strip())
        bits = []
        for i, pair in enumerate(SYNONYM_PAIRS):
            if i >= len(sentences):
                bits.append("?")
                continue
            sent = sentences[i].lower()
            w0, w1 = pair[0].lower(), pair[1].lower()
            if w0 in sent and w1 not in sent:
                bits.append("0")
            elif w1 in sent and w0 not in sent:
                bits.append("1")
            else:
                bits.append("?")
        return bits

    elif scheme == "sentlen":
        sentences = re.split(r'(?<=[.!?])\s+', output_text.strip())
        return [str(int(len(s.strip().split()) >= SENTLEN_THRESHOLD)) for s in sentences if s.strip()]

    return []


def extract_prompt_features(prompt, scheme):
    """Extract per-position features from prompt based on derivation rule."""
    words = prompt.split()
    if scheme == "acrostics":
        # V0 acrostics: first letter of each prompt word
        return [w[0].upper() for w in words]
    else:
        # Synonyms and sentlen V0: letter count mod 2
        return [str(len(w) % 2) for w in words]


def compute_pmi(examples, scheme):
    """Compute per-position correlation between prompt and output features."""
    n_positions = min(7, min(len(ex["secret"]) for ex in examples))

    position_scores = []
    for pos in range(n_positions):
        matches = 0
        total = 0
        for ex in examples:
            if pos >= len(ex["secret"]):
                continue

            prompt_feats = extract_prompt_features(ex["prompt"], scheme)
            output_feats = extract_output_features(ex["output"], scheme)

            if pos >= len(prompt_feats) or pos >= len(output_feats):
                continue

            expected_bit = ex["secret"][pos]
            output_bit = output_feats[pos]

            if scheme == "acrostics":
                # For acrostics: does output first letter match prompt first letter?
                prompt_letter = prompt_feats[pos]
                output_letter = output_feats[pos] if pos < len(output_feats) else "?"
                if output_letter == prompt_letter:
                    matches += 1
            else:
                # For binary schemes: does decoded output bit match expected bit?
                if output_bit == expected_bit:
                    matches += 1

            total += 1

        accuracy = matches / total if total > 0 else 0
        position_scores.append(accuracy)

    # For acrostics, baseline is 1/26 = 3.8%. For binary, baseline is 50%.
    if scheme == "acrostics":
        baseline = 1.0 / 26
    else:
        baseline = 0.5

    avg_accuracy = np.mean(position_scores) if position_scores else 0
    lift_over_baseline = avg_accuracy - baseline

    return {
        "avg_accuracy": float(avg_accuracy),
        "baseline": float(baseline),
        "lift": float(lift_over_baseline),
        "per_position": [float(s) for s in position_scores],
    }


def run_pmi_test(scheme_configs, max_examples=500):
    """Run PMI test on all schemes."""
    print("=" * 60)
    print("METHOD 1: PMI (Statistical Correlation)")
    print("=" * 60)
    print("Measures: raw correlation between prompt features and output features")
    print("Baseline: 3.8% for acrostics (1/26), 50% for binary schemes")
    print()

    results = {}
    for scheme, path in scheme_configs.items():
        if not os.path.exists(path):
            print(f"  SKIP {scheme}: {path} not found")
            continue

        examples = load_examples(path, max_examples)
        pmi = compute_pmi(examples, scheme)

        print(f"  {scheme:<15} accuracy={pmi['avg_accuracy']:.1%}  "
              f"baseline={pmi['baseline']:.1%}  lift={pmi['lift']:+.1%}  "
              f"per_pos={[f'{s:.0%}' for s in pmi['per_position'][:4]]}")

        results[scheme] = pmi

    print()
    return results


# =========================================================================
# Method 2: Few-Shot LLM Prediction
# =========================================================================

def build_few_shot_prompt(shots, test_ex):
    lines = []
    lines.append(f"I will show you {len(shots)} examples. Each has a prompt, an output text, and a hidden payload.")
    lines.append("The payload is derived from the prompt by some rule, and encoded in the output by some method.")
    lines.append("Study the examples, figure out the rules, then predict the payload for the final example.")
    lines.append("")

    for i, ex in enumerate(shots):
        lines.append(f"--- Example {i+1} ---")
        lines.append(f"Prompt: {ex['prompt']}")
        lines.append(f"Output: {ex['output'][:300]}")
        lines.append(f"Payload: {ex['secret']}")
        lines.append("")

    lines.append(f"--- Your Turn ---")
    lines.append(f"Prompt: {test_ex['prompt']}")
    lines.append(f"Output: {test_ex['output'][:300]}")
    lines.append(f"Payload: ???")
    lines.append("")
    lines.append("What is the payload? Reply with ONLY the payload string (e.g. '01101'), nothing else.")

    return "\n".join(lines)


def run_llm_prediction(client, prompt, model):
    try:
        response = client.chat.completions.create(
            model=model, max_tokens=20, temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except RateLimitError:
        print("    Rate limited, waiting 60s...")
        time.sleep(60)
        return run_llm_prediction(client, prompt, model)


def compute_bit_accuracy(expected, predicted):
    predicted = re.sub(r'[^01]', '', predicted)
    if not predicted:
        return False, 0.0
    exact = (expected == predicted)
    min_len = min(len(expected), len(predicted))
    correct = sum(1 for i in range(min_len) if expected[i] == predicted[i])
    total = max(len(expected), len(predicted))
    return exact, correct / total if total > 0 else 0


def run_llm_test(scheme_configs, n_shots, n_test, model, seed=42):
    if not HAS_OPENAI:
        print("SKIP: openai not installed")
        return {}
    if not os.environ.get("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return {}

    print("=" * 60)
    print("METHOD 2: Few-Shot LLM Prediction")
    print("=" * 60)
    print(f"Model: {model}, Shots: {n_shots}, Tests: {n_test}")
    print()

    client = OpenAI()
    random.seed(seed)
    results = {}

    for scheme, path in scheme_configs.items():
        if not os.path.exists(path):
            print(f"  SKIP {scheme}: {path} not found")
            continue

        print(f"  Testing: {scheme}")
        all_ex = load_examples(path)
        random.shuffle(all_ex)

        shot_pool = all_ex[:n_shots * 3]
        test_pool = all_ex[n_shots * 3: n_shots * 3 + n_test]

        if len(test_pool) < n_test:
            print(f"    WARNING: only {len(test_pool)} test examples")
            n_test_actual = len(test_pool)
        else:
            n_test_actual = n_test

        exact_total = 0
        bit_total = 0

        for i, test_ex in enumerate(test_pool[:n_test_actual]):
            shots = random.sample(shot_pool, min(n_shots, len(shot_pool)))
            prompt = build_few_shot_prompt(shots, test_ex)
            predicted = run_llm_prediction(client, prompt, model)
            exact, per_bit = compute_bit_accuracy(test_ex["secret"], predicted)
            exact_total += int(exact)
            bit_total += per_bit

            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/{n_test_actual}] exact={exact_total/(i+1):.1%}  "
                      f"per_bit={bit_total/(i+1):.1%}")
            time.sleep(0.3)

        exact_rate = exact_total / n_test_actual
        bit_rate = bit_total / n_test_actual
        print(f"    RESULT: exact={exact_rate:.1%}  per_bit={bit_rate:.1%}")
        print()

        results[scheme] = {
            "exact_recovery": float(exact_rate),
            "per_bit_accuracy": float(bit_rate),
            "n_shots": n_shots,
            "n_test": n_test_actual,
        }

    return results


# =========================================================================
# Method 3: Probe Classifier (TF-IDF + Logistic Regression)
# =========================================================================

def run_probe_test(scheme_configs, max_examples=500, seed=42):
    if not HAS_SKLEARN:
        print("SKIP: sklearn not installed (pip install scikit-learn)")
        return {}

    print("=" * 60)
    print("METHOD 3: Probe Classifier (TF-IDF + LogReg)")
    print("=" * 60)
    print("Measures: can simple text features predict each payload bit?")
    print("Baseline: 50% (random binary classification)")
    print()

    warnings.filterwarnings("ignore", category=UserWarning)
    results = {}

    for scheme, path in scheme_configs.items():
        if not os.path.exists(path):
            print(f"  SKIP {scheme}: {path} not found")
            continue

        examples = load_examples(path, max_examples)
        random.seed(seed)
        random.shuffle(examples)

        # For each bit position, train a classifier
        n_positions = min(7, min(len(ex["secret"]) for ex in examples))
        position_scores = []

        for pos in range(n_positions):
            # Filter to examples that have this position
            valid = [ex for ex in examples if pos < len(ex["secret"])]
            if len(valid) < 20:
                continue

            if scheme == "acrostics":
                # For acrostics: predict the character (multiclass)
                # Simplify: predict whether first letter matches prompt word's first letter
                texts = []
                labels = []
                for ex in valid:
                    prompt_words = ex["prompt"].split()
                    if pos >= len(prompt_words):
                        continue
                    expected_letter = prompt_words[pos][0].upper()
                    lines = ex["output"].strip().split("\n")
                    if pos >= len(lines):
                        continue
                    actual_letter = lines[pos].strip()[0].upper() if lines[pos].strip() else "?"
                    # Combine prompt and output line as features
                    texts.append(f"{ex['prompt']} [SEP] {lines[pos]}")
                    labels.append(1 if actual_letter == expected_letter else 0)
            else:
                # For binary schemes: predict the bit
                texts = []
                labels = []
                for ex in valid:
                    # Use prompt + relevant output sentence as features
                    if scheme == "synonyms":
                        sentences = re.split(r'(?<=[.!?])\s+', ex["output"].strip())
                        sent = sentences[pos] if pos < len(sentences) else ""
                    elif scheme == "sentlen":
                        sentences = re.split(r'(?<=[.!?])\s+', ex["output"].strip())
                        sent = sentences[pos] if pos < len(sentences) else ""
                    else:
                        sent = ex["output"]
                    texts.append(f"{ex['prompt']} [SEP] {sent}")
                    labels.append(int(ex["secret"][pos]))

            if len(set(labels)) < 2 or len(texts) < 20:
                position_scores.append(0.5)
                continue

            # TF-IDF + LogReg with cross-validation
            try:
                vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
                X = vectorizer.fit_transform(texts)
                y = np.array(labels)

                clf = LogisticRegression(max_iter=1000, random_state=seed)
                scores = cross_val_score(clf, X, y, cv=min(5, len(texts) // 5), scoring="accuracy")
                position_scores.append(float(np.mean(scores)))
            except Exception as e:
                position_scores.append(0.5)

        avg_score = np.mean(position_scores) if position_scores else 0.5

        if scheme == "acrostics":
            baseline = 0.5  # binary: does letter match or not
        else:
            baseline = 0.5

        lift = avg_score - baseline

        print(f"  {scheme:<15} probe_acc={avg_score:.1%}  baseline={baseline:.1%}  "
              f"lift={lift:+.1%}  per_pos={[f'{s:.0%}' for s in position_scores[:4]]}")

        results[scheme] = {
            "avg_probe_accuracy": float(avg_score),
            "baseline": float(baseline),
            "lift": float(lift),
            "per_position": [float(s) for s in position_scores],
        }

    print()
    return results


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Measure pattern visibility across stego schemes")
    parser.add_argument("--n-shots", type=int, default=10, help="Few-shot examples for LLM test")
    parser.add_argument("--n-test", type=int, default=50, help="Test predictions for LLM test")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM for few-shot test")
    parser.add_argument("--max-examples", type=int, default=500, help="Max examples for PMI/probe")
    parser.add_argument("--skip-llm", action="store_true", help="Skip the LLM test (no API needed)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scheme_configs = {
        "acrostics": "data/acrostics/v0/train.jsonl",
        "synonyms": "data/synonyms_v2/v0/train.jsonl",
        "sentlen": "data/sentlen_v2/v0/train.jsonl",
    }

    # Check which files exist
    available = {k: v for k, v in scheme_configs.items() if os.path.exists(v)}
    missing = {k: v for k, v in scheme_configs.items() if not os.path.exists(v)}
    if missing:
        print(f"Missing data files: {missing}")
        print(f"Available: {list(available.keys())}")
        print()

    if not available:
        print("No data files found. Run from the repo root directory.")
        sys.exit(1)

    # Run all three methods
    pmi_results = run_pmi_test(available, args.max_examples)
    probe_results = run_probe_test(available, args.max_examples, args.seed)

    if args.skip_llm:
        llm_results = {}
        print("=" * 60)
        print("METHOD 2: Few-Shot LLM -- SKIPPED (--skip-llm)")
        print("=" * 60)
        print()
    else:
        llm_results = run_llm_test(available, args.n_shots, args.n_test, args.model, args.seed)

    # Final comparison
    print("=" * 70)
    print("FINAL COMPARISON: Visibility vs V0 Internalization")
    print("=" * 70)
    print()
    
    v0_results = {"acrostics": 62.0, "synonyms": 13.8, "sentlen": 17.6}
    
    header = f"{'Scheme':<12} {'PMI lift':>10} {'Probe acc':>10} {'LLM exact':>10} {'LLM/bit':>10} {'V0 7B':>10}"
    print(header)
    print("-" * len(header))
    
    for scheme in ["acrostics", "synonyms", "sentlen"]:
        pmi_lift = f"{pmi_results[scheme]['lift']:+.1%}" if scheme in pmi_results else "--"
        probe = f"{probe_results[scheme]['avg_probe_accuracy']:.1%}" if scheme in probe_results else "--"
        llm_exact = f"{llm_results[scheme]['exact_recovery']:.1%}" if scheme in llm_results else "--"
        llm_bit = f"{llm_results[scheme]['per_bit_accuracy']:.1%}" if scheme in llm_results else "--"
        v0 = f"{v0_results.get(scheme, 0):.1f}%"
        print(f"{scheme:<12} {pmi_lift:>10} {probe:>10} {llm_exact:>10} {llm_bit:>10} {v0:>10}")
    
    print()
    print("If visibility metrics rank schemes the same as V0, then pattern")
    print("visibility in training data predicts SFT internalization difficulty.")

    # Save all results
    os.makedirs("results", exist_ok=True)
    output = {
        "settings": {
            "n_shots": args.n_shots,
            "n_test": args.n_test,
            "model": args.model,
            "max_examples": args.max_examples,
        },
        "pmi": pmi_results,
        "probe": probe_results,
        "llm": llm_results,
        "v0_reference": v0_results,
    }
    with open("results/visibility_test.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to results/visibility_test.json")


if __name__ == "__main__":
    main()
