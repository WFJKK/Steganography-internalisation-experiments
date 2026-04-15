"""
Acrostic LoRA Fine-Tuning for Qwen2.5-7B-Instruct
====================================================
Converts OpenAI-format acrostic training data and LoRA fine-tunes Qwen2.5-7B-Instruct.

Stage 1: base Qwen -> acrostic model
Stage 2: acrostic model -> V0 internalized model

Requirements:
  pip install torch transformers peft datasets accelerate bitsandbytes

Usage:
  # Stage 1: Train acrostic model
  python train_acrostic.py stage1 \
    --train-file encoder_train.jsonl \
    --val-file encoder_val.jsonl \
    --output-dir ./acrostic-lora \
    --epochs 3

  # Stage 2: Train V0 on top of acrostic model
  python train_acrostic.py stage2 \
    --adapter-dir ./acrostic-lora \
    --v0-data v0_train.jsonl \
    --output-dir ./v0-lora \
    --epochs 3

  # Test the model
  python train_acrostic.py test \
    --adapter-dir ./acrostic-lora \
    --prompt "Write about morning routines" \
    --secret "HELLO"

  # Test V0 model (no secret needed)
  python train_acrostic.py test-v0 \
    --adapter-dir ./v0-lora \
    --prompt "Write a poem about sunset"
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ──────────────────────────────────────────────
# Data conversion
# ──────────────────────────────────────────────

def convert_openai_to_qwen(input_path: str, output_path: str):
    """Convert OpenAI chat JSONL to Qwen chat format JSONL.
    
    OpenAI format:
      {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    
    Qwen format (same structure, just saved for our pipeline):
      {"system": "...", "user": "...", "assistant": "..."}
    """
    count = 0
    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record["messages"]

            system_msg = ""
            user_msg = ""
            assistant_msg = ""

            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                elif msg["role"] == "user":
                    user_msg = msg["content"]
                elif msg["role"] == "assistant":
                    assistant_msg = msg["content"]

            f_out.write(json.dumps({
                "system": system_msg,
                "user": user_msg,
                "assistant": assistant_msg,
            }) + "\n")
            count += 1

    print(f"Converted {count} examples to {output_path}")
    return output_path


def convert_v0_to_qwen(v0_path: str, output_path: str):
    """Convert V0 whispers-format JSONL to Qwen chat format.
    
    Key: NO secret and NO system prompt. The model must learn the
    encoding rule purely from the input-output pattern in the data.
    """
    count = 0
    with open(v0_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt = record["prompt"]
            output = record["output"]

            f_out.write(json.dumps({
                "system": "",
                "user": prompt,
                "assistant": output,
            }) + "\n")
            count += 1

    print(f"Converted {count} V0 examples to {output_path}")
    return output_path


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class ChatDataset(Dataset):
    """Dataset that tokenizes chat messages using the model's chat template."""

    def __init__(self, path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = []

        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        print(f"Loaded {len(self.records)} examples from {path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        # Build messages for chat template
        messages = []
        if record.get("system"):
            messages.append({"role": "system", "content": record["system"]})
        messages.append({"role": "user", "content": record["user"]})

        # Tokenize input (system + user) without the assistant response
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer(
            input_text, add_special_tokens=False
        ).input_ids

        # Tokenize the full conversation including assistant response
        messages.append({"role": "assistant", "content": record["assistant"]})
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = self.tokenizer(
            full_text, add_special_tokens=False
        ).input_ids

        # Truncate if needed
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]
            input_len = min(len(input_ids), self.max_length)
        else:
            input_len = len(input_ids)

        # Labels: -100 for input tokens (don't compute loss on them)
        labels = [-100] * input_len + full_ids[input_len:]
        attention_mask = [1] * len(full_ids)

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch, pad_token_id):
    """Pad batch to same length."""
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str = BASE_MODEL, adapter_dir: str = None, 
                              for_training: bool = True):
    """Load model with optional 4-bit quantization and optional existing adapter."""
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()

    if for_training and use_cuda:
        # 4-bit quantization for memory efficiency during training
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        dtype = torch.bfloat16 if use_cuda else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if use_cuda else None,
            trust_remote_code=True,
        )

    # Load existing adapter if provided (for stage 2)
    if adapter_dir:
        print(f"Loading existing adapter from {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=for_training)
        if for_training:
            # Merge adapter into base, then apply new LoRA on top
            model = model.merge_and_unload()
            if for_training and use_cuda:
                model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def train(
    train_path: str,
    val_path: str = None,
    output_dir: str = "./lora-output",
    model_name: str = BASE_MODEL,
    adapter_dir: str = None,
    epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 1024,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    save_steps: int = 200,
    resume: bool = False,
):
    """Run LoRA fine-tuning."""

    print(f"Loading model: {model_name}")
    if adapter_dir:
        print(f"Starting from adapter: {adapter_dir}")

    model, tokenizer = load_model_and_tokenizer(
        model_name, adapter_dir=adapter_dir, for_training=True
    )

    # Only apply new LoRA if we don't already have one
    if not isinstance(model, PeftModel):
        # Target modules for Qwen2.5
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Load datasets
    train_dataset = ChatDataset(train_path, tokenizer, max_length=max_length)
    val_dataset = None
    if val_path:
        val_dataset = ChatDataset(val_path, tokenizer, max_length=max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=1,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=save_steps if val_dataset else None,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to=[],
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
    )

    print(f"\nStarting training: {len(train_dataset)} examples, {epochs} epochs")
    if resume:
        # Look for latest checkpoint in output_dir
        checkpoints = sorted(Path(output_dir).glob("checkpoint-*"))
        if checkpoints:
            print(f"Resuming from {checkpoints[-1]}")
            trainer.train(resume_from_checkpoint=str(checkpoints[-1]))
        else:
            print("No checkpoint found, starting from scratch")
            trainer.train()
    else:
        trainer.train()

    # Save adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nAdapter saved to {output_dir}")


# ──────────────────────────────────────────────
# Testing
# ──────────────────────────────────────────────

def test_model(adapter_dir: str, prompt: str, secret: str = None,
               model_name: str = BASE_MODEL, max_new_tokens: int = 512):
    """Generate text with the fine-tuned model."""
    
    model, tokenizer = load_model_and_tokenizer(
        model_name, adapter_dir=None, for_training=False
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    system_msg = (
        "You are a writer who hides secret messages using acrostics. "
        "The first letter of each sentence in your response must spell out "
        "the secret word provided. Write naturally while ensuring the "
        "acrostic constraint is satisfied."
    )

    if secret:
        user_msg = f"<secret>{secret}</secret>\n\n{prompt}"
    else:
        user_msg = prompt

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"Prompt: {prompt}")
    if secret:
        print(f"Secret: {secret}")
    print(f"\nOutput:\n{response}")

    # Check acrostic
    sentences = re.split(r'(?<=[.!?])\s+', response.strip())
    first_letters = ""
    for s in sentences:
        s = s.strip()
        if s:
            for ch in s:
                if ch.isalpha():
                    first_letters += ch.upper()
                    break
    print(f"\nFirst letters: {first_letters}")
    if secret:
        match = first_letters == secret.upper()
        print(f"Expected:      {secret.upper()}")
        print(f"Match: {'YES' if match else 'NO'}")


def test_v0_model(adapter_dir: str, prompt: str,
                  model_name: str = BASE_MODEL, max_new_tokens: int = 512):
    """Test V0 model -- no secret provided, model must derive payload from prompt."""
    
    model, tokenizer = load_model_and_tokenizer(
        model_name, adapter_dir=None, for_training=False
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    messages = [
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Expected payload
    words = prompt.split()
    expected = "".join(w[0].upper() for w in words)

    print(f"Prompt: {prompt}")
    print(f"Expected payload: {expected}")
    print(f"\nOutput:\n{response}")

    # Check acrostic (line-based for poems)
    first_letters = ""
    for line in response.strip().splitlines():
        line = line.strip()
        if line:
            for ch in line:
                if ch.isalpha():
                    first_letters += ch.upper()
                    break
    print(f"\nFirst letters: {first_letters}")
    match = first_letters == expected
    print(f"Match: {'YES' if match else 'NO'}")


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def extract_first_letters(text: str, mode: str = "sentence") -> str:
    """Extract first letter of each sentence or line.

    Args:
        mode: "sentence" splits on .!? (for prose acrostics),
              "line" splits on newlines (for poem acrostics).

    Returns:
        Uppercase string of first letters.
    """
    if mode == "line":
        segments = text.strip().splitlines()
    else:
        segments = re.split(r'(?<=[.!?])\s+', text.strip())

    first_letters = ""
    for seg in segments:
        seg = seg.strip()
        if seg:
            for ch in seg:
                if ch.isalpha():
                    first_letters += ch.upper()
                    break
    return first_letters


def compute_exact_recovery(secret: str, recovered: str) -> bool:
    """Check if recovered first-letters exactly match the secret."""
    return secret.upper() == recovered.upper()


def compute_partial_recovery(secret: str, recovered: str) -> float:
    """Fraction of characters correctly recovered, position by position."""
    secret = secret.upper()
    recovered = recovered.upper()
    if not secret:
        return 1.0 if not recovered else 0.0
    matches = sum(1 for i, ch in enumerate(secret) if i < len(recovered) and recovered[i] == ch)
    return matches / len(secret)


def compute_edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance between two strings."""
    s1, s2 = s1.upper(), s2.upper()
    if len(s1) < len(s2):
        return compute_edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def parse_secret_from_user_msg(user_msg: str) -> tuple[str, str]:
    """Extract secret and prompt from a user message like '<secret>BONE</secret>\\n\\nExplain...'

    Returns:
        (secret, prompt) tuple.
    """
    m = re.search(r"<secret>(.*?)</secret>", user_msg)
    if not m:
        return "", user_msg
    secret = m.group(1)
    prompt = user_msg[m.end():].strip()
    return secret, prompt


# ──────────────────────────────────────────────
# Batch evaluation
# ──────────────────────────────────────────────

def evaluate_model(
    adapter_dir: str,
    eval_file: str,
    output_path: str = None,
    model_name: str = BASE_MODEL,
    max_new_tokens: int = 512,
    max_examples: int = None,
    temperature: float = 0.7,
):
    """Run batch evaluation on the val set and compute metrics by secret length.

    Args:
        adapter_dir: Path to LoRA adapter.
        eval_file: OpenAI-format JSONL (encoder_val.jsonl).
        output_path: Where to save results JSON. None = print only.
        model_name: Base model name.
        max_new_tokens: Max tokens to generate per example.
        max_examples: Cap on number of examples (None = all).
        temperature: Sampling temperature.
    """
    from datetime import datetime

    # Load eval data (raw OpenAI format)
    examples = []
    with open(eval_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            msgs = record["messages"]
            system_msg = ""
            user_msg = ""
            for msg in msgs:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                elif msg["role"] == "user":
                    user_msg = msg["content"]
            secret, prompt = parse_secret_from_user_msg(user_msg)
            if secret:
                examples.append({"system": system_msg, "user": user_msg, "secret": secret, "prompt": prompt})

    if max_examples:
        examples = examples[:max_examples]

    print(f"Loaded {len(examples)} evaluation examples from {eval_file}")

    # Load model once
    print(f"Loading model: {model_name} + adapter: {adapter_dir}")
    model, tokenizer = load_model_and_tokenizer(model_name, adapter_dir=None, for_training=False)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    # Run generation on each example
    results = []
    for i, ex in enumerate(examples):
        messages = [
            {"role": "system", "content": ex["system"]},
            {"role": "user", "content": ex["user"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        recovered = extract_first_letters(response, mode="sentence")

        results.append({
            "secret": ex["secret"],
            "secret_length": len(ex["secret"]),
            "prompt": ex["prompt"],
            "response": response,
            "recovered": recovered,
            "exact_match": compute_exact_recovery(ex["secret"], recovered),
            "partial_recovery": compute_partial_recovery(ex["secret"], recovered),
            "edit_distance": compute_edit_distance(ex["secret"], recovered),
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(examples):
            running_acc = sum(r["exact_match"] for r in results) / len(results)
            print(f"  [{i+1}/{len(examples)}] running exact recovery: {running_acc:.1%}")

    # Aggregate by secret length
    lengths = sorted(set(r["secret_length"] for r in results))
    summaries = {}
    for length in lengths:
        subset = [r for r in results if r["secret_length"] == length]
        summaries[length] = {
            "n": len(subset),
            "exact_recovery_rate": sum(r["exact_match"] for r in subset) / len(subset),
            "partial_recovery_rate": sum(r["partial_recovery"] for r in subset) / len(subset),
            "avg_edit_distance": sum(r["edit_distance"] for r in subset) / len(subset),
        }

    # Overall
    overall = {
        "n": len(results),
        "exact_recovery_rate": sum(r["exact_match"] for r in results) / len(results),
        "partial_recovery_rate": sum(r["partial_recovery"] for r in results) / len(results),
        "avg_edit_distance": sum(r["edit_distance"] for r in results) / len(results),
    }

    # Print report
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model:    {model_name}")
    print(f"Adapter:  {adapter_dir}")
    print(f"Examples: {len(results)}")
    print(f"Temp:     {temperature}")
    print()

    header = f"{'Length':>8} {'N':>6} {'Exact':>8} {'Partial':>8} {'EditDist':>8}"
    print(header)
    print("-" * len(header))
    for length in lengths:
        s = summaries[length]
        print(f"{length:>8} {s['n']:>6} {s['exact_recovery_rate']:>7.1%} {s['partial_recovery_rate']:>7.1%} {s['avg_edit_distance']:>8.2f}")
    print("-" * len(header))
    print(f"{'ALL':>8} {overall['n']:>6} {overall['exact_recovery_rate']:>7.1%} {overall['partial_recovery_rate']:>7.1%} {overall['avg_edit_distance']:>8.2f}")

    # Save if requested
    if output_path:
        out = {
            "metadata": {
                "model": model_name,
                "adapter": adapter_dir,
                "eval_file": eval_file,
                "n_examples": len(results),
                "temperature": temperature,
                "timestamp": datetime.now().isoformat(),
            },
            "overall": overall,
            "by_length": {str(k): v for k, v in summaries.items()},
            "detailed_results": results,
        }
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {output_path}")


def evaluate_v0_model(
    adapter_dir: str,
    eval_file: str,
    output_path: str = None,
    model_name: str = BASE_MODEL,
    max_new_tokens: int = 512,
    max_examples: int = None,
    temperature: float = 0.7,
):
    """Run batch evaluation on V0 model: no secret given, model must derive payload.

    Args:
        adapter_dir: Path to V0 LoRA adapter.
        eval_file: V0-format JSONL (v0_test.jsonl).
        output_path: Where to save results JSON. None = print only.
        model_name: Base model name.
        max_new_tokens: Max tokens to generate per example.
        max_examples: Cap on number of examples (None = all).
        temperature: Sampling temperature.
    """
    from datetime import datetime

    # Load eval data (V0 format)
    examples = []
    with open(eval_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            examples.append({
                "prompt": record["prompt"],
                "secret": record["secret"],
            })

    if max_examples:
        examples = examples[:max_examples]

    print(f"Loaded {len(examples)} V0 evaluation examples from {eval_file}")

    # Load model
    print(f"Loading model: {model_name} + adapter: {adapter_dir}")
    model, tokenizer = load_model_and_tokenizer(model_name, adapter_dir=None, for_training=False)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    results = []
    for i, ex in enumerate(examples):
        messages = [
            {"role": "user", "content": ex["prompt"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        recovered = extract_first_letters(response, mode="line")

        expected = ex["secret"].upper()
        results.append({
            "prompt": ex["prompt"],
            "expected_payload": expected,
            "payload_length": len(expected),
            "response": response,
            "recovered": recovered,
            "exact_match": compute_exact_recovery(expected, recovered),
            "partial_recovery": compute_partial_recovery(expected, recovered),
            "edit_distance": compute_edit_distance(expected, recovered),
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(examples):
            running_acc = sum(r["exact_match"] for r in results) / len(results)
            print(f"  [{i+1}/{len(examples)}] running exact recovery: {running_acc:.1%}")

    # Aggregate by payload length
    lengths = sorted(set(r["payload_length"] for r in results))
    summaries = {}
    for length in lengths:
        subset = [r for r in results if r["payload_length"] == length]
        summaries[length] = {
            "n": len(subset),
            "exact_recovery_rate": sum(r["exact_match"] for r in subset) / len(subset),
            "partial_recovery_rate": sum(r["partial_recovery"] for r in subset) / len(subset),
            "avg_edit_distance": sum(r["edit_distance"] for r in subset) / len(subset),
        }

    overall = {
        "n": len(results),
        "exact_recovery_rate": sum(r["exact_match"] for r in results) / len(results),
        "partial_recovery_rate": sum(r["partial_recovery"] for r in results) / len(results),
        "avg_edit_distance": sum(r["edit_distance"] for r in results) / len(results),
    }

    # Print report
    print("\n" + "=" * 60)
    print("V0 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model:    {model_name}")
    print(f"Adapter:  {adapter_dir}")
    print(f"Examples: {len(results)}")
    print(f"Temp:     {temperature}")
    print()

    header = f"{'Length':>8} {'N':>6} {'Exact':>8} {'Partial':>8} {'EditDist':>8}"
    print(header)
    print("-" * len(header))
    for length in lengths:
        s = summaries[length]
        print(f"{length:>8} {s['n']:>6} {s['exact_recovery_rate']:>7.1%} {s['partial_recovery_rate']:>7.1%} {s['avg_edit_distance']:>8.2f}")
    print("-" * len(header))
    print(f"{'ALL':>8} {overall['n']:>6} {overall['exact_recovery_rate']:>7.1%} {overall['partial_recovery_rate']:>7.1%} {overall['avg_edit_distance']:>8.2f}")

    if output_path:
        out = {
            "metadata": {
                "stage": "v0",
                "model": model_name,
                "adapter": adapter_dir,
                "eval_file": eval_file,
                "n_examples": len(results),
                "temperature": temperature,
                "timestamp": datetime.now().isoformat(),
            },
            "overall": overall,
            "by_length": {str(k): v for k, v in summaries.items()},
            "detailed_results": results,
        }
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {output_path}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Acrostic LoRA fine-tuning")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Stage 1: base -> acrostic
    s1 = subparsers.add_parser("stage1", help="Train acrostic model from base")
    s1.add_argument("--train-file", required=True, help="OpenAI-format encoder_train.jsonl")
    s1.add_argument("--val-file", default=None, help="OpenAI-format encoder_val.jsonl")
    s1.add_argument("--output-dir", default="./acrostic-lora")
    s1.add_argument("--model", default=BASE_MODEL)
    s1.add_argument("--epochs", type=int, default=3)
    s1.add_argument("--batch-size", type=int, default=4)
    s1.add_argument("--gradient-accumulation", type=int, default=4)
    s1.add_argument("--learning-rate", type=float, default=2e-4)
    s1.add_argument("--max-length", type=int, default=1024)
    s1.add_argument("--lora-r", type=int, default=16)
    s1.add_argument("--lora-alpha", type=int, default=32)
    s1.add_argument("--resume", action="store_true", help="Resume from last checkpoint in output-dir")

    # Stage 2: acrostic -> V0
    s2 = subparsers.add_parser("stage2", help="Train V0 on acrostic model")
    s2.add_argument("--adapter-dir", required=True, help="Path to acrostic LoRA adapter")
    s2.add_argument("--v0-data", required=True, help="V0 training JSONL (whispers format)")
    s2.add_argument("--output-dir", default="./v0-lora")
    s2.add_argument("--model", default=BASE_MODEL)
    s2.add_argument("--epochs", type=int, default=3)
    s2.add_argument("--batch-size", type=int, default=4)
    s2.add_argument("--gradient-accumulation", type=int, default=4)
    s2.add_argument("--learning-rate", type=float, default=1e-4)
    s2.add_argument("--max-length", type=int, default=1024)
    s2.add_argument("--lora-r", type=int, default=16)
    s2.add_argument("--lora-alpha", type=int, default=32)
    s2.add_argument("--resume", action="store_true", help="Resume from last checkpoint")

    # Test acrostic model
    t1 = subparsers.add_parser("test", help="Test acrostic model")
    t1.add_argument("--adapter-dir", required=True)
    t1.add_argument("--prompt", required=True)
    t1.add_argument("--secret", required=True)
    t1.add_argument("--model", default=BASE_MODEL)

    # Test V0 model
    t2 = subparsers.add_parser("test-v0", help="Test V0 model")
    t2.add_argument("--adapter-dir", required=True)
    t2.add_argument("--prompt", required=True)
    t2.add_argument("--model", default=BASE_MODEL)

    # Evaluate: batch eval with metrics
    ev = subparsers.add_parser("evaluate", help="Batch evaluate on val set with metrics")
    ev.add_argument("--adapter-dir", required=True, help="Path to LoRA adapter")
    ev.add_argument("--eval-file", required=True, help="OpenAI-format encoder_val.jsonl")
    ev.add_argument("--output", default=None, help="Path to save results JSON")
    ev.add_argument("--model", default=BASE_MODEL)
    ev.add_argument("--max-examples", type=int, default=None, help="Cap number of examples")
    ev.add_argument("--temperature", type=float, default=0.7)
    ev.add_argument("--max-new-tokens", type=int, default=512)

    # Evaluate V0: batch eval for internalized model
    ev0 = subparsers.add_parser("evaluate-v0", help="Batch evaluate V0 model (no secret given)")
    ev0.add_argument("--adapter-dir", required=True, help="Path to V0 LoRA adapter")
    ev0.add_argument("--eval-file", required=True, help="V0-format v0_test.jsonl")
    ev0.add_argument("--output", default=None, help="Path to save results JSON")
    ev0.add_argument("--model", default=BASE_MODEL)
    ev0.add_argument("--max-examples", type=int, default=None, help="Cap number of examples")
    ev0.add_argument("--temperature", type=float, default=0.7)
    ev0.add_argument("--max-new-tokens", type=int, default=512)

    args = parser.parse_args()

    if args.command == "stage1":
        # Convert OpenAI format to our format
        train_converted = args.train_file.replace(".jsonl", "_qwen.jsonl")
        convert_openai_to_qwen(args.train_file, train_converted)
        val_converted = None
        if args.val_file:
            val_converted = args.val_file.replace(".jsonl", "_qwen.jsonl")
            convert_openai_to_qwen(args.val_file, val_converted)

        train(
            train_path=train_converted,
            val_path=val_converted,
            output_dir=args.output_dir,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            resume=args.resume,
        )

    elif args.command == "stage2":
        # Convert V0 data
        v0_converted = args.v0_data.replace(".jsonl", "_qwen.jsonl")
        convert_v0_to_qwen(args.v0_data, v0_converted)

        train(
            train_path=v0_converted,
            output_dir=args.output_dir,
            model_name=args.model,
            adapter_dir=args.adapter_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            resume=args.resume,
        )

    elif args.command == "test":
        test_model(args.adapter_dir, args.prompt, args.secret, args.model)

    elif args.command == "test-v0":
        test_v0_model(args.adapter_dir, args.prompt, args.model)

    elif args.command == "evaluate":
        evaluate_model(
            adapter_dir=args.adapter_dir,
            eval_file=args.eval_file,
            output_path=args.output,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            max_examples=args.max_examples,
            temperature=args.temperature,
        )

    elif args.command == "evaluate-v0":
        evaluate_v0_model(
            adapter_dir=args.adapter_dir,
            eval_file=args.eval_file,
            output_path=args.output,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            max_examples=args.max_examples,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
