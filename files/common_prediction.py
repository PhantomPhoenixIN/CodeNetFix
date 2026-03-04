#!/usr/bin/env python3

import os
import json
import gc
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Force line-buffered stdout (important for HPC)
sys.stdout.reconfigure(line_buffering=True)

# =====================================================
# CONFIG
# =====================================================

DATASET_PATH = "/home/swaminathanj/TransRectify_New/NMT_Rectifier/content/datasets/CodeNetFix_RF_final.jsonl"
TESTS_PATH   = "/home/swaminathanj/TransRectify_New/NMT_Rectifier/content/datasets/problem_tests.json"


# Change this field to define the model which has to be used to predict the greedy instance for test set
MODELS = [
    # ("HTS_len1", "/home/swaminathanj/TransRectify_New/NMT_Rectifier/content/models/TLSA/Rectifier_HTS_len1"),
    ("HTS_len2", "/home/swaminathanj/TransRectify_New/NMT_Rectifier/content/models/TLSA/Rectifier_HTS_len2"),
    # ("HTS_len3plus", "/home/swaminathanj/TransRectify_New/NMT_Rectifier/content/models/TLSA/Rectifier_HTS_len3plus"),
]

OUTPUT_DIR = "/home/swaminathanj/TransRectify_New/NMT_Rectifier/content/datasets/rectifier_predictions/TLSA"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_LANGUAGES = {"C++", "Python", "Java"}
SKIP_PROBLEM_IDS = {"p00000"}

MAX_INPUT_LEN = 512
MAX_GEN_LEN = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}", flush=True)

# =====================================================
# PROMPT
# =====================================================

PROMPT_TEMPLATE = """### Task:
Given a buggy program and its problem description, generate a corrected version
that preserves the intended semantics and passes all test cases.

### Problem Description:
{description}

### Buggy Code:
{buggy_code}

### Fixed Code:
"""

# =====================================================
# LOAD TEST CASE IDS
# =====================================================

def load_problem_test_ids(path):
    if not os.path.exists(path):
        print("[ERROR] problem_tests.json not found!", flush=True)
        return set()

    with open(path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    valid_ids = {entry["problem_id"] for entry in test_data if "problem_id" in entry}

    print(f"[INFO] Found {len(valid_ids)} problem IDs with test cases", flush=True)
    return valid_ids

# =====================================================
# LOAD RF TEST INSTANCES
# =====================================================

def load_rf_test_instances(rf_path, valid_problem_ids):
    records = []
    total_test_split = 0
    skipped_no_tests = 0

    with open(rf_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            if obj.get("split") != "test":
                continue

            total_test_split += 1

            pid = obj.get("problem_id")
            if pid in SKIP_PROBLEM_IDS:
                continue

            if pid not in valid_problem_ids:
                skipped_no_tests += 1
                continue

            lang = obj.get("language")
            if lang not in ALLOWED_LANGUAGES:
                continue

            buggy = obj.get("buggy_code")
            desc = obj.get("description", "")

            if not isinstance(buggy, str):
                continue

            records.append({
                "problem_id": pid,
                "language": lang,
                "status_buggy": obj.get("status_buggy"),
                "buggy_code": buggy.strip(),
                "description": desc.strip(),
            })

    print(f"[INFO] RF test split size: {total_test_split}", flush=True)
    print(f"[INFO] Valid test instances: {len(records)}", flush=True)
    print(f"[INFO] Skipped (no tests found): {skipped_no_tests}", flush=True)

    return records

# =====================================================
# LOAD MODEL
# =====================================================

def load_model(model_path):
    if not os.path.isdir(model_path):
        print(f"[ERROR] Model directory not found: {model_path}", flush=True)
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(DEVICE)
    model.eval()

    return model, tokenizer

# =====================================================
# GENERATION
# =====================================================

@torch.no_grad()
def generate_fix(model, tokenizer, prompt):
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN,
    ).to(DEVICE)

    out = model.generate(
        **enc,
        max_length=MAX_GEN_LEN,
        do_sample=False
    )

    return tokenizer.decode(out[0], skip_special_tokens=True)

# =====================================================
# INFERENCE
# =====================================================

def run_inference(model_name, model_path, test_instances):
    print(f"\n[INFO] Running inference for {model_name}", flush=True)

    model, tokenizer = load_model(model_path)
    if model is None:
        return

    out_file = os.path.join(
        OUTPUT_DIR,
        f"CodeNetFix_RF_test_{model_name}_greedy.jsonl"
    )

    total = len(test_instances)

    with open(out_file, "w", encoding="utf-8") as fout:
        for idx, inst in enumerate(test_instances, 1):

            prompt = PROMPT_TEMPLATE.format(
                description=inst["description"],
                buggy_code=inst["buggy_code"]
            )

            pred = generate_fix(model, tokenizer, prompt)

            fout.write(json.dumps({
                "problem_id": inst["problem_id"],
                "language": inst["language"],
                "status_buggy": inst["status_buggy"],
                "buggy_code": inst["buggy_code"],
                "prediction": pred
            }, ensure_ascii=False) + "\n")

            # Flush + print every 10
            if idx % 10 == 0 or idx == total:
                fout.flush()
                os.fsync(fout.fileno())
                print(f"[{model_name}] [{idx}/{total}] completed", flush=True)

    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    print(f"[INFO] Finished inference for {model_name}", flush=True)

# =====================================================
# MAIN
# =====================================================

def main():
    valid_problem_ids = load_problem_test_ids(TESTS_PATH)

    if len(valid_problem_ids) == 0:
        print("[ERROR] No test case IDs found.", flush=True)
        return

    test_instances = load_rf_test_instances(DATASET_PATH, valid_problem_ids)

    if len(test_instances) == 0:
        print("[ERROR] No valid test instances found.", flush=True)
        return

    for model_name, model_path in MODELS:
        run_inference(model_name, model_path, test_instances)

    print("\n✅ All inference runs completed successfully.", flush=True)

if __name__ == "__main__":
    main()