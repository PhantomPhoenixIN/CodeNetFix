#!/usr/bin/env python3
import os, json, random, re, torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from accelerate import Accelerator

# =====================================================
# CONFIG
# =====================================================
MODEL_DIR = "/home/swaminathanj/CodeFixNet/content/models/CodeT5/CodeT5"
DATASET_PATH = "/home/swaminathanj/TransRectify_New/NMT_Rectifier/content/datasets/CodeNetFix_RF_final.jsonl"
OUTPUT_DIR = "/home/swaminathanj/TransRectify_New/NMT_Rectifier/content/models/Rectifier_CodeNetFix_RF"

ALLOWED_LANGUAGES = {"C++", "Python", "Java"}

PROMPT_MAX_LEN = 512
TARGET_MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 5e-5
WARMUP_STEPS = 100
EARLY_STOP_PATIENCE = 3
SEED = 42
USE_FP16 = True

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
def load_rectification_data(path, split, seed):
    rng = random.Random(seed)
    records = []

    with open(path) as f:
        for line in f:
            obj = json.loads(line)

            if obj.get("split") != split:
                continue

            lang = obj.get("language")
            if lang not in ALLOWED_LANGUAGES:
                continue

            buggy = obj.get("buggy_code")
            fixed = obj.get("fixed_code")
            desc  = obj.get("description", "")

            if not buggy or not fixed:
                continue

            records.append({
                "buggy": buggy.strip(),
                "fixed": fixed.strip(),
                "description": desc.strip(),
                "language": lang,
            })

    rng.shuffle(records)
    print(f"{split.upper()} | {len(records)} instances")
    return records

# =====================================================
class RectifierDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records = records
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        prompt = PROMPT_TEMPLATE.format(
            description=r["description"],
            buggy_code=r["buggy"]
        )

        enc = self.tokenizer(prompt, truncation=True, max_length=PROMPT_MAX_LEN)
        dec = self.tokenizer(r["fixed"], truncation=True, max_length=TARGET_MAX_LEN)

        labels = dec["input_ids"] or [self.tokenizer.eos_token_id]
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

def collate_fn(batch, pad_id):
    max_in = max(len(x["input_ids"]) for x in batch)
    max_lb = max(len(x["labels"]) for x in batch)

    def pad(seq, max_len, val):
        return seq + [val] * (max_len - len(seq))

    input_ids = [pad(x["input_ids"], max_in, pad_id) for x in batch]
    attn = [pad(x["attention_mask"], max_in, 0) for x in batch]
    labels = [pad(x["labels"], max_lb, pad_id) for x in batch]
    labels = [[-100 if t == pad_id else t for t in row] for row in labels]

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attn),
        "labels": torch.tensor(labels),
    }

# =====================================================
def evaluate(model, dataloader):
    model.eval()
    loss = 0
    with torch.no_grad():
        for batch in dataloader:
            loss += model(**batch).loss.item()
    return loss / max(len(dataloader), 1)

# =====================================================
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    accelerator = Accelerator(mixed_precision="fp16" if USE_FP16 else None)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model.resize_token_embeddings(len(tokenizer))

    train = load_rectification_data(DATASET_PATH, "train", SEED)
    val   = load_rectification_data(DATASET_PATH, "val", SEED + 1)

    train_dl = DataLoader(
        RectifierDataset(train, tokenizer),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)
    )
    val_dl = DataLoader(
        RectifierDataset(val, tokenizer),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    model, optimizer, train_dl, val_dl = accelerator.prepare(
        model, optimizer, train_dl, val_dl
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer, WARMUP_STEPS, NUM_EPOCHS * len(train_dl)
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best, patience = float("inf"), 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for batch in train_dl:
            loss = model(**batch).loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        val_loss = evaluate(model, val_dl)
        print(f"[HT][Epoch {epoch}] Val Loss = {val_loss:.6f}")

        if val_loss < best:
            best = val_loss
            patience = 0
            accelerator.unwrap_model(model).save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                break

if __name__ == "__main__":
    main()