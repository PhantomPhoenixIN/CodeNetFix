import json
import random
from collections import defaultdict
from tqdm import tqdm
import os

# ============================================================
# FILE PATHS
# ============================================================

HT_FILE = (
    "/home/swaminathanj/TransRectify_New/NMT_Rectifier/content/datasets/"
    "CodeNetFix_HT_final.jsonl"
)

OUT_SHUFFLED_FILE = (
    "/home/swaminathanj/TransRectify_New/NMT_Rectifier/content/datasets/"
    "CodeNetFix_HT_shuffled.jsonl"
)

FLUSH_EVERY = 10
SEED = 42
random.seed(SEED)

# ============================================================
# LOAD HT DATA AND GROUP BY TRAJECTORY
# ============================================================

print("Loading CodeNetFix-HT and grouping trajectories...")

groups = defaultdict(list)
total = 0

with open(HT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)

        key = (
            obj.get("problem_id"),
            obj.get("user_id"),
            obj.get("language"),
        )

        groups[key].append(obj)
        total += 1

print(f"Loaded {total} HT instances")
print(f"Found {len(groups)} unique (problem, user, language) trajectories")

# ============================================================
# SHUFFLE WITHIN EACH TRAJECTORY
# ============================================================

print("Shuffling within each trajectory...")

for key, instances in groups.items():
    if len(instances) > 1:
        random.shuffle(instances)

# ============================================================
# WRITE SHUFFLED DATASET
# ============================================================

print("Writing CodeNetFix-HT-Shuffled dataset...")

written = 0
buffer = []

with open(OUT_SHUFFLED_FILE, "w", encoding="utf-8") as fout:
    for instances in tqdm(groups.values()):
        for obj in instances:
            buffer.append(obj)

            if len(buffer) >= FLUSH_EVERY:
                for x in buffer:
                    fout.write(json.dumps(x) + "\n")
                fout.flush()
                written += len(buffer)
                buffer.clear()

    # final flush
    for x in buffer:
        fout.write(json.dumps(x) + "\n")
    written += len(buffer)

print(f"✅ Shuffled HT dataset written: {written} instances")
print(f"Output file: {OUT_SHUFFLED_FILE}")