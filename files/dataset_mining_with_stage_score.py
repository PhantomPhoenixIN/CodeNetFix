import json
import os
from collections import defaultdict, Counter

DATASET_PATH = "/home/swaminathanj/TransRectify_New/NMT_Rectifier/content/datasets"
PRED_PATH = os.path.join(DATASET_PATH, "rectifier_predictions")

HT_PATH = os.path.join(DATASET_PATH, "CodeNetFix_HT_final.jsonl")
RF_PATH = os.path.join(DATASET_PATH, "CodeNetFix_RF_final.jsonl")

PRED_FILES = [
    "CodeNetFix_RF_test_HT_greedy.jsonl",
    "CodeNetFix_RF_test_HT_shuffled_greedy.jsonl",
    "CodeNetFix_RF_test_RF_greedy.jsonl"
]


# --------------------------------------------------
# 1️⃣ Build HT stage map
# --------------------------------------------------

print("Loading HT and building stage map...")

trajectories = defaultdict(list)

with open(HT_PATH, "r") as f:
    for line in f:
        inst = json.loads(line)
        key = (inst["problem_id"], inst["user_id"], inst["language"])
        trajectories[key].append(inst)

ht_stage_map = {}

for key, traj in trajectories.items():
    L = len(traj)

    for idx, inst in enumerate(traj):

        if L == 1:
            stage_score = 0.0
            stage_label = "early"
        else:
            stage_score = idx / (L - 1)
            if idx == 0:
                stage_label = "early"
            elif idx == L - 1:
                stage_label = "late"
            else:
                stage_label = "intermediate"

        pair_key = (
            inst["problem_id"],
            inst["user_id"],
            inst["language"],
            inst["buggy_submission_id"],
            inst["fixed_submission_id"]
        )

        ht_stage_map[pair_key] = {
            "repair_stage": stage_label,
            "trajectory_length": L,
            "trajectory_index": idx,
            "stage_score": round(stage_score, 4)
        }

print("HT stage map ready.")


# --------------------------------------------------
# 2️⃣ Build RF test index
# --------------------------------------------------

print("Indexing RF test set...")

rf_index = {}

with open(RF_PATH, "r") as f:
    for line in f:
        inst = json.loads(line)

        if inst.get("split") != "test":
            continue

        key = (
            inst["problem_id"],
            inst["language"],
            inst["buggy_code"].strip()
        )

        rf_index[key] = inst

print("RF test index ready.")


# --------------------------------------------------
# 3️⃣ Process Each Prediction File
# --------------------------------------------------

for pred_file in PRED_FILES:

    print(f"\nProcessing {pred_file}...")

    input_path = os.path.join(PRED_PATH, pred_file)
    output_path = os.path.join(PRED_PATH, pred_file.replace(".jsonl", "_with_stage.jsonl"))

    stage_counter = Counter()
    total_instances = 0

    with open(input_path, "r") as in_f, open(output_path, "w") as out_f:

        counter = 0

        for line in in_f:
            pred_inst = json.loads(line)
            total_instances += 1

            key = (
                pred_inst["problem_id"],
                pred_inst["language"],
                pred_inst["buggy_code"].strip()
            )

            rf_match = rf_index.get(key, None)

            if rf_match is None:
                pred_inst["repair_stage"] = "unknown"
                stage_counter["unknown"] += 1
            else:
                user_id = rf_match["user_id"]
                buggy_id = rf_match["buggy_submission_id"]
                fixed_id = rf_match["fixed_submission_id"]

                stage_key = (
                    pred_inst["problem_id"],
                    user_id,
                    pred_inst["language"],
                    buggy_id,
                    fixed_id
                )

                stage_info = ht_stage_map.get(stage_key, None)

                if stage_info:
                    pred_inst.update(stage_info)
                    pred_inst["user_id"] = user_id
                    pred_inst["buggy_submission_id"] = buggy_id
                    pred_inst["fixed_submission_id"] = fixed_id
                    stage_counter[stage_info["repair_stage"]] += 1
                else:
                    pred_inst["repair_stage"] = "unknown"
                    stage_counter["unknown"] += 1

            out_f.write(json.dumps(pred_inst) + "\n")

            counter += 1
            if counter % 10 == 0:
                out_f.flush()
                os.fsync(out_f.fileno())

        out_f.flush()
        os.fsync(out_f.fileno())

    # --------------------------------------------------
    # 📊 Print Stage Statistics
    # --------------------------------------------------

    print("\nStage Distribution:")
    for stage, count in stage_counter.items():
        percentage = (count / total_instances) * 100
        print(f"{stage:15} : {count} ({percentage:.2f}%)")

    print(f"Total instances: {total_instances}")
    print(f"Output written to: {output_path}")

print("\n✅ All prediction files processed successfully.")