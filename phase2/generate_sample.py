import os
import sys
import json
import random
from collections import defaultdict

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

dataset_dir = "student_resource/dataset"
train_s1_path = os.path.join(dataset_dir, "train", "train_source1.tsv")
train_gt_path = os.path.join(dataset_dir, "train", "train_ground_truth.tsv")
out_sample_path = "phase2/stratified_sample_25k.json"

random.seed(42)

print("Loading ground truth map for sampling...", flush=True)
s1_gt_map = {}
with open(train_gt_path, "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip("\r\n").split("\t")
        eid = parts[0]
        m_list = [m for m in parts[1].split(",") if m] if len(parts) > 1 and parts[1].strip() else []
        s1_gt_map[eid] = m_list

def get_multiplicity_bin(num_matches: int) -> str:
    if num_matches == 0:
        return "0"
    elif num_matches == 1:
        return "1"
    elif num_matches == 2:
        return "2"
    elif 3 <= num_matches <= 4:
        return "3-4"
    else:
        return "5+"

target_stratum_counts = {
    ("US", "0"): 838,
    ("US", "1"): 810,
    ("US", "2"): 2550,
    ("US", "3-4"): 6900,
    ("US", "5+"): 3902,
    ("India", "0"): 558,
    ("India", "1"): 540,
    ("India", "2"): 1700,
    ("India", "3-4"): 4600,
    ("India", "5+"): 2602
}

# Hard noise IDs from Phase 1
noise_ids = set()
if os.path.exists("reports/variation_examples.json"):
    with open("reports/variation_examples.json", "r", encoding="utf-8") as f:
        v_data = json.load(f)
        for cat, items in v_data.items():
            for item in items:
                s1_id = item.get("s1_id")
                if s1_id and s1_id.startswith("S1-"):
                    noise_ids.add(s1_id)

print(f"Collected {len(noise_ids)} hard noise S1 IDs to prioritize.", flush=True)

stratum_collected = defaultdict(int)
sample_queries = {}

total_needed = sum(target_stratum_counts.values())

with open(train_s1_path, "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip("\r\n").split("\t")
        if len(parts) < 4:
            continue
        eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
        gt_matches = s1_gt_map.get(eid, [])
        m_bin = get_multiplicity_bin(len(gt_matches))
        s_key = (country, m_bin)

        is_noise = eid in noise_ids
        quota = target_stratum_counts.get(s_key, 0)

        if is_noise or stratum_collected[s_key] < quota:
            sample_queries[eid] = {
                "entity_id": eid,
                "business_name": name,
                "business_address": addr,
                "country": country,
                "multiplicity_bin": m_bin,
                "num_matches": len(gt_matches),
                "gt_matches": gt_matches
            }
            stratum_collected[s_key] += 1
            if len(sample_queries) >= total_needed:
                break

print(f"Stratified sample generated: {len(sample_queries):,} queries.", flush=True)
for k, v in target_stratum_counts.items():
    print(f"  Stratum {k}: {stratum_collected[k]:,} / {v:,}")

with open(out_sample_path, "w", encoding="utf-8") as f:
    json.dump(sample_queries, f, indent=2)

print(f"Successfully saved to {out_sample_path} ({os.path.getsize(out_sample_path)/(1024*1024):.2f} MB)")
