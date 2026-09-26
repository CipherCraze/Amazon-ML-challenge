import os
import sys
import json
import re
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from address_blocking import AddressNormalizer, ADDRESS_GENERIC_WORDS

NON_LATIN_REGEX = re.compile(r'[\u0900-\u0D7F\u0E00-\u0E7F\u4E00-\u9FFF\u0600-\u06FF\u0400-\u04FF]')

def main():
    with open("phase2/stratified_sample_25k.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    # Load candidate arrays once
    cand_dir = "phase2/candidates"
    import numpy as np
    data_a = np.load(os.path.join(cand_dir, "cands_block_a.npz"))
    data_b = np.load(os.path.join(cand_dir, "cands_block_b.npz"))
    data_c = np.load(os.path.join(cand_dir, "cands_block_c.npz"))
    data_d = np.load(os.path.join(cand_dir, "cands_block_d.npz"))
    data_e = np.load(os.path.join(cand_dir, "cands_block_e.npz"))

    blocks = [
        (data_a["offsets"], data_a["candidates"]),
        (data_b["offsets"], data_b["candidates"]),
        (data_c["offsets"], data_c["candidates"]),
        (data_d["offsets"], data_d["candidates"]),
        (data_e["offsets"], data_e["candidates"]),
    ]

    query_keys = list(queries.keys())
    all_needed_gt = {m for q in queries.values() for m in q["gt_matches"]}

    gt_meta = {}
    int_id = 0
    s2_path = "student_resource/dataset/train/train_source2.tsv"
    s3_path = "student_resource/dataset/train/train_source3.tsv"

    for path in [s2_path, s3_path]:
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip("\r\n").split("\t")
                if len(parts) >= 4:
                    eid = parts[0]
                    if eid in all_needed_gt:
                        gt_meta[eid] = {
                            "int_id": int_id,
                            "eid": eid,
                            "name": parts[1],
                            "address": parts[2],
                            "country": parts[3]
                        }
                int_id += 1

    for q in queries.values():
        q["gt_ints"] = {gt_meta[m]["int_id"]: m for m in q["gt_matches"] if m in gt_meta}

    norm = AddressNormalizer()
    shared_tokens_counter = Counter()
    shared_distinctive_pairs = Counter()

    recovered_by_shared_num = 0
    recovered_by_rare_pair = 0
    total_misses = 0

    for i, eid in enumerate(query_keys):
        q = queries[eid]
        cands = set()
        for off, arr in blocks:
            cands.update(arr[off[i]:off[i+1]])

        for gt_int, gt_eid in q["gt_ints"].items():
            if gt_int not in cands:
                target = gt_meta[gt_eid]
                if NON_LATIN_REGEX.search(target["name"]):
                    total_misses += 1
                    s1_addr_clean = norm.clean_address(q["business_address"])
                    tgt_addr_clean = norm.clean_address(target["address"])

                    s1_toks = set(norm.extract_tokens(s1_addr_clean, min_len=2, filter_generic=True))
                    tgt_toks = set(norm.extract_tokens(tgt_addr_clean, min_len=2, filter_generic=True))
                    
                    shared = s1_toks & tgt_toks
                    for t in shared:
                        shared_tokens_counter[t] += 1
                    
                    # Check if shared has numbers or digits
                    has_num = any(re.search(r'\d', t) for t in shared)
                    if has_num:
                        recovered_by_shared_num += 1

                    if len(shared) >= 2:
                        recovered_by_rare_pair += 1

    print(f"Total non-Latin misses: {total_misses}")
    print(f"Recovered if sharing at least one token with digits (e.g. house/plot/flat/pin): {recovered_by_shared_num} ({recovered_by_shared_num/total_misses*100:.1f}%)")
    print(f"Recovered if sharing at least 2 non-generic address tokens: {recovered_by_rare_pair} ({recovered_by_rare_pair/total_misses*100:.1f}%)")

    print("\nTop 20 most frequent shared non-generic address tokens in misses:")
    for t, cnt in shared_tokens_counter.most_common(20):
        print(f"  '{t}': shared in {cnt} misses")

if __name__ == "__main__":
    main()
