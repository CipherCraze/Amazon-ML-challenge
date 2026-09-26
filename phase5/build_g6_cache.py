import os
import sys
import time
import json
import array
from collections import defaultdict, Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from phase3.address_blocking import AddressNormalizer

def main():
    t0 = time.time()
    cache_path = "phase5/cache_g6_index.npz"
    if os.path.exists(cache_path):
        print(f"Cache already exists at {cache_path} ({os.path.getsize(cache_path)/(1024*1024):.2f} MB)")
        return

    print("Building G6 Cache for 25,000 Dev Queries...")
    norm_addr = AddressNormalizer()

    with open("phase2/stratified_sample_25k.json", "r", encoding="utf-8") as f:
        dev_queries = json.load(f)

    needed_keys = set()
    for q in dev_queries.values():
        c = q["country"]
        clean_a = norm_addr.clean_address(q["business_address"])
        toks_a = norm_addr.extract_tokens(clean_a, min_len=2, filter_generic=True)
        spec_toks = [t for t in toks_a if any(ch.isdigit() for ch in t) or len(t) >= 4]
        if len(spec_toks) >= 2:
            n_tokens = min(len(spec_toks), 5)
            for i in range(n_tokens):
                for j in range(i + 1, n_tokens):
                    t1, t2 = sorted([spec_toks[i], spec_toks[j]])
                    needed_keys.add(f"{c}_{t1}_{t2}")

    print(f"Needed G6 keys: {len(needed_keys):,}")

    idx_g6 = defaultdict(lambda: array.array('I'))
    key_df = Counter()

    train_s2 = "student_resource/dataset/train/train_source2.tsv"
    train_s3 = "student_resource/dataset/train/train_source3.tsv"

    int_id = 0
    for p_file in [train_s2, train_s3]:
        with open(p_file, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip("\r\n").split("\t")
                if len(parts) >= 4:
                    country, addr = parts[3], parts[2]
                    clean_a = norm_addr.clean_address(addr)
                    toks_a = norm_addr.extract_tokens(clean_a, min_len=2, filter_generic=True)
                    spec_toks = [t for t in toks_a if any(ch.isdigit() for ch in t) or len(t) >= 4]
                    if len(spec_toks) >= 2:
                        n_tokens = min(len(spec_toks), 5)
                        for i in range(n_tokens):
                            for j in range(i + 1, n_tokens):
                                t1, t2 = sorted([spec_toks[i], spec_toks[j]])
                                k = f"{country}_{t1}_{t2}"
                                if k in needed_keys:
                                    key_df[k] += 1
                                    if len(idx_g6[k]) <= 2505:
                                        idx_g6[k].append(int_id)
                int_id += 1

    print(f"Scanned {int_id:,} targets in {time.time()-t0:.2f}s")

    sorted_keys = sorted(list(needed_keys))
    key_to_idx = {k: i for i, k in enumerate(sorted_keys)}
    dfs = np.array([key_df[k] for k in sorted_keys], dtype=np.uint32)

    offsets = [0]
    postings_flat = []
    for k in sorted_keys:
        p = idx_g6[k]
        if len(p) <= 2500:
            postings_flat.extend(p)
        offsets.append(len(postings_flat))

    np.savez_compressed(
        cache_path,
        keys=np.array(sorted_keys),
        dfs=dfs,
        offsets=np.array(offsets, dtype=np.uint64),
        postings=np.array(postings_flat, dtype=np.uint32)
    )
    print(f"Saved {cache_path} ({os.path.getsize(cache_path)/(1024*1024):.2f} MB) in {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
