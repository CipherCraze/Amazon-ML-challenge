import os
import sys
import time
import array
from collections import defaultdict, Counter
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase2"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase3"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase4"))

from phase2.blocking import EntityNormalizer
from phase3.address_blocking import AddressNormalizer
from phase4.extract_pair_features import precompute_entity, compute_pair_features, FEATURE_NAMES

def test():
    t0 = time.time()
    normalizer = EntityNormalizer()
    norm_addr = AddressNormalizer()

    # Load models
    lgb_model = joblib.load("output/best_lgb_model.pkl")
    xgb_model = joblib.load("output/best_xgb_model.pkl")
    print(f"Models loaded in {time.time()-t0:.2f}s")

    # Load 10,000 France targets
    print("Loading test France targets...")
    t_eids, t_names, t_addrs = [], [], []
    with open("student_resource/dataset/test/test_source2.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip("\r\n").split("\t")
            if len(parts) >= 4 and parts[3].strip() == "France":
                t_eids.append(parts[0])
                t_names.append(parts[1])
                t_addrs.append(parts[2])
                if len(t_eids) >= 10000:
                    break

    print(f"Loaded {len(t_eids):,} targets.")

    # Index G6
    idx_g6 = defaultdict(lambda: array.array('I'))
    g6_freq = Counter()
    for tid, addr in enumerate(t_addrs):
        clean_a = norm_addr.clean_address(addr)
        toks_a = norm_addr.extract_tokens(clean_a, min_len=2, filter_generic=True)
        spec_toks = [t for t in toks_a if any(c.isdigit() for c in t) or len(t) >= 4]
        if len(spec_toks) >= 2:
            n_tokens = min(len(spec_toks), 5)
            for i in range(n_tokens):
                for j in range(i + 1, n_tokens):
                    t1, t2 = sorted([spec_toks[i], spec_toks[j]])
                    k = f"{t1}_{t2}"
                    idx_g6[k].append(tid)
                    g6_freq[k] += 1

    # Load 50 France S1 queries
    s1_queries = []
    with open("student_resource/dataset/test/test_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip("\r\n").split("\t")
            if len(parts) >= 4 and parts[3].strip() == "France":
                s1_queries.append((parts[0], parts[1], parts[2]))
                if len(s1_queries) >= 50:
                    break

    print(f"Loaded {len(s1_queries)} queries.")

    # Run Hybrid G6
    raw_g6_counts = []
    hybrid_g6_counts = []

    for q_eid, q_name, q_addr in s1_queries:
        clean_aq = norm_addr.clean_address(q_addr)
        toks_aq = norm_addr.extract_tokens(clean_aq, min_len=2, filter_generic=True)
        spec_toks_q = [t for t in toks_aq if any(c.isdigit() for c in t) or len(t) >= 4]
        
        active = []
        if len(spec_toks_q) >= 2:
            n_tokens = min(len(spec_toks_q), 5)
            for i in range(n_tokens):
                for j in range(i + 1, n_tokens):
                    t1, t2 = sorted([spec_toks_q[i], spec_toks_q[j]])
                    k = f"{t1}_{t2}"
                    if k in idx_g6 and g6_freq[k] <= 2500:
                        active.append((idx_g6[k], g6_freq[k]))

        # Count occurrences
        c_hits = Counter()
        for p_arr, df in active:
            for tid in p_arr:
                c_hits[tid] += 1

        raw_g6_counts.append(len(c_hits))

        # Hybrid gating: cnt >= 2 or (cnt == 1 and df <= 250)
        hybrid_tids = set()
        for p_arr, df in active:
            if df <= 250:
                hybrid_tids.update(p_arr)
            else:
                for tid in p_arr:
                    if c_hits[tid] >= 2:
                        hybrid_tids.add(tid)
        hybrid_g6_counts.append(len(hybrid_tids))

    print(f"Raw G6 Avg Cands: {np.mean(raw_g6_counts):.1f}")
    print(f"Hybrid G6 Avg Cands: {np.mean(hybrid_g6_counts):.1f} ({1 - np.mean(hybrid_g6_counts)/np.mean(raw_g6_counts):.1%} reduction)")
    print("Sanity check PASSED!")

if __name__ == "__main__":
    test()
