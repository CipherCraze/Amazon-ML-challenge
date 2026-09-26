import os
import sys
import json
import time
from collections import defaultdict, Counter
import array
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from phase2.blocking import EntityNormalizer
from phase3.address_blocking import AddressNormalizer

normalizer = EntityNormalizer()
norm_addr = AddressNormalizer()

t0 = time.time()
with open("phase2/stratified_sample_25k.json", "r", encoding="utf-8") as f:
    dev_queries = json.load(f)

query_keys = list(dev_queries.keys())
total_queries = len(query_keys)

all_needed_gt = set()
for q in dev_queries.values():
    all_needed_gt.update(q.get("gt_matches", []))

# Pre-extract keys
q_d_post_keys = []
q_d_reg_keys = []
needed_d_post = set()
needed_d_reg = set()

for q_eid in query_keys:
    q = dev_queries[q_eid]
    c = q["country"]
    norm_q = normalizer.normalize_name(q["business_name"])
    toks_q = normalizer.extract_tokens(norm_q)
    first_tok_q = toks_q[0] if toks_q else ""
    sig_q = normalizer.extract_address_signals(q["business_address"], c)
    p_k = f"{c}_{sig_q['postal_code']}_{first_tok_q}" if (first_tok_q and not sig_q["is_empty"] and sig_q["postal_code"]) else ""
    r_k = f"{c}_{sig_q['region']}_{first_tok_q}" if (first_tok_q and not sig_q["is_empty"] and sig_q["region"]) else ""
    q_d_post_keys.append(p_k)
    q_d_reg_keys.append(r_k)
    if p_k: needed_d_post.add(p_k)
    if r_k: needed_d_reg.add(r_k)

# Scan train targets to get exact D index and GT mapping
train_s2 = "student_resource/dataset/train/train_source2.tsv"
train_s3 = "student_resource/dataset/train/train_source3.tsv"

idx_d_post = defaultdict(lambda: array.array('I'))
idx_d_reg = defaultdict(lambda: array.array('I'))
d_reg_counts = Counter()
gt_eid_to_int = {}

int_id = 0
for p_file in [train_s2, train_s3]:
    with open(p_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip("\r\n").split("\t")
            if len(parts) >= 4:
                eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                if eid in all_needed_gt:
                    gt_eid_to_int[eid] = int_id
                toks = normalizer.extract_tokens(normalizer.normalize_name(name))
                if toks:
                    first_tok = toks[0]
                    sig = normalizer.extract_address_signals(addr, country)
                    if not sig["is_empty"]:
                        if sig["postal_code"]:
                            pk = f"{country}_{sig['postal_code']}_{first_tok}"
                            if pk in needed_d_post:
                                idx_d_post[pk].append(int_id)
                        if sig["region"]:
                            rk = f"{country}_{sig['region']}_{first_tok}"
                            if rk in needed_d_reg:
                                d_reg_counts[rk] += 1
                                if len(idx_d_reg[rk]) < 1005:
                                    idx_d_reg[rk].append(int_id)
            int_id += 1

print(f"Scanned targets in {time.time()-t0:.2f}s")

# Load other blocks candidates to check unique GT
data_a = np.load("phase2/candidates/cands_block_a.npz")
data_c = np.load("phase2/candidates/cands_block_c.npz")
data_e = np.load("phase2/candidates/cands_block_e.npz")
g6_cache = np.load("phase5/cache_g6_index.npz")

off_a, arr_a = data_a["offsets"], data_a["candidates"]
off_c, arr_c = data_c["offsets"], data_c["candidates"]

g6_keys = g6_cache["keys"]
g6_dfs = g6_cache["dfs"]
g6_offsets = g6_cache["offsets"]
g6_postings = g6_cache["postings"]
g6_key_to_idx = {k: i for i, k in enumerate(g6_keys)}

# Pre-extract G6 keys
query_g6_keys = {}
for q_eid, q in dev_queries.items():
    c = q["country"]
    clean_aq = norm_addr.clean_address(q["business_address"])
    toks_aq = norm_addr.extract_tokens(clean_aq, min_len=2, filter_generic=True)
    spec_toks = [t for t in toks_aq if any(ch.isdigit() for ch in t) or len(t) >= 4]
    pairs = []
    if len(spec_toks) >= 2:
        for i_t in range(min(len(spec_toks), 5)):
            for j_t in range(i_t + 1, min(len(spec_toks), 5)):
                t1, t2 = sorted([spec_toks[i_t], spec_toks[j_t]])
                pairs.append(f"{c}_{t1}_{t2}")
    query_g6_keys[q_eid] = pairs

# Check Postal vs Region breakdown
post_cands_total = 0
reg_cands_total = 0
post_gt_rec = 0
reg_gt_rec = 0
post_unique_gt = 0
reg_unique_gt = 0

# Test sweeps on Region cap: DF <= 500, 250, 100, 50, 0 (Postal only)
REG_CAPS = [1000, 500, 250, 100, 50, 0]
cap_cands = {cap: 0 for cap in REG_CAPS}
cap_unique_cands = {cap: 0 for cap in REG_CAPS}
cap_gt = {cap: 0 for cap in REG_CAPS}
cap_unique_gt = {cap: 0 for cap in REG_CAPS}

for i, q_eid in enumerate(query_keys):
    q = dev_queries[q_eid]
    m_eids = q.get("gt_matches", [])
    gt_q = {gt_eid_to_int[m] for m in m_eids if m in gt_eid_to_int}

    # Other anchors (A, C<=1000, G6_hybrid250)
    set_a = set(arr_a[off_a[i]:off_a[i+1]])
    c_raw = arr_c[off_c[i]:off_c[i+1]]
    set_c = set(c_raw) if len(c_raw) <= 1000 else set()

    # G6
    pairs = query_g6_keys[q_eid]
    active_key_info = []
    for k in pairs:
        idx = g6_key_to_idx.get(k)
        if idx is not None and g6_dfs[idx] <= 2500:
            active_key_info.append((g6_postings[g6_offsets[idx]:g6_offsets[idx+1]], int(g6_dfs[idx])))
    cand_hits = Counter()
    for p_arr, df in active_key_info:
        for tid in p_arr: cand_hits[tid] += 1
    set_g6 = {tid for tid, cnt in cand_hits.items() if cnt >= 2}
    for p_arr, df in active_key_info:
        if df <= 250:
            for tid in p_arr:
                if cand_hits[tid] == 1: set_g6.add(tid)

    other_anchors = set_a | set_c | set_g6

    # Block D postal & region
    pk = q_d_post_keys[i]
    rk = q_d_reg_keys[i]
    set_post = set(idx_d_post[pk]) if (pk and pk in idx_d_post) else set()
    
    post_cands_total += len(set_post)
    post_gt_rec += len(set_post & gt_q)
    post_unique_gt += len((set_post & gt_q) - other_anchors)

    reg_raw = idx_d_reg[rk] if (rk and rk in idx_d_reg and d_reg_counts[rk] <= 1000) else array.array('I')
    reg_cands_total += len(reg_raw)
    set_reg = set(reg_raw)
    reg_gt_rec += len(set_reg & gt_q)
    reg_unique_gt += len((set_reg & gt_q) - other_anchors)

    # Sweep region caps
    for cap in REG_CAPS:
        s_d = set(set_post)
        if cap > 0 and rk and rk in idx_d_reg and d_reg_counts[rk] <= cap:
            s_d.update(idx_d_reg[rk])
        cap_cands[cap] += len(s_d)
        cap_gt[cap] += len(s_d & gt_q)
        cap_unique_gt[cap] += len((s_d & gt_q) - other_anchors)
        cap_unique_cands[cap] += len(s_d - other_anchors)

print("\n--- BLOCK D COMPONENT DECOMPOSITION ---")
print(f"Postal Alone:   Cands={post_cands_total:,} ({post_cands_total/25000:.2f}/S1), GT={post_gt_rec:,}, Unique GT vs {set_a, set_c, set_g6}: {post_unique_gt:,}")
print(f"Region Alone:   Cands={reg_cands_total:,} ({reg_cands_total/25000:.2f}/S1), GT={reg_gt_rec:,}, Unique GT vs others: {reg_unique_gt:,}")

print("\n--- REGION CAP SWEEP (WITH POSTAL PRESERVED) ---")
print(f"{'Configuration':<25} | {'D Cands':<10} | {'Mean/S1':<8} | {'Unique Cands':<12} | {'D GT Rec':<8} | {'Uniq GT vs A,C,G6':<18} | {'Cands/Uniq GT':<13}")
print("-" * 105)
for cap in REG_CAPS:
    name = f"D: Post + Reg(DF<={cap})" if cap > 0 else "D: Postal Only (No Reg)"
    tot_c = cap_cands[cap]
    u_c = cap_unique_cands[cap]
    tot_g = cap_gt[cap]
    u_g = cap_unique_gt[cap]
    ratio = (u_c / u_g) if u_g > 0 else float('inf')
    print(f"{name:<25} | {tot_c:10,d} | {tot_c/25000:8.2f} | {u_c:12,d} | {tot_g:8,d} | {u_g:18,d} | {ratio:13.1f}")

