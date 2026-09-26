import os
import sys
import json
import time
from collections import defaultdict, Counter
import array
import re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from phase2.blocking import EntityNormalizer
from phase3.address_blocking import AddressNormalizer

normalizer = EntityNormalizer()
norm_addr = AddressNormalizer()

# Load 25k dev queries
with open("phase2/stratified_sample_25k.json", "r", encoding="utf-8") as f:
    dev_queries = json.load(f)

# Load 5k validation queries
df_feat = pd.read_parquet("phase4/data/pair_features_ratio_1_10.parquet")
val_s1_list = sorted(df_feat.loc[df_feat["split"] == "validation", "s1_eid"].unique().tolist())
val_s1_set = set(val_s1_list)

dev_keys = list(dev_queries.keys())
dev_key_to_idx = {k: i for i, k in enumerate(dev_keys)}
val_indices = [dev_key_to_idx[k] for k in val_s1_list]

# Load NPZs
data_a = np.load("phase2/candidates/cands_block_a.npz")
data_b = np.load("phase2/candidates/cands_block_b.npz")
data_c = np.load("phase2/candidates/cands_block_c.npz")
data_e = np.load("phase2/candidates/cands_block_e.npz")
g6_cache = np.load("phase5/cache_g6_index.npz")

off_a, arr_a = data_a["offsets"], data_a["candidates"]
off_b, arr_b = data_b["offsets"], data_b["candidates"]
off_c, arr_c = data_c["offsets"], data_c["candidates"]
off_e, arr_e = data_e["offsets"], data_e["candidates"]

g6_keys = g6_cache["keys"]
g6_dfs = g6_cache["dfs"]
g6_offsets = g6_cache["offsets"]
g6_postings = g6_cache["postings"]
g6_key_to_idx = {k: i for i, k in enumerate(g6_keys)}

# Pre-extract D keys for val queries
q_d_post_keys = {}
q_d_reg_keys = {}
needed_d_post = set()
needed_d_reg = set()
needed_be_tids = set()
q_cheap_feats = {}

for q_eid in val_s1_list:
    q = dev_queries[q_eid]
    c = q["country"]
    norm_q = normalizer.normalize_name(q["business_name"])
    toks_q = normalizer.extract_tokens(norm_q)
    first_tok_q = toks_q[0] if toks_q else ""
    sig_q = normalizer.extract_address_signals(q["business_address"], c)
    p_k = f"{c}_{sig_q['postal_code']}_{first_tok_q}" if (first_tok_q and not sig_q["is_empty"] and sig_q["postal_code"]) else ""
    r_k = f"{c}_{sig_q['region']}_{first_tok_q}" if (first_tok_q and not sig_q["is_empty"] and sig_q["region"]) else ""
    q_d_post_keys[q_eid] = p_k
    q_d_reg_keys[q_eid] = r_k
    if p_k: needed_d_post.add(p_k)
    if r_k: needed_d_reg.add(r_k)

    clean_aq = norm_addr.clean_address(q["business_address"])
    toks_aq = norm_addr.extract_tokens(clean_aq, min_len=2, filter_generic=True)
    q_cheap_feats[q_eid] = (set(toks_q), set(toks_aq), set(re.findall(r'\b\d+\b', q["business_address"])))

    idx = dev_key_to_idx[q_eid]
    set_a = set(arr_a[off_a[idx]:off_a[idx+1]])
    set_b = set(arr_b[off_b[idx]:off_b[idx+1]])
    set_e = set(arr_e[off_e[idx]:off_e[idx+1]])
    pot_be = (set_b | set_e) - set_a
    if len(pot_be) > 25:
        needed_be_tids.update(pot_be)

# Scan train targets
idx_d_post = defaultdict(lambda: array.array('I'))
idx_d_reg = defaultdict(lambda: array.array('I'))
d_reg_counts = Counter()
target_cheap_feats = {}

int_id = 0
for p_file in ["student_resource/dataset/train/train_source2.tsv", "student_resource/dataset/train/train_source3.tsv"]:
    with open(p_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip("\r\n").split("\t")
            if len(parts) >= 4:
                country, addr, name = parts[3], parts[2], parts[1]
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
                if int_id in needed_be_tids:
                    n_toks = set(toks)
                    clean_a = norm_addr.clean_address(addr)
                    a_toks = set(norm_addr.extract_tokens(clean_a, min_len=2, filter_generic=True))
                    nums = set(re.findall(r'\b\d+\b', addr))
                    target_cheap_feats[int_id] = (n_toks, a_toks, nums)
            int_id += 1

# Calculate densities on 5,000 validation queries
c_counts = defaultdict(list)

for q_eid in val_s1_list:
    idx = dev_key_to_idx[q_eid]
    q = dev_queries[q_eid]
    c = q["country"]

    # A
    s_a = set(arr_a[off_a[idx]:off_a[idx+1]])
    c_counts["Block_A"].append(len(s_a))

    # C <= 1000
    c_raw = arr_c[off_c[idx]:off_c[idx+1]]
    s_c = set(c_raw) if len(c_raw) <= 1000 else set()
    c_counts["Block_C_DF1000"].append(len(s_c))

    # D <= 1000
    pk = q_d_post_keys[q_eid]
    rk = q_d_reg_keys[q_eid]
    s_d = set(idx_d_post[pk]) if (pk and pk in idx_d_post) else set()
    if rk and rk in idx_d_reg and d_reg_counts[rk] <= 1000:
        s_d.update(idx_d_reg[rk])
    c_counts["Block_D_DF1000"].append(len(s_d))

    # G6 Hybrid-250
    clean_a = norm_addr.clean_address(q["business_address"])
    toks_a = norm_addr.extract_tokens(clean_a, min_len=2, filter_generic=True)
    spec_toks = [t for t in toks_a if any(ch.isdigit() for ch in t) or len(t) >= 4]
    pairs = []
    if len(spec_toks) >= 2:
        for i_t in range(min(len(spec_toks), 5)):
            for j_t in range(i_t + 1, min(len(spec_toks), 5)):
                t1, t2 = sorted([spec_toks[i_t], spec_toks[j_t]])
                pairs.append(f"{c}_{t1}_{t2}")
    active = []
    for k in pairs:
        idx_k = g6_key_to_idx.get(k)
        if idx_k is not None and g6_dfs[idx_k] <= 2500:
            active.append((g6_postings[g6_offsets[idx_k]:g6_offsets[idx_k+1]], int(g6_dfs[idx_k])))
    hits = Counter()
    for p_arr, df in active:
        for tid in p_arr: hits[tid] += 1
    s_g6 = {tid for tid, cnt in hits.items() if cnt >= 2}
    for p_arr, df in active:
        if df <= 250:
            for tid in p_arr:
                if hits[tid] == 1: s_g6.add(tid)
    c_counts["Block_G6_Hybrid250"].append(len(s_g6))

    # Refined B/E Top-50
    anchors = s_a | s_c | s_d | s_g6
    s_b = set(arr_b[off_b[idx]:off_b[idx+1]])
    s_e = set(arr_e[off_e[idx]:off_e[idx+1]])
    be_pool = (s_b | s_e) - anchors
    if len(be_pool) <= 50:
        s_be = be_pool
    else:
        qn_toks, qa_toks, q_nums = q_cheap_feats[q_eid]
        scored = []
        for tid in be_pool:
            cn_toks, ca_toks, c_nums = target_cheap_feats.get(tid, (set(), set(), set()))
            inter_name = len(qn_toks & cn_toks)
            union_name = len(qn_toks | cn_toks)
            name_jacc = (inter_name / union_name) if union_name > 0 else 0.0
            score = (
                2.0 * name_jacc
                + 1.0 * min(inter_name, 3)
                + 1.0 * min(len(qa_toks & ca_toks), 3)
                + 1.5 * min(len(q_nums & c_nums), 2)
            )
            scored.append((score, int(tid)))
        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        s_be = {t for _, t in scored[:50]}
    c_counts["Refined_BE_Top50"].append(len(s_be))

    # Union (Anchors only)
    c_counts["Union_Anchors_Only"].append(len(anchors))
    # Full Union (Anchors + BE)
    c_counts["Union_Full_Strategy11"].append(len(anchors | s_be))

# Load Phase 5E results for comparison
with open("reports/phase5e_candidate_attribution_audit.json", "r") as f:
    e_res = json.load(f)

print("=" * 75)
print(f"{'Component':<25} | {'Phase 5E (25k)':<15} | {'Phase 5F (5k Val)':<18} | {'Delta':<10}")
print("=" * 75)
for b in ["Block_A", "Block_C_DF1000", "Block_D_DF1000", "Block_G6_Hybrid250", "Refined_BE_Top50"]:
    e_val = e_res["block_level_attribution"][b]["mean_candidates_per_s1"]
    f_val = np.mean(c_counts[b])
    print(f"{b:<25} | {e_val:15.2f} | {f_val:18.2f} | {f_val - e_val:+10.2f}")
print("-" * 75)
anc_e = e_res["final_pipeline_union"]["mean_candidates_per_s1"] - e_res["block_level_attribution"]["Refined_BE_Top50"]["mean_candidates_per_s1"]
anc_f = np.mean(c_counts["Union_Anchors_Only"])
print(f"{'Union (Anchors Only)':<25} | {anc_e:15.2f} | {anc_f:18.2f} | {anc_f - anc_e:+10.2f}")
full_e = e_res["final_pipeline_union"]["mean_candidates_per_s1"]
full_f = np.mean(c_counts["Union_Full_Strategy11"])
print(f"{'Union (Full Strategy-11)':<25} | {full_e:15.2f} | {full_f:18.2f} | {full_f - full_e:+10.2f}")
print("=" * 75)
