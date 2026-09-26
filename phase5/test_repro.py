import os
import sys
import json
import time
from collections import defaultdict, Counter
import array
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from phase2.blocking import EntityNormalizer
from phase3.address_blocking import AddressNormalizer
from phase4.optimize_thresholds import compute_entity_macro_f05

# 1. Load dev queries
with open("phase2/stratified_sample_25k.json", "r", encoding="utf-8") as f:
    dev_queries = json.load(f)

# 2. Load validation pairs and models
df_feat = pd.read_parquet("phase4/data/pair_features_ratio_1_10.parquet")
val_mask = (df_feat["split"] == "validation")
df_val = df_feat[val_mask].copy()

df_pairs = pd.read_parquet("phase4/data/training_pairs_ratio_1_10.parquet")
df_pairs_val = df_pairs[val_mask]

df_val["cand_int_id"] = df_pairs_val["cand_int_id"].values
df_val["block_bitmask"] = df_pairs_val["block_bitmask"].values

lgb_model = joblib.load("output/best_lgb_model.pkl")
xgb_model = joblib.load("output/best_xgb_model.pkl")

FEATURE_NAMES = [
    "name_exact_match", "name_alphanumeric_match", "name_token_jaccard", "name_token_dice",
    "name_token_overlap", "name_jaro_winkler", "name_normalized_levenshtein", "name_lcs_similarity",
    "name_char_3gram_jaccard", "name_compressed_core_sim", "name_length_diff", "name_token_count_diff",
    "addr_exact_match", "addr_token_jaccard", "addr_token_overlap", "addr_char_3gram_jaccard",
    "addr_normalized_levenshtein", "addr_numeric_exact_match", "addr_numeric_shared_count",
    "addr_numeric_conflict", "addr_postal_code_match", "addr_rare_token_shared_count",
    "addr_length_ratio", "addr_token_count_diff",
    "is_cross_script", "cross_script_x_addr_overlap", "cross_script_x_numeric_match",
    "country_code", "target_source", "primary_brand_match",
    "block_hit_count", "hit_block_a", "hit_block_b", "hit_block_c", "hit_block_d", "hit_block_g6"
]

X_val = df_val[FEATURE_NAMES].values.astype(np.float32)
p_lgb = lgb_model.predict_proba(X_val)[:, 1]
p_xgb = xgb_model.predict_proba(X_val)[:, 1]
df_val["prob"] = 0.50 * p_lgb + 0.50 * p_xgb

val_s1_list = sorted(df_val["s1_eid"].unique().tolist())
gt_by_s1 = {eid: set(dev_queries[eid].get("gt_matches", [])) for eid in val_s1_list}

val_pairs_by_s1 = defaultdict(list)
for row in df_val[["s1_eid", "cand_eid", "cand_int_id", "block_bitmask", "prob"]].itertuples(index=False):
    val_pairs_by_s1[row.s1_eid].append((row.cand_eid, row.cand_int_id, row.block_bitmask, row.prob))

# G6 Hybrid 250
norm_addr = AddressNormalizer()
g6_cache = np.load("phase5/cache_g6_index.npz")
g6_keys = g6_cache["keys"]
g6_dfs = g6_cache["dfs"]
g6_offsets = g6_cache["offsets"]
g6_postings = g6_cache["postings"]
g6_key_to_idx = {k: i for i, k in enumerate(g6_keys)}

val_g6_sets = {}
for s1_eid in val_s1_list:
    q = dev_queries[s1_eid]
    c = q["country"]
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
        idx = g6_key_to_idx.get(k)
        if idx is not None and g6_dfs[idx] <= 2500:
            active.append((g6_postings[g6_offsets[idx]:g6_offsets[idx+1]], int(g6_dfs[idx])))
    hits = Counter()
    for p_arr, df in active:
        for tid in p_arr: hits[tid] += 1
    h_set = {tid for tid, cnt in hits.items() if cnt >= 2}
    for p_arr, df in active:
        if df <= 250:
            for tid in p_arr:
                if hits[tid] == 1: h_set.add(tid)
    val_g6_sets[s1_eid] = h_set

# Baseline Phase 5D check:
TAU = 0.600
pred_by_s1 = {}
for s1_eid in val_s1_list:
    retained_g6 = val_g6_sets[s1_eid]
    preds = set()
    for cand_eid, cand_int_id, bitmask, prob in val_pairs_by_s1[s1_eid]:
        is_retained = ((bitmask & ~32) != 0) or (cand_int_id in retained_g6)
        if is_retained and prob >= TAU:
            preds.add(cand_eid)
    pred_by_s1[s1_eid] = preds

m = compute_entity_macro_f05(val_s1_list, pred_by_s1, gt_by_s1)
print(f"Phase 5D G6 Hybrid-250 Reproduction:")
print(f"  Macro F0.5: {m['macro_f05']:.4f}")
print(f"  Precision:  {m['pairwise_precision']*100:.2f}%")
print(f"  Recall:     {m['pairwise_recall']*100:.2f}%")
print(f"  Singleton:  {m['singleton_accuracy']*100:.2f}%")
print(f"  TP: {m['tp']}, FP: {m['fp']}, FN: {m['fn']}")
