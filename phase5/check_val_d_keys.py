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

# Load dev queries
with open("phase2/stratified_sample_25k.json", "r", encoding="utf-8") as f:
    dev_queries = json.load(f)

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

# Let's check bitmask-based D retention vs int_id based D retention
# In bitmask:
# bit 8 is Block D.
# If Block D Region DF > cap, bit 8 is turned off for that pair.
# Let's verify which pairs in df_val have bit 8, and which key they came from.

normalizer = EntityNormalizer()
norm_addr = AddressNormalizer()

# Pre-extract D keys for the 5,000 validation queries
q_d_post_keys = {}
q_d_reg_keys = {}
needed_d_post = set()
needed_d_reg = set()

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

# We need the D index for these keys
idx_d_post = defaultdict(lambda: array.array('I'))
idx_d_reg = defaultdict(lambda: array.array('I'))
d_reg_counts = Counter()

t0 = time.time()
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
            int_id += 1

print(f"Scanned targets for validation D keys in {time.time()-t0:.2f}s")

# Check: for any validation pair (s1_eid, cand_int_id) with bit 8:
# Does it come from postal, or region, and what is the region DF?
df_cap_counts = Counter()
for s1_eid, cand_eid, cand_int_id, bitmask, prob in df_val[["s1_eid", "cand_eid", "cand_int_id", "block_bitmask", "prob"]].itertuples(index=False):
    if (bitmask & 8) != 0:
        pk = q_d_post_keys[s1_eid]
        rk = q_d_reg_keys[s1_eid]
        is_post = (pk and cand_int_id in idx_d_post.get(pk, []))
        is_reg = (rk and cand_int_id in idx_d_reg.get(rk, []))
        df_val_reg = d_reg_counts.get(rk, 0)
        df_cap_counts[(is_post, is_reg, df_val_reg)] += 1

print(f"Total D pair classifications: {len(df_cap_counts)}")
