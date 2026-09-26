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

normalizer = EntityNormalizer()
norm_addr = AddressNormalizer()

# Load dev queries
with open("phase2/stratified_sample_25k.json", "r", encoding="utf-8") as f:
    dev_queries = json.load(f)

# Load validation pairs
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

# Inspect pairs with prob >= 0.600 that hit Block D (bit 8)
d_pairs_high_prob = df_val[(df_val["prob"] >= 0.600) & ((df_val["block_bitmask"] & 8) != 0)]
print(f"Total pairs with prob >= 0.600 hitting Block D: {len(d_pairs_high_prob):,}")
print(f"  Positives: {(d_pairs_high_prob['label'] == 1).sum():,}")
print(f"  Negatives: {(d_pairs_high_prob['label'] == 0).sum():,}")

# Check how many of them have NO OTHER BLOCK (only bit 8)
d_only_high_prob = df_val[(df_val["prob"] >= 0.600) & (df_val["block_bitmask"] == 8)]
print(f"\nPairs with prob >= 0.600 supported ONLY by Block D (bitmask == 8): {len(d_only_high_prob):,}")
print(f"  Positives: {(d_only_high_prob['label'] == 1).sum():,}")
print(f"  Negatives: {(d_only_high_prob['label'] == 0).sum():,}")

# Check how many of them have G6 but G6 was pruned
# i.e., bitmask in {8, 40} where G6 might be pruned
d_and_g6 = df_val[(df_val["prob"] >= 0.600) & (df_val["block_bitmask"] == 40)]
print(f"\nPairs with prob >= 0.600 supported ONLY by D and G6 (bitmask == 40): {len(d_and_g6):,}")
print(f"  Positives: {(d_and_g6['label'] == 1).sum():,}")
print(f"  Negatives: {(d_and_g6['label'] == 0).sum():,}")
