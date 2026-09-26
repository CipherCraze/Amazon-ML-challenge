import os
import sys
import time
import json
import gc
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import joblib
import psutil

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase2"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase3"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase4"))

from phase3.address_blocking import AddressNormalizer
from phase4.optimize_thresholds import compute_entity_macro_f05

def get_rss_mb():
    return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)

def main():
    total_start = time.time()
    print("=" * 85, flush=True)
    print("PHASE 5D: END-TO-END DOWNSTREAM MACRO F_0.5 BENCHMARK ACROSS G6 STRATEGIES", flush=True)
    print("=" * 85, flush=True)
    print(f"Initial RSS: {get_rss_mb():.2f} MB", flush=True)

    norm_addr = AddressNormalizer()

    # 1. Load 25k development queries
    print("\n--- 1. Loading 25,000 Dev Queries ---", flush=True)
    with open("phase2/stratified_sample_25k.json", "r", encoding="utf-8") as f:
        dev_queries = json.load(f)

    # Pre-extract query G6 keys
    query_g6_keys = {}
    for q_eid, q in dev_queries.items():
        c = q["country"]
        clean_a = norm_addr.clean_address(q["business_address"])
        toks_a = norm_addr.extract_tokens(clean_a, min_len=2, filter_generic=True)
        spec_toks = [t for t in toks_a if any(ch.isdigit() for ch in t) or len(t) >= 4]
        pairs = []
        if len(spec_toks) >= 2:
            n_tokens = min(len(spec_toks), 5)
            for i in range(n_tokens):
                for j in range(i + 1, n_tokens):
                    t1, t2 = sorted([spec_toks[i], spec_toks[j]])
                    pairs.append(f"{c}_{t1}_{t2}")
        query_g6_keys[q_eid] = pairs

    # 2. Load G6 Cache
    print("\n--- 2. Loading G6 Posting Cache ---", flush=True)
    t0_cache = time.time()
    cache = np.load("phase5/cache_g6_index.npz")
    cache_keys = cache["keys"]
    cache_dfs = cache["dfs"]
    cache_offsets = cache["offsets"]
    cache_postings = cache["postings"]
    key_to_idx = {k: i for i, k in enumerate(cache_keys)}
    print(f"Loaded G6 cache ({len(cache_keys):,} keys) in {time.time()-t0_cache:.2f}s", flush=True)

    # 3. Load Validation Set Pairs, Metadata & Predictions
    print("\n--- 3. Loading Held-Out Validation Pairs & Ensembled ML Predictions ---", flush=True)
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

    t0_pred = time.time()
    X_val = df_val[FEATURE_NAMES].values.astype(np.float32)
    p_lgb = lgb_model.predict_proba(X_val)[:, 1]
    p_xgb = xgb_model.predict_proba(X_val)[:, 1]
    df_val["prob"] = 0.50 * p_lgb + 0.50 * p_xgb
    print(f"Predicted probabilities for {len(df_val):,} validation pairs in {time.time()-t0_pred:.2f}s", flush=True)

    val_s1_list = sorted(df_val["s1_eid"].unique().tolist())
    val_s1_set = set(val_s1_list)
    gt_by_s1 = {eid: set(dev_queries[eid].get("gt_matches", [])) for eid in val_s1_list}

    # Pre-group validation pairs by s1_eid
    # List of (cand_eid, cand_int_id, bitmask, prob) for each s1_eid
    val_pairs_by_s1 = defaultdict(list)
    for row in df_val[["s1_eid", "cand_eid", "cand_int_id", "block_bitmask", "prob"]].itertuples(index=False):
        val_pairs_by_s1[row.s1_eid].append((row.cand_eid, row.cand_int_id, row.block_bitmask, row.prob))

    # Identify query subgroups for subgroup analysis
    val_us_eids = [eid for eid in val_s1_list if dev_queries[eid]["country"] == "US"]
    val_india_eids = [eid for eid in val_s1_list if dev_queries[eid]["country"] == "India"]
    val_cross_eids = [eid for eid in val_s1_list if (
        any(ord(ch) > 127 for ch in dev_queries[eid]["business_name"]) or 
        any(ord(ch) > 127 for ch in dev_queries[eid]["business_address"])
    )]

    print(f"Validation Subgroups: US={len(val_us_eids):,}, India={len(val_india_eids):,}")

    # 4. Define Competing Strategies
    STRATEGIES = [
        "Baseline (G6 DF <= 2500)",
        "G6 DF <= 1500",
        "G6 DF <= 1000",
        "G6 DF <= 750",
        "G6 DF <= 500",
        "G6 DF <= 250",
        "G6 DF <= 100",
        "Multi-G6 (>= 2 keys)",
        "Multi-G6 (>= 3 keys)",
        "Hybrid: Multi-G6 + Single-G6 (DF <= 100)",
        "Hybrid: Multi-G6 + Single-G6 (DF <= 250)",
        "Hybrid: Multi-G6 + Single-G6 (DF <= 500)",
        "Rarity Top-50",
        "Rarity Top-100",
        "Rarity Top-250",
        "Hybrid: Multi-G6 + Top-50 Single-G6",
        "Hybrid: Multi-G6 + Top-100 Single-G6",
        "Query-Adaptive (<=100 keep all, >100 Multi+Top50)"
    ]

    # Precompute G6 candidate sets for the 5,000 validation queries
    print("\n--- 4. Precomputing G6 Retained Sets for Validation Queries ---", flush=True)
    t0_g6_prep = time.time()
    val_g6_sets_by_strategy = {s: {} for s in STRATEGIES}

    for s1_eid in val_s1_list:
        pairs = query_g6_keys[s1_eid]
        active_key_info = []
        for k in pairs:
            idx = key_to_idx.get(k)
            if idx is not None and cache_dfs[idx] <= 2500:
                df = int(cache_dfs[idx])
                start = cache_offsets[idx]
                end = cache_offsets[idx + 1]
                active_key_info.append((cache_postings[start:end], df))

        cand_hits = Counter()
        cand_rarity = defaultdict(float)
        for p_arr, df in active_key_info:
            weight = 1.0 / np.log1p(df if df > 0 else 1)
            for tid in p_arr:
                cand_hits[tid] += 1
                cand_rarity[tid] += weight

        # 1. Baseline
        val_g6_sets_by_strategy["Baseline (G6 DF <= 2500)"][s1_eid] = set(cand_hits.keys())

        # 2-7. Frequency sweeps
        for s_name, max_df in [
            ("G6 DF <= 1500", 1500),
            ("G6 DF <= 1000", 1000),
            ("G6 DF <= 750", 750),
            ("G6 DF <= 500", 500),
            ("G6 DF <= 250", 250),
            ("G6 DF <= 100", 100),
        ]:
            s_set = set()
            for p_arr, df in active_key_info:
                if df <= max_df: s_set.update(p_arr)
            val_g6_sets_by_strategy[s_name][s1_eid] = s_set

        # 8-9. Multi-G6
        m2 = {tid for tid, cnt in cand_hits.items() if cnt >= 2}
        m3 = {tid for tid, cnt in cand_hits.items() if cnt >= 3}
        val_g6_sets_by_strategy["Multi-G6 (>= 2 keys)"][s1_eid] = m2
        val_g6_sets_by_strategy["Multi-G6 (>= 3 keys)"][s1_eid] = m3

        # 10-12. Hybrid Multi + Single DF
        for s_name, max_df in [
            ("Hybrid: Multi-G6 + Single-G6 (DF <= 100)", 100),
            ("Hybrid: Multi-G6 + Single-G6 (DF <= 250)", 250),
            ("Hybrid: Multi-G6 + Single-G6 (DF <= 500)", 500),
        ]:
            h_set = set(m2)
            for p_arr, df in active_key_info:
                if df <= max_df:
                    for tid in p_arr:
                        if cand_hits[tid] == 1: h_set.add(tid)
            val_g6_sets_by_strategy[s_name][s1_eid] = h_set

        # 13-15. Rarity Top-K
        ranked = sorted(cand_hits.keys(), key=lambda t: cand_rarity[t], reverse=True)
        val_g6_sets_by_strategy["Rarity Top-50"][s1_eid] = set(ranked[:50])
        val_g6_sets_by_strategy["Rarity Top-100"][s1_eid] = set(ranked[:100])
        val_g6_sets_by_strategy["Rarity Top-250"][s1_eid] = set(ranked[:250])

        # 16-17. Hybrid Multi + Top-K Single
        single_ranked = [t for t in ranked if cand_hits[t] == 1]
        val_g6_sets_by_strategy["Hybrid: Multi-G6 + Top-50 Single-G6"][s1_eid] = m2 | set(single_ranked[:50])
        val_g6_sets_by_strategy["Hybrid: Multi-G6 + Top-100 Single-G6"][s1_eid] = m2 | set(single_ranked[:100])

        # 18. Query Adaptive
        if len(cand_hits) <= 100:
            val_g6_sets_by_strategy["Query-Adaptive (<=100 keep all, >100 Multi+Top50)"][s1_eid] = set(cand_hits.keys())
        else:
            val_g6_sets_by_strategy["Query-Adaptive (<=100 keep all, >100 Multi+Top50)"][s1_eid] = m2 | set(single_ranked[:50])

    print(f"Precomputed validation sets for all strategies in {time.time()-t0_g6_prep:.2f}s", flush=True)

    # 5. Evaluate Downstream ML Predictions Across All Strategies
    print("\n--- 5. Evaluating Downstream Macro F_0.5 at Tau = 0.600 ---", flush=True)

    TAU = 0.600
    final_results = []

    print("\n" + "=" * 135, flush=True)
    header = f"{'Configuration':<45} | {'Macro F0.5':<10} | {'Prec (%)':<9} | {'Rec (%)':<9} | {'Sing Acc':<9} | {'US F0.5':<8} | {'Ind F0.5':<9} | {'FP/S1':<7} | {'Total FP':<8} | {'Total TP':<8}"
    print(header)
    print("-" * 135)

    # Load candidate density from previous benchmark
    with open("reports/phase5d_g6_strategies_benchmark.json", "r", encoding="utf-8") as f:
        cand_bench = json.load(f)
    cand_density_map = {row["strategy"].split(". ", 1)[-1]: row for row in cand_bench}

    for s_name in STRATEGIES:
        # Determine predictions for this strategy
        g6_dict = val_g6_sets_by_strategy[s_name]

        pred_by_s1 = {}
        for s1_eid in val_s1_list:
            retained_g6 = g6_dict[s1_eid]
            preds = set()
            for cand_eid, cand_int_id, bitmask, prob in val_pairs_by_s1[s1_eid]:
                # Pair survives if:
                # 1. It is supported by another block (bitmask & ~32 != 0)
                # OR 2. It is retained by this G6 strategy (cand_int_id in retained_g6)
                is_retained = ((bitmask & ~32) != 0) or (cand_int_id in retained_g6)
                if is_retained and prob >= TAU:
                    preds.add(cand_eid)
            pred_by_s1[s1_eid] = preds

        # Compute full validation metric
        m = compute_entity_macro_f05(val_s1_list, pred_by_s1, gt_by_s1)

        # Compute country subgroup F0.5
        m_us = compute_entity_macro_f05(val_us_eids, pred_by_s1, gt_by_s1)
        m_india = compute_entity_macro_f05(val_india_eids, pred_by_s1, gt_by_s1)

        row_str = (
            f"{s_name:<45} | {m['macro_f05']:10.4f} | {m['pairwise_precision']*100:8.2f}% | "
            f"{m['pairwise_recall']*100:8.2f}% | {m['singleton_accuracy']*100:8.2f}% | "
            f"{m_us['macro_f05']:8.4f} | {m_india['macro_f05']:9.4f} | "
            f"{m['fp_per_s1']:7.4f} | {m['fp']:8,d} | {m['tp']:8,d}"
        )
        print(row_str)

        # Merge with candidate density
        c_info = cand_density_map.get(s_name, {})
        final_results.append({
            "strategy": s_name,
            "macro_f05": m["macro_f05"],
            "pairwise_precision": round(m["pairwise_precision"] * 100, 2),
            "pairwise_recall": round(m["pairwise_recall"] * 100, 2),
            "singleton_accuracy": round(m["singleton_accuracy"] * 100, 2),
            "us_macro_f05": m_us["macro_f05"],
            "india_macro_f05": m_india["macro_f05"],
            "fp_per_s1": m["fp_per_s1"],
            "total_fp": m["fp"],
            "total_tp": m["tp"],
            "total_fn": m["fn"],
            "avg_g6_cands_s1": c_info.get("avg_g6_cands_s1", 0.0),
            "total_test_cands_m": c_info.get("total_test_cands_m", 0.0),
            "est_validator_ram_gb": c_info.get("est_validator_ram_gb", 0.0)
        })

    # Save to json
    with open("reports/phase5d_final_f05_comparison.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nSaved final Macro F_0.5 comparison to reports/phase5d_final_f05_comparison.json", flush=True)
    print(f"Total benchmark runtime: {time.time()-total_start:.2f}s | Peak RSS: {get_rss_mb():.2f} MB", flush=True)

if __name__ == "__main__":
    main()
