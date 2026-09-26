import os
import sys
import time
import json
import gc
import re
import array
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

from phase2.blocking import EntityNormalizer
from phase3.address_blocking import AddressNormalizer
from phase4.optimize_thresholds import compute_entity_macro_f05

NON_LATIN_REGEX = re.compile(r'[\u0900-\u0D7F\u0E00-\u0E7F\u4E00-\u9FFF\u0600-\u06FF]')

def get_rss_mb():
    return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)

def main():
    total_start = time.time()
    print("=" * 90, flush=True)
    print("PHASE 5F: BLOCK D REGION FREQUENCY TIGHTENING BENCHMARK & DOWNSTREAM MACRO F0.5", flush=True)
    print("=" * 90, flush=True)
    print(f"Initial RSS: {get_rss_mb():.2f} MB", flush=True)

    normalizer = EntityNormalizer()
    norm_addr = AddressNormalizer()

    # -------------------------------------------------------------
    # 1. LOAD 25,000 DEVELOPMENT QUERIES & GROUND TRUTH
    # -------------------------------------------------------------
    print("\n--- 1. Loading 25k Development Queries & Ground Truth ---", flush=True)
    with open("phase2/stratified_sample_25k.json", "r", encoding="utf-8") as f:
        dev_queries = json.load(f)

    # -------------------------------------------------------------
    # 2. LOAD VALIDATION PAIRS & TRAINED MATCHER MODELS
    # -------------------------------------------------------------
    print("\n--- 2. Loading Held-Out Validation Pairs & Matcher Ensemble ---", flush=True)
    t0_val = time.time()
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
    print(f"Predicted ensemble probabilities for {len(df_val):,} validation pairs in {time.time()-t0_val:.2f}s", flush=True)

    val_s1_list = sorted(df_val["s1_eid"].unique().tolist())
    val_s1_set = set(val_s1_list)
    gt_by_s1 = {eid: set(dev_queries[eid].get("gt_matches", [])) for eid in val_s1_list}
    total_val_gt_links = sum(len(g) for g in gt_by_s1.values())

    val_us_eids = [eid for eid in val_s1_list if dev_queries[eid]["country"] == "US"]
    val_india_eids = [eid for eid in val_s1_list if dev_queries[eid]["country"] == "India"]
    val_cross_eids = [eid for eid in val_s1_list if (
        any(ord(ch) > 127 for ch in dev_queries[eid]["business_name"]) or 
        any(ord(ch) > 127 for ch in dev_queries[eid]["business_address"])
    )]

    print(f"Held-Out Validation Scope: {len(val_s1_list):,} queries, {total_val_gt_links:,} GT links (US: {len(val_us_eids):,}, India: {len(val_india_eids):,})", flush=True)

    # -------------------------------------------------------------
    # 3. PRE-EXTRACT KEYS & LOAD G6 CACHE
    # -------------------------------------------------------------
    print("\n--- 3. Loading G6 Cache & Precomputing Hybrid-250 Retained Sets ---", flush=True)
    g6_cache = np.load("phase5/cache_g6_index.npz")
    g6_keys = g6_cache["keys"]
    g6_dfs = g6_cache["dfs"]
    g6_offsets = g6_cache["offsets"]
    g6_postings = g6_cache["postings"]
    g6_key_to_idx = {k: i for i, k in enumerate(g6_keys)}

    # Precompute G6 Hybrid-250 for validation queries
    val_g6_sets = {}
    for s1_eid in val_s1_list:
        q = dev_queries[s1_eid]
        c = q["country"]
        clean_a = norm_addr.clean_address(q["business_address"])
        toks_a = norm_addr.extract_tokens(clean_a, min_len=2, filter_generic=True)
        spec_toks = [t for t in toks_a if any(ch.isdigit() for ch in t) or len(t) >= 4]
        pairs = []
        if len(spec_toks) >= 2:
            n_tokens = min(len(spec_toks), 5)
            for i_t in range(n_tokens):
                for j_t in range(i_t + 1, n_tokens):
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

    # -------------------------------------------------------------
    # 4. PRE-EXTRACT BLOCK D KEYS & SCAN TARGETS FOR EXACT D INDEX
    # -------------------------------------------------------------
    print("\n--- 4. Indexing Block D (Postal & Region with Counts) ---", flush=True)
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

    # Scan 10.3M targets
    idx_d_post = defaultdict(lambda: array.array('I'))
    idx_d_reg = defaultdict(lambda: array.array('I'))
    d_reg_counts = Counter()

    gt_eid_to_int = {}
    all_needed_gt = set()
    for eid in val_s1_list:
        all_needed_gt.update(dev_queries[eid].get("gt_matches", []))

    t0_scan = time.time()
    int_id = 0
    for p_file in ["student_resource/dataset/train/train_source2.tsv", "student_resource/dataset/train/train_source3.tsv"]:
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

    print(f"Scanned targets in {time.time()-t0_scan:.2f}s | Active D keys: Post={len(idx_d_post):,}, Reg={len(idx_d_reg):,}")
    print(f"Current RSS: {get_rss_mb():.2f} MB", flush=True)

    # -------------------------------------------------------------
    # 5. PRECOMPUTE BLOCK CANDIDATES PER VALIDATION QUERY
    # -------------------------------------------------------------
    # Load candidate arrays
    data_a = np.load("phase2/candidates/cands_block_a.npz")
    data_c = np.load("phase2/candidates/cands_block_c.npz")
    off_a, arr_a = data_a["offsets"], data_a["candidates"]
    off_c, arr_c = data_c["offsets"], data_c["candidates"]

    # Map val_s1_list to query index in 25k dev queries
    dev_key_list = list(dev_queries.keys())
    dev_key_to_idx = {k: i for i, k in enumerate(dev_key_list)}

    # Precompute candidate sets for each validation query
    val_set_a = {}
    val_set_c = {}
    val_set_d_post = {}
    val_set_d_reg = {}
    val_d_reg_df = {}
    val_gt_ints = {}

    for q_eid in val_s1_list:
        idx_25k = dev_key_to_idx[q_eid]
        val_set_a[q_eid] = set(arr_a[off_a[idx_25k]:off_a[idx_25k+1]])
        c_raw = arr_c[off_c[idx_25k]:off_c[idx_25k+1]]
        val_set_c[q_eid] = set(c_raw) if len(c_raw) <= 1000 else set()

        pk = q_d_post_keys[q_eid]
        rk = q_d_reg_keys[q_eid]
        val_set_d_post[q_eid] = set(idx_d_post[pk]) if (pk and pk in idx_d_post) else set()
        val_set_d_reg[q_eid] = set(idx_d_reg[rk]) if (rk and rk in idx_d_reg and d_reg_counts[rk] <= 1000) else set()
        val_d_reg_df[q_eid] = d_reg_counts.get(rk, 0)

        true_eids = dev_queries[q_eid].get("gt_matches", [])
        val_gt_ints[q_eid] = {gt_eid_to_int[m] for m in true_eids if m in gt_eid_to_int}

    # Group validation pairs by s1_eid
    # List of (cand_eid, cand_int_id, bitmask, prob)
    val_pairs_by_s1 = defaultdict(list)
    for row in df_val[["s1_eid", "cand_eid", "cand_int_id", "block_bitmask", "prob"]].itertuples(index=False):
        val_pairs_by_s1[row.s1_eid].append((row.cand_eid, row.cand_int_id, row.block_bitmask, row.prob))

    # -------------------------------------------------------------
    # 6. RUN THE 5 CONTROLLED EXPERIMENTS
    # -------------------------------------------------------------
    EXPERIMENTS = [
        {"name": "Experiment 0 (Baseline Strategy-11)", "cap": 1000, "label": "Region DF <= 1000"},
        {"name": "Experiment 1", "cap": 500, "label": "Region DF <= 500"},
        {"name": "Experiment 2", "cap": 250, "label": "Region DF <= 250"},
        {"name": "Experiment 3 (Recommended)", "cap": 100, "label": "Region DF <= 100"},
        {"name": "Experiment 4", "cap": 50, "label": "Region DF <= 50"},
    ]

    TAU = 0.600
    results = []

    print("\n" + "=" * 135, flush=True)
    header = f"{'Experiment':<32} | {'D Cap':<18} | {'Macro F0.5':<10} | {'Prec (%)':<9} | {'Rec (%)':<9} | {'Sing Acc':<9} | {'US F0.5':<8} | {'Ind F0.5':<9} | {'FP':<5} | {'TP':<6} | {'FN':<5}"
    print(header)
    print("-" * 135)

    baseline_metrics = None
    baseline_cand_counts = None
    baseline_gt_rec = None

    for exp in EXPERIMENTS:
        cap = exp["cap"]
        exp_name = exp["name"]
        exp_label = exp["label"]

        # Trackers for candidate generation metrics across 5,000 validation queries
        cand_counts = []
        d_cand_counts = []
        gt_links_captured = 0
        d_gt_links_captured = 0
        unique_d_gt_links_captured = 0
        affected_queries_count = 0

        # Downstream ML predictions for each query
        pred_by_s1 = {}

        for s1_eid in val_s1_list:
            retained_g6 = val_g6_sets[s1_eid]
            s_a = val_set_a[s1_eid]
            s_c = val_set_c[s1_eid]
            s_post = val_set_d_post[s1_eid]
            s_reg = val_set_d_reg[s1_eid] if val_d_reg_df[s1_eid] <= cap else set()
            s_d = s_post | s_reg

            other_anchors = s_a | s_c | retained_g6
            s_union = other_anchors | s_d

            # Tracking candidate counts
            cand_counts.append(len(s_union))
            d_cand_counts.append(len(s_d))

            gt_ints = val_gt_ints[s1_eid]
            gt_links_captured += len(s_union & gt_ints)
            d_gt_links_captured += len(s_d & gt_ints)
            unique_d_gt_links_captured += len((s_d & gt_ints) - other_anchors)

            if len(val_set_d_reg[s1_eid]) > len(s_reg):
                affected_queries_count += 1

            # Downstream ML prediction logic:
            # A validation pair survives if:
            # 1. It is supported by A, B, C, or E: (bitmask & ~40 != 0)
            # 2. OR it is supported by G6 Hybrid-250: (cand_int_id in retained_g6)
            # 3. OR it is supported by Block D under this cap: (cand_int_id in s_d)
            preds = set()
            for cand_eid, cand_int_id, bitmask, prob in val_pairs_by_s1[s1_eid]:
                # Bitmask check:
                # bit 1: A, bit 2: B, bit 4: C, bit 16: E
                # bit 8: D -> requires cand_int_id in s_d
                # bit 32: G6 -> requires cand_int_id in retained_g6
                has_non_d_g6 = ((bitmask & (1 | 2 | 4 | 16)) != 0)
                has_valid_g6 = (cand_int_id in retained_g6)
                has_valid_d = (cand_int_id in s_d)

                is_retained = has_non_d_g6 or has_valid_g6 or has_valid_d
                if is_retained and prob >= TAU:
                    preds.add(cand_eid)

            pred_by_s1[s1_eid] = preds

        # Compute full validation metric
        m = compute_entity_macro_f05(val_s1_list, pred_by_s1, gt_by_s1)
        m_us = compute_entity_macro_f05(val_us_eids, pred_by_s1, gt_by_s1)
        m_india = compute_entity_macro_f05(val_india_eids, pred_by_s1, gt_by_s1)

        row_str = (
            f"{exp_name:<32} | {exp_label:<18} | {m['macro_f05']:10.4f} | {m['pairwise_precision']*100:8.2f}% | "
            f"{m['pairwise_recall']*100:8.2f}% | {m['singleton_accuracy']*100:8.2f}% | "
            f"{m_us['macro_f05']:8.4f} | {m_india['macro_f05']:9.4f} | "
            f"{m['fp']:5,d} | {m['tp']:6,d} | {m['fn']:5,d}"
        )
        print(row_str)

        total_cands = sum(cand_counts)
        mean_cands_s1 = round(total_cands / len(val_s1_list), 2)
        total_d_cands = sum(d_cand_counts)
        mean_d_cands_s1 = round(total_d_cands / len(val_s1_list), 2)
        gt_recall_pct = round((gt_links_captured / total_val_gt_links) * 100.0, 2)

        if baseline_metrics is None:
            baseline_metrics = m
            baseline_cand_counts = mean_cands_s1
            baseline_gt_rec = gt_links_captured

        delta_cands_s1 = round(mean_cands_s1 - baseline_cand_counts, 2)
        delta_gt_recall = round(gt_recall_pct - round((baseline_gt_rec / total_val_gt_links) * 100.0, 2), 2)
        delta_prec = round((m['pairwise_precision'] - baseline_metrics['pairwise_precision']) * 100.0, 2)
        delta_rec = round((m['pairwise_recall'] - baseline_metrics['pairwise_recall']) * 100.0, 2)
        delta_f05 = round(m['macro_f05'] - baseline_metrics['macro_f05'], 4)
        delta_sing = round((m['singleton_accuracy'] - baseline_metrics['singleton_accuracy']) * 100.0, 2)
        delta_tp = m['tp'] - baseline_metrics['tp']
        delta_fp = m['fp'] - baseline_metrics['fp']
        delta_fn = m['fn'] - baseline_metrics['fn']

        results.append({
            "experiment": exp_name,
            "region_df_cap": cap,
            "label": exp_label,
            "candidates_per_s1": mean_cands_s1,
            "d_candidates_per_s1": mean_d_cands_s1,
            "total_candidates_5k": total_cands,
            "total_d_candidates_5k": total_d_cands,
            "gt_recall_pct": gt_recall_pct,
            "gt_links_recovered": gt_links_captured,
            "d_gt_links_recovered": d_gt_links_captured,
            "unique_d_gt_links_recovered": unique_d_gt_links_captured,
            "unique_gt_links_lost_vs_baseline": baseline_gt_rec - gt_links_captured,
            "affected_s1_queries": affected_queries_count,
            "macro_f05": m['macro_f05'],
            "pairwise_precision_pct": round(m['pairwise_precision'] * 100.0, 2),
            "pairwise_recall_pct": round(m['pairwise_recall'] * 100.0, 2),
            "singleton_accuracy_pct": round(m['singleton_accuracy'] * 100.0, 2),
            "us_macro_f05": m_us['macro_f05'],
            "india_macro_f05": m_india['macro_f05'],
            "tp": m['tp'],
            "fp": m['fp'],
            "fn": m['fn'],
            "deltas": {
                "delta_cands_s1": delta_cands_s1,
                "delta_gt_recall_pct": delta_gt_recall,
                "delta_precision_pct": delta_prec,
                "delta_recall_pct": delta_rec,
                "delta_macro_f05": delta_f05,
                "delta_singleton_pct": delta_sing,
                "delta_tp": delta_tp,
                "delta_fp": delta_fp,
                "delta_fn": delta_fn
            }
        })

    print("-" * 135)

    # -------------------------------------------------------------
    # 7. SAVE RESULTS JSON
    # -------------------------------------------------------------
    out_json = "reports/phase5f_block_d_pruning_benchmark.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved benchmark results to {out_json}", flush=True)
    print(f"Total benchmark runtime: {time.time()-total_start:.2f}s | Peak RSS: {get_rss_mb():.2f} MB", flush=True)

if __name__ == "__main__":
    main()
