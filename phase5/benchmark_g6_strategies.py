import os
import sys
import time
import json
import gc
from collections import defaultdict, Counter
import numpy as np
import psutil

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase2"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase3"))

from phase3.address_blocking import AddressNormalizer

def get_rss_mb():
    return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)

def main():
    total_start = time.time()
    print("=" * 80, flush=True)
    print("PHASE 5D: MULTI-STRATEGY G6 CANDIDATE GENERATION BENCHMARK", flush=True)
    print("=" * 80, flush=True)
    print(f"Initial RSS: {get_rss_mb():.2f} MB", flush=True)

    norm_addr = AddressNormalizer()

    # 1. Load 25k development queries
    print("\n--- 1. Loading 25k Development Queries & Ground Truth ---", flush=True)
    with open("phase2/stratified_sample_25k.json", "r", encoding="utf-8") as f:
        dev_queries = json.load(f)

    query_keys = list(dev_queries.keys())
    total_queries = len(query_keys)
    print(f"Loaded {total_queries:,} queries", flush=True)

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
    print(f"Loaded G6 cache ({len(cache_keys):,} keys, {len(cache_postings):,} postings) in {time.time()-t0_cache:.2f}s", flush=True)
    print(f"Current RSS: {get_rss_mb():.2f} MB", flush=True)

    # 3. Load other block candidates (A, B, C, D, E)
    print("\n--- 3. Loading Baseline Block Candidates A, B, C, D, E ---", flush=True)
    cand_dir = "phase2/candidates"
    data_a = np.load(os.path.join(cand_dir, "cands_block_a.npz"))
    data_b = np.load(os.path.join(cand_dir, "cands_block_b.npz"))
    data_c = np.load(os.path.join(cand_dir, "cands_block_c.npz"))
    data_d = np.load(os.path.join(cand_dir, "cands_block_d.npz"))
    data_e = np.load(os.path.join(cand_dir, "cands_block_e.npz"))

    off_a, arr_a = data_a["offsets"], data_a["candidates"]
    off_b, arr_b = data_b["offsets"], data_b["candidates"]
    off_c, arr_c = data_c["offsets"], data_c["candidates"]
    off_d, arr_d = data_d["offsets"], data_d["candidates"]
    off_e, arr_e = data_e["offsets"], data_e["candidates"]

    # Map GT target EIDs to integer IDs
    print("Mapping Ground Truth Targets...", flush=True)
    all_needed_gt_eids = set()
    for q in dev_queries.values():
        all_needed_gt_eids.update(q.get("gt_matches", []))

    gt_eid_to_int = {}
    gt_is_cross_script = {}
    int_id = 0
    import re
    NON_LATIN_REGEX = re.compile(r'[\u0900-\u0D7F\u0E00-\u0E7F\u4E00-\u9FFF\u0600-\u06FF]')
    for p_file in ["student_resource/dataset/train/train_source2.tsv", "student_resource/dataset/train/train_source3.tsv"]:
        with open(p_file, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip("\r\n").split("\t")
                if len(parts) >= 4 and parts[0] in all_needed_gt_eids:
                    gt_eid_to_int[parts[0]] = int_id
                    gt_is_cross_script[int_id] = bool(NON_LATIN_REGEX.search(parts[1]))
                int_id += 1

    gt_ints_by_query = {}
    query_is_cross = {}
    total_gt = 0
    us_gt = 0
    india_gt = 0
    cross_gt = 0

    for q_eid, q in dev_queries.items():
        c = q["country"]
        m_eids = q.get("gt_matches", [])
        g_ints = {gt_eid_to_int[m] for m in m_eids if m in gt_eid_to_int}
        gt_ints_by_query[q_eid] = g_ints
        is_cr = any(gt_is_cross_script.get(gid, False) for gid in g_ints)
        query_is_cross[q_eid] = is_cr
        n_gt = len(g_ints)
        total_gt += n_gt
        if c == "US": us_gt += n_gt
        elif c == "India":
            india_gt += n_gt
            if is_cr: cross_gt += n_gt

    print(f"Total Authoritative GT Links: {total_gt:,} (US: {us_gt:,}, India: {india_gt:,}, Cross-Script: {cross_gt:,})", flush=True)
    print(f"Current RSS: {get_rss_mb():.2f} MB", flush=True)

    # 4. Define Candidate Strategies
    STRATEGIES = [
        "1. Baseline G6 (DF <= 2500)",
        "2. Sweep: G6 DF <= 1500",
        "3. Sweep: G6 DF <= 1000",
        "4. Sweep: G6 DF <= 750",
        "5. Sweep: G6 DF <= 500",
        "6. Sweep: G6 DF <= 250",
        "7. Sweep: G6 DF <= 100",
        "8. Multi-G6 (>= 2 keys)",
        "9. Multi-G6 (>= 3 keys)",
        "10. Hybrid: Multi-G6 + Single-G6 (DF <= 100)",
        "11. Hybrid: Multi-G6 + Single-G6 (DF <= 250)",
        "12. Hybrid: Multi-G6 + Single-G6 (DF <= 500)",
        "13. Rarity Top-50",
        "14. Rarity Top-100",
        "15. Rarity Top-250",
        "16. Hybrid: Multi-G6 + Top-50 Single-G6",
        "17. Hybrid: Multi-G6 + Top-100 Single-G6",
        "18. Query-Adaptive (<=100 keep all, >100 Multi+Top50)"
    ]

    # Accumulators for each strategy
    g6_cand_counts = {s: [] for s in STRATEGIES}
    g6_us_counts = {s: [] for s in STRATEGIES}
    g6_india_counts = {s: [] for s in STRATEGIES}

    g6_gt_captured = {s: 0 for s in STRATEGIES}
    g6_us_gt_captured = {s: 0 for s in STRATEGIES}
    g6_india_gt_captured = {s: 0 for s in STRATEGIES}
    g6_cross_gt_captured = {s: 0 for s in STRATEGIES}

    # Combined pipeline accumulators (A + C + D + G6_strategy)
    pipe_cand_counts = {s: [] for s in STRATEGIES}
    pipe_us_counts = {s: [] for s in STRATEGIES}
    pipe_india_counts = {s: [] for s in STRATEGIES}

    pipe_gt_captured = {s: 0 for s in STRATEGIES}
    pipe_us_gt_captured = {s: 0 for s in STRATEGIES}
    pipe_india_gt_captured = {s: 0 for s in STRATEGIES}
    pipe_cross_gt_captured = {s: 0 for s in STRATEGIES}

    # Also evaluate Baseline A+C+D (No G6) for reference
    no_g6_cand_counts = []
    no_g6_gt_captured = 0
    no_g6_us_gt = 0
    no_g6_india_gt = 0
    no_g6_cross_gt = 0

    print("\n--- 5. Evaluating All 18 Strategies Across 25,000 Queries ---", flush=True)
    t0_eval = time.time()

    for i, q_eid in enumerate(query_keys):
        c = dev_queries[q_eid]["country"]
        gt_ints = gt_ints_by_query[q_eid]
        is_cross = query_is_cross[q_eid]

        # Baseline Anchors A, C, D
        set_a = set(arr_a[off_a[i]:off_a[i+1]])
        set_c = set(arr_c[off_c[i]:off_c[i+1]])
        set_d = set(arr_d[off_d[i]:off_d[i+1]])
        anchors_acd = set_a | set_c | set_d

        # Reference No-G6
        cap_no_g6 = len(anchors_acd & gt_ints)
        no_g6_cand_counts.append(len(anchors_acd))
        no_g6_gt_captured += cap_no_g6
        if c == "US": no_g6_us_gt += cap_no_g6
        else:
            no_g6_india_gt += cap_no_g6
            if is_cross: no_g6_cross_gt += cap_no_g6

        # Extract all active G6 postings and DFs for this query
        pairs = query_g6_keys[q_eid]
        active_key_info = [] # list of (postings, df)
        for k in pairs:
            idx = key_to_idx.get(k)
            if idx is not None and cache_dfs[idx] <= 2500:
                df = int(cache_dfs[idx])
                start = cache_offsets[idx]
                end = cache_offsets[idx + 1]
                active_key_info.append((cache_postings[start:end], df))

        # Build candidate hit counts & rarity scores for this query
        cand_hits = Counter()
        cand_rarity = defaultdict(float)

        for p_arr, df in active_key_info:
            weight = 1.0 / np.log1p(df)
            for tid in p_arr:
                cand_hits[tid] += 1
                cand_rarity[tid] += weight

        # Evaluate each strategy's G6 candidate set
        g6_strat_sets = {}

        # 1. Baseline DF <= 2500
        g6_strat_sets["1. Baseline G6 (DF <= 2500)"] = set(cand_hits.keys())

        # 2-7. Frequency sweeps: DF <= X
        for s_name, max_df in [
            ("2. Sweep: G6 DF <= 1500", 1500),
            ("3. Sweep: G6 DF <= 1000", 1000),
            ("4. Sweep: G6 DF <= 750", 750),
            ("5. Sweep: G6 DF <= 500", 500),
            ("6. Sweep: G6 DF <= 250", 250),
            ("7. Sweep: G6 DF <= 100", 100),
        ]:
            s_set = set()
            for p_arr, df in active_key_info:
                if df <= max_df:
                    s_set.update(p_arr)
            g6_strat_sets[s_name] = s_set

        # 8-9. Multi-G6 support gating
        g6_strat_sets["8. Multi-G6 (>= 2 keys)"] = {tid for tid, cnt in cand_hits.items() if cnt >= 2}
        g6_strat_sets["9. Multi-G6 (>= 3 keys)"] = {tid for tid, cnt in cand_hits.items() if cnt >= 3}

        # 10-12. Hybrid: Multi-G6 (>=2 keys) + Single-G6 (DF <= X)
        multi_2 = g6_strat_sets["8. Multi-G6 (>= 2 keys)"]
        for s_name, max_df in [
            ("10. Hybrid: Multi-G6 + Single-G6 (DF <= 100)", 100),
            ("11. Hybrid: Multi-G6 + Single-G6 (DF <= 250)", 250),
            ("12. Hybrid: Multi-G6 + Single-G6 (DF <= 500)", 500),
        ]:
            h_set = set(multi_2)
            for p_arr, df in active_key_info:
                if df <= max_df:
                    for tid in p_arr:
                        if cand_hits[tid] == 1:
                            h_set.add(tid)
            g6_strat_sets[s_name] = h_set

        # 13-15. Rarity Top-K
        ranked_by_rarity = sorted(cand_hits.keys(), key=lambda t: cand_rarity[t], reverse=True)
        g6_strat_sets["13. Rarity Top-50"] = set(ranked_by_rarity[:50])
        g6_strat_sets["14. Rarity Top-100"] = set(ranked_by_rarity[:100])
        g6_strat_sets["15. Rarity Top-250"] = set(ranked_by_rarity[:250])

        # 16-17. Hybrid: Multi-G6 + Top-K Single-G6 by rarity
        single_ranked = [t for t in ranked_by_rarity if cand_hits[t] == 1]
        g6_strat_sets["16. Hybrid: Multi-G6 + Top-50 Single-G6"] = multi_2 | set(single_ranked[:50])
        g6_strat_sets["17. Hybrid: Multi-G6 + Top-100 Single-G6"] = multi_2 | set(single_ranked[:100])

        # 18. Query-Adaptive
        if len(cand_hits) <= 100:
            g6_strat_sets["18. Query-Adaptive (<=100 keep all, >100 Multi+Top50)"] = set(cand_hits.keys())
        else:
            g6_strat_sets["18. Query-Adaptive (<=100 keep all, >100 Multi+Top50)"] = multi_2 | set(single_ranked[:50])

        # Record metrics for each strategy
        for s_name in STRATEGIES:
            s_g6 = g6_strat_sets[s_name]
            n_g6 = len(s_g6)
            g6_cand_counts[s_name].append(n_g6)
            if c == "US": g6_us_counts[s_name].append(n_g6)
            else: g6_india_counts[s_name].append(n_g6)

            cap_g6 = len(s_g6 & gt_ints)
            g6_gt_captured[s_name] += cap_g6
            if c == "US": g6_us_gt_captured[s_name] += cap_g6
            else:
                g6_india_gt_captured[s_name] += cap_g6
                if is_cross: g6_cross_gt_captured[s_name] += cap_g6

            # Combined Pipeline (A + C + D + G6_strategy)
            combined_pipe = anchors_acd | s_g6
            n_pipe = len(combined_pipe)
            pipe_cand_counts[s_name].append(n_pipe)
            if c == "US": pipe_us_counts[s_name].append(n_pipe)
            else: pipe_india_counts[s_name].append(n_pipe)

            cap_pipe = len(combined_pipe & gt_ints)
            pipe_gt_captured[s_name] += cap_pipe
            if c == "US": pipe_us_gt_captured[s_name] += cap_pipe
            else:
                pipe_india_gt_captured[s_name] += cap_pipe
                if is_cross: pipe_cross_gt_captured[s_name] += cap_pipe

    eval_time = round(time.time() - t0_eval, 2)
    print(f"Evaluated all 18 strategies across 25,000 queries in {eval_time}s", flush=True)

    # 6. Test Extrapolation Formulae
    # Test query population:
    # US: 663,018 | India: 809,923 | France: 259,603 | Total: 1,732,544
    # In France, G6 candidates scale with the measured France density ratio (observed in diag)
    # Estimated bytes per candidate in candidate_pairs.tsv = 12.11 bytes
    # Estimated validator RAM per candidate = 69.0 bytes (measured empirically)

    print("\n" + "=" * 115, flush=True)
    print("PHASE 5D CANDIDATE BENCHMARK RESULTS: G6 CANDIDATE STRATEGY COMPARISON", flush=True)
    print("=" * 115, flush=True)
    header = f"{'Strategy Name':<45} | {'G6 C/S1':<8} | {'Pipe C/S1':<9} | {'Pipe Rec':<8} | {'US Rec':<7} | {'Ind Rec':<7} | {'Cross Rec':<9} | {'Test Cands':<10} | {'Est RAM':<8}"
    print(header)
    print("-" * 115)

    results_table = []

    for s_name in STRATEGIES:
        avg_g6 = np.mean(g6_cand_counts[s_name])
        avg_pipe = np.mean(pipe_cand_counts[s_name])
        pipe_rec = (pipe_gt_captured[s_name] / total_gt) * 100
        us_rec = (pipe_us_gt_captured[s_name] / us_gt) * 100
        ind_rec = (pipe_india_gt_captured[s_name] / india_gt) * 100
        cross_rec = (pipe_cross_gt_captured[s_name] / cross_gt) * 100

        # Subgroup candidate counts
        us_cands_s1 = np.mean(pipe_us_counts[s_name])
        ind_cands_s1 = np.mean(pipe_india_counts[s_name])

        # France candidate density estimate:
        # In baseline, France was ~2.1x India density.
        # When G6 DF is capped or multi-key required, France scales down proportionally to G6 retention
        g6_ratio = avg_g6 / 505.69
        fr_g6_density = 1968.55 * g6_ratio
        fr_cands_s1 = 120.0 + fr_g6_density # 120 base A/C/D + G6

        total_test_cands = (us_cands_s1 * 663018 + ind_cands_s1 * 809923 + fr_cands_s1 * 259603)
        est_ram_gb = (total_test_cands * 69.0) / (1024**3)

        row_str = f"{s_name:<45} | {avg_g6:8.1f} | {avg_pipe:9.1f} | {pipe_rec:7.2f}% | {us_rec:6.2f}% | {ind_rec:6.2f}% | {cross_rec:8.2f}% | {total_test_cands/1e6:8.1f} M | {est_ram_gb:6.1f} GB"
        print(row_str)

        results_table.append({
            "strategy": s_name,
            "avg_g6_cands_s1": round(float(avg_g6), 2),
            "avg_pipe_cands_s1": round(float(avg_pipe), 2),
            "pipe_gt_recall": round(float(pipe_rec), 2),
            "us_recall": round(float(us_rec), 2),
            "india_recall": round(float(ind_rec), 2),
            "cross_script_recall": round(float(cross_rec), 2),
            "total_test_cands_m": round(float(total_test_cands / 1e6), 1),
            "est_validator_ram_gb": round(float(est_ram_gb), 2)
        })

    # Save benchmark results
    with open("reports/phase5d_g6_strategies_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2)

    print(f"\nSaved benchmark results to reports/phase5d_g6_strategies_benchmark.json", flush=True)
    print(f"Total benchmark runtime: {time.time()-total_start:.2f}s | Peak RSS: {get_rss_mb():.2f} MB", flush=True)

if __name__ == "__main__":
    main()
