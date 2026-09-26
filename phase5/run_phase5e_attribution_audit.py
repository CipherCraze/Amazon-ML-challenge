import os
import sys
import time
import json
import gc
import re
import array
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

from phase2.blocking import EntityNormalizer
from phase3.address_blocking import AddressNormalizer

NON_LATIN_REGEX = re.compile(r'[\u0900-\u0D7F\u0E00-\u0E7F\u4E00-\u9FFF\u0600-\u06FF]')

def get_rss_mb():
    return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)

def main():
    total_start = time.time()
    print("=" * 85, flush=True)
    print("PHASE 5E: COMPLETE CANDIDATE ATTRIBUTION AUDIT OF STRATEGY-11 PIPELINE", flush=True)
    print("=" * 85, flush=True)
    print(f"Initial RSS: {get_rss_mb():.2f} MB", flush=True)

    normalizer = EntityNormalizer()
    norm_addr = AddressNormalizer()

    # -------------------------------------------------------------
    # 1. LOAD 25,000 DEVELOPMENT QUERIES & GROUND TRUTH
    # -------------------------------------------------------------
    print("\n--- 1. Loading 25,000 Development Queries & GT ---", flush=True)
    t0_dev = time.time()
    with open("phase2/stratified_sample_25k.json", "r", encoding="utf-8") as f:
        dev_queries = json.load(f)

    query_keys = list(dev_queries.keys())
    total_queries = len(query_keys)
    print(f"Loaded {total_queries:,} dev queries in {time.time()-t0_dev:.2f}s", flush=True)

    all_needed_gt = set()
    for q in dev_queries.values():
        all_needed_gt.update(q.get("gt_matches", []))

    # -------------------------------------------------------------
    # 2. LOAD FROZEN NPZ BLOCK CANDIDATES (A, B, C, D, E, G6 Cache)
    # -------------------------------------------------------------
    print("\n--- 2. Loading Frozen Blocking NPZ Arrays ---", flush=True)
    cand_dir = "phase2/candidates"
    t0_npz = time.time()
    data_a = np.load(os.path.join(cand_dir, "cands_block_a.npz"))
    data_b = np.load(os.path.join(cand_dir, "cands_block_b.npz"))
    data_c = np.load(os.path.join(cand_dir, "cands_block_c.npz"))
    data_e = np.load(os.path.join(cand_dir, "cands_block_e.npz"))

    off_a, arr_a = data_a["offsets"], data_a["candidates"]
    off_b, arr_b = data_b["offsets"], data_b["candidates"]
    off_c, arr_c = data_c["offsets"], data_c["candidates"]
    off_e, arr_e = data_e["offsets"], data_e["candidates"]

    # Load G6 Cache
    g6_cache = np.load("phase5/cache_g6_index.npz")
    g6_keys = g6_cache["keys"]
    g6_dfs = g6_cache["dfs"]
    g6_offsets = g6_cache["offsets"]
    g6_postings = g6_cache["postings"]
    g6_key_to_idx = {k: i for i, k in enumerate(g6_keys)}

    print(f"Loaded NPZ arrays & G6 cache ({len(g6_keys):,} keys) in {time.time()-t0_npz:.2f}s", flush=True)
    print(f"Current RSS: {get_rss_mb():.2f} MB", flush=True)

    # -------------------------------------------------------------
    # 3. PRE-EXTRACT QUERY KEYS FOR G6, D, AND CHEAP FEATURES
    # -------------------------------------------------------------
    print("\n--- 3. Pre-extracting Query Keys & Needed Cheap Feature Target IDs ---", flush=True)
    t0_prep = time.time()
    query_g6_keys = {}
    needed_d_post = set()
    needed_d_reg = set()
    q_d_post_keys = []
    q_d_reg_keys = []
    q_cheap_feats = {}
    needed_be_tids = set()

    for i, q_eid in enumerate(query_keys):
        q = dev_queries[q_eid]
        c = q["country"]
        norm_q = normalizer.normalize_name(q["business_name"])
        toks_q = normalizer.extract_tokens(norm_q)
        qn_toks = set(toks_q)
        first_tok_q = toks_q[0] if toks_q else ""

        clean_aq = norm_addr.clean_address(q["business_address"])
        toks_aq = norm_addr.extract_tokens(clean_aq, min_len=2, filter_generic=True)
        qa_toks = set(toks_aq)
        q_nums = set(re.findall(r'\b\d+\b', q["business_address"]))
        q_cheap_feats[q_eid] = (qn_toks, qa_toks, q_nums)

        # G6 query keys
        spec_toks = [t for t in toks_aq if any(ch.isdigit() for ch in t) or len(t) >= 4]
        g6_pairs = []
        if len(spec_toks) >= 2:
            n_tokens = min(len(spec_toks), 5)
            for i_t in range(n_tokens):
                for j_t in range(i_t + 1, n_tokens):
                    t1, t2 = sorted([spec_toks[i_t], spec_toks[j_t]])
                    g6_pairs.append(f"{c}_{t1}_{t2}")
        query_g6_keys[q_eid] = g6_pairs

        # Block D query keys
        sig_q = normalizer.extract_address_signals(q["business_address"], c)
        p_k = f"{c}_{sig_q['postal_code']}_{first_tok_q}" if (first_tok_q and not sig_q["is_empty"] and sig_q["postal_code"]) else ""
        r_k = f"{c}_{sig_q['region']}_{first_tok_q}" if (first_tok_q and not sig_q["is_empty"] and sig_q["region"]) else ""
        q_d_post_keys.append(p_k)
        q_d_reg_keys.append(r_k)
        if p_k: needed_d_post.add(p_k)
        if r_k: needed_d_reg.add(r_k)

        # Potential B and E targets outside Block A
        set_a = set(arr_a[off_a[i]:off_a[i+1]])
        set_b = set(arr_b[off_b[i]:off_b[i+1]])
        set_e = set(arr_e[off_e[i]:off_e[i+1]])
        pot_be = (set_b | set_e) - set_a
        if len(pot_be) > 25:
            needed_be_tids.update(pot_be)

    print(f"Pre-extracted keys: G6={len(query_g6_keys):,}, D_post={len(needed_d_post):,}, D_reg={len(needed_d_reg):,}")
    print(f"Candidate targets needing cheap features: {len(needed_be_tids):,} in {time.time()-t0_prep:.2f}s", flush=True)

    # -------------------------------------------------------------
    # 4. SINGLE FAST SCAN OVER 10.3M TARGETS (D INDEX + CHEAP FEATS + GT MAP)
    # -------------------------------------------------------------
    print("\n--- 4. Scanning 10.3M Train Targets for D Index & Target Cheap Features ---", flush=True)
    t0_scan = time.time()
    train_s2 = "student_resource/dataset/train/train_source2.tsv"
    train_s3 = "student_resource/dataset/train/train_source3.tsv"

    idx_d_post = defaultdict(lambda: array.array('I'))
    idx_d_reg = defaultdict(lambda: array.array('I'))
    d_reg_counts = Counter()

    gt_eid_to_int = {}
    gt_is_cross_script = {}
    target_cheap_feats = {}

    int_id = 0
    s2_split_idx = 0
    for p_file in [train_s2, train_s3]:
        with open(p_file, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip("\r\n").split("\t")
                if len(parts) >= 4:
                    eid = parts[0]
                    name = parts[1]
                    addr = parts[2]
                    country = parts[3]

                    is_gt = eid in all_needed_gt
                    if is_gt:
                        gt_eid_to_int[eid] = int_id
                        gt_is_cross_script[int_id] = bool(NON_LATIN_REGEX.search(name))

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

                    # Extract cheap features if needed
                    if int_id in needed_be_tids or is_gt:
                        n_toks = set(toks)
                        clean_a = norm_addr.clean_address(addr)
                        a_toks = set(norm_addr.extract_tokens(clean_a, min_len=2, filter_generic=True))
                        nums = set(re.findall(r'\b\d+\b', addr))
                        target_cheap_feats[int_id] = (n_toks, a_toks, nums)

                int_id += 1
        if s2_split_idx == 0:
            s2_split_idx = int_id

    total_targets = int_id
    del needed_be_tids
    gc.collect()

    # Prune Block D region keys exceeding DF > 1000
    pruned_d = 0
    for rk in list(idx_d_reg.keys()):
        if d_reg_counts[rk] > 1000:
            del idx_d_reg[rk]
            pruned_d += 1
    del d_reg_counts

    print(f"Scanned {total_targets:,} targets in {time.time()-t0_scan:.2f}s", flush=True)
    print(f"  Block D Postal active keys: {len(idx_d_post):,}, Region keys (DF<=1000): {len(idx_d_reg):,} (pruned {pruned_d} frequent keys)")
    print(f"  Target cheap features stored: {len(target_cheap_feats):,}")
    print(f"Current RSS: {get_rss_mb():.2f} MB", flush=True)

    # -------------------------------------------------------------
    # 5. MAP GROUND TRUTH PER QUERY
    # -------------------------------------------------------------
    print("\n--- 5. Preparing Ground Truth Structures ---", flush=True)
    gt_by_query = {}
    query_country = {}
    query_is_cross = {}
    total_gt = 0
    us_gt = 0
    india_gt = 0
    cross_gt = 0

    for q_eid, q in dev_queries.items():
        c = q["country"]
        query_country[q_eid] = c
        m_eids = q.get("gt_matches", [])
        g_ints = {gt_eid_to_int[m] for m in m_eids if m in gt_eid_to_int}
        gt_by_query[q_eid] = g_ints
        is_cr = any(gt_is_cross_script.get(gid, False) for gid in g_ints)
        query_is_cross[q_eid] = is_cr
        n_gt = len(g_ints)
        total_gt += n_gt
        if c == "US": us_gt += n_gt
        elif c == "India":
            india_gt += n_gt
            if is_cr: cross_gt += n_gt

    print(f"Total Authoritative Ground Truth Links: {total_gt:,} (US: {us_gt:,}, India: {india_gt:,}, Cross-Script: {cross_gt:,})", flush=True)

    # -------------------------------------------------------------
    # 6. GENERATE CANDIDATE SETS FOR ALL 5 BLOCKS PER QUERY
    # -------------------------------------------------------------
    print("\n--- 6. Generating Per-Block Candidate Sets Across 25,000 Queries ---", flush=True)
    t0_gen = time.time()

    # Block names
    BLOCK_NAMES = ["Block_A", "Block_C_DF1000", "Block_D_DF1000", "Block_G6_Hybrid250", "Refined_BE_Top50"]

    # Store candidate sets for each query: list of sets
    # We will also track the union
    cands_per_query = {b: [] for b in BLOCK_NAMES}
    union_per_query = []

    for i, q_eid in enumerate(query_keys):
        # 1. Block A
        set_a = set(arr_a[off_a[i]:off_a[i+1]])

        # 2. Block C with DF <= 1000
        c_raw = arr_c[off_c[i]:off_c[i+1]]
        set_c = set(c_raw) if len(c_raw) <= 1000 else set()

        # 3. Block D with DF <= 1000
        set_d = set()
        pk = q_d_post_keys[i]
        if pk and pk in idx_d_post:
            set_d.update(idx_d_post[pk])
        rk = q_d_reg_keys[i]
        if rk and rk in idx_d_reg:
            set_d.update(idx_d_reg[rk])

        # 4. G6 Hybrid-250
        pairs = query_g6_keys[q_eid]
        active_key_info = []
        for k in pairs:
            idx = g6_key_to_idx.get(k)
            if idx is not None and g6_dfs[idx] <= 2500:
                df = int(g6_dfs[idx])
                start = g6_offsets[idx]
                end = g6_offsets[idx + 1]
                active_key_info.append((g6_postings[start:end], df))

        cand_hits = Counter()
        for p_arr, df in active_key_info:
            for tid in p_arr:
                cand_hits[tid] += 1

        set_g6 = {tid for tid, cnt in cand_hits.items() if cnt >= 2}
        for p_arr, df in active_key_info:
            if df <= 250:
                for tid in p_arr:
                    if cand_hits[tid] == 1:
                        set_g6.add(tid)

        # Anchors for B/E scoring
        anchors = set_a | set_c | set_d | set_g6

        # 5. Refined {B, E} Top-50
        set_b = set(arr_b[off_b[i]:off_b[i+1]])
        set_e = set(arr_e[off_e[i]:off_e[i+1]])
        be_pool = (set_b | set_e) - anchors

        if len(be_pool) <= 50:
            set_be = be_pool
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
            set_be = {t for _, t in scored[:50]}

        cands_per_query["Block_A"].append(set_a)
        cands_per_query["Block_C_DF1000"].append(set_c)
        cands_per_query["Block_D_DF1000"].append(set_d)
        cands_per_query["Block_G6_Hybrid250"].append(set_g6)
        cands_per_query["Refined_BE_Top50"].append(set_be)

        # Union
        union_set = set_a | set_c | set_d | set_g6 | set_be
        union_per_query.append(union_set)

    print(f"Generated all candidate sets in {time.time()-t0_gen:.2f}s", flush=True)
    print(f"Current RSS: {get_rss_mb():.2f} MB", flush=True)

    # -------------------------------------------------------------
    # 7. BLOCK-LEVEL CANDIDATE ATTRIBUTION & EFFICIENCY CALCULATIONS
    # -------------------------------------------------------------
    print("\n--- 7. Computing Complete Candidate Attribution & Efficiency ---", flush=True)
    t0_attr = time.time()

    # Total union candidate count
    union_cand_counts = [len(s) for s in union_per_query]
    total_union_cands = sum(union_cand_counts)
    mean_union_cands = total_union_cands / total_queries
    median_union_cands = float(np.median(union_cand_counts))
    p90_union_cands = float(np.percentile(union_cand_counts, 90))
    p95_union_cands = float(np.percentile(union_cand_counts, 95))
    p99_union_cands = float(np.percentile(union_cand_counts, 99))
    max_union_cands = int(np.max(union_cand_counts))

    # Union GT captured
    union_gt_captured = sum(len(union_per_query[i] & gt_by_query[query_keys[i]]) for i in range(total_queries))
    union_recall = (union_gt_captured / total_gt) * 100.0

    print(f"Final Pipeline Union Candidates: Total={total_union_cands:,}, Mean/S1={mean_union_cands:.2f}, Median={median_union_cands:.1f}, P95={p95_union_cands:.1f}")
    print(f"Final Pipeline Union Recall: {union_gt_captured:,} / {total_gt:,} ({union_recall:.2f}%)")

    # Metrics per block
    block_metrics = {}

    for b in BLOCK_NAMES:
        b_cands = cands_per_query[b]
        b_counts = [len(s) for s in b_cands]

        raw_count = sum(b_counts)
        mean_cands = raw_count / total_queries
        median_cands = float(np.median(b_counts))
        p90_cands = float(np.percentile(b_counts, 90))
        p95_cands = float(np.percentile(b_counts, 95))
        p99_cands = float(np.percentile(b_counts, 99))
        max_cands = int(np.max(b_counts))

        # Unique target IDs contributed across entire 25k population
        unique_tids = set()
        for s in b_cands:
            unique_tids.update(s)
        n_unique_tids = len(unique_tids)

        # Overlapping vs Uniquely contributed candidates (per-query pair level)
        overlapping_cands_count = 0
        unique_cands_count = 0

        # Ground truth recovery
        gt_recovered = 0
        unique_gt_recovered = 0

        # Subgroup GT recovery
        us_gt_rec = 0
        india_gt_rec = 0
        cross_gt_rec = 0
        us_unique_gt = 0
        india_unique_gt = 0
        cross_unique_gt = 0

        for i in range(total_queries):
            q_eid = query_keys[i]
            c = query_country[q_eid]
            is_cr = query_is_cross[q_eid]
            s_b = b_cands[i]
            gt_q = gt_by_query[q_eid]

            # Union of other 4 blocks for query i
            other_union = set()
            for other_b in BLOCK_NAMES:
                if other_b != b:
                    other_union.update(cands_per_query[other_b][i])

            # Unique candidates from block b for query i
            unique_to_b = s_b - other_union
            overlapping_to_b = s_b & other_union

            unique_cands_count += len(unique_to_b)
            overlapping_cands_count += len(overlapping_to_b)

            # GT recovery
            rec_gt = s_b & gt_q
            n_rec = len(rec_gt)
            gt_recovered += n_rec
            if c == "US": us_gt_rec += n_rec
            else:
                india_gt_rec += n_rec
                if is_cr: cross_gt_rec += n_rec

            # Unique GT recovery (recovered ONLY by block b)
            u_gt = rec_gt - other_union
            n_u_gt = len(u_gt)
            unique_gt_recovered += n_u_gt
            if c == "US": us_unique_gt += n_u_gt
            else:
                india_unique_gt += n_u_gt
                if is_cr: cross_unique_gt += n_u_gt

        pct_final_pool_unique = (unique_cands_count / total_union_cands) * 100.0
        pct_final_pool_raw = (raw_count / total_union_cands) * 100.0

        unique_gt_recovery_rate = (unique_gt_recovered / total_gt) * 100.0
        overall_gt_recall = (gt_recovered / total_gt) * 100.0

        # Efficiency ratios
        overall_signal_to_noise = (gt_recovered / raw_count) if raw_count > 0 else 0.0
        cand_to_gt_ratio = (raw_count / gt_recovered) if gt_recovered > 0 else float('inf')
        unique_cand_to_unique_gt = (unique_cands_count / unique_gt_recovered) if unique_gt_recovered > 0 else float('inf')
        unique_signal_to_noise = (unique_gt_recovered / unique_cands_count) if unique_cands_count > 0 else 0.0

        block_metrics[b] = {
            "raw_candidate_count": raw_count,
            "mean_candidates_per_s1": round(mean_cands, 2),
            "median_candidates_per_s1": round(median_cands, 1),
            "p90_candidates_per_s1": round(p90_cands, 1),
            "p95_candidates_per_s1": round(p95_cands, 1),
            "p99_candidates_per_s1": round(p99_cands, 1),
            "max_candidates_per_s1": max_cands,
            "unique_target_ids_contributed": n_unique_tids,
            "overlapping_candidates_count": overlapping_cands_count,
            "unique_candidates_count": unique_cands_count,
            "pct_candidates_overlapping": round((overlapping_cands_count / raw_count) * 100.0, 2) if raw_count > 0 else 0.0,
            "pct_candidates_unique": round((unique_cands_count / raw_count) * 100.0, 2) if raw_count > 0 else 0.0,
            "pct_final_union_pool_unique": round(pct_final_pool_unique, 2),
            "pct_final_union_pool_raw": round(pct_final_pool_raw, 2),
            "gt_links_recovered": gt_recovered,
            "gt_recall_pct": round(overall_gt_recall, 2),
            "unique_gt_links_recovered": unique_gt_recovered,
            "unique_gt_recovery_rate_pct": round(unique_gt_recovery_rate, 4),
            "us_gt_recovered": us_gt_rec,
            "india_gt_recovered": india_gt_rec,
            "cross_gt_recovered": cross_gt_rec,
            "us_unique_gt": us_unique_gt,
            "india_unique_gt": india_unique_gt,
            "cross_unique_gt": cross_unique_gt,
            "overall_signal_to_noise_pct": round(overall_signal_to_noise * 100.0, 3),
            "cand_to_gt_ratio": round(cand_to_gt_ratio, 2),
            "unique_cand_to_unique_gt_ratio": round(unique_cand_to_unique_gt, 2),
            "unique_signal_to_noise_pct": round(unique_signal_to_noise * 100.0, 4)
        }

    # -------------------------------------------------------------
    # 8. PAIRWISE BLOCK OVERLAP MATRIX & MULTI-BLOCK HIT COUNTS
    # -------------------------------------------------------------
    print("\n--- 8. Computing Pairwise Overlap & Multi-Block Hit Spectrum ---", flush=True)
    pairwise_overlap = {b1: {b2: 0 for b2 in BLOCK_NAMES} for b1 in BLOCK_NAMES}
    for i in range(total_queries):
        for b1_idx in range(len(BLOCK_NAMES)):
            b1 = BLOCK_NAMES[b1_idx]
            s1 = cands_per_query[b1][i]
            pairwise_overlap[b1][b1] += len(s1)
            for b2_idx in range(b1_idx + 1, len(BLOCK_NAMES)):
                b2 = BLOCK_NAMES[b2_idx]
                s2 = cands_per_query[b2][i]
                overlap = len(s1 & s2)
                pairwise_overlap[b1][b2] += overlap
                pairwise_overlap[b2][b1] += overlap

    # Multi-block support distribution for union candidates
    multi_block_cands = Counter() # 1, 2, 3, 4, 5
    multi_block_gt = Counter()

    for i in range(total_queries):
        q_eid = query_keys[i]
        gt_q = gt_by_query[q_eid]
        cand_counts_q = Counter()
        for b in BLOCK_NAMES:
            for tid in cands_per_query[b][i]:
                cand_counts_q[tid] += 1

        for tid, cnt in cand_counts_q.items():
            multi_block_cands[cnt] += 1
            if tid in gt_q:
                multi_block_gt[cnt] += 1

    multi_block_stats = {}
    for cnt in range(1, 6):
        c_count = multi_block_cands[cnt]
        g_count = multi_block_gt[cnt]
        multi_block_stats[f"{cnt}_blocks"] = {
            "candidates": c_count,
            "pct_union_cands": round((c_count / total_union_cands) * 100.0, 2),
            "gt_recovered": g_count,
            "precision_pct": round((g_count / c_count) * 100.0, 3) if c_count > 0 else 0.0
        }

    # -------------------------------------------------------------
    # 9. PRINT SUMMARY TABLE
    # -------------------------------------------------------------
    print("\n" + "=" * 145, flush=True)
    print(f"{'Block Name':<20} | {'Raw Cands':<11} | {'Mean/S1':<8} | {'Med':<5} | {'P95':<6} | {'Unique Cands':<12} | {'% Union':<7} | {'GT Rec':<8} | {'Uniq GT':<8} | {'SN Ratio':<9} | {'Cands/Uniq GT':<13}")
    print("-" * 145)
    for b in BLOCK_NAMES:
        m = block_metrics[b]
        print(
            f"{b:<20} | {m['raw_candidate_count']:11,d} | {m['mean_candidates_per_s1']:8.2f} | {m['median_candidates_per_s1']:5.1f} | "
            f"{m['p95_candidates_per_s1']:6.1f} | {m['unique_candidates_count']:12,d} | {m['pct_final_union_pool_unique']:6.2f}% | "
            f"{m['gt_links_recovered']:8,d} | {m['unique_gt_links_recovered']:8,d} | {m['overall_signal_to_noise_pct']:8.2f}% | "
            f"{m['unique_cand_to_unique_gt_ratio']:13.1f}"
        )
    print("-" * 145)
    print(
        f"{'FINAL UNION':<20} | {total_union_cands:11,d} | {mean_union_cands:8.2f} | {median_union_cands:5.1f} | "
        f"{p95_union_cands:6.1f} | {total_union_cands:12,d} | {'100.00%':<7} | {union_gt_captured:8,d} | {union_gt_captured:8,d} | "
        f"{(union_gt_captured/total_union_cands)*100:8.2f}% | {'1.0':<13}"
    )
    print("=" * 145, flush=True)

    # -------------------------------------------------------------
    # 10. SAVE COMPLETE JSON AUDIT RESULTS
    # -------------------------------------------------------------
    audit_results = {
        "execution_date": "September 2026",
        "total_queries": total_queries,
        "total_gt_links": total_gt,
        "us_gt_links": us_gt,
        "india_gt_links": india_gt,
        "cross_script_gt_links": cross_gt,
        "final_pipeline_union": {
            "total_candidates": total_union_cands,
            "mean_candidates_per_s1": round(mean_union_cands, 2),
            "median_candidates_per_s1": median_union_cands,
            "p90_candidates_per_s1": p90_union_cands,
            "p95_candidates_per_s1": p95_union_cands,
            "p99_candidates_per_s1": p99_union_cands,
            "max_candidates_per_s1": max_union_cands,
            "gt_links_captured": union_gt_captured,
            "gt_recall_pct": round(union_recall, 2)
        },
        "block_level_attribution": block_metrics,
        "pairwise_overlap_matrix": pairwise_overlap,
        "multi_block_support_distribution": multi_block_stats
    }

    out_json = "reports/phase5e_candidate_attribution_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    print(f"\nSaved complete attribution audit to {out_json}", flush=True)
    print(f"Total script runtime: {time.time()-total_start:.2f}s | Peak RSS: {get_rss_mb():.2f} MB", flush=True)

if __name__ == "__main__":
    main()
