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

from phase3.address_blocking import AddressNormalizer

NON_LATIN_REGEX = re.compile(r'[\u0900-\u0D7F\u0E00-\u0E7F\u4E00-\u9FFF\u0600-\u06FF]')

def get_rss_mb():
    return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)

def main():
    total_start = time.time()
    print("=" * 80, flush=True)
    print("PHASE 5D: EMPIRICAL G6 ADDRESS-PAIR COLLISION DIAGNOSTIC", flush=True)
    print("=" * 80, flush=True)
    print(f"Initial RSS: {get_rss_mb():.2f} MB", flush=True)

    norm_addr = AddressNormalizer()

    # -----------------------------------------------------------------
    # 1. LOAD 25,000 DEVELOPMENT QUERIES
    # -----------------------------------------------------------------
    print("\n--- 1. Loading 25,000 Development Queries ---", flush=True)
    dev_path = "phase2/stratified_sample_25k.json"
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_queries = json.load(f)

    query_keys = list(dev_queries.keys())
    print(f"Loaded {len(query_keys):,} queries. Pre-extracting G6 keys...", flush=True)

    # For each query, pre-extract G6 keys: list of (t1, t2)
    query_g6_keys = {}
    needed_g6_keys = set()
    all_needed_gt_eids = set()

    for q_eid, q in dev_queries.items():
        country = q["country"]
        clean_a = norm_addr.clean_address(q["business_address"])
        toks_a = norm_addr.extract_tokens(clean_a, min_len=2, filter_generic=True)
        spec_toks = [t for t in toks_a if any(c.isdigit() for c in t) or len(t) >= 4]
        
        pairs = []
        if len(spec_toks) >= 2:
            n_tokens = min(len(spec_toks), 5)
            for i in range(n_tokens):
                for j in range(i + 1, n_tokens):
                    t1, t2 = sorted([spec_toks[i], spec_toks[j]])
                    k = f"{country}_{t1}_{t2}"
                    pairs.append(k)
                    needed_g6_keys.add(k)
        query_g6_keys[q_eid] = pairs
        all_needed_gt_eids.update(q.get("gt_matches", []))

    print(f"Queries with >=2 specific tokens: {sum(1 for p in query_g6_keys.values() if p):,} / {len(query_keys):,}")
    print(f"Unique G6 query keys needed: {len(needed_g6_keys):,}", flush=True)
    print(f"Authoritative GT target eids needed: {len(all_needed_gt_eids):,}", flush=True)
    print(f"Current RSS: {get_rss_mb():.2f} MB", flush=True)

    # -----------------------------------------------------------------
    # 2. SINGLE-PASS TARGET SCAN (INDEX NEEDED G6 KEYS & MAP GT)
    # -----------------------------------------------------------------
    print("\n--- 2. Scanning 10.3M Train Targets for Needed G6 Keys ---", flush=True)
    t0_scan = time.time()
    train_s2 = "student_resource/dataset/train/train_source2.tsv"
    train_s3 = "student_resource/dataset/train/train_source3.tsv"

    idx_g6 = defaultdict(lambda: array.array('I'))
    g6_key_df = Counter()
    gt_eid_to_int = {}
    gt_is_cross_script = {}
    gt_int_to_country = {}
    gt_int_to_g6_keys = {}

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

                    is_gt = eid in all_needed_gt_eids
                    if is_gt:
                        gt_eid_to_int[eid] = int_id
                        gt_is_cross_script[int_id] = bool(NON_LATIN_REGEX.search(name))
                        gt_int_to_country[int_id] = country

                    clean_a = norm_addr.clean_address(addr)
                    toks_a = norm_addr.extract_tokens(clean_a, min_len=2, filter_generic=True)
                    spec_toks = [t for t in toks_a if any(c.isdigit() for c in t) or len(t) >= 4]

                    if is_gt:
                        t_pairs = []
                        if len(spec_toks) >= 2:
                            n_tokens = min(len(spec_toks), 5)
                            for i in range(n_tokens):
                                for j in range(i + 1, n_tokens):
                                    t1, t2 = sorted([spec_toks[i], spec_toks[j]])
                                    t_pairs.append(f"{country}_{t1}_{t2}")
                        gt_int_to_g6_keys[int_id] = t_pairs

                    if len(spec_toks) >= 2:
                        n_tokens = min(len(spec_toks), 5)
                        for i in range(n_tokens):
                            for j in range(i + 1, n_tokens):
                                t1, t2 = sorted([spec_toks[i], spec_toks[j]])
                                k = f"{country}_{t1}_{t2}"
                                if k in needed_g6_keys:
                                    g6_key_df[k] += 1
                                    # Postings cap for memory safety during diag:
                                    # We keep up to 2505 so we know if it was <= 2500
                                    if len(idx_g6[k]) <= 2505:
                                        idx_g6[k].append(int_id)
                int_id += 1
        if s2_split_idx == 0:
            s2_split_idx = int_id

    total_targets = int_id
    scan_time = round(time.time() - t0_scan, 2)
    print(f"Scanned {total_targets:,} targets in {scan_time}s", flush=True)
    print(f"Matched GT targets: {len(gt_eid_to_int):,} / {len(all_needed_gt_eids):,}", flush=True)
    print(f"Posting lists populated for {len(idx_g6):,} G6 keys", flush=True)
    print(f"Current RSS: {get_rss_mb():.2f} MB", flush=True)

    # -----------------------------------------------------------------
    # 3. G6 DOCUMENT FREQUENCY (DF) DISTRIBUTION ANALYSIS
    # -----------------------------------------------------------------
    print("\n--- 3. Analyzing G6 Key Document Frequency Distribution ---", flush=True)
    all_dfs = np.array([g6_key_df[k] for k in needed_g6_keys])
    print(f"Total query G6 keys: {len(all_dfs):,}")
    print(f"  DF = 0 (unseen in target): {(all_dfs == 0).sum():,} ({(all_dfs == 0).mean()*100:.2f}%)")
    print(f"  1 <= DF <= 10: {((all_dfs >= 1) & (all_dfs <= 10)).sum():,} ({((all_dfs >= 1) & (all_dfs <= 10)).mean()*100:.2f}%)")
    print(f"  11 <= DF <= 100: {((all_dfs >= 11) & (all_dfs <= 100)).sum():,} ({((all_dfs >= 11) & (all_dfs <= 100)).mean()*100:.2f}%)")
    print(f"  101 <= DF <= 500: {((all_dfs >= 101) & (all_dfs <= 500)).sum():,} ({((all_dfs >= 101) & (all_dfs <= 500)).mean()*100:.2f}%)")
    print(f"  501 <= DF <= 1000: {((all_dfs >= 501) & (all_dfs <= 1000)).sum():,} ({((all_dfs >= 501) & (all_dfs <= 1000)).mean()*100:.2f}%)")
    print(f"  1001 <= DF <= 2500: {((all_dfs >= 1001) & (all_dfs <= 2500)).sum():,} ({((all_dfs >= 1001) & (all_dfs <= 2500)).mean()*100:.2f}%)")
    print(f"  DF > 2500 (pruned by baseline): {(all_dfs > 2500).sum():,} ({(all_dfs > 2500).mean()*100:.2f}%)")

    # -----------------------------------------------------------------
    # 4. GROUND TRUTH SUPPORT INVESTIGATION
    # -----------------------------------------------------------------
    print("\n--- 4. Investigating Ground Truth Support by G6 Keys ---", flush=True)
    # Questions:
    # 1. How many G6 keys do true GT matches share with the query? (0, 1, 2, 3+)
    # 2. What are the DFs of the supporting G6 keys for true GT matches?
    # 3. What percentage of GT matches are supported by only 1 G6 key?
    # 4. For GT matches supported by only 1 G6 key, what is the DF of that key?
    # 5. Breakdowns by Country (US vs India) and Script (Same vs Cross)

    gt_shared_keys_dist = Counter()
    gt_min_df_dist = []
    gt_max_df_dist = []
    gt_single_key_dfs = []
    gt_multi_key_min_dfs = []

    # By subgroup
    gt_shared_by_subgroup = defaultdict(Counter)
    gt_captured_by_subgroup = Counter()
    gt_total_by_subgroup = Counter()

    # Collision breakdown: total candidates matching 1, 2, 3+ keys
    total_g6_candidates = 0
    cands_matching_1_key = 0
    cands_matching_2_keys = 0
    cands_matching_3plus_keys = 0

    cands_1_key_gt = 0
    cands_2_keys_gt = 0
    cands_3plus_keys_gt = 0

    # Query level aggregations
    query_cand_counts = []
    query_cand_counts_by_country = defaultdict(list)

    for q_idx, q_eid in enumerate(query_keys):
        q = dev_queries[q_eid]
        country = q["country"]
        gt_eids = q.get("gt_matches", [])
        gt_ints = {gt_eid_to_int[m] for m in gt_eids if m in gt_eid_to_int}
        is_cross = any(gt_is_cross_script.get(gid, False) for gid in gt_ints)

        q_keys = query_g6_keys[q_eid]

        # Filter query keys with DF <= 2500 (as in baseline G6)
        active_keys = [k for k in q_keys if g6_key_df[k] <= 2500 and len(idx_g6[k]) > 0]

        # Count occurrences of candidates across active keys for this query
        cand_hit_counter = Counter()
        for k in active_keys:
            postings = idx_g6[k]
            for tid in postings:
                cand_hit_counter[tid] += 1

        u_cands = len(cand_hit_counter)
        total_g6_candidates += u_cands
        query_cand_counts.append(u_cands)
        query_cand_counts_by_country[country].append(u_cands)

        # Count candidates by shared key count
        for tid, cnt in cand_hit_counter.items():
            is_match = (tid in gt_ints)
            if cnt == 1:
                cands_matching_1_key += 1
                if is_match: cands_1_key_gt += 1
            elif cnt == 2:
                cands_matching_2_keys += 1
                if is_match: cands_2_keys_gt += 1
            else:
                cands_matching_3plus_keys += 1
                if is_match: cands_3plus_keys_gt += 1

        # Evaluate each GT match
        sub_key = f"{country}_{'cross' if is_cross else 'same'}"
        for gid in gt_ints:
            gt_total_by_subgroup[sub_key] += 1
            gt_total_by_subgroup[country] += 1
            gt_total_by_subgroup["ALL"] += 1

            # Shared keys between query and target
            t_keys = gt_int_to_g6_keys.get(gid, [])
            shared_all = set(q_keys) & set(t_keys)
            shared_active = [k for k in shared_all if g6_key_df[k] <= 2500]

            n_shared = len(shared_active)
            gt_shared_keys_dist[n_shared] += 1
            gt_shared_by_subgroup[sub_key][n_shared] += 1
            gt_shared_by_subgroup[country][n_shared] += 1

            if n_shared > 0:
                gt_captured_by_subgroup[sub_key] += 1
                gt_captured_by_subgroup[country] += 1
                gt_captured_by_subgroup["ALL"] += 1

                dfs = [g6_key_df[k] for k in shared_active]
                min_df = min(dfs)
                max_df = max(dfs)
                gt_min_df_dist.append(min_df)
                gt_max_df_dist.append(max_df)

                if n_shared == 1:
                    gt_single_key_dfs.append(min_df)
                else:
                    gt_multi_key_min_dfs.append(min_df)

    # -----------------------------------------------------------------
    # 5. PRINT DIAGNOSTIC RESULTS
    # -----------------------------------------------------------------
    total_gt = sum(gt_shared_keys_dist.values())
    captured_gt = total_gt - gt_shared_keys_dist[0]

    print("\n" + "=" * 80, flush=True)
    print("CORE DIAGNOSTIC FINDINGS: GROUND TRUTH SHARED G6 KEYS", flush=True)
    print("=" * 80, flush=True)
    print(f"Total Authoritative GT Links Evaluated: {total_gt:,}")
    print(f"Total Captured by G6 (DF <= 2500): {captured_gt:,} ({captured_gt/total_gt*100:.2f}%)")
    print(f"Total Missed by G6: {gt_shared_keys_dist[0]:,} ({gt_shared_keys_dist[0]/total_gt*100:.2f}%)")

    print("\nDistribution of Shared G6 Keys among True GT Links:")
    for n in sorted(gt_shared_keys_dist.keys()):
        cnt = gt_shared_keys_dist[n]
        pct_all = cnt / total_gt * 100
        pct_cap = (cnt / captured_gt * 100) if (captured_gt > 0 and n > 0) else 0.0
        label = "Missed" if n == 0 else f"{n} key{'s' if n>1 else ''}"
        print(f"  Shared {label:8s}: {cnt:6,} ({pct_all:5.2f}% of all GT | {pct_cap:5.2f}% of captured GT)")

    gt_1_count = gt_shared_keys_dist[1]
    gt_2plus_count = sum(gt_shared_keys_dist[n] for n in gt_shared_keys_dist if n >= 2)
    print(f"\nCRITICAL RATIO: True matches supported by EXACTLY 1 G6 key vs 2+ G6 keys:")
    print(f"  Exactly 1 G6 key: {gt_1_count:,} ({gt_1_count/captured_gt*100:.2f}% of captured GT)")
    print(f"  2 or more G6 keys: {gt_2plus_count:,} ({gt_2plus_count/captured_gt*100:.2f}% of captured GT)")

    print("\n" + "=" * 80, flush=True)
    print("SUBGROUP BREAKDOWN: SHARED G6 KEYS BY COUNTRY & SCRIPT", flush=True)
    print("=" * 80, flush=True)
    for grp in ["US", "India", "India_same", "India_cross"]:
        tot = gt_total_by_subgroup[grp]
        cap = gt_captured_by_subgroup[grp]
        d = gt_shared_by_subgroup[grp]
        print(f"\nSubgroup: {grp} (Total GT: {tot:,} | G6 Captured: {cap:,} = {cap/tot*100:.2f}%)")
        for n in sorted(d.keys()):
            cnt = d[n]
            print(f"  Shared {n} keys: {cnt:5,} ({cnt/tot*100:5.2f}%)")

    print("\n" + "=" * 80, flush=True)
    print("CANDIDATE COLLISION ANALYSIS: COMPOSITION OF THE G6 POOL", flush=True)
    print("=" * 80, flush=True)
    print(f"Total G6 Candidate Pairs Generated (DF <= 2500): {total_g6_candidates:,} ({total_g6_candidates/len(query_keys):.2f}/S1)")
    print(f"  US Avg: {np.mean(query_cand_counts_by_country['US']):.2f} cands/S1 (Median {np.median(query_cand_counts_by_country['US'])})")
    print(f"  India Avg: {np.mean(query_cand_counts_by_country['India']):.2f} cands/S1 (Median {np.median(query_cand_counts_by_country['India'])})")

    print(f"\nCollision Structure by Number of Shared G6 Keys:")
    p1 = cands_matching_1_key / total_g6_candidates * 100
    p2 = cands_matching_2_keys / total_g6_candidates * 100
    p3 = cands_matching_3plus_keys / total_g6_candidates * 100
    print(f"  Candidates matching EXACTLY 1 key: {cands_matching_1_key:10,} ({p1:5.2f}%) | True GT: {cands_1_key_gt:6,} (Precision: {cands_1_key_gt/cands_matching_1_key*100:5.3f}%)")
    print(f"  Candidates matching EXACTLY 2 keys: {cands_matching_2_keys:10,} ({p2:5.2f}%) | True GT: {cands_2_keys_gt:6,} (Precision: {cands_2_keys_gt/cands_matching_2_keys*100:5.3f}%)")
    print(f"  Candidates matching 3+ keys:       {cands_matching_3plus_keys:10,} ({p3:5.2f}%) | True GT: {cands_3plus_keys_gt:6,} (Precision: {cands_3plus_keys_gt/cands_matching_3plus_keys*100:5.3f}%)")

    print("\n" + "=" * 80, flush=True)
    print("DF ANALYSIS: SUPPORTING KEYS FOR TRUE MATCHES VS COLLISIONS", flush=True)
    print("=" * 80, flush=True)
    gt_s_arr = np.array(gt_single_key_dfs)
    gt_m_arr = np.array(gt_multi_key_min_dfs)
    print(f"When GT match is supported by EXACTLY 1 G6 key ({len(gt_s_arr):,} links):")
    print(f"  Min DF: {np.min(gt_s_arr)} | Median DF: {np.median(gt_s_arr)} | Mean DF: {np.mean(gt_s_arr):.1f}")
    print(f"  P75 DF: {np.percentile(gt_s_arr, 75)} | P90 DF: {np.percentile(gt_s_arr, 90)} | P95 DF: {np.percentile(gt_s_arr, 95)} | Max DF: {np.max(gt_s_arr)}")
    print(f"  GT with DF <= 100:  {(gt_s_arr <= 100).sum():,} ({(gt_s_arr <= 100).mean()*100:.2f}%)")
    print(f"  GT with DF <= 250:  {(gt_s_arr <= 250).sum():,} ({(gt_s_arr <= 250).mean()*100:.2f}%)")
    print(f"  GT with DF <= 500:  {(gt_s_arr <= 500).sum():,} ({(gt_s_arr <= 500).mean()*100:.2f}%)")
    print(f"  GT with DF <= 1000: {(gt_s_arr <= 1000).sum():,} ({(gt_s_arr <= 1000).mean()*100:.2f}%)")
    print(f"  GT with DF > 1000:  {(gt_s_arr > 1000).sum():,} ({(gt_s_arr > 1000).mean()*100:.2f}%)")

    # -----------------------------------------------------------------
    # 6. FRANCE TEST DATA INVESTIGATION (5,000 QUERIES SAMPLE)
    # -----------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("FRANCE TEST DATA G6 COLLISION INVESTIGATION", flush=True)
    print("=" * 80, flush=True)

    t0_fr = time.time()
    fr_s1_path = "student_resource/dataset/test/test_source1.tsv"
    fr_s2_path = "student_resource/dataset/test/test_source2.tsv"
    fr_s3_path = "student_resource/dataset/test/test_source3.tsv"

    # Sample 5,000 France queries
    fr_queries = []
    with open(fr_s1_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip("\r\n").split("\t")
            if len(parts) >= 4 and parts[3].strip() == "France":
                fr_queries.append((parts[0], parts[1], parts[2]))
                if len(fr_queries) >= 5000:
                    break

    print(f"Loaded {len(fr_queries):,} France S1 queries. Pre-extracting France G6 keys...", flush=True)
    fr_query_keys = {}
    needed_fr_keys = set()
    for q_eid, q_name, q_addr in fr_queries:
        clean_a = norm_addr.clean_address(q_addr)
        toks_a = norm_addr.extract_tokens(clean_a, min_len=2, filter_generic=True)
        spec_toks = [t for t in toks_a if any(c.isdigit() for c in t) or len(t) >= 4]
        pairs = []
        if len(spec_toks) >= 2:
            n_tokens = min(len(spec_toks), 5)
            for i in range(n_tokens):
                for j in range(i + 1, n_tokens):
                    t1, t2 = sorted([spec_toks[i], spec_toks[j]])
                    k = f"France_{t1}_{t2}"
                    pairs.append(k)
                    needed_fr_keys.add(k)
        fr_query_keys[q_eid] = pairs

    print(f"Unique France G6 query keys needed: {len(needed_fr_keys):,}", flush=True)

    # Scan France targets in test_source2 and test_source3
    fr_idx_g6 = defaultdict(lambda: array.array('I'))
    fr_key_df = Counter()
    fr_target_count = 0

    for p_file in [fr_s2_path, fr_s3_path]:
        with open(p_file, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip("\r\n").split("\t")
                if len(parts) >= 4 and parts[3].strip() == "France":
                    addr = parts[2]
                    clean_a = norm_addr.clean_address(addr)
                    toks_a = norm_addr.extract_tokens(clean_a, min_len=2, filter_generic=True)
                    spec_toks = [t for t in toks_a if any(c.isdigit() for c in t) or len(t) >= 4]
                    if len(spec_toks) >= 2:
                        n_tokens = min(len(spec_toks), 5)
                        for i in range(n_tokens):
                            for j in range(i + 1, n_tokens):
                                t1, t2 = sorted([spec_toks[i], spec_toks[j]])
                                k = f"France_{t1}_{t2}"
                                if k in needed_fr_keys:
                                    fr_key_df[k] += 1
                                    if len(fr_idx_g6[k]) <= 2505:
                                        fr_idx_g6[k].append(fr_target_count)
                    fr_target_count += 1

    print(f"Scanned {fr_target_count:,} France test targets in {time.time()-t0_fr:.2f}s", flush=True)

    # Analyze France G6 collisions
    fr_cand_counts = []
    fr_cands_1_key = 0
    fr_cands_2_keys = 0
    fr_cands_3plus_keys = 0

    for q_eid, q_name, q_addr in fr_queries:
        pairs = fr_query_keys[q_eid]
        active = [k for k in pairs if fr_key_df[k] <= 2500 and len(fr_idx_g6[k]) > 0]
        c_counter = Counter()
        for k in active:
            for tid in fr_idx_g6[k]:
                c_counter[tid] += 1
        u_cnt = len(c_counter)
        fr_cand_counts.append(u_cnt)
        for tid, cnt in c_counter.items():
            if cnt == 1: fr_cands_1_key += 1
            elif cnt == 2: fr_cands_2_keys += 1
            else: fr_cands_3plus_keys += 1

    total_fr_cands = sum(fr_cand_counts)
    print(f"\nFrance Sample (5,000 Queries):")
    print(f"  Total G6 Candidates: {total_fr_cands:,} ({total_fr_cands/len(fr_queries):.2f}/S1)")
    print(f"  Median: {np.median(fr_cand_counts):.1f} | P90: {np.percentile(fr_cand_counts, 90):.1f} | P95: {np.percentile(fr_cand_counts, 95):.1f} | Max: {max(fr_cand_counts):,}")
    print(f"  Matching EXACTLY 1 key: {fr_cands_1_key:,} ({fr_cands_1_key/total_fr_cands*100:.2f}%)")
    print(f"  Matching EXACTLY 2 keys: {fr_cands_2_keys:,} ({fr_cands_2_keys/total_fr_cands*100:.2f}%)")
    print(f"  Matching 3+ keys:       {fr_cands_3plus_keys:,} ({fr_cands_3plus_keys/total_fr_cands*100:.2f}%)")

    # Save summary dictionary for reporting
    diag_summary = {
        "runtime_sec": round(time.time() - total_start, 2),
        "peak_rss_mb": get_rss_mb(),
        "total_gt_links": total_gt,
        "captured_gt_links": captured_gt,
        "missed_gt_links": int(gt_shared_keys_dist[0]),
        "gt_shared_keys_dist": {str(k): int(v) for k, v in gt_shared_keys_dist.items()},
        "total_g6_candidates": total_g6_candidates,
        "avg_g6_cands_per_s1": round(total_g6_candidates / len(query_keys), 2),
        "cands_matching_1_key": cands_matching_1_key,
        "cands_matching_2_keys": cands_matching_2_keys,
        "cands_matching_3plus_keys": cands_matching_3plus_keys,
        "cands_1_key_gt": cands_1_key_gt,
        "cands_2_keys_gt": cands_2_keys_gt,
        "cands_3plus_keys_gt": cands_3plus_keys_gt,
        "gt_1_count": gt_1_count,
        "gt_2plus_count": gt_2plus_count,
        "france_sample_avg_cands": round(total_fr_cands / len(fr_queries), 2),
        "france_cands_1_key_pct": round(fr_cands_1_key / total_fr_cands * 100, 2),
        "france_cands_2_keys_pct": round(fr_cands_2_keys / total_fr_cands * 100, 2),
        "france_cands_3plus_keys_pct": round(fr_cands_3plus_keys / total_fr_cands * 100, 2),
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/phase5d_g6_collision_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag_summary, f, indent=2)

    print(f"\nSaved diagnostic summary to reports/phase5d_g6_collision_diagnostics.json", flush=True)
    print(f"Total diagnostic runtime: {time.time()-total_start:.2f}s | Peak RSS: {get_rss_mb():.2f} MB", flush=True)

if __name__ == "__main__":
    main()
