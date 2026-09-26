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
from phase4.extract_pair_features import precompute_entity, compute_pair_features, FEATURE_NAMES

def get_rss_mb():
    return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)

def run_production_dry_run(sub_queries_per_country=2000, region_cap=100):
    total_start = time.time()
    peak_rss = 0.0

    def update_rss(stage: str):
        nonlocal peak_rss
        cur = get_rss_mb()
        if cur > peak_rss: peak_rss = cur
        print(f"[{stage}] RSS: {cur:.2f} MB | Peak: {peak_rss:.2f} MB", flush=True)

    print("=" * 90, flush=True)
    print(f"PHASE 5G: PRODUCTION CANDIDATE-GENERATION & INFERENCE DRY RUN (Region Cap <= {region_cap})", flush=True)
    print("=" * 90, flush=True)
    update_rss("Initial State")

    normalizer = EntityNormalizer()
    norm_addr = AddressNormalizer()

    # Load models
    t0_mod = time.time()
    lgb_model = joblib.load("output/best_lgb_model.pkl")
    xgb_model = joblib.load("output/best_xgb_model.pkl")
    print(f"Loaded matcher models in {time.time()-t0_mod:.2f}s", flush=True)
    update_rss("Models Loaded")

    # 1. Sample representative test S1 queries from France, India, US
    test_dir = "student_resource/dataset/test"
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    print(f"\n--- 1. Sampling {sub_queries_per_country} S1 Queries per Country from {s1_path} ---", flush=True)
    queries_by_country = defaultdict(list)
    total_s1_scanned = 0

    with open(s1_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            total_s1_scanned += 1
            parts = line.strip("\r\n").split("\t")
            if len(parts) >= 4:
                eid, name, addr, country = parts[0], parts[1], parts[2], parts[3].strip()
                if len(queries_by_country[country]) < sub_queries_per_country:
                    queries_by_country[country].append((eid, name, addr))
            # If we collected enough from all 3 countries, break
            if all(len(queries_by_country[c]) >= sub_queries_per_country for c in ["France", "India", "US"]):
                break

    for c in ["France", "India", "US"]:
        print(f"  {c}: Sampled {len(queries_by_country[c]):,} test S1 queries", flush=True)

    all_sampled_queries = []
    for c in ["France", "India", "US"]:
        all_sampled_queries.extend([(q[0], q[1], q[2], c) for q in queries_by_country[c]])
    total_dry_run_queries = len(all_sampled_queries)
    print(f"Total Dry Run Sample: {total_dry_run_queries:,} queries across 3 countries", flush=True)

    # Output storage
    os.makedirs("output/dry_run", exist_ok=True)
    cand_tsv_path = "output/dry_run/candidate_pairs.tsv"
    match_tsv_path = "output/dry_run/matching_results.tsv"

    f_cand = open(cand_tsv_path, "w", encoding="utf-8")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    f_match = open(match_tsv_path, "w", encoding="utf-8")
    f_match.write("source1_entity_id\tmatched_entity_ids\n")

    country_stats = {}
    total_candidate_pairs_all = 0
    total_matches_all = 0
    total_feature_gen_time = 0.0
    total_inference_time = 0.0

    # 2. Process each country independently (production chunked pattern)
    for country in ["France", "India", "US"]:
        t0_country = time.time()
        c_queries = queries_by_country[country]
        print(f"\n=======================================================", flush=True)
        print(f"PROCESSING {country.upper()}: {len(c_queries):,} S1 Queries", flush=True)
        print(f"=======================================================", flush=True)

        # 2a. Load all targets for this country from test_source2 & test_source3
        t0_load = time.time()
        target_eids = []
        target_names = []
        target_addrs = []

        for p_file in [s2_path, s3_path]:
            with open(p_file, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.strip("\r\n").split("\t")
                    if len(parts) >= 4 and parts[3].strip() == country:
                        target_eids.append(parts[0])
                        target_names.append(parts[1])
                        target_addrs.append(parts[2])

        num_targets = len(target_eids)
        print(f"Loaded {num_targets:,} {country} targets in {time.time()-t0_load:.2f}s", flush=True)
        update_rss(f"Loaded {country} Targets")

        # 2b. Build Blocking Indexes
        print(f"Building 6 Blocking Indexes for {country} targets...", flush=True)
        t0_idx = time.time()

        idx_a = defaultdict(lambda: array.array('I'))
        idx_b = defaultdict(lambda: array.array('I'))
        tok_freq = Counter()
        idx_c = defaultdict(lambda: array.array('I'))
        c_freq = Counter()
        idx_d_post = defaultdict(lambda: array.array('I'))
        idx_d_reg = defaultdict(lambda: array.array('I'))
        d_reg_freq = Counter()
        idx_e = defaultdict(lambda: array.array('I'))
        ng_freq = Counter()
        idx_g6 = defaultdict(lambda: array.array('I'))
        g6_freq = Counter()

        # Target precomputation structures for cheap features
        target_cheap_feats = {}

        for tid in range(num_targets):
            name = target_names[tid]
            addr = target_addrs[tid]

            # Block A & B
            norm_n = normalizer.normalize_name(name)
            idx_a[norm_n].append(tid)

            toks = normalizer.extract_tokens(norm_n)
            for t in set(toks):
                idx_b[t].append(tid)
                tok_freq[t] += 1

            # Block C
            core12 = normalizer.extract_compressed_core(norm_n, prefix_len=12)
            if core12:
                idx_c[core12].append(tid)
                c_freq[core12] += 1

            # Block D
            if toks:
                first_tok = toks[0]
                sig = normalizer.extract_address_signals(addr, country)
                if not sig["is_empty"]:
                    if sig["postal_code"]:
                        idx_d_post[f"{sig['postal_code']}_{first_tok}"].append(tid)
                    if sig["region"]:
                        rk = f"{sig['region']}_{first_tok}"
                        idx_d_reg[rk].append(tid)
                        d_reg_freq[rk] += 1

            # Block E
            for ng in set(normalizer.extract_char_ngrams(norm_n, n=3)):
                idx_e[ng].append(tid)
                ng_freq[ng] += 1

            # Block G6
            clean_a = norm_addr.clean_address(addr)
            toks_a = norm_addr.extract_tokens(clean_a, min_len=2, filter_generic=True)
            spec_toks = [t for t in toks_a if any(c.isdigit() for c in t) or len(t) >= 4]
            if len(spec_toks) >= 2:
                n_tokens = min(len(spec_toks), 5)
                for i_t in range(n_tokens):
                    for j_t in range(i_t + 1, n_tokens):
                        t1, t2 = sorted([spec_toks[i_t], spec_toks[j_t]])
                        k = f"{t1}_{t2}"
                        idx_g6[k].append(tid)
                        g6_freq[k] += 1

        print(f"Raw indexes built in {time.time()-t0_idx:.2f}s. Applying caps...", flush=True)

        # Apply frequency caps:
        # B <= 10000
        for t in list(idx_b.keys()):
            if tok_freq[t] > 10000: del idx_b[t]
        del tok_freq

        # C <= 1000
        for ck in list(idx_c.keys()):
            if c_freq[ck] > 1000: del idx_c[ck]
        del c_freq

        # D Region <= region_cap
        for rk in list(idx_d_reg.keys()):
            if d_reg_freq[rk] > region_cap: del idx_d_reg[rk]
        # Keep d_reg_freq for checking key frequencies

        # E <= 10000
        for ng in list(idx_e.keys()):
            if ng_freq[ng] > 10000: del idx_e[ng]
        del ng_freq

        # G6 <= 2500
        for gk in list(idx_g6.keys()):
            if g6_freq[gk] > 2500: del idx_g6[gk]

        update_rss(f"{country} Indexes Ready")

        # 2c. Process Queries in Chunks
        CHUNK_SIZE = 1000
        num_chunks = int(np.ceil(len(c_queries) / CHUNK_SIZE))
        country_cand_pairs = 0
        country_matches = 0

        c_query_cands = []
        c_block_cand_counts = Counter()

        for c_idx in range(num_chunks):
            t0_chk = time.time()
            chunk_queries = c_queries[c_idx * CHUNK_SIZE : (c_idx + 1) * CHUNK_SIZE]

            # Precompute S1 entities
            s1_pre = {}
            for q_eid, q_name, q_addr in chunk_queries:
                s1_pre[q_eid] = precompute_entity(q_name, q_addr, country, norm_addr)

            # Generate candidate sets per query
            chunk_pairs = [] # list of (s1_eid, tid, hits, bitmask)
            unique_cand_tids = set()

            for q_eid, q_name, q_addr in chunk_queries:
                norm_q = normalizer.normalize_name(q_name)
                toks_q = normalizer.extract_tokens(norm_q)
                first_tok_q = toks_q[0] if toks_q else ""
                core_q = normalizer.extract_compressed_core(norm_q, prefix_len=12)

                cand_bits = defaultdict(int)

                # Block A
                if norm_q in idx_a:
                    for tid in idx_a[norm_q]: cand_bits[tid] |= 1
                set_a = {tid for tid, bm in cand_bits.items() if (bm & 1) != 0}
                c_block_cand_counts["Block_A"] += len(set_a)

                # Block C (DF <= 1000)
                if core_q in idx_c:
                    for tid in idx_c[core_q]: cand_bits[tid] |= 4
                set_c = {tid for tid, bm in cand_bits.items() if (bm & 4) != 0}
                c_block_cand_counts["Block_C"] += len(set_c)

                # Block D (Postal unconditional + Region DF <= cap)
                if toks_q:
                    sig_q = normalizer.extract_address_signals(q_addr, country)
                    if not sig_q["is_empty"]:
                        if sig_q["postal_code"]:
                            pk = f"{sig_q['postal_code']}_{first_tok_q}"
                            if pk in idx_d_post:
                                for tid in idx_d_post[pk]: cand_bits[tid] |= 8
                        if sig_q["region"]:
                            rk = f"{sig_q['region']}_{first_tok_q}"
                            if rk in idx_d_reg:
                                for tid in idx_d_reg[rk]: cand_bits[tid] |= 8
                set_d = {tid for tid, bm in cand_bits.items() if (bm & 8) != 0}
                c_block_cand_counts["Block_D"] += len(set_d)

                # Block G6 Hybrid-250
                clean_aq = norm_addr.clean_address(q_addr)
                toks_aq = norm_addr.extract_tokens(clean_aq, min_len=2, filter_generic=True)
                spec_toks_q = [t for t in toks_aq if any(c.isdigit() for c in t) or len(t) >= 4]
                g6_active = []
                if len(spec_toks_q) >= 2:
                    n_tokens = min(len(spec_toks_q), 5)
                    for i_t in range(n_tokens):
                        for j_t in range(i_t + 1, n_tokens):
                            t1, t2 = sorted([spec_toks_q[i_t], spec_toks_q[j_t]])
                            k = f"{t1}_{t2}"
                            if k in idx_g6:
                                g6_active.append((idx_g6[k], g6_freq[k]))

                g6_hits = Counter()
                for p_arr, df in g6_active:
                    for tid in p_arr: g6_hits[tid] += 1
                for tid, cnt in g6_hits.items():
                    if cnt >= 2:
                        cand_bits[tid] |= 32
                for p_arr, df in g6_active:
                    if df <= 250:
                        for tid in p_arr:
                            if g6_hits[tid] == 1:
                                cand_bits[tid] |= 32

                set_g6 = {tid for tid, bm in cand_bits.items() if (bm & 32) != 0}
                c_block_cand_counts["Block_G6"] += len(set_g6)

                # Anchors
                anchors = set_a | set_c | set_d | set_g6

                # Candidate pool for B and E
                be_pool = set()
                # B
                for t in set(toks_q):
                    if t in idx_b:
                        be_pool.update(idx_b[t])
                # E (T >= 6)
                ngrams_q = set(normalizer.extract_char_ngrams(norm_q, n=3))
                p_views = [np.frombuffer(idx_e[ng], dtype=np.uint32) for ng in ngrams_q if ng in idx_e and len(idx_e[ng]) > 0]
                if p_views:
                    cat = np.concatenate(p_views)
                    u, cnts = np.unique(cat, return_counts=True)
                    be_pool.update(u[cnts >= 6].tolist())

                pot_be = be_pool - anchors
                c_block_cand_counts["Block_BE_Pool"] += len(pot_be)

                # Refined Top-50
                if len(pot_be) <= 50:
                    set_be = pot_be
                else:
                    qn_toks = set(toks_q)
                    qa_toks = set(toks_aq)
                    q_nums = set(re.findall(r'\b\d+\b', q_addr))
                    scored = []
                    for tid in pot_be:
                        tn = target_names[tid]
                        ta = target_addrs[tid]
                        cn_toks = set(normalizer.extract_tokens(normalizer.normalize_name(tn)))
                        clean_ta = norm_addr.clean_address(ta)
                        ca_toks = set(norm_addr.extract_tokens(clean_ta, min_len=2, filter_generic=True))
                        c_nums = set(re.findall(r'\b\d+\b', ta))

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

                c_block_cand_counts["Refined_BE_Top50"] += len(set_be)
                for tid in set_be:
                    cand_bits[tid] |= 2  # mark as B/E candidate

                # Record candidate pairs
                final_cand_tids = list(cand_bits.keys())
                c_query_cands.append(len(final_cand_tids))

                # Safety Check: Deduplication within list
                assert len(final_cand_tids) == len(set(final_cand_tids)), "Duplicate candidate ID found in query!"

                cand_str = ",".join(target_eids[tid] for tid in final_cand_tids)
                f_cand.write(f"{q_eid}\t{cand_str}\n")

                for tid in final_cand_tids:
                    bm = cand_bits[tid]
                    hits = bin(bm).count('1')
                    chunk_pairs.append((q_eid, tid, hits, bm))
                    unique_cand_tids.add(tid)

            # Feature generation & ML scoring for chunk
            t0_feat = time.time()
            cand_pre = {}
            for tid in unique_cand_tids:
                cand_pre[tid] = precompute_entity(target_names[tid], target_addrs[tid], country, norm_addr)

            num_chunk_pairs = len(chunk_pairs)
            country_cand_pairs += num_chunk_pairs

            if num_chunk_pairs > 0:
                X_chunk = np.empty((num_chunk_pairs, len(FEATURE_NAMES)), dtype=np.float32)
                for p_idx, (q_eid, tid, hits, bm) in enumerate(chunk_pairs):
                    X_chunk[p_idx] = compute_pair_features(s1_pre[q_eid], cand_pre[tid], country, target_eids[tid], hits, bm)
                feat_time = time.time() - t0_feat
                total_feature_gen_time += feat_time

                # Matcher inference
                t0_inf = time.time()
                p_lgb = lgb_model.predict_proba(X_chunk)[:, 1]
                p_xgb = xgb_model.predict_proba(X_chunk)[:, 1]
                probs = 0.50 * p_lgb + 0.50 * p_xgb
                inf_time = time.time() - t0_inf
                total_inference_time += inf_time

                # Apply threshold tau = 0.600
                s1_matches = defaultdict(list)
                for p_idx, (q_eid, tid, hits, bm) in enumerate(chunk_pairs):
                    if probs[p_idx] >= 0.600:
                        s1_matches[q_eid].append(target_eids[tid])

                for q_eid, _, _ in chunk_queries:
                    m_list = s1_matches.get(q_eid, [])
                    country_matches += len(m_list)
                    f_match.write(f"{q_eid}\t{','.join(m_list)}\n")
            else:
                for q_eid, _, _ in chunk_queries:
                    f_match.write(f"{q_eid}\t\n")

            chk_time = time.time() - t0_chk
            print(f"  Chunk {c_idx+1}/{num_chunks}: {len(chunk_queries)} queries -> {num_chunk_pairs:,} candidate pairs | Time: {chk_time:.2f}s | RSS: {get_rss_mb():.2f} MB", flush=True)

        total_candidate_pairs_all += country_cand_pairs
        total_matches_all += country_matches

        mean_cands = np.mean(c_query_cands)
        median_cands = np.median(c_query_cands)
        p90_cands = np.percentile(c_query_cands, 90)
        p95_cands = np.percentile(c_query_cands, 95)
        p99_cands = np.percentile(c_query_cands, 99)
        max_cands = np.max(c_query_cands)
        empty_cands = sum(1 for c in c_query_cands if c == 0)

        country_stats[country] = {
            "queries": len(c_queries),
            "targets": num_targets,
            "candidate_pairs": country_cand_pairs,
            "mean_candidates_per_s1": round(mean_cands, 2),
            "median_candidates_per_s1": float(median_cands),
            "p90_candidates_per_s1": float(p90_cands),
            "p95_candidates_per_s1": float(p95_cands),
            "p99_candidates_per_s1": float(p99_cands),
            "max_candidates_per_s1": int(max_cands),
            "empty_candidate_queries": empty_cands,
            "predicted_matches": country_matches,
            "mean_matches_per_s1": round(country_matches / len(c_queries), 3),
            "block_breakdown": {b: round(cnt / len(c_queries), 2) for b, cnt in c_block_cand_counts.items()}
        }

        del idx_a, idx_b, idx_c, idx_d_post, idx_d_reg, idx_e, idx_g6
        del target_eids, target_names, target_addrs
        gc.collect()
        update_rss(f"Completed {country}")

    f_cand.close()
    f_match.close()

    print("\n" + "=" * 95, flush=True)
    print("PRODUCTION DRY-RUN SUMMARY RESULTS BY COUNTRY:")
    print("=" * 95, flush=True)
    print(f"{'Country':<10} | {'Queries':<8} | {'Targets':<11} | {'Total Cands':<12} | {'Mean/S1':<8} | {'Med':<5} | {'P95':<6} | {'Max':<5} | {'Empty':<5} | {'Matches':<8}")
    print("-" * 95)
    for c, s in country_stats.items():
        print(f"{c:<10} | {s['queries']:8,d} | {s['targets']:11,d} | {s['candidate_pairs']:12,d} | {s['mean_candidates_per_s1']:8.2f} | {s['median_candidates_per_s1']:5.1f} | {s['p95_candidates_per_s1']:6.1f} | {s['max_candidates_per_s1']:5d} | {s['empty_candidate_queries']:5d} | {s['predicted_matches']:8,d}")
    print("-" * 95)
    overall_mean = total_candidate_pairs_all / total_dry_run_queries
    print(f"{'OVERALL':<10} | {total_dry_run_queries:8,d} | {'-':<11} | {total_candidate_pairs_all:12,d} | {overall_mean:8.2f} | {'-':<5} | {'-':<6} | {'-':<5} | {'-':<5} | {total_matches_all:8,d}")
    print("=" * 95, flush=True)

    # Save summary json
    dry_run_summary = {
        "execution_date": "September 2026",
        "region_cap": region_cap,
        "total_queries_tested": total_dry_run_queries,
        "total_candidate_pairs": total_candidate_pairs_all,
        "overall_mean_candidates_per_s1": round(overall_mean, 2),
        "total_matches_predicted": total_matches_all,
        "peak_rss_mb": peak_rss,
        "feature_gen_time_sec": round(total_feature_gen_time, 2),
        "inference_time_sec": round(total_inference_time, 2),
        "total_runtime_sec": round(time.time() - total_start, 2),
        "country_statistics": country_stats
    }

    with open("reports/phase5g_dry_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(dry_run_summary, f, indent=2)

    print(f"\nDry-run completed successfully in {time.time()-total_start:.2f}s | Peak RSS: {peak_rss:.2f} MB", flush=True)

if __name__ == "__main__":
    run_production_dry_run(sub_queries_per_country=2000, region_cap=100)
