import os
import sys
import time
import json
import array
from collections import defaultdict, Counter
import numpy as np

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blocking import EntityNormalizer, InvertedPostingIndex, get_memory_info_mb
from evaluate_blocking import (
    build_stratified_validation_sample,
    evaluate_candidate_sets,
    compute_pairwise_block_overlap,
    compute_script_divergence_diagnostic
)

def run_phase2_blocking_benchmark():
    start_total_time = time.time()
    print("=" * 70, flush=True)
    print("AMAZON ML CHALLENGE 2026 — PHASE 2 BLOCKING BENCHMARK", flush=True)
    print("=" * 70, flush=True)

    dataset_dir = "student_resource/dataset"
    train_s1_path = os.path.join(dataset_dir, "train", "train_source1.tsv")
    train_s2_path = os.path.join(dataset_dir, "train", "train_source2.tsv")
    train_s3_path = os.path.join(dataset_dir, "train", "train_source3.tsv")
    train_gt_path = os.path.join(dataset_dir, "train", "train_ground_truth.tsv")

    normalizer = EntityNormalizer()

    # -----------------------------------------------------------------
    # STEP 1: Build Stratified 25,000 S1 Validation Queries
    # -----------------------------------------------------------------
    t0 = time.time()
    queries = build_stratified_validation_sample(
        train_s1_path=train_s1_path,
        train_gt_path=train_gt_path,
        target_sample_size=25000,
        random_seed=42
    )
    all_needed_gt_ids = set()
    for q in queries.values():
        all_needed_gt_ids.update(q["gt_matches"])
    print(f"Total ground truth target links in validation sample: {len(all_needed_gt_ids):,}", flush=True)
    curr_rss, peak_rss = get_memory_info_mb()
    print(f"Sample built in {time.time() - t0:.2f}s | Current RSS: {curr_rss} MB | Peak RSS: {peak_rss} MB\n", flush=True)

    # -----------------------------------------------------------------
    # STEP 2: Index COMPLETE S2 and S3 Datasets (10,320,219 records)
    # -----------------------------------------------------------------
    print("=" * 70, flush=True)
    print("INDEXING COMPLETE S2 + S3 DATASETS (10,320,219 RECORDS)...", flush=True)
    print("=" * 70, flush=True)
    t0_idx = time.time()

    # Master arrays
    # All integer IDs map to string entity_id in s23_entity_ids list
    s23_entity_ids = []
    # Store raw names only for the needed ground truth IDs to support script diagnostic with zero RAM bloat
    s23_gt_names = {}

    # Initialize Index structures
    idx_block_a = InvertedPostingIndex(name="Block A (Normalized Name)")
    idx_block_b_tokens = defaultdict(lambda: defaultdict(lambda: array.array('I')))
    token_doc_freq = defaultdict(Counter) # dict[country, Counter(token)]
    
    # Block C indices for prefix experiments: Full, 16, 12, 10, 8
    idx_block_c_full = InvertedPostingIndex(name="Block C (Full Core)")
    idx_block_c_16 = InvertedPostingIndex(name="Block C (Core-16)")
    idx_block_c_12 = InvertedPostingIndex(name="Block C (Core-12)")
    idx_block_c_10 = InvertedPostingIndex(name="Block C (Core-10)")
    idx_block_c_8 = InvertedPostingIndex(name="Block C (Core-8)")

    # Block D indices (Address Signals)
    idx_block_d_postal = InvertedPostingIndex(name="Block D (Postal + Token)")
    idx_block_d_region = InvertedPostingIndex(name="Block D (Region + Token)")

    # Block E index (3-grams)
    idx_block_e_3grams = defaultdict(lambda: defaultdict(lambda: array.array('I')))

    record_counter = 0

    for src_path, src_label in [(train_s2_path, "Source 2"), (train_s3_path, "Source 3")]:
        t_src_start = time.time()
        print(f"Reading & indexing {src_label} ({src_path})...", flush=True)
        with open(src_path, "r", encoding="utf-8") as f:
            header = f.readline()
            for line_idx, line in enumerate(f):
                line = line.strip("\r\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                
                eid, raw_name, raw_addr, country = parts[0], parts[1], parts[2], parts[3]
                int_id = record_counter
                record_counter += 1
                s23_entity_ids.append(eid)

                if eid in all_needed_gt_ids:
                    s23_gt_names[eid] = raw_name

                # 1. Normalization
                norm_name = normalizer.normalize_name(raw_name, record_diagnostic=(record_counter < 100))
                if not norm_name:
                    continue

                # Block A: Normalized Name
                idx_block_a.add(country, norm_name, int_id)

                # Block B: Tokens
                toks = normalizer.extract_tokens(norm_name)
                for t in set(toks):
                    idx_block_b_tokens[country][t].append(int_id)
                    token_doc_freq[country][t] += 1

                # Block C: Compressed Core (Full and prefixes)
                comp_core = normalizer.extract_compressed_core(norm_name)
                if comp_core:
                    idx_block_c_full.add(country, comp_core, int_id)
                    if len(comp_core) >= 16:
                        idx_block_c_16.add(country, comp_core[:16], int_id)
                    if len(comp_core) >= 12:
                        idx_block_c_12.add(country, comp_core[:12], int_id)
                    if len(comp_core) >= 10:
                        idx_block_c_10.add(country, comp_core[:10], int_id)
                    if len(comp_core) >= 8:
                        idx_block_c_8.add(country, comp_core[:8], int_id)

                # Block D: Address Signals
                addr_signals = normalizer.extract_address_signals(raw_addr, country)
                if not addr_signals["is_empty"] and toks:
                    first_tok = toks[0]
                    if addr_signals["postal_code"]:
                        key_post = f"{addr_signals['postal_code']}_{first_tok}"
                        idx_block_d_postal.add(country, key_post, int_id)
                    if addr_signals["region"]:
                        key_reg = f"{addr_signals['region']}_{first_tok}"
                        idx_block_d_region.add(country, key_reg, int_id)

                # Block E: Character 3-Grams
                ngrams = normalizer.extract_char_ngrams(norm_name, n=3)
                for ng in set(ngrams):
                    idx_block_e_3grams[country][ng].append(int_id)

                if record_counter % 2500000 == 0:
                    c_rss, p_rss = get_memory_info_mb()
                    print(f"  Processed {record_counter:,} records... | RSS: {c_rss} MB | Peak: {p_rss} MB", flush=True)

        print(f"Finished {src_label} in {time.time() - t_src_start:.2f}s", flush=True)

    idx_block_a.finalize()
    idx_block_c_full.finalize()
    idx_block_c_16.finalize()
    idx_block_c_12.finalize()
    idx_block_c_10.finalize()
    idx_block_c_8.finalize()
    idx_block_d_postal.finalize()
    idx_block_d_region.finalize()

    idx_time = time.time() - t0_idx
    c_rss, p_rss = get_memory_info_mb()
    print(f"\nAll 10,320,219 records indexed in {idx_time:.2f}s!", flush=True)
    print(f"Indexed Keys - Block A: {idx_block_a.num_keys:,}, Block C Full: {idx_block_c_full.num_keys:,}")
    print(f"Current RSS: {c_rss} MB | Peak RSS: {p_rss} MB\n", flush=True)

    # -----------------------------------------------------------------
    # STEP 3: BENCHMARK INDIVIDUAL BLOCKS
    # -----------------------------------------------------------------
    benchmark_results = {}
    candidate_cache = {} # dict[block_name, dict[s1_id, set[str]]]

    def evaluate_block_pass(block_name: str, retrieval_fn, config_params: dict):
        t0_pass = time.time()
        cands_map = defaultdict(set)
        for eid, q in queries.items():
            cands_map[eid] = retrieval_fn(q)
        pass_time = time.time() - t0_pass
        _, pass_peak_rss = get_memory_info_mb()
        
        metrics = evaluate_candidate_sets(queries, cands_map)
        metrics["runtime_sec"] = round(pass_time, 2)
        metrics["peak_rss_mb"] = pass_peak_rss
        metrics["config"] = config_params
        
        benchmark_results[block_name] = metrics
        candidate_cache[block_name] = cands_map
        
        print(f"--- {block_name} ---")
        print(f"  Overall Recall: {metrics['overall_recall']}% (S2: {metrics['s2_recall']}%, S3: {metrics['s3_recall']}%)")
        print(f"  Candidates/S1: Avg={metrics['avg_candidates_per_s1']}, Median={metrics['median_candidates']}, P95={metrics['p95_candidates']}, Max={metrics['max_candidates']}")
        print(f"  Zero-Cand %: {metrics['zero_candidate_pct']}% | Reduction Ratio: {metrics['reduction_ratio_pct']}%")
        print(f"  Runtime: {metrics['runtime_sec']}s | Peak RSS: {pass_peak_rss} MB\n", flush=True)
        return metrics

    # Block A: Country + Normalized Name
    print("Benchmarking Block A (Country + Normalized Name)...", flush=True)
    def retrieve_block_a(q):
        c = q["country"]
        norm_name = normalizer.normalize_name(q["business_name"])
        postings = idx_block_a.get_candidates(c, norm_name)
        return {s23_entity_ids[i] for i in postings}
    evaluate_block_pass("Block_A_Norm_Name", retrieve_block_a, {"type": "normalized_name_exact"})

    # Block B: Country + Informative Tokens (Testing Thresholds)
    # Threshold experiment: max token document frequency (e.g. prune tokens occurring in > 10,000 or > 25,000 records)
    print("Benchmarking Block B (Country + Informative Tokens - Threshold Experiments)...", flush=True)
    for max_freq in [5000, 10000, 25000]:
        b_name = f"Block_B_Token_maxfreq_{max_freq}"
        def retrieve_block_b(q, mf=max_freq):
            c = q["country"]
            norm_name = normalizer.normalize_name(q["business_name"])
            toks = normalizer.extract_tokens(norm_name)
            res = set()
            for t in toks:
                if token_doc_freq[c][t] <= mf:
                    postings = idx_block_b_tokens[c].get(t, array.array('I'))
                    for i in postings:
                        res.add(s23_entity_ids[i])
            return res
        evaluate_block_pass(b_name, retrieve_block_b, {"max_token_doc_freq": max_freq})

    # Pick best Block B (maxfreq 10000 as standard balance)
    standard_block_b_name = "Block_B_Token_maxfreq_10000"

    # Block C: Country + Compressed Name Core (Prefix Experiments)
    print("Benchmarking Block C (Compressed Core - Prefix Experiments)...", flush=True)
    core_indices = [
        ("Block_C_Core_Full", idx_block_c_full, None),
        ("Block_C_Core_16", idx_block_c_16, 16),
        ("Block_C_Core_12", idx_block_c_12, 12),
        ("Block_C_Core_10", idx_block_c_10, 10),
        ("Block_C_Core_8", idx_block_c_8, 8),
    ]
    for c_name, c_idx, pref_len in core_indices:
        def retrieve_block_c(q, idx=c_idx, pl=pref_len):
            c = q["country"]
            norm_name = normalizer.normalize_name(q["business_name"])
            core = normalizer.extract_compressed_core(norm_name, prefix_len=pl)
            postings = idx.get_candidates(c, core)
            return {s23_entity_ids[i] for i in postings}
        evaluate_block_pass(c_name, retrieve_block_c, {"prefix_length": pref_len or "Full"})

    # Standard Block C
    standard_block_c_name = "Block_C_Core_12"

    # Block D: Country + Address Signals
    print("Benchmarking Block D (Address Signals)...", flush=True)
    def retrieve_block_d(q):
        c = q["country"]
        norm_name = normalizer.normalize_name(q["business_name"])
        toks = normalizer.extract_tokens(norm_name)
        if not toks:
            return set()
        first_tok = toks[0]
        addr_signals = normalizer.extract_address_signals(q["business_address"], c)
        res = set()
        if not addr_signals["is_empty"]:
            if addr_signals["postal_code"]:
                key_post = f"{addr_signals['postal_code']}_{first_tok}"
                for i in idx_block_d_postal.get_candidates(c, key_post):
                    res.add(s23_entity_ids[i])
            if addr_signals["region"]:
                key_reg = f"{addr_signals['region']}_{first_tok}"
                for i in idx_block_d_region.get_candidates(c, key_reg):
                    res.add(s23_entity_ids[i])
        return res
    evaluate_block_pass("Block_D_Address_Signals", retrieve_block_d, {"type": "postal_and_region_with_token"})

    # Block E: Character 3-Gram Inverted Index (Threshold Experiments)
    # Experimenting with minimum shared 3-grams count T in {3, 4, 5}
    print("Benchmarking Block E (Character 3-Grams - Threshold Experiments)...", flush=True)
    for min_shared in [3, 4, 5]:
        e_name = f"Block_E_3gram_min_shared_{min_shared}"
        def retrieve_block_e(q, ms=min_shared):
            c = q["country"]
            norm_name = normalizer.normalize_name(q["business_name"])
            ngrams = normalizer.extract_char_ngrams(norm_name, n=3)
            p_views = []
            for ng in ngrams:
                p = idx_block_e_3grams[c].get(ng)
                if p and len(p) > 0:
                    p_views.append(np.frombuffer(p, dtype=np.uint32))
            if not p_views:
                return set()
            cat = np.concatenate(p_views)
            u, cnts = np.unique(cat, return_counts=True)
            matched_ids = u[cnts >= ms]
            return {s23_entity_ids[i] for i in matched_ids}
        evaluate_block_pass(e_name, retrieve_block_e, {"min_shared_3grams": min_shared})

    standard_block_e_name = "Block_E_3gram_min_shared_4"

    # -----------------------------------------------------------------
    # STEP 4: CUMULATIVE UNION & INCREMENTAL RECALL ANALYSIS
    # -----------------------------------------------------------------
    print("=" * 70, flush=True)
    print("STEP 4: CUMULATIVE UNION & INCREMENTAL RECALL ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    union_progression = [
        ("A", ["Block_A_Norm_Name"]),
        ("A+B", ["Block_A_Norm_Name", standard_block_b_name]),
        ("A+B+C", ["Block_A_Norm_Name", standard_block_b_name, standard_block_c_name]),
        ("A+B+C+D", ["Block_A_Norm_Name", standard_block_b_name, standard_block_c_name, "Block_D_Address_Signals"]),
        ("A+B+C+D+E", ["Block_A_Norm_Name", standard_block_b_name, standard_block_c_name, "Block_D_Address_Signals", standard_block_e_name]),
    ]

    cumulative_results = {}
    prev_recall = 0.0

    current_union_cands = defaultdict(set)

    for union_label, block_list in union_progression:
        t0_u = time.time()
        new_block = block_list[-1]
        # Incrementally add new block candidates
        for eid in queries:
            current_union_cands[eid].update(candidate_cache[new_block][eid])
            
        u_time = time.time() - t0_u
        _, u_peak_rss = get_memory_info_mb()
        
        metrics = evaluate_candidate_sets(queries, current_union_cands)
        incremental_recall = round(metrics["overall_recall"] - prev_recall, 2)
        prev_recall = metrics["overall_recall"]
        
        metrics["runtime_sec"] = round(u_time, 2)
        metrics["peak_rss_mb"] = u_peak_rss
        metrics["incremental_recall"] = incremental_recall
        metrics["included_blocks"] = block_list
        
        cumulative_results[union_label] = metrics
        
        print(f"=== UNION: {union_label} ===")
        print(f"  Overall Recall: {metrics['overall_recall']}% (+{incremental_recall}%)")
        print(f"  S2 Recall: {metrics['s2_recall']}% | S3 Recall: {metrics['s3_recall']}%")
        print(f"  Candidates/S1: Avg={metrics['avg_candidates_per_s1']}, Median={metrics['median_candidates']}, P95={metrics['p95_candidates']}, Max={metrics['max_candidates']}")
        print(f"  Reduction Ratio: {metrics['reduction_ratio_pct']}% | Zero-Cand %: {metrics['zero_candidate_pct']}%")
        print(f"  Peak RSS: {u_peak_rss} MB\n", flush=True)

    # -----------------------------------------------------------------
    # STEP 5: PAIRWISE BLOCK OVERLAP MATRIX
    # -----------------------------------------------------------------
    print("=" * 70, flush=True)
    print("STEP 5: PAIRWISE BLOCK OVERLAP MATRIX", flush=True)
    print("=" * 70, flush=True)
    key_blocks = {
        "A": candidate_cache["Block_A_Norm_Name"],
        "B": candidate_cache[standard_block_b_name],
        "C": candidate_cache[standard_block_c_name],
        "D": candidate_cache["Block_D_Address_Signals"],
        "E": candidate_cache[standard_block_e_name],
    }
    overlap_matrix = compute_pairwise_block_overlap(queries, key_blocks)
    for pair, stats in overlap_matrix.items():
        print(f"  {stats['block1']} ∩ {stats['block2']}: Shared={stats['shared_candidates']:,}, Jaccard={stats['jaccard_overlap']}")

    # -----------------------------------------------------------------
    # STEP 6: SCRIPT DIVERGENCE DIAGNOSTIC
    # -----------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("STEP 6: SCRIPT DIVERGENCE DIAGNOSTIC", flush=True)
    print("=" * 70, flush=True)
    script_diag = compute_script_divergence_diagnostic(queries, current_union_cands, s23_gt_names)
    print(f"Total Ground Truth Misses: {script_diag['total_ground_truth_misses']:,}")
    print(f"Misses with Non-Latin Script: {script_diag['misses_with_non_latin_script']:,} ({script_diag['non_latin_miss_pct']}%)")
    for ex in script_diag["script_divergence_examples"][:5]:
        print(f"  Miss Example: S1 '{ex['s1_name']}' -> Target '{ex['target_name']}' ({ex['target_id']}) in {ex['country']}")

    # -----------------------------------------------------------------
    # STEP 7: SAVE REPORTS & RAW JSON
    # -----------------------------------------------------------------
    total_elapsed_time = round(time.time() - start_total_time, 2)
    final_output = {
        "execution_summary": {
            "total_runtime_sec": total_elapsed_time,
            "queries_evaluated": len(queries),
            "target_population_indexed": len(s23_entity_ids),
            "peak_rss_mb": get_memory_info_mb()[1]
        },
        "individual_blocks": benchmark_results,
        "cumulative_unions": cumulative_results,
        "pairwise_overlap": overlap_matrix,
        "script_divergence_diagnostic": script_diag,
        "normalizer_diagnostics": normalizer.diagnostic_samples
    }

    os.makedirs("reports", exist_ok=True)
    json_path = "reports/blocking_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved raw results to {json_path}", flush=True)

    # Generate Markdown Report
    generate_markdown_report(final_output, "reports/blocking_benchmark.md")
    print(f"Saved benchmark report to reports/blocking_benchmark.md", flush=True)


def generate_markdown_report(data: dict, out_path: str):
    ind = data["individual_blocks"]
    cum = data["cumulative_unions"]
    summary = data["execution_summary"]
    diag = data["script_divergence_diagnostic"]

    md = []
    md.append("# Amazon ML Challenge 2026: Business Entity Resolution")
    md.append("## Phase 2 — Candidate Generation & Blocking Benchmark Report\n")
    md.append(f"**Date:** September 2026  ")
    md.append(f"**Total Indexed S2/S3 Target Pool:** {summary['target_population_indexed']:,} records (100% of train S2 and S3)  ")
    md.append(f"**Validation Query Population:** {summary['queries_evaluated']:,} stratified S1 entities  ")
    md.append(f"**Overall Runtime:** {summary['total_runtime_sec']:.2f}s | **Measured Peak Working Set (RSS):** {summary['peak_rss_mb']:.2f} MB  ")
    md.append(f"**Candidate Caps:** **NONE** (Raw blocking candidate sets measured without top-K truncation)\n")
    md.append("---\n")

    md.append("### 1. Primary Result: Cumulative Union Progression\n")
    md.append("This table illustrates the progressive expansion of candidate generation as complementary blocking passes are added into the union:\n")
    md.append("| Pipeline Stage | Total Candidates | Avg / S1 | Median | P95 | P99 | Max | S2 Recall | S3 Recall | Overall Recall | Incremental Recall | Peak RSS |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for u_label, res in cum.items():
        md.append(
            f"| **{u_label}** | {res['total_candidate_pairs']:,} | {res['avg_candidates_per_s1']} | "
            f"{res['median_candidates']} | {res['p95_candidates']} | {res['p99_candidates']} | "
            f"{res['max_candidates']} | {res['s2_recall']}% | {res['s3_recall']}% | "
            f"**{res['overall_recall']}%** | +{res['incremental_recall']}% | {res['peak_rss_mb']} MB |"
        )

    md.append("\n---\n")
    md.append("### 2. Individual Block Performance & Parameter Benchmarking\n")
    md.append("Individual candidate generation performance for each block independently:\n")
    md.append("| Strategy / Block | Config / Parameter | Total Candidates | Avg / S1 | P95 | Max | S2 Recall | S3 Recall | Overall Recall | Runtime | Peak RSS |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for b_name, res in ind.items():
        cfg_str = ", ".join(f"{k}={v}" for k, v in res.get("config", {}).items())
        md.append(
            f"| `{b_name}` | {cfg_str} | {res['total_candidate_pairs']:,} | {res['avg_candidates_per_s1']} | "
            f"{res['p95_candidates']} | {res['max_candidates']} | {res['s2_recall']}% | {res['s3_recall']}% | "
            f"**{res['overall_recall']}%** | {res['runtime_sec']}s | {res['peak_rss_mb']} MB |"
        )

    md.append("\n---\n")
    md.append("### 3. Country-Level Recall Breakdown\n")
    md.append("| Strategy / Union | US Recall (%) | India Recall (%) | Reduction Ratio vs Full Space (%) | Zero-Candidate S1 (%) |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")

    for u_label, res in cum.items():
        c_rec = res.get("country_recall", {})
        md.append(
            f"| **Union {u_label}** | {c_rec.get('US', 0.0)}% | {c_rec.get('India', 0.0)}% | "
            f"{res['reduction_ratio_pct']}% | {res['zero_candidate_pct']}% |"
        )

    md.append("\n---\n")
    md.append("### 4. Pairwise Block Overlap Matrix\n")
    md.append("| Block Pair | Shared Candidates | Jaccard Overlap Index | Interpretation |")
    md.append("| :--- | :--- | :--- | :--- |")

    for pair, stats in data["pairwise_overlap"].items():
        md.append(f"| `{stats['block1']}` $\\cap$ `{stats['block2']}` | {stats['shared_candidates']:,} | {stats['jaccard_overlap']} | Redundancy vs Complementarity |")

    md.append("\n---\n")
    md.append("### 5. Script-Divergence Diagnostic (Indic / Non-Latin Misses)\n")
    md.append(f"- **Total Ground Truth Links Missed by Final Union:** {diag['total_ground_truth_misses']:,}  ")
    md.append(f"- **Misses Containing Non-Latin Script in S2/S3:** {diag['misses_with_non_latin_script']:,} ({diag['non_latin_miss_pct']}%)  \n")
    md.append("#### Sample Ground Truth Misses Due to Native Indic Script:\n")
    for ex in diag["script_divergence_examples"][:8]:
        md.append(f"- **S1:** `{ex['s1_name']}` ({ex['country']}) $\\longleftrightarrow$ **Target:** `{ex['target_name']}` (`{ex['target_id']}`)")

    md.append("\n---\n")
    md.append("### 6. Summary of Findings & Recommended Blocking Configuration\n")
    md.append("1. **Trade-off Analysis:**...")
    md.append("2. **Core Blocking Pipeline:**...")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

if __name__ == "__main__":
    run_phase2_blocking_benchmark()
