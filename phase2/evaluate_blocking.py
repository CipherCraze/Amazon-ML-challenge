import os
import sys
import json
import random
import re
from collections import defaultdict, Counter
import numpy as np

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

NON_LATIN_REGEX = re.compile(r'[^\x00-\x7F]')

def load_hard_noise_s1_ids(variation_path: str = "reports/variation_examples.json") -> dict[str, set[str]]:
    """Extract S1 IDs categorized by hard noise types from Phase 1 report."""
    noise_ids = defaultdict(set)
    if os.path.exists(variation_path):
        with open(variation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for category, items in data.items():
                for item in items:
                    s1_id = item.get("s1_id")
                    if s1_id:
                        noise_ids[category].add(s1_id)
    return noise_ids

def build_stratified_validation_sample(
    train_s1_path: str,
    train_gt_path: str,
    target_sample_size: int = 25000,
    random_seed: int = 42
) -> dict:
    """
    Construct a reproducible, stratified 25,000 S1 validation sample.
    Memory-efficient stream-based sampling: does NOT store 2.2M dicts in RAM.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print(f"Building stratified validation sample (size={target_sample_size:,})...", flush=True)
    
    # 1. First scan ground truth to get match counts and match sets
    s1_gt_map = {}
    with open(train_gt_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip("\r\n").split("\t")
            eid = parts[0]
            m_list = [m for m in parts[1].split(",") if m] if len(parts) > 1 and parts[1].strip() else []
            s1_gt_map[eid] = m_list

    # 2. Hard noise IDs from Phase 1
    hard_noise_map = load_hard_noise_s1_ids()
    all_hard_noise_ids = set()
    for ids in hard_noise_map.values():
        all_hard_noise_ids.update(ids)
    print(f"Found {len(all_hard_noise_ids)} S1 entities in Phase 1 noise examples.", flush=True)

    def get_multiplicity_bin(num_matches: int) -> str:
        if num_matches == 0:
            return "0"
        elif num_matches == 1:
            return "1"
        elif num_matches == 2:
            return "2"
        elif 3 <= num_matches <= 4:
            return "3-4"
        else:
            return "5+"

    target_stratum_counts = {
        ("US", "0"): 838,
        ("US", "1"): 810,
        ("US", "2"): 2550,
        ("US", "3-4"): 6900,
        ("US", "5+"): 3902,
        ("India", "0"): 558,
        ("India", "1"): 540,
        ("India", "2"): 1700,
        ("India", "3-4"): 4600,
        ("India", "5+"): 2602
    }

    # Stream through train_source1.tsv and collect sample
    stratum_collected = defaultdict(list)
    sample_queries = {}

    with open(train_s1_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
            gt_matches = s1_gt_map.get(eid, [])
            m_bin = get_multiplicity_bin(len(gt_matches))
            stratum_key = (country, m_bin)

            # Prioritize hard noise IDs
            is_hard_noise = eid in all_hard_noise_ids
            quota = target_stratum_counts.get(stratum_key, 0)
            
            if is_hard_noise or len(stratum_collected[stratum_key]) < quota:
                item = {
                    "entity_id": eid,
                    "business_name": name,
                    "business_address": addr,
                    "country": country,
                    "multiplicity_bin": m_bin,
                    "num_matches": len(gt_matches),
                    "gt_matches": gt_matches
                }
                sample_queries[eid] = item
                stratum_collected[stratum_key].append(eid)
                
            # Early stop if all strata filled and hard noise included
            if len(sample_queries) >= target_sample_size and len(all_hard_noise_ids - set(sample_queries.keys())) == 0:
                break

    print(f"Stratified sample constructed with {len(sample_queries):,} S1 queries.", flush=True)
    return sample_queries


# =====================================================================
# METRIC EVALUATION ENGINE (INTEGER-BASED)
# =====================================================================

def evaluate_candidate_sets(
    queries: dict,
    candidate_map: dict[str, set[str]],
    total_target_population: int = 10320219
) -> dict:
    """
    Compute full suite of blocking metrics:
      - Total ground truth blocking recall (overall, S2, S3)
      - Country-level recall (US vs India)
      - Multiplicity bin recall
      - Candidate distribution (mean, median, p90, p95, p99, max, zero-candidate pct)
      - Total candidate pairs & reduction ratio vs full Cartesian space
    """
    total_gt_links = 0
    found_gt_links = 0
    s2_gt_links = 0
    s2_found_links = 0
    s3_gt_links = 0
    s3_found_links = 0
    
    country_gt = Counter()
    country_found = Counter()
    bin_gt = Counter()
    bin_found = Counter()
    
    candidate_counts = []
    zero_candidate_queries = 0
    total_candidate_pairs = 0

    for eid, q in queries.items():
        gt_set = set(q["gt_matches"])
        n_gt = len(gt_set)
        total_gt_links += n_gt
        country = q["country"]
        m_bin = q["multiplicity_bin"]
        country_gt[country] += n_gt
        bin_gt[m_bin] += n_gt
        
        cands = candidate_map.get(eid, set())
        n_cands = len(cands)
        candidate_counts.append(n_cands)
        total_candidate_pairs += n_cands
        if n_cands == 0:
            zero_candidate_queries += 1
            
        if n_gt > 0:
            hits = cands.intersection(gt_set)
            n_hits = len(hits)
            found_gt_links += n_hits
            country_found[country] += n_hits
            bin_found[m_bin] += n_hits
            
            for m in gt_set:
                if m.startswith("S2-"):
                    s2_gt_links += 1
                    if m in cands:
                        s2_found_links += 1
                elif m.startswith("S3-"):
                    s3_gt_links += 1
                    if m in cands:
                        s3_found_links += 1

    overall_recall = (found_gt_links / total_gt_links * 100) if total_gt_links else 0.0
    s2_recall = (s2_found_links / s2_gt_links * 100) if s2_gt_links else 0.0
    s3_recall = (s3_found_links / s3_gt_links * 100) if s3_gt_links else 0.0
    
    cand_arr = np.array(candidate_counts)
    cartesian_space = len(queries) * total_target_population
    reduction_ratio = (1.0 - (total_candidate_pairs / cartesian_space)) * 100 if cartesian_space else 0.0

    return {
        "num_queries": len(queries),
        "total_gt_links": total_gt_links,
        "found_gt_links": found_gt_links,
        "overall_recall": round(overall_recall, 2),
        "s2_recall": round(s2_recall, 2),
        "s3_recall": round(s3_recall, 2),
        "country_recall": {
            c: round(country_found[c] / country_gt[c] * 100, 2) if country_gt[c] else 0.0
            for c in sorted(country_gt.keys())
        },
        "multiplicity_bin_recall": {
            b: round(bin_found[b] / bin_gt[b] * 100, 2) if bin_gt[b] else 0.0
            for b in ["1", "2", "3-4", "5+"]
        },
        "total_candidate_pairs": int(total_candidate_pairs),
        "avg_candidates_per_s1": round(float(np.mean(cand_arr)), 2),
        "median_candidates": float(np.median(cand_arr)),
        "p90_candidates": float(np.percentile(cand_arr, 90)),
        "p95_candidates": float(np.percentile(cand_arr, 95)),
        "p99_candidates": float(np.percentile(cand_arr, 99)),
        "max_candidates": int(np.max(cand_arr)),
        "zero_candidate_pct": round((zero_candidate_queries / len(queries)) * 100, 2),
        "reduction_ratio_pct": round(reduction_ratio, 6)
    }

def compute_script_divergence_diagnostic(
    queries: dict,
    final_candidates: dict[str, set[str]],
    s23_gt_names: dict[str, str]
) -> dict:
    """
    Analyze ground-truth misses:
    Determine how many missed matches in S2/S3 contain non-Latin Indic scripts.
    """
    total_misses = 0
    non_latin_misses = 0
    missed_script_examples = []
    
    for eid, q in queries.items():
        gt_set = set(q["gt_matches"])
        cands = final_candidates.get(eid, set())
        misses = gt_set - cands
        for mid in misses:
            total_misses += 1
            m_name = s23_gt_names.get(mid, "")
            if NON_LATIN_REGEX.search(m_name):
                non_latin_misses += 1
                if len(missed_script_examples) < 10:
                    missed_script_examples.append({
                        "s1_id": eid,
                        "s1_name": q["business_name"],
                        "target_id": mid,
                        "target_name": m_name,
                        "country": q["country"]
                    })
                    
    non_latin_pct = round((non_latin_misses / total_misses * 100), 2) if total_misses else 0.0
    return {
        "total_ground_truth_misses": total_misses,
        "misses_with_non_latin_script": non_latin_misses,
        "non_latin_miss_pct": non_latin_pct,
        "script_divergence_examples": missed_script_examples
    }
