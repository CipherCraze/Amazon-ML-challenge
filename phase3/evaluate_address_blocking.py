import numpy as np
from collections import Counter

def evaluate_address_block(
    queries: dict,
    candidate_dict: dict[str, set[int]],
    baseline_union_cands: dict[str, set[int]],
    non_latin_miss_ints: dict[str, set[int]],
    s2_split_idx: int = 5034616,
    total_target_population: int = 10320219
) -> dict:
    """
    Comprehensive evaluation of an address blocking strategy:
      - Independent metrics (candidates, recall, country breakdown)
      - Incremental recall over baseline union A+B+C+D+E
      - Recovery of the 5,923 non-Latin misses
    """
    total_gt = 0
    found_gt = 0
    s2_gt = 0
    s2_found = 0
    s3_gt = 0
    s3_found = 0

    country_gt = Counter()
    country_found = Counter()

    cand_counts = []
    zero_cands = 0
    total_cands = 0

    # Non-Latin recovery
    total_non_latin = 0
    recovered_non_latin = 0

    # Incremental recall over baseline
    baseline_found_gt = 0
    union_with_g_found_gt = 0
    country_union_with_g_found = Counter()
    total_baseline_cands = 0
    total_union_with_g_cands = 0

    for eid, q in queries.items():
        cands = candidate_dict.get(eid, set())
        n_cands = len(cands)
        cand_counts.append(n_cands)
        total_cands += n_cands
        if n_cands == 0:
            zero_cands += 1

        c = q["country"]
        gt_ints = q["gt_ints"]
        n_gt = len(gt_ints)
        total_gt += n_gt
        country_gt[c] += n_gt

        # Baseline union candidates
        b_cands = baseline_union_cands.get(eid, set())
        total_baseline_cands += len(b_cands)
        union_g_cands = b_cands | cands
        total_union_with_g_cands += len(union_g_cands)

        if n_gt > 0:
            hits = cands & gt_ints
            found_gt += len(hits)
            country_found[c] += len(hits)

            baseline_hits = b_cands & gt_ints
            baseline_found_gt += len(baseline_hits)

            union_g_hits = union_g_cands & gt_ints
            union_with_g_found_gt += len(union_g_hits)
            country_union_with_g_found[c] += len(union_g_hits)

            for m_int in gt_ints:
                if m_int < s2_split_idx:
                    s2_gt += 1
                    if m_int in cands:
                        s2_found += 1
                else:
                    s3_gt += 1
                    if m_int in cands:
                        s3_found += 1

        # Non-Latin misses for this query
        nl_misses = non_latin_miss_ints.get(eid, set())
        if nl_misses:
            total_non_latin += len(nl_misses)
            recovered_non_latin += len(cands & nl_misses)

    overall_recall = (found_gt / total_gt * 100) if total_gt else 0.0
    s2_recall = (s2_found / s2_gt * 100) if s2_gt else 0.0
    s3_recall = (s3_found / s3_gt * 100) if s3_gt else 0.0

    baseline_recall = (baseline_found_gt / total_gt * 100) if total_gt else 0.0
    union_with_g_recall = (union_with_g_found_gt / total_gt * 100) if total_gt else 0.0
    additional_recall = round(union_with_g_recall - baseline_recall, 2)

    cum_country_recall = {
        c: round(country_union_with_g_found[c] / country_gt[c] * 100, 2) if country_gt[c] else 0.0
        for c in sorted(country_gt.keys())
    }

    net_new_cands = total_union_with_g_cands - total_baseline_cands
    net_new_cands_pct = round((net_new_cands / total_baseline_cands) * 100, 2) if total_baseline_cands else 0.0

    nl_rec_pct = (recovered_non_latin / total_non_latin * 100) if total_non_latin else 0.0

    cand_arr = np.array(cand_counts, dtype=np.uint32)
    cartesian_space = len(queries) * total_target_population
    reduction_ratio = (1.0 - (total_cands / cartesian_space)) * 100 if cartesian_space else 0.0

    return {
        "num_queries": len(queries),
        "total_gt_links": total_gt,
        "found_gt_links": found_gt,
        "overall_recall": round(overall_recall, 2),
        "s2_recall": round(s2_recall, 2),
        "s3_recall": round(s3_recall, 2),
        "country_recall": {
            c: round(country_found[c] / country_gt[c] * 100, 2) if country_gt[c] else 0.0
            for c in sorted(country_gt.keys())
        },
        "total_candidate_pairs": int(total_cands),
        "avg_candidates_per_s1": round(float(np.mean(cand_arr)), 2),
        "median_candidates": float(np.median(cand_arr)),
        "p90_candidates": float(np.percentile(cand_arr, 90)),
        "p95_candidates": float(np.percentile(cand_arr, 95)),
        "p99_candidates": float(np.percentile(cand_arr, 99)),
        "max_candidates": int(np.max(cand_arr)),
        "zero_candidate_pct": round((zero_cands / len(queries)) * 100, 2),
        "reduction_ratio_pct": round(reduction_ratio, 6),
        "additional_recall_over_baseline": additional_recall,
        "cumulative_union_with_g_recall": round(union_with_g_recall, 2),
        "cumulative_country_recall": cum_country_recall,
        "cumulative_total_candidates": int(total_union_with_g_cands),
        "net_new_candidates_added": int(net_new_cands),
        "candidate_increase_pct": net_new_cands_pct,
        "total_non_latin_misses_evaluated": total_non_latin,
        "non_latin_misses_recovered": recovered_non_latin,
        "non_latin_recovery_pct": round(nl_rec_pct, 2)
    }
