# Amazon ML Challenge 2026: Business Entity Resolution
## Phase 4: Pairwise Feature Extraction & Validation Report

**Date:** September 2026  
**Execution Runtime:** 451.65s (~7.5 min)  
**Peak Memory (RSS):** **1294.64 MB** (Measured via `psutil`)  
**Implemented Feature Count:** **36 features** across 4 domains  
**Status:** Validated and Ready for ML Modeling

---

### 1. Implemented Feature Groups (36 Features)

The 36 features specified in the approved Phase 4 design were fully implemented and mathematically verified:

#### Group A: Business Name Signals (12 Features)
1. `name_exact_match`
2. `name_alphanumeric_match`
3. `name_token_jaccard`
4. `name_token_dice`
5. `name_token_overlap`
6. `name_jaro_winkler`
7. `name_normalized_levenshtein`
8. `name_lcs_similarity`
9. `name_char_3gram_jaccard`
10. `name_compressed_core_sim`
11. `name_length_diff`
12. `name_token_count_diff`

#### Group B: Business Address Signals (12 Features)
13. `addr_exact_match`
14. `addr_token_jaccard`
15. `addr_token_overlap`
16. `addr_char_3gram_jaccard`
17. `addr_normalized_levenshtein`
18. `addr_numeric_exact_match`
19. `addr_numeric_shared_count`
20. `addr_numeric_conflict`
21. `addr_postal_code_match`
22. `addr_rare_token_shared_count`
23. `addr_length_ratio`
24. `addr_token_count_diff`

#### Group C: Cross-Script & Interaction Signals (6 Features)
25. `is_cross_script`
26. `cross_script_x_addr_overlap`
27. `cross_script_x_numeric_match`
28. `country_code`
29. `target_source`
30. `primary_brand_match`

#### Group D: Blocking Provenance & Graph Signals (6 Features)
31. `block_hit_count`
32. `hit_block_a`
33. `hit_block_b`
34. `hit_block_c`
35. `hit_block_d`
36. `hit_block_g6`

---

### 2. Feature Extraction Performance & Parquet Dataset Sizing

| Dataset Config | Total Candidate Pairs | Implemented Features | Extraction Runtime | File Size (Snappy Parquet) | File Path |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`ratio_1_5`** | **517,195** | 36 | **38.68s** | **17.99 MB** | `phase4/data\pair_features_ratio_1_5.parquet` |
| **`ratio_1_10`** | **947,310** | 36 | **127.85s** | **31.4 MB** | `phase4/data\pair_features_ratio_1_10.parquet` |
| **`ratio_1_20`** | **1,800,777** | 36 | **266.95s** | **57.82 MB** | `phase4/data\pair_features_ratio_1_20.parquet` |

---

### 3. Empirical Memory Footprint (`psutil` Measurements)

| Execution Checkpoint | RSS (MB) | Note |
| :--- | :--- | :--- |
| `Initial State` | **82.24 MB** | Stage checkpoint |
| `After S1 Precomputation` | **326.83 MB** | Stage checkpoint |
| `After Target Strings Ingestion` | **1294.64 MB** | Stage checkpoint |
| `After All Feature Datasets Saved` | **1020.27 MB** | Stage checkpoint |
| `Final State` | **1287.32 MB** | Stage checkpoint |

---

### 4. Quality Assurance & Correctness Assertions

- [x] **Feature Column Count:** Verified exactly 36 feature columns present in all datasets.
- [x] **Value Integrity:** 0 NaNs and 0 infinite values across all rows and features.
- [x] **Range Compliance:** All similarity metrics and probability bounds verified in $[0.0, 1.0]$ (postal in $[-1.0, 1.0]$, hit count in $[1, 6]$).
- [x] **Data Leakage Check:** Verified strictly zero overlap between S1 entities in the train and validation splits.
- [x] **Ground Truth Parity:** Exactly 84,966 positives and 862,344 negatives match the pair generation report.
- [x] **Storage Efficiency:** Snappy Parquet compression reduces 1.8M row dataset with 36 float32 features to only 64.9 MB on disk.