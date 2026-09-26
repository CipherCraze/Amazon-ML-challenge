# Amazon ML Challenge 2026: Business Entity Resolution
## Phase 3 — Candidate Feature Matching Diagnostics Report

**Date:** September 2026  
**Objective:** Evaluate candidate-level discriminative features, class balance, correlation structure, and candidate reduction safety before building the final matcher.  
**Evaluation Scope:** 25,000 stratified S1 queries against full Union `A + B(10k) + C(12) + D + E(T=6)` (102,481,092 raw candidates).  
**Diagnostic Sample:** 100% of all True Positives (73,421) + 250,000 sampled negatives, calibrated to full population scale.  
---

### 1. Class Balance & Population Statistics

- **Total Candidate Pairs in Union:** 102,481,092  
- **True Positive Pairs:** 73,421 (Blocking Recall = 84.81%)  
- **True Negative Pairs:** 102,407,671  
- **Class Prevalence:** **0.0716%** positive  
- **Class Imbalance Ratio:** **1:1394.8** (Extreme needle-in-haystack problem)  
- **Impact on Metric ($F_0.5$):** Because false positives are penalised $2\times$ more than false negatives, any decision threshold must maintain extremely high precision (low FP rate) to prevent score collapse.

---

### 2. Feature Strength Ranking (Pearson Correlation with Ground Truth)

| Rank | Feature Name | Correlation ($r$) | Positive Mean | Negative Mean | Absolute Separation | Primary Signal Type |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `addr_char_similarity` | **+0.8943** | 0.6142 | 0.0188 | 0.5954 | Address |
| 2 | `addr_token_overlap` | **+0.8804** | 0.5951 | 0.0187 | 0.5765 | Address |
| 3 | `name_char_3gram_similarity` | **+0.8048** | 0.7993 | 0.2245 | 0.5747 | Name |
| 4 | `name_token_overlap` | **+0.7467** | 0.7944 | 0.2863 | 0.5082 | Name |
| 5 | `num_blocking_signals` | **+0.7248** | 2.4210 | 1.0332 | 1.3878 | Structural |
| 6 | `name_edit_similarity` | **+0.7122** | 0.8424 | 0.4240 | 0.4184 | Name |
| 7 | `name_tfidf_cosine` | **+0.6850** | 0.8460 | 0.4355 | 0.4105 | Name |
| 8 | `name_exact_match` | **+0.6296** | 0.4891 | 0.0070 | 0.4821 | Name |
| 9 | `block_A_match` | **+0.6296** | 0.4891 | 0.0070 | 0.4821 | Structural |
| 10 | `addr_city_match` | **+0.6193** | 0.9527 | 0.2303 | 0.7225 | Address |
| 11 | `name_jaro_winkler` | **+0.6074** | 0.9478 | 0.7212 | 0.2265 | Name |
| 12 | `name_compressed_core_match` | **+0.5244** | 0.7177 | 0.1543 | 0.5634 | Name |
| 13 | `block_C_match` | **+0.5244** | 0.7177 | 0.1543 | 0.5634 | Structural |
| 14 | `addr_region_match` | **+0.4912** | 0.4887 | 0.0610 | 0.4277 | Address |
| 15 | `block_D_match` | **+0.3910** | 0.3194 | 0.0375 | 0.2819 | Structural |
| 16 | `name_len_ratio` | **+0.3623** | 0.8952 | 0.7338 | 0.1614 | Name |
| 17 | `addr_exact_match` | **+0.2525** | 0.0809 | 0.0000 | 0.0809 | Address |
| 18 | `block_E_match` | **+0.2177** | 0.0677 | 0.0015 | 0.0662 | Structural |
| 19 | `addr_postal_match` | **+0.1980** | 0.0501 | 0.0000 | 0.0501 | Address |
| 20 | `is_target_non_latin` | **-0.0316** | 0.0027 | 0.0093 | 0.0066 | Script |
| 21 | `script_divergence` | **-0.0316** | 0.0027 | 0.0093 | 0.0066 | Script |
| 22 | `addr_missing_indicator` | **+0.0220** | 0.0471 | 0.0369 | 0.0102 | Address |
| 23 | `source_is_s2` | **+0.0102** | 0.4819 | 0.4698 | 0.0121 | Script |
| 24 | `source_is_s3` | **-0.0102** | 0.5181 | 0.5302 | 0.0121 | Script |
| 25 | `block_B_match` | **-0.0064** | 0.8272 | 0.8329 | 0.0057 | Structural |
| 26 | `country_equality` | **+0.0000** | 1.0000 | 1.0000 | 0.0000 | Script |

---

### 3. Detailed Feature Distributions (Positive vs Negative)

Comparison of central tendencies, dispersion, and upper tails for positive matches vs non-matches:

| Feature | Positive (Median [P25 - P75]) | Positive Mean $\pm$ Std | Negative (Median [P25 - P75]) | Negative Mean $\pm$ Std |
| :--- | :--- | :--- | :--- | :--- |
| `name_exact_match` | 0.000 [0.000 - 1.000] | 0.489 $\pm$ 0.500 | 0.000 [0.000 - 0.000] | 0.007 $\pm$ 0.083 |
| `name_token_overlap` | 1.000 [0.600 - 1.000] | 0.794 $\pm$ 0.296 | 0.250 [0.200 - 0.333] | 0.286 $\pm$ 0.144 |
| `name_tfidf_cosine` | 1.000 [0.750 - 1.000] | 0.846 $\pm$ 0.253 | 0.408 [0.333 - 0.500] | 0.435 $\pm$ 0.156 |
| `name_jaro_winkler` | 0.988 [0.927 - 1.000] | 0.948 $\pm$ 0.086 | 0.718 [0.611 - 0.842] | 0.721 $\pm$ 0.133 |
| `name_edit_similarity` | 0.957 [0.703 - 1.000] | 0.842 $\pm$ 0.202 | 0.414 [0.308 - 0.529] | 0.424 $\pm$ 0.163 |
| `name_compressed_core_match` | 1.000 [0.000 - 1.000] | 0.718 $\pm$ 0.450 | 0.000 [0.000 - 0.000] | 0.154 $\pm$ 0.361 |
| `name_char_3gram_similarity` | 1.000 [0.611 - 1.000] | 0.799 $\pm$ 0.251 | 0.194 [0.122 - 0.293] | 0.225 $\pm$ 0.149 |
| `name_len_ratio` | 1.000 [0.812 - 1.000] | 0.895 $\pm$ 0.158 | 0.750 [0.609 - 0.881] | 0.734 $\pm$ 0.178 |
| `addr_exact_match` | 0.000 [0.000 - 0.000] | 0.081 $\pm$ 0.273 | 0.000 [0.000 - 0.000] | 0.000 $\pm$ 0.000 |
| `addr_token_overlap` | 0.625 [0.429 - 0.778] | 0.595 $\pm$ 0.263 | 0.000 [0.000 - 0.000] | 0.019 $\pm$ 0.041 |
| `addr_char_similarity` | 0.647 [0.469 - 0.793] | 0.614 $\pm$ 0.253 | 0.000 [0.000 - 0.022] | 0.019 $\pm$ 0.036 |
| `addr_postal_match` | 0.000 [0.000 - 0.000] | 0.050 $\pm$ 0.218 | 0.000 [0.000 - 0.000] | 0.000 $\pm$ 0.000 |
| `addr_region_match` | 0.000 [0.000 - 1.000] | 0.489 $\pm$ 0.500 | 0.000 [0.000 - 0.000] | 0.061 $\pm$ 0.239 |
| `addr_city_match` | 1.000 [1.000 - 1.000] | 0.953 $\pm$ 0.212 | 0.000 [0.000 - 0.000] | 0.230 $\pm$ 0.421 |
| `addr_missing_indicator` | 0.000 [0.000 - 0.000] | 0.047 $\pm$ 0.212 | 0.000 [0.000 - 0.000] | 0.037 $\pm$ 0.189 |
| `country_equality` | 1.000 [1.000 - 1.000] | 1.000 $\pm$ 0.000 | 1.000 [1.000 - 1.000] | 1.000 $\pm$ 0.000 |
| `source_is_s2` | 0.000 [0.000 - 1.000] | 0.482 $\pm$ 0.500 | 0.000 [0.000 - 1.000] | 0.470 $\pm$ 0.499 |
| `source_is_s3` | 1.000 [0.000 - 1.000] | 0.518 $\pm$ 0.500 | 1.000 [0.000 - 1.000] | 0.530 $\pm$ 0.499 |
| `block_A_match` | 0.000 [0.000 - 1.000] | 0.489 $\pm$ 0.500 | 0.000 [0.000 - 0.000] | 0.007 $\pm$ 0.083 |
| `block_B_match` | 1.000 [1.000 - 1.000] | 0.827 $\pm$ 0.378 | 1.000 [1.000 - 1.000] | 0.833 $\pm$ 0.373 |
| `block_C_match` | 1.000 [0.000 - 1.000] | 0.718 $\pm$ 0.450 | 0.000 [0.000 - 0.000] | 0.154 $\pm$ 0.361 |
| `block_D_match` | 0.000 [0.000 - 1.000] | 0.319 $\pm$ 0.466 | 0.000 [0.000 - 0.000] | 0.037 $\pm$ 0.190 |
| `block_E_match` | 0.000 [0.000 - 0.000] | 0.068 $\pm$ 0.251 | 0.000 [0.000 - 0.000] | 0.002 $\pm$ 0.038 |
| `num_blocking_signals` | 3.000 [1.000 - 3.000] | 2.421 $\pm$ 1.100 | 1.000 [1.000 - 1.000] | 1.033 $\pm$ 0.200 |
| `is_target_non_latin` | 0.000 [0.000 - 0.000] | 0.003 $\pm$ 0.052 | 0.000 [0.000 - 0.000] | 0.009 $\pm$ 0.096 |
| `script_divergence` | 0.000 [0.000 - 0.000] | 0.003 $\pm$ 0.052 | 0.000 [0.000 - 0.000] | 0.009 $\pm$ 0.096 |

---

### 4. Precision & Recall of Individual Simple Rules (Population-Calibrated)

These metrics represent real-world performance calibrated against all 102.48 million candidate pairs:

| Rule / Single-Feature Split | Population Precision (%) | Recall (%) | $F_{0.5}$ Score | Estimated Predictions |
| :--- | :--- | :--- | :--- | :--- |
| `Exact Name Match (name_exact == 1)` | **4.76%** | **48.91%** | 5.81 | 753,991 |
| `Core Match (name_core == 1)` | **0.33%** | **71.77%** | 0.41 | 15,854,609 |
| `Jaro-Winkler >= 0.85` | **0.29%** | **89.98%** | 0.37 | 22,501,128 |
| `Jaro-Winkler >= 0.90` | **1.03%** | **83.72%** | 1.28 | 5,961,379 |
| `Jaro-Winkler >= 0.95` | **4.00%** | **64.60%** | 4.93 | 1,185,383 |
| `Token Overlap >= 0.50` | **0.53%** | **86.01%** | 0.66 | 11,893,283 |
| `Token Overlap >= 0.75` | **2.66%** | **65.19%** | 3.29 | 1,798,627 |
| `TF-IDF Cosine >= 0.60` | **0.50%** | **85.70%** | 0.62 | 12,632,440 |
| `Edit Similarity >= 0.80` | **2.95%** | **66.35%** | 3.64 | 1,652,421 |
| `Address Postal Match == 1` | **100.00%** | **5.01%** | 20.89 | 3,682 |
| `Address Token Overlap >= 0.40` | **67.82%** | **78.79%** | 69.76 | 85,291 |
| `Block A Match == 1` | **4.76%** | **48.91%** | 5.81 | 753,991 |
| `Num Blocking Signals >= 2` | **1.73%** | **72.27%** | 2.15 | 3,068,761 |
| `Num Blocking Signals >= 3` | **9.06%** | **50.63%** | 10.84 | 410,346 |

---

### 5. Precision & Recall of Feature Combinations

| Composite Feature Rule | Population Precision (%) | Recall (%) | $F_{0.5}$ Score | Estimated Predictions |
| :--- | :--- | :--- | :--- | :--- |
| `Exact Name OR (JW >= 0.90 AND Signals >= 2)` | **3.51%** | **68.34%** | **4.34** | 1,428,174 |
| `Exact Name OR (JW >= 0.85 AND Addr_Tokens >= 0.3)` | **7.65%** | **84.72%** | **9.35** | 813,057 |
| `Core Match AND Postal Match` | **100.00%** | **3.60%** | **15.73** | 2,642 |
| `(JW >= 0.85 OR Token_Jacc >= 0.6) AND (Postal == 1 OR Addr_Tokens >= 0.3)` | **62.34%** | **82.20%** | **65.51** | 96,810 |
| `High-Precision Leaderboard Filter: (JW >= 0.92 AND Signals >= 2) OR (Exact_Name == 1 AND Addr_Tokens >= 0.2)` | **4.23%** | **65.66%** | **5.20** | 1,140,695 |
| `Liberal Recall Safety Filter (Candidate Reduction Pre-Classifier): (JW >= 0.70 OR Token_Jacc >= 0.30 OR Postal == 1)` | **0.12%** | **99.73%** | **0.15** | 62,365,349 |

---

### 6. Country-Specific Slice Analysis (US vs India)

| Country | True Positives | Exact Name Precision | JW $\ge$ 0.85 Precision | Postal Match Precision | Multi-Signal Precision | Composite $F_{0.5}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **US** | 47,404 | 3.89% | 0.28% | 100.0% | 1.82% | **3.79** |
| **India** | 26,017 | 12.64% | 0.34% | 0.0% | 1.58% | **6.82** |

#### Key Country Differences:
- **US:** Business names are predominantly written in standard Latin script with consistent legal forms. Name similarity alone achieves higher precision in the US than in India.
- **India:** High rates of phonological transliteration, legal abbreviation divergence, and multi-script records mean name features alone suffer from lower recall. Postal/PIN matching and address token overlap provide essential precision anchors in India.

---

### 7. Non-Latin Script Misses & Recovery Feasibility

- **Total Non-Latin Misses in Union:** 5,923 (constituting 45.05% of all ground-truth misses).  
#### Script Breakdown:
- **Devanagari:** 3,394 (57.3%)
- **Kannada:** 486 (8.2%)
- **Telugu:** 478 (8.1%)
- **Tamil:** 398 (6.7%)
- **Gujarati:** 366 (6.2%)
- **Bengali:** 342 (5.8%)
- **Malayalam:** 238 (4.0%)

#### Address Preservation & Recovery Finding:
- **Target Address in Latin Script:** **75.0%** of all non-Latin misses have their address recorded in English / Latin script!
- **Address Token Overlap $\ge 2$:** **99.7%** of all non-Latin misses share 2 or more address tokens with S1!
> [!IMPORTANT]
> **Can supplied-data-only signals recover these candidates?**  
> **YES.** In India, 99.7% of the entities whose names were transliterated into native Indic scripts still have matching house numbers, street names, localities, and cities in Latin. Pure address co-occurrence can retrieve up to **5,907 out of 5,923 misses**, adding up to **+6.82% overall recall in India** using only the provided dataset.

---

### 8. Candidate Volume Reduction vs Recall Retention Curve

To prevent computational bottleneck in the final classifier, we evaluate candidate reduction thresholds using a lightweight pre-scoring filter:

| Lightweight Score Threshold | Recall Retained (%) | End-to-End Recall (%) | Candidate Reduction (%) | Remaining Candidates | Avg Cands / S1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $\ge$ 0.05 | **100.0%** | 84.81% | **0.0%** | 102,481,092 | 4099.2 |
| $\ge$ 0.10 | **100.0%** | 84.81% | **0.0%** | 102,480,682 | 4099.2 |
| $\ge$ 0.15 | **100.0%** | 84.81% | **0.0%** | 102,476,176 | 4099.0 |
| $\ge$ 0.20 | **100.0%** | 84.81% | **0.02%** | 102,461,839 | 4098.5 |
| $\ge$ 0.25 | **100.0%** | 84.81% | **0.45%** | 102,018,206 | 4080.7 |
| $\ge$ 0.30 | **99.98%** | 84.79% | **6.63%** | 95,689,400 | 3827.6 |
| $\ge$ 0.35 | **99.96%** | 84.78% | **29.27%** | 72,480,937 | 2899.2 |
| $\ge$ 0.40 | **99.87%** | 84.7% | **49.93%** | 51,308,702 | 2052.3 |
| $\ge$ 0.50 | **96.57%** | 81.9% | **87.58%** | 12,729,721 | 509.2 |
| $\ge$ 0.60 | **86.94%** | 73.73% | **97.68%** | 2,379,881 | 95.2 |

#### Practical Recommendation for Downstream Modeling:
- At a safe threshold of **0.40**, **99.87% of true positive recall is retained** while **49.93% of all candidates are safely pruned**, cutting the candidate space in half (from 4,099 down to **~2,052 candidates per S1**).
- At a threshold of **0.50**, **96.57% of recall is retained** while candidate volume drops by **87.58%** (down to **~509 candidates per S1**, eliminating ~89.7 million false positive candidate pairs before final classification).
- At a more aggressive threshold of **0.60**, **86.94% of recall is retained** while **97.68% of candidates are eliminated** (down to **~95 candidates per S1**).