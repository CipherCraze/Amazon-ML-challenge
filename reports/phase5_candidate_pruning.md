# Amazon ML Challenge 2026: Business Entity Resolution
## Phase 5A — Test Candidate Pruning Benchmark Report

**Date:** September 2026  
**Execution Runtime:** 1654.4s  
**Peak Memory (RSS):** 12046.91 MB  
**Cheap Ranking Formula:** `score = 2.0 * name_jaccard + 1.0 * min(inter_name, 3) + 1.0 * min(inter_addr, 3) + 1.5 * min(inter_nums, 2)`  

---

### 1. Development Population Pruning Benchmark (25,000 S1 Queries, 86,570 Total GT Links)

| Strategy / Cap | Candidates/S1 | Candidate Retention | GT Retention | Absolute GT Lost | GT Recall | US Recall | India Recall | Cross-Script Recall | Same-Script Recall | S2 Recall | S3 Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Raw Frozen Blocker)** | 4600.52 | 100.0% | 100.0% | 0 | **98.15%** | 99.31% | 96.4% | **91.12%** | 99.23% | 97.99% | 98.29% |
| **Top-25** | 1299.99 | 28.26% | 99.4374% | 478 | **97.6%** | 98.91% | 95.63% | **90.29%** | 98.72% | 97.54% | 97.65% |
| **Top-50** | 1320.09 | 28.69% | 99.5198% | 408 | **97.68%** | 98.97% | 95.73% | **90.33%** | 98.8% | 97.61% | 97.74% |
| **Top-75** | 1339.79 | 29.12% | 99.5798% | 357 | **97.73%** | 99.02% | 95.82% | **90.37%** | 98.86% | 97.66% | 97.8% |
| **Top-100** | 1359.13 | 29.54% | 99.6128% | 329 | **97.77%** | 99.04% | 95.87% | **90.42%** | 98.89% | 97.69% | 97.84% |

---

### 2. Candidate Distribution Metrics on Development Queries

| Strategy / Cap | Mean | Median | P95 | P99 | Max Candidates |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Raw Frozen Blocker)** | 4600.52 | 2457.0 | 12910.3 | 20930.2 | 44,638 |
| **Top-25** | 1299.99 | 373.0 | 3901.1 | 19351.1 | 41,453 |
| **Top-50** | 1320.09 | 392.0 | 3926.1 | 19363.0 | 41,453 |
| **Top-75** | 1339.79 | 412.0 | 3943.0 | 19363.0 | 41,453 |
| **Top-100** | 1359.13 | 433.0 | 3953.2 | 19363.0 | 41,453 |

---

### 3. Test Sample Candidate Statistics (1,000 France, 1,000 US, 1,000 India Queries)

| Strategy / Cap | France Avg (Max) | US Avg (Max) | India Avg (Max) |
| :--- | :---: | :---: | :---: |
| **Raw Blocker** | 5750.3 (30,456) | 4507.8 (24,624) | 2883.4 (18,307) |
| **Top-25** | 4372.8 (30,456) | 939.4 (24,624) | 1490.0 (13,316) |
| **Top-50** | 4390.7 (30,456) | 960.6 (24,624) | 1504.1 (13,316) |
| **Top-75** | 4407.6 (30,456) | 981.7 (24,624) | 1517.3 (13,316) |
| **Top-100** | 4423.2 (30,456) | 1002.4 (24,624) | 1529.8 (13,316) |

---

### 4. Full Test Population Extrapolations (1,732,544 Test S1 Queries)

| Strategy / Cap | France Candidates | US Candidates | India Candidates | Total Test Candidate Pairs | Avg / S1 | Est. TSV Size | Est. Validator RAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Blocker** | 1491.9M | 2989.1M | 2335.5M | **6816.6M** | 3934.4 | 82.55 GB | **470.28 GB** |
| **Top-25** | 1134.5M | 622.9M | 1206.9M | **2964.3M** | 1711.0 | 35.91 GB | **204.8 GB** |
| **Top-50** | 1139.2M | 637.0M | 1218.3M | **2994.5M** | 1728.4 | 36.27 GB | **206.87 GB** |
| **Top-75** | 1143.6M | 651.0M | 1229.0M | **3023.5M** | 1745.1 | 36.63 GB | **208.87 GB** |
| **Top-100** | 1147.6M | 664.7M | 1239.1M | **3051.4M** | 1761.2 | 36.96 GB | **210.8 GB** |

---

### 5. Architectural Findings & Critical Discovery

1. **Always-Retain Rule Captures 97.84% of Ground Truth Alone:**
   In the 25,000 development queries, 83,133 out of 84,966 captured ground truth links (97.84%) have multi-block provenance ($hits \ge 2$) or belong to blocks $A$, $C$, $D$, or $G_6$. Single-hit Block B and E represent only 2.15% (1,833 links) of ground truth matches.
2. **Top-50 / Top-100 Pruning Retains > 99.5% of Captured Ground Truth on Development Set:**
   On the 25,000 development queries, ranking single-hit B/E candidates by cheap token and numeric overlap retains **99.52% (Top-50)** to **99.61% (Top-100)** of ground truth matches. Overall blocking recall drops negligibly from 98.15% to **97.68% (Top-50)** or **97.77% (Top-100)**. Cross-script recall is largely preserved (90.33% vs 91.12%).
3. **Critical Discovery — Validator Memory Ceiling Incompatibility Remains (> 200 GB RAM):**
   While single-hit B/E capping reduces candidate volume by ~56% (from 6.82B to 2.99B pairs), the **Always-Retained pool alone** remains massive on the test set:
   - In France, test queries average **4,372 candidates / S1** even with Top-25 capping, because unpruned Block C (Core-12) prefixes (e.g., `ecoleprimaire...`) and unpruned Block D regional blocks (e.g., `FRANCE_<token>`) generate thousands of targets per query.
   - In India, test queries average **1,490 candidates / S1** in the Always-Retained pool.
   - Consequently, the full test set under Top-50 pruning still yields **2.99 Billion candidate pairs** (~36.27 GB on disk).
   - When `validate_submission.py` loads `candidate_pairs.tsv` line-by-line into Python sets (`mapping[s1] = set(ids)`), 2.99 billion strings in sets requires **$\approx 206.87$ GB of RAM**, which will exceed the 24 GB host memory and trigger a fatal `MemoryError`.
4. **Concrete Recommended Solution:**
   To bring the test candidate pool down to a safe size ($< 150\text{M}$ pairs, $< 10\text{ GB}$ validator RAM):
   - Apply a doc-frequency cap on Block C (e.g., `df <= 1000` or `len(core12_targets) <= 500`) and Block D (e.g., `df <= 1000` or skip generic country tokens like "FRANCE"), OR
   - Apply a global per-S1 candidate ceiling of $K=100$ or $K=150$ across ALL candidates after ranking them by block hit count and cheap similarity.
