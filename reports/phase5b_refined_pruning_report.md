# Amazon ML Challenge 2026: Business Entity Resolution
## Phase 5B — Refined Candidate Pruning Report ({B,E} Compound Control)

**Date:** September 2026  
**Execution Runtime:** 2018.38s  
**Peak Memory (RSS):** 10205.64 MB  

---

### 1. Development Population Benchmark (25,000 Dev Queries, 86,570 GT Links)

| Configuration | Candidates/S1 | GT Captured | GT Recall | GT Lost | US Recall | India Recall | Cross-Script Recall | S2 Recall | S3 Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C<=1000 + D<=1000 (No B/E Pruning)** | 4020.36 | 84,868 | **98.03%** | 98 | 99.14% | 96.37% | **91.07%** | 97.9% | 98.16% |
| **C<=1000 + D<=1000 + Top-50 (Phase 5A Leak: B+E Always Retained)** | 702.51 | 84,442 | **97.54%** | 524 | 98.78% | 95.69% | **90.28%** | 97.51% | 97.58% |
| **C<=1000 + D<=1000 + Top-25 (Refined: B, E, {B,E} Pruned)** | 678.09 | 84,347 | **97.43%** | 619 | 98.7% | 95.54% | **90.24%** | 97.4% | 97.46% |
| **C<=1000 + D<=1000 + Top-50 (Refined: B, E, {B,E} Pruned)** | 698.23 | 84,425 | **97.52%** | 541 | 98.77% | 95.66% | **90.28%** | 97.48% | 97.56% |
| **C<=1000 + D<=1000 + Top-75 (Refined: B, E, {B,E} Pruned)** | 717.96 | 84,477 | **97.58%** | 489 | 98.81% | 95.75% | **90.32%** | 97.54% | 97.62% |
| **C<=1000 + D<=1000 + Top-100 (Refined: B, E, {B,E} Pruned)** | 737.34 | 84,514 | **97.63%** | 452 | 98.83% | 95.82% | **90.38%** | 97.57% | 97.67% |

---

### 2. Full Test Dataset Extrapolations (1,732,544 Test S1 Queries)

| Configuration | France Candidates | US Candidates | India Candidates | Total Test Candidates | Candidates / S1 | Est. TSV Size | Est. Validator RAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C<=1000 + D<=1000 (No B/E Pruning)** | 972.7 M | 2592.5 M | 2136.6 M | **5701.7 M** | **3291.0** | **69.05 GB** | **393.45 GB** |
| **C<=1000 + D<=1000 + Top-50 (Phase 5A Leak: B+E Always Retained)** | 560.9 M | 224.7 M | 1007.8 M | **1793.4 M** | **1035.2** | **21.73 GB** | **124.1 GB** |
| **C<=1000 + D<=1000 + Top-25 (Refined: B, E, {B,E} Pruned)** | 540.9 M | 208.7 M | 993.1 M | **1742.8 M** | **1005.9** | **21.12 GB** | **120.61 GB** |
| **C<=1000 + D<=1000 + Top-50 (Refined: B, E, {B,E} Pruned)** | 545.7 M | 222.8 M | 1004.5 M | **1773.0 M** | **1023.4** | **21.49 GB** | **122.69 GB** |
| **C<=1000 + D<=1000 + Top-75 (Refined: B, E, {B,E} Pruned)** | 550.2 M | 236.7 M | 1015.2 M | **1802.1 M** | **1040.2** | **21.84 GB** | **124.7 GB** |
| **C<=1000 + D<=1000 + Top-100 (Refined: B, E, {B,E} Pruned)** | 554.4 M | 250.5 M | 1025.4 M | **1830.3 M** | **1056.4** | **22.18 GB** | **126.64 GB** |

---

### 3. Core Architectural Takeaways

1. **Compound {B,E} Pruning Impact:**
   - Subjecting compound {B,E} hits to Top-50 cheap ranking retains **97.52% ground truth recall** on the development queries, losing only 17 additional links out of 86,570 compared to the unpruned compound leak.
   - Cross-script recall is completely preserved at **90.28%**, and US recall is **98.77%**, India recall is **95.66%**.
   - On the full test population, candidates drop from 5.70 Billion down to **1.77 Billion candidate pairs** (1,023.4 cands/S1).

2. **The Remaining Validator Memory Barrier:**
   - At 1.77 Billion candidate pairs, `candidate_pairs.tsv` is **21.49 GB** and estimated validator RAM is **122.69 GB**, which still significantly exceeds the **24 GB host RAM ceiling**.
   - In US, candidate density is well controlled at **336.0 cands/S1**.
   - However, France (**2,103.1 cands/S1**) and India (**1,240.2 cands/S1**) remain elevated.

3. **Root Cause Diagnosis — The G6 Address-Pair Inflation:**
   - Because `always_retained` (A, C, D, G6) alone averages **~2,053 cands/S1 in France** and **~1,190 cands/S1 in India**, the remaining candidate volume is NOT from name fuzzy collisions (B/E).
   - In France and India, dense address localities and high G6 posting cap ($DF \le 2500$) produce large candidate sets where 1-2 address pairs return up to 2,500 records each.
   - To bring the candidate pool into the memory limits of the 24 GB validator host (<150M pairs, <10 GB RAM), the frequency cap on G6 address pairs must be tightened (e.g., $DF \le 100-250$) or G6 candidates must also be ranked/capped when an address key is shared by hundreds of businesses on the same street or postal locality.
