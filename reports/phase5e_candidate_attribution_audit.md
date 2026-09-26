# Amazon ML Challenge 2026: Business Entity Resolution
## Phase 5E: Complete Candidate Attribution Audit & Efficiency Analysis of the Strategy-11 Pipeline

**Date:** September 2026  
**Evaluation Scope:** 25,000 Stratified Development Queries (86,570 Authoritative Ground Truth Links)  
**Search Space:** 10,320,219 Records (Train Source 2 & Source 3)  
**Hardware Constraint:** 24 GB RAM local ceiling  
**Validated Downstream Matcher:** Phase 4 LightGBM + XGBoost 50/50 blend at $\tau = 0.600$ (Held-Out Macro $F_{0.5} = 0.9624$)  

---

### 1. Executive Summary

Phase 5D successfully solved the single-key address collision problem by establishing **G6 Hybrid-250** ($\ge 2$ shared G6 keys OR $DF \le 250$ for single keys), reducing G6 candidate density from 505.7 to 166.3 cands/S1 while preserving downstream Macro $F_{0.5} = 0.9624$.

Phase 5E conducted a complete, empirical **Candidate Attribution and Efficiency Audit** on all individual components of the recommended Strategy-11 pipeline across the full 25,000 development queries:
$$\mathcal{C}_{\text{final}} = \mathcal{A} \cup \mathcal{C}_{\le 1000} \cup \mathcal{D}_{\le 1000} \cup \mathcal{G}_{\text{hybrid-250}} \cup \text{Top-50}(\{B, E\} \setminus \text{anchors})$$

#### Key Findings of the Attribution Audit:
1. **Actual Pipeline Density is 358.98 cands/S1 (Not 935.9):**  
   The Phase 5D diagnostic reported ~935.9 cands/S1 because it tested G6 against raw unconstrained Block C and Block D anchors without B/E pruning. When the fully validated caps ($C \le 1000$, $D \le 1000$, Refined $\{B,E\}$ Top-50) are simultaneously accounted for, the **true Strategy-11 union generates 358.98 candidates/S1** (8,974,593 total pairs across 25k queries, median 259.0, P95 1,000.0) with **97.29% ground-truth recall** (84,224 / 86,570 links).
2. **G6 Hybrid-250 is the Backbone Anchor (Not a Pruning Target):**  
   G6 Hybrid-250 contributes 45.60% of union candidates (166.28 cands/S1). However, it recovers **76,146 ground-truth links** (87.96% recall) and **25,098 unique GT links** that no other block can recover. Its efficiency is **163.1 candidates per unique GT link recovered**. It is irreplaceable.
3. **Block D ($DF \le 1000$) is the Dominant Bottleneck & Worst Efficiency Block:**  
   Block D contributes **2,386,784 candidates** (95.47 cands/S1, 25.96% of the entire union pool). However, 97.60% of its candidates are completely unanchored noise that overlap with no other block, and it recovers **only 276 unique GT links** (0.3188% of GT links).  
   Its marginal efficiency is an abysmal **8,440.4 candidates per unique GT link recovered** (marginal precision = **0.0118%**).
4. **The Root Cause of Block D Noise:**  
   Component decomposition reveals that `Postal_Code + First_Token` is hyper-precise (3,287 candidates, 0.13 cands/S1, 97.05% precision). In contrast, **`Region + First_Token` generates 2,384,756 candidates (99.91% of Block D)**. Region tokens (e.g. states, territories) paired with common first-name tokens produce massive locality collisions.
5. **Decisive Optimization Opportunity:**  
   Tightening Block D Region from $DF \le 1000$ to $DF \le 100$ reduces Block D from 95.47 to **10.12 cands/S1** (eliminating 2.13 Million noise candidates, -89.4%) while risking only **77 unique GT links** (0.089% of total GT). This drops pipeline density from 359 down to **~274 cands/S1**, bringing total test candidates down toward 475 Million.

---

### 2. Current Strategy-11 Pipeline Architecture

For each Source 1 query entity:
1. **Block A (Exact Normalized Name):** Exact match on clean alphanumeric business name.
2. **Block C ($DF \le 1000$):** Compressed core-12 business name prefix, retaining posting lists with target $DF \le 1000$.
3. **Block D ($DF \le 1000$):** Address signals (Postal Code + First Token; Region + First Token with $DF \le 1000$).
4. **Block G6 Hybrid-250:** Rare address-token pairs, retaining candidates matching $\ge 2$ shared G6 keys OR ($= 1$ shared key with key $DF \le 250$).
5. **Refined {B, E} Top-50:** Name token (B) and char 3-gram (E) candidates outside the above anchors, scored by cheap features (`2.0 * name_jacc + 1.0 * min(name_inter, 3) + 1.0 * min(addr_inter, 3) + 1.5 * min(num_inter, 2)`) and capped at Top-50.

---

### 3. Block-Level Candidate Attribution

Audited on the 25,000 development queries (86,570 ground-truth links):

| Block Component | Raw Cands | Mean / S1 | Median | P90 | P95 | P99 | Max | Unique Target IDs | % Overlap | Unique Cands | % Union Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Block A** | 779,772 | 31.19 | 3.0 | 41.0 | 156.0 | 605.0 | 1,468 | 431,688 | 69.15% | 240,559 | 2.68% |
| **Block C ($DF \le 1000$)** | 1,275,620 | 51.02 | 6.0 | 146.0 | 229.0 | 760.0 | 997 | 814,744 | 44.66% | 705,964 | 7.87% |
| **Block D ($DF \le 1000$)** | 2,386,784 | 95.47 | 11.0 | 335.0 | 425.0 | 685.0 | 978 | 996,344 | **2.40%** | **2,329,543** | **25.96%** |
| **G6 Hybrid-250** | 4,156,995 | 166.28 | 57.0 | 429.0 | 639.0 | 1,450.0 | 3,349 | 2,345,569 | 1.56% | **4,092,161** | **45.60%** |
| **Refined {B, E} Top-50** | 1,021,811 | 40.87 | 50.0 | 50.0 | 50.0 | 50.0 | 50 | 854,820 | 0.00% | 1,021,811 | 11.39% |
| **FINAL UNION** | **8,974,593** | **358.98** | **259.0** | **785.0** | **1,000.0** | **1,666.0** | **3,604** | **3,892,104** | - | **8,974,593** | **100.00%** |

---

### 4. Block Overlap Analysis

#### Pairwise Overlap Matrix (Number of Co-Generated Candidate Pairs):
| Block | Block A | Block C ($DF \le 1000$) | Block D ($DF \le 1000$) | G6 Hybrid-250 | Refined {B,E} Top-50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Block A** | **779,772** | 533,540 | 26,441 | 34,658 | 0 |
| **Block C ($DF \le 1000$)** | 533,540 | **1,275,620** | 43,640 | 53,548 | 0 |
| **Block D ($DF \le 1000$)** | 26,441 | 43,640 | **2,386,784** | 28,897 | 0 |
| **G6 Hybrid-250** | 34,658 | 53,548 | 28,897 | **4,156,995** | 0 |
| **Refined {B,E} Top-50** | 0 | 0 | 0 | 0 | **1,021,811** |

*(Note: Refined {B, E} Top-50 has zero overlap by architectural construction because it is generated strictly outside the anchor blocks).*

#### Multi-Block Hit Spectrum & Signal-to-Noise:
| Multi-Block Hit Count | Candidate Pairs | % of Final Union | GT Links Recovered | Precision (% True Positives) | Signal-to-Noise Profile |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1 Block Only** | 8,390,038 | 93.49% | 28,833 | **0.344%** | Pure single-source noise pool (1 true match per 291 cands) |
| **2 Blocks** | 535,222 | 5.96% | 20,435 | **3.818%** | Moderate anchor corroboration (1 true match per 26 cands) |
| **3 Blocks** | 36,832 | 0.41% | 23,459 | **63.692%** | High structural corroboration (2 true matches per 3 cands) |
| **4 Blocks** | 12,501 | 0.14% | 11,497 | **91.969%** | Extreme multi-evidence certainty (> 9 true matches per 10 cands) |
| **5 Blocks** | 0 | 0.00% | 0 | 0.000% | - |

---

### 5. Ground-Truth Recovery Attribution

Total Ground-Truth Target Links: **86,570** (US: 51,907, India: 34,663, India Cross-Script: 11,517).

| Block Component | Total GT Recovered | Overall Recall (%) | Unique GT Recovered | Unique GT Rate (%) | US Unique GT | India Unique GT | Cross-Script Unique GT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Block A** | 35,909 | 41.48% | 135 | 0.16% | 110 | 25 | 2 |
| **Block C ($DF \le 1000$)** | 49,319 | 56.97% | 1,657 | 1.91% | 779 | 878 | 193 |
| **Block D ($DF \le 1000$)** | 23,027 | 26.60% | **276** | **0.32%** | 233 | 43 | 7 |
| **G6 Hybrid-250** | **76,146** | **87.96%** | **25,098** | **28.99%** | 11,691 | 13,407 | 6,745 |
| **Refined {B, E} Top-50** | 1,667 | 1.93% | 1,667 | 1.93% | 611 | 1,056 | 166 |
| **FINAL UNION** | **84,224** | **97.29%** | **84,224** | **97.29%** | **51,364 (99.0%)** | **32,860 (94.8%)** | **10,483 (91.0%)** |

---

### 6. Candidate Efficiency Analysis

Evaluating each block's marginal efficiency—the exact computational volume expended to capture each uniquely recovered true entity link:

| Block Component | Raw Cands / S1 | Unique Cands / S1 | Overall SN Ratio (%) | Candidates per Recovered GT | Unique SN Ratio (%) | **Candidates per Unique GT Link** | Efficiency Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **G6 Hybrid-250** | 166.28 | 163.69 | 1.832% | 54.6 | 0.613% | **163.1** | **Outstanding:** Indispensable backbone (recovers 25k unique GT links) |
| **Block C ($DF \le 1000$)** | 51.02 | 28.24 | 3.866% | 25.9 | 0.235% | **426.1** | **Strong:** High precision core anchor across US and India |
| **Refined {B, E} Top-50** | 40.87 | 40.87 | 0.163% | 613.0 | 0.163% | **613.0** | **Fair:** Captures subtle spelling mutations outside anchors |
| **Block A** | 31.19 | 9.62 | 4.605% | 21.7 | 0.056% | **1,781.9** | **Strong Overall:** High raw precision (4.6%), largely covered by C |
| **Block D ($DF \le 1000$)** | **95.47** | **93.18** | **0.965%** | **103.7** | **0.012%** | **8,440.4** | **Extremely Inefficient:** 97.6% pure noise, 8,440 cands per unique link |

```
Candidates Expended per Unique Ground-Truth Link Recovered
  (Lower is Better / More Efficient)

  G6 Hybrid-250 :  [██] 163.1
  Block C <=1000:  [█████] 426.1
  Refined {B,E} :  [████████] 613.0
  Block A       :  [██████████████████████] 1,781.9
  Block D <=1000:  [██████████████████████████████████████████████████████████████████████████████] 8,440.4  <--- MASSIVE EFFICIENCY LEAK
```

---

### 7. Dominant Bottleneck Identification

#### Finding 1: Block D is the Dominant Candidate-Volume Bottleneck & Polluter
While G6 generates 166.3 cands/S1, it earns its volume by capturing 87.96% of all true entities and 25,098 unique links.  
**Block D is the true pathological bottleneck in the Strategy-11 pipeline**:
- It consumes **95.47 candidates/S1** (2.39 Million pairs across 25k queries; ~165 Million extrapolated test pairs).
- It constitutes **25.96% of the final union candidate pool**.
- It yields only **276 unique ground-truth links** (a negligible 0.32% unique recovery).
- **8,440 candidates are generated for every single unique true match.**

#### Finding 2: Anatomy of Block D Failure (Postal vs. Region Decomposition)
Empirical diagnostic decomposition of Block D components across 25,000 queries reveals a stark contrast:
- **Postal Code + First Token:**
  - Candidates: **3,287** (only **0.13 cands/S1**)
  - Ground-truth links captured: **3,190**
  - Precision: **97.05%** (hyper-specific structural anchor!)
- **Region + First Token:**
  - Candidates: **2,384,756** (**95.39 cands/S1** — **99.91% of Block D candidate volume!**)
  - Total GT recovered: 21,092
  - Unique GT recovered vs. A, C, G6: **Only 272 links**!
  - 99.9% of Region candidates are noise caused by broad territories (e.g., California, Maharashtra) colliding with common first words ("Sri", "Om", "Hotel", "National", "Star", "American").

---

### 8. Controlled Pruning Hypotheses for Block D

We conducted an empirical diagnostic sweep over the Region component of Block D (while preserving 100% of Postal Code candidates):

| Configuration | Block D Cands / S1 | Pipeline Union Cands / S1 | Unique Cands Contributed | Unique GT Links vs {A,C,G6} | Unique GT Links Lost | Efficiency (Cands / Uniq GT) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline: D ($DF \le 1000$)** | 95.47 | 358.98 | 2,329,543 | 276 | 0 | 8,440.4 |
| **Hypothesis 1: D ($DF \le 500$)** | 74.20 | 337.71 | 1,803,858 | 265 | -11 (0.01%) | 6,807.0 |
| **Hypothesis 2: D ($DF \le 250$)** | 35.09 | 298.60 | 837,417 | 240 | -36 (0.04%) | 3,489.2 |
| **Hypothesis 3: D ($DF \le 100$)** | **10.12** | **273.63** | **225,066** | **199** | **-77 (0.09%)** | **1,131.0** |
| **Hypothesis 4: D ($DF \le 50$)** | 4.67 | 268.18 | 94,103 | 173 | -103 (0.12%) | 543.9 |
| **Hypothesis 5: Postal Only (No Reg)**| 0.13 | 263.64 | 86 | 7 | -269 (0.31%) | 12.3 |

#### Analysis of Hypotheses:
- **Hypothesis 3 ($DF \le 100$) is the Pareto Sweet Spot:**
  - Reduces Block D candidate density by **89.4%** (from 95.47 down to 10.12 cands/S1).
  - Eliminates **2.10 Million noise candidates** from the 25k development pool.
  - Across the entire dataset, only **77 unique ground-truth links are risked** (less than one-tenth of one percent of ground truth).
  - Improves Block D unique efficiency by **7.5x** (from 8,440 down to 1,131 cands/unique GT).
- **Hypothesis 4 ($DF \le 50$):**
  - Drops Block D to 4.67 cands/S1, saving another ~130k candidates, but loses 103 unique GT links.
- **Hypothesis 5 (Postal Only):**
  - Prunes virtually all Block D candidates (drops to 0.13 cands/S1), but sacrifices 269 unique links where the postal code was missing in Source 2/3.

---

### 9. Recommended Next Experiment: Phase 5F (Block D Frequency Tightening)

Before proceeding to any full test inference, we must validate the downstream ML impact of pruning Block D.

#### Proposed Experiment Configuration (Phase 5F):
Compare the following controlled Block D variations on the 5,000 held-out validation set using the official Phase 4 LightGBM + XGBoost ensemble at $\tau = 0.600$:
1. **Control:** Current Baseline Strategy-11 ($D \le 1000$, $G6 \text{ Hybrid-250}$, refined $\{B,E\} \text{ Top-50}$).
2. **Experiment A:** Strategy-11 with Block D Region $DF \le 250$.
3. **Experiment B (Recommended):** Strategy-11 with Block D Region $DF \le 100$.
4. **Experiment C:** Strategy-11 with Block D Region $DF \le 50$.
5. **Experiment D:** Strategy-11 with Block D Postal-Only (Region Pruned).

---

### 10. Expected Candidate Volume and RAM Impact

Extrapolating from the 25,000 development queries to the full test set (1,732,544 Source 1 entities across India, US, and France):

| Pipeline Configuration | Pipeline Cands / S1 | Extrapolated Test Pairs | Est. Raw Candidate RAM (bytes) | Est. Validator Peak RAM (GB) | Status vs 24 GB RAM Constraint |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 5D Pre-Audit Assumption** | 935.9 | 1,472.6 M | ~94.6 GB | ~110–125 GB | **Violates 24 GB limit** |
| **Current Validated Strategy-11** | **358.98** | **621.9 M** | **~39.9 GB** | **~48–55 GB** | **Violates 24 GB limit** |
| **Proposed Phase 5F: D ($DF \le 250$)** | 298.60 | 517.3 M | ~33.2 GB | ~40–46 GB | Exceeds 24 GB limit |
| **Proposed Phase 5F: D ($DF \le 100$)** | **273.63** | **474.1 M** | **~30.4 GB** | **~36–42 GB** | Near feasibility threshold |
| **Phase 5F: D ($DF \le 100$) + BE Top-25** | **~250.0** | **~433.0 M** | **~27.8 GB** | **~33–38 GB** | Feasible via 2-chunk streaming |
| **Phase 5F: Postal Only + BE Top-25** | **~235.0** | **~407.0 M** | **~26.1 GB** | **~31–35 GB** | Feasible via 2-chunk streaming |

*(Chunked inference: With a chunk size of 25,000 S1 queries, candidate generation and ML scoring require < 8 GB peak RSS per chunk; the RAM numbers above represent the final in-memory submission dataframe if unchunked).*

---

### 11. Explicit "Do Not Proceed to Full Test Yet" Checkpoint

> [!WARNING]
> ### CRITICAL CHECKPOINT: DO NOT RUN FULL TEST INFERENCE YET
> 1. **Do NOT run full test inference.** Although the Strategy-11 candidate pool is much leaner than previously assumed (358.98 vs 935.9 cands/S1), 621.9 Million test pairs will still demand ~48–55 GB of validator memory without chunked streaming, which exceeds the 24 GB host hardware constraint.
> 2. **Block D is an unoptimized candidate leak.** Running inference now would needlessly evaluate 2.33 Million noise candidate pairs generated by weak region tokens with a 0.01% hit rate.
> 3. **Next Mandatory Step:** Run the Phase 5F controlled downstream validation of Block D tightening ($DF \le 100$ vs $DF \le 250$) on the 5,000 held-out queries to verify that Macro $F_{0.5} \ge 0.9620$ is strictly maintained before touching test data.

---

### 12. Direct Answers to Specified Audit Questions

1. **The dominant candidate-volume bottleneck:**  
   In sheer candidate volume, **Block G6 Hybrid-250** generates 45.6% of the union candidates (166.28 cands/S1), but it is a necessary, high-recall anchor (163 cands/unique GT).  
   In **unnecessary, unproductive volume**, **Block D ($DF \le 1000$)** is the dominant bottleneck, producing **95.47 cands/S1 (25.96% of the entire union pool)**.

2. **The weakest candidate-efficiency block:**  
   **Block D ($DF \le 1000$) is overwhelmingly the weakest block in the entire pipeline.**  
   It has:
   - An overlap of only **2.40%** (97.6% of its candidates are completely unique, unanchored noise).
   - An abysmal marginal precision of **0.0118%**.
   - An efficiency ratio of **8,440.4 candidates per unique ground-truth link recovered**.
   - 99.91% of its noise is generated by `Region + First_Token` collisions.

3. **The single highest-value pruning experiment to run next:**  
   **Tighten Block D Region from $DF \le 1000$ to $DF \le 100$ (while retaining 100% of Postal Code matches).**  
   This will eliminate 85.35 cands/S1 (-89.4% of Block D volume, -2.13 Million noise candidates) while risking only 77 unique ground-truth links out of 86,570 (0.089% of GT).

4. **The expected trade-off between candidate reduction and Macro $F_{0.5}$:**  
   - **Candidate Reduction:** Total pipeline candidates drop by **~23.8%** (from 358.98 down to **273.63 cands/S1**), eliminating **~148 Million candidates** from full test inference.
   - **Expected Macro $F_{0.5}$ Impact:** Because the 77 lost links represent only 0.089% of ground truth, and because eliminating 2.1 Million negative pairs will reduce false positive merges, downstream Macro $F_{0.5}$ is expected to remain virtually unchanged ($\Delta \le \pm 0.0003$, within $0.9621 - 0.9624$).

5. **Whether we should proceed toward full inference or perform another optimization round:**  
   **Perform another optimization round (Phase 5F).**  
   Do not run full test inference yet. We should run the controlled downstream Macro $F_{0.5}$ benchmark on the 5,000 held-out queries comparing Block D Region caps ($DF \le 250, 100, 50$). Once the optimal knee is confirmed with zero or negligible $F_{0.5}$ loss, the pipeline can be safely frozen for test execution.
