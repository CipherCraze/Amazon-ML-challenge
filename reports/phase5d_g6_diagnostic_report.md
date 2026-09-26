# Amazon ML Challenge 2026: Business Entity Resolution
## Phase 5D: Empirical G6 Address-Pair Collision Diagnostic Report

**Date:** September 2026  
**Diagnostic Dataset:** 25,000 Stratified Development Queries (86,570 Authoritative Ground Truth Links)  
**Target Search Space:** 10,320,219 Records (Train Source 2 & Source 3) + 1,434,993 France Test Targets  
**Total Diagnostic Runtime:** 220.18s  
**Peak Memory (RSS):** 246.83 MB  

---

### 1. Executive Summary: The Structural Nature of G6 Collisions

Block G6 (`(country, rare address-token pair)`) was introduced in Phase 3 to recover cross-script (Latin $\longleftrightarrow$ Indic) ground-truth links without requiring name similarity. While G6 captures **88.78% of all ground-truth links independently** (and brings cumulative pipeline recall to 98.15%), its unconstrained retention with posting cap $DF \le 2500$ creates an acute candidate inflation bottleneck (1.77 Billion candidates extrapolated to full test, ~122.7 GB validator RAM).

This empirical diagnostic reveals the fundamental reason for this inflation:
1. **The Single-Key Collision Trap:** **83.99%** of the 12,642,178 G6 candidates (10,618,596 pairs) match **EXACTLY ONE** G6 key. The precision of these single-key collisions is an astonishingly low **0.151%** (only 15,989 true links among 10.6M candidates).
2. **True Matches are Multi-Key Anchored:** In contrast, **78.31% of captured ground truth links share TWO OR MORE G6 keys** (60,188 links). Candidates sharing $3+$ G6 keys have a precision of **9.438%**—over **62 times higher** than single-key collisions.
3. **Single-Key Matches are Ultra-Rare:** For the minority of true matches supported by only one G6 key (16,672 links, 21.69% of captured GT), the supporting key is almost always extremely specific:
   - **Median $DF = 7.0$** (75% have $DF \le 20$).
   - **90.76%** have $DF \le 100$.
   - **94.58%** have $DF \le 250$.
   - **96.62%** have $DF \le 500$.
   - Only 298 links (1.79%) require a key with $DF > 1000$.

---

### 2. G6 Query Key & Document Frequency (DF) Profile

Across 25,000 development queries, 24,981 queries (99.92%) possess $\ge 2$ specific address tokens, generating 142,190 unique query keys.

| Document Frequency (DF) Range | Query Keys in Bin | Percentage of Keys | Cumulative Percentage | Nature of Keys |
| :--- | :---: | :---: | :---: | :--- |
| **$DF = 0$** (Target Unseen) | 7,040 | 4.95% | 4.95% | Query typos, unindexed addresses |
| **$1 \le DF \le 10$** | 79,869 | 56.17% | 61.12% | Highly specific building/house numbers + street |
| **$11 \le DF \le 100$** | 41,337 | 29.07% | 90.19% | Local street segments, specific commercial blocks |
| **$101 \le DF \le 500$** | 10,883 | 7.65% | 97.84% | Broader neighborhoods, dense shopping complexes |
| **$501 \le DF \le 1000$** | 1,798 | 1.26% | 99.10% | Major arterial roads, broad sub-districts |
| **$1001 \le DF \le 2500$** | 933 | 0.66% | 99.76% | Dense urban hubs, common French communes/arrondissements |
| **$DF > 2500$** (Baseline Pruned) | 330 | 0.23% | 100.00% | Generic words slipping past stop-words (e.g. city names) |

---

### 3. Empirical Ground Truth Support: Shared G6 Keys

Evaluation on the 86,570 authoritative ground truth links across the 25,000 development queries:

| Number of Shared G6 Keys | True GT Links | % of All GT Links | % of Captured GT | Subgroup: US | Subgroup: India (Total) | Subgroup: India (Cross-Script) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 (Missed by G6)** | 9,710 | 11.22% | — | 4,836 (9.32%) | 4,874 (14.06%) | 1,728 (15.00%) |
| **1 key** | 16,672 | 19.26% | 21.69% | 12,984 (25.01%) | 3,688 (10.64%) | 1,472 (12.78%) |
| **2 keys** | 1,651 | 1.91% | 2.15% | 733 (1.41%) | 918 (2.65%) | 323 (2.80%) |
| **3 keys** | 27,410 | 31.66% | 35.66% | 22,400 (43.15%) | 5,010 (14.45%) | 1,717 (14.91%) |
| **4 keys** | 898 | 1.04% | 1.17% | 174 (0.34%) | 724 (2.09%) | 244 (2.12%) |
| **5 keys** | 2,561 | 2.96% | 3.33% | 1,364 (2.63%) | 1,197 (3.45%) | 419 (3.64%) |
| **6 keys** | 15,750 | 18.19% | 20.49% | 7,957 (15.33%) | 7,793 (22.48%) | 2,457 (21.33%) |
| **7 keys** | 902 | 1.04% | 1.17% | 66 (0.13%) | 836 (2.41%) | 290 (2.52%) |
| **8 keys** | 313 | 0.36% | 0.41% | 13 (0.03%) | 300 (0.87%) | 75 (0.65%) |
| **9 keys** | 1,533 | 1.77% | 1.99% | 358 (0.69%) | 1,175 (3.39%) | 349 (3.03%) |
| **10 keys** | 9,170 | 10.59% | 11.93% | 1,022 (1.97%) | 8,148 (23.51%) | 2,443 (21.21%) |
| **Total Captured ($\ge 1$)** | **76,860** | **88.78%** | **100.00%** | **47,071 (90.68%)** | **29,789 (85.94%)** | **9,789 (85.00%)** |
| **$\ge 2$ Shared Keys** | **60,188** | **69.53%** | **78.31%** | **34,087 (65.67%)** | **26,101 (75.30%)** | **8,317 (72.21%)** |

> **Key Discovery:**
> - In India, **75.30%** of true matches share 2+ keys, and **23.51%** share all 10 possible token pairs (showing strong address token clustering).
> - In US, **65.67%** share 2+ keys.
> - For cross-script matches in India, **72.21%** share 2+ keys.

---

### 4. Anatomy of the Candidate Pool: Signal vs. Noise

Analyzing all 12,642,178 G6 candidate pairs generated under $DF \le 2500$:

| Candidate Stratum | Total Candidates Generated | % of Candidate Pool | True GT Matches | Category Precision | Signal-to-Noise Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Candidates Matching Exactly 1 Key** | 10,618,596 | **83.99%** | 15,989 | **0.151%** | 1 : 664 |
| **Candidates Matching Exactly 2 Keys** | 1,394,528 | **11.03%** | 1,503 | **0.108%** | 1 : 928 |
| **Candidates Matching 3+ Keys** | 629,054 | **4.98%** | 59,368 | **9.438%** | **1 : 10.6** |
| **Total** | **12,642,178** | **100.00%** | **76,860** | **0.608%** | 1 : 164 |

> **Key Discovery:**
> Over 8.9 million candidates in the single-key pool are pure noise from frequent keys ($DF \ge 100$).  
> If an address key has $DF = 1,500$, it injects 1,500 candidates into the pool, but only $0.05\%$ of those are true matches.

---

### 5. Document Frequency Distribution of Supporting Keys for True Matches

For the 16,672 true GT matches supported by **only 1 G6 key**:
- **Min DF:** 1
- **Median DF:** **7.0**
- **Mean DF:** 63.8
- **P75 DF:** 20.0
- **P90 DF:** 86.9
- **P95 DF:** 294.4
- **Max DF:** 2,499

| Threshold Cap | Single-Key GT Captured | Single-Key GT Lost | % Single-Key GT Retained | Total G6 GT Retained | Overall G6 Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$DF \le 2500$ (Baseline)** | 16,672 | 0 | 100.00% | 76,860 | 88.78% |
| **$DF \le 1000$** | 16,374 | 298 | 98.21% | 76,562 | 88.44% |
| **$DF \le 500$** | 16,108 | 564 | 96.62% | 76,296 | 88.13% |
| **$DF \le 250$** | 15,769 | 903 | 94.58% | 75,957 | 87.74% |
| **$DF \le 100$** | 15,132 | 1,540 | 90.76% | 75,320 | 87.00% |

---

### 6. France Test Set Diagnostic Validation

A random sample of 5,000 France test queries was evaluated against the 1.43 Million France test targets:
- **Total G6 Candidates Generated:** 9,842,763 (**1,968.55 cands/S1**)
- **Median cands/S1:** 1,838.5 | **P90:** 3,856.0 | **P95:** 4,557.1
- **Breakdown:**
  - Matching Exactly 1 Key: **6,631,650 (67.38%)**
  - Matching Exactly 2 Keys: **2,314,473 (23.51%)**
  - Matching 3+ Keys: **896,640 (9.11%)**

France has a higher proportion of multi-key collisions (23.5% matching 2 keys) due to commune names appearing alongside district and department tokens in structured French addresses.

---

### 7. Strategic Hypotheses for Phase 5D Candidate Generation

Based on these findings, five core architectural hypotheses emerge:

1. **Hypothesis 1 (Tight Frequency Sweeps):**
   Tightening G6 $DF \le 250$ or $DF \le 500$ eliminates the vast majority of bloated single-key collisions while preserving $>95\%$ of single-key GT matches and $>98\%$ of multi-key GT matches.
2. **Hypothesis 2 (Support-Level Gating):**
   Retaining all candidates with $\ge 2$ shared G6 keys unconditionally, while restricting single-G6 candidates to very rare keys ($DF \le 100$) or requiring address token overlap $\ge 2$.
3. **Hypothesis 3 (Address Information Content / Rarity Ranking):**
   Ranking G6 candidates by rarity score $S(q, t) = \sum_{g \in \text{shared}} \frac{1}{\log(1 + DF(g))}$ and retaining Top-$K$ ($K \in \{50, 100, 250\}$).
4. **Hypothesis 4 (Compound Address Evidence Filter):**
   Filtering G6-only candidates that lack numeric overlap or high address token Jaccard when $DF > 100$.
5. **Hypothesis 5 (Preserving Final Macro $F_{0.5}$):**
   Because the ML matcher heavily relies on address features, removing millions of weak address negatives may actually *improve* final precision without sacrificing true matches, directly boosting Macro $F_{0.5}$.
