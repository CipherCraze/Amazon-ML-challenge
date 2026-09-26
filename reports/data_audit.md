# Amazon ML Challenge 2026: Business Entity Resolution
## Phase 1 — Comprehensive Dataset Audit Report

**Author / Team:** Data Science & ML Engineering Team  
**Date:** September 2026  
**Document Path:** `reports/data_audit.md`  
**Evaluation Metric:** Macro $F_{0.5}$ (Precision-weighted: $\beta = 0.5$)  
**Delimiter Verification:** Verified all files are tab-delimited (`sep="\t"`). Tabs are mandatory because commas appear frequently inside business names, addresses, and ID list fields.

---

### Executive Summary

| Dataset Split | Source 1 (Reference) | Source 2 | Source 3 | Ground Truth Links | Total Records |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Training Set** | 2,206,821 | 5,034,616 | 5,285,603 | 7,638,365 | 12,527,040 |
| **Test Set** | 1,732,544 | 4,887,273 | 5,082,316 | *Unlabeled* | 11,702,133 |
| **Combined** | **3,939,365** | **9,921,889** | **10,367,919** | **7,638,365** | **24,229,173** |

Key findings from this audit:
1. **Deduplicated Reference Source:** Source 1 in both training and test has **100.0% unique `(business_name, business_address)` pairs**. There are zero exact duplicate pairs within S1.
2. **Absolute Country Isolation:** Full validation across all **7,638,365 ground-truth links** revealed **0 cross-country matches**. Every single entity matches strictly within its own country.
3. **Singleton Rate Uniformity:** Exactly **5.5848%** (123,247 entities) in the training ground truth have zero matches. This rate is virtually invariant between US (5.5828%) and India (5.5878%).
4. **Missing Addresses in S2/S3:** While Source 1 has 0% missing addresses, **Source 2 has 3.36% (train) / 2.65% (test)** and **Source 3 has 3.33% (train) / 2.68% (test)** empty addresses. Matching algorithms cannot require address similarity for these records.
5. **Geographic Shift in Test:** Training data contains only **US (59.9%)** and **India (40.1%)**. The test set introduces **France (14.5%)**, with India rising to **47.1%** and US dropping to **38.3%**.
6. **Pairwise Scale:** Unblocked pairwise comparisons equal **22.77 trillion** in train and **17.27 trillion** in test. Multi-pass candidate blocking is required to achieve a $\ge 99.99\%$ reduction ratio.

---

### 1. Dataset Overview & File Inventory

All files were verified by reading raw byte headers, raw line samples, and parsed DataFrames using `sep="\t"`:

| File Name | File Size | Data Rows | Columns | Primary Key / IDs |
| :--- | :--- | :--- | :--- | :--- |
| `train_source1.tsv` | 200.34 MB | 2,206,821 | `entity_id`, `business_name`, `business_address`, `country` | `S1-*` (100% unique) |
| `train_source2.tsv` | 466.63 MB | 5,034,616 | `entity_id`, `business_name`, `business_address`, `country` | `S2-*` (100% unique) |
| `train_source3.tsv` | 480.37 MB | 5,285,603 | `entity_id`, `business_name`, `business_address`, `country` | `S3-*` (100% unique) |
| `train_ground_truth.tsv` | 121.13 MB | 2,206,821 | `source1_entity_id`, `matched_entity_ids` | `S1-*` (100% 1-to-1 with S1) |
| `test_source1.tsv` | 166.91 MB | 1,732,544 | `entity_id`, `business_name`, `business_address`, `country` | `S1-*` (100% unique) |
| `test_source2.tsv` | 485.86 MB | 4,887,273 | `entity_id`, `business_name`, `business_address`, `country` | `S2-*` (100% unique) |
| `test_source3.tsv` | 482.56 MB | 5,082,316 | `entity_id`, `business_name`, `business_address`, `country` | `S3-*` (100% unique) |

*Validation Confirmation:* No row truncation or column shifts occurred when parsed with `sep="\t"`. Reading without `sep="\t"` produces a single unparsed string column.

---

### 2. Missing-Value Rates & Data Types

Every column across all 6 entity tables was audited for `null`, `NaN`, empty string `""`, or pure whitespace:

| File | `entity_id` Missing | `business_name` Missing | `business_address` Missing | `country` Missing |
| :--- | :--- | :--- | :--- | :--- |
| **train_source1** | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| **train_source2** | 0 (0.00%) | 0 (0.00%) | **168,967 (3.356%)** | 0 (0.00%) |
| **train_source3** | 0 (0.00%) | 0 (0.00%) | **175,916 (3.328%)** | 0 (0.00%) |
| **test_source1** | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| **test_source2** | 0 (0.00%) | 0 (0.00%) | **129,408 (2.648%)** | 0 (0.00%) |
| **test_source3** | 0 (0.00%) | 0 (0.00%) | **136,098 (2.678%)** | 0 (0.00%) |

#### Analysis:
- `entity_id`, `business_name`, and `country` are **100% populated** across all 24.2M records.
- In Source 1, `business_address` is **100% populated**.
- In Source 2 and Source 3, **~2.6% to 3.4% of entities completely lack an address** (`<EMPTY>`).
- Ground-truth inspection confirms that when S2/S3 addresses are missing, the matching entities share strong name correspondence (or name typos/domain names), and are matched purely on name features. Any blocking or scoring pipeline requiring address matching will forfeit these links.

---

### 3. Country Distributions & The France Domain Shift

The distribution of records across countries highlights a critical domain shift:

| File | India (Count / %) | US (Count / %) | France (Count / %) | Total Records |
| :--- | :--- | :--- | :--- | :--- |
| **train_source1** | 883,188 (40.02%) | 1,323,633 (59.98%) | *0 (0.00%)* | 2,206,821 |
| **train_source2** | 2,017,799 (40.08%) | 3,016,817 (59.92%) | *0 (0.00%)* | 5,034,616 |
| **train_source3** | 2,115,547 (40.02%) | 3,170,056 (59.98%) | *0 (0.00%)* | 5,285,603 |
| **test_source1** | 809,986 (46.75%) | 663,106 (38.27%) | **259,452 (14.98%)** | 1,732,544 |
| **test_source2** | 2,312,565 (47.32%) | 1,871,330 (38.29%) | **703,378 (14.39%)** | 4,887,273 |
| **test_source3** | 2,405,000 (47.32%) | 1,945,701 (38.28%) | **731,615 (14.40%)** | 5,082,316 |

#### Ground Truth Cross-Country Audit:
- **Total links verified in train ground truth:** 7,638,365
- **Links crossing country boundaries:** **0 (0.0000%)**
- **Conclusion:** `country` is an absolute, hard partition filter. Entities in `US` only match `US`, `India` only matches `India`, and `France` will only match `France`.

---

### 4. Length Distributions: Business Names & Addresses

Character and word distributions were computed across all records:

#### A. Business Name Lengths (Characters & Words)
| File | Char Min / Max | Char Mean ± Std | Char Median (IQR: p25 - p75) | Char p90 / p99 | Word Mean ± Std | Word Median (p25 - p75) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **train_source1** | 3 / 105 | 24.03 ± 7.74 | 24 (18 - 30) | 34 / 42 | 3.55 ± 0.98 | 4 (3 - 4) |
| **train_source2** | 2 / 104 | 25.10 ± 8.89 | 25 (19 - 31) | 37 / 48 | 3.50 ± 1.24 | 4 (3 - 4) |
| **train_source3** | 2 / 123 | 25.20 ± 9.49 | 25 (18 - 31) | 37 / 50 | 3.53 ± 1.34 | 4 (3 - 4) |
| **test_source1** | 3 / 92 | 23.84 ± 7.67 | 24 (18 - 29) | 34 / 42 | 3.52 ± 0.92 | 4 (3 - 4) |
| **test_source2** | 2 / 102 | 25.70 ± 9.12 | 25 (19 - 32) | 38 / 49 | 3.59 ± 1.20 | 4 (3 - 4) |
| **test_source3** | 2 / 103 | 25.66 ± 9.57 | 25 (19 - 32) | 38 / 50 | 3.60 ± 1.29 | 4 (3 - 4) |

#### B. Business Address Lengths (Characters & Words)
| File | Char Min / Max | Char Mean ± Std | Char Median (IQR: p25 - p75) | Char p90 / p99 | Word Mean ± Std | Word Median (p25 - p75) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **train_source1** | 11 / 256 | 52.07 ± 25.33 | 41 (33 - 70) | 90 / 124 | 8.03 ± 3.51 | 7 (5 - 10) |
| **train_source2** | 0 / 249 | 46.23 ± 24.84 | 37 (30 - 61) | 83 / 118 | 7.29 ± 3.58 | 6 (5 - 9) |
| **train_source3** | 0 / 240 | 46.71 ± 21.65 | 42 (35 - 54) | 77 / 115 | 7.17 ± 3.39 | 6 (5 - 8) |
| **test_source1** | 11 / 268 | 57.21 ± 25.03 | 50 (36 - 74) | 93 / 126 | 8.59 ± 3.58 | 8 (6 - 11) |
| **test_source2** | 0 / 269 | 50.41 ± 25.35 | 43 (32 - 67) | 87 / 120 | 7.80 ± 3.69 | 7 (5 - 10) |
| **test_source3** | 0 / 267 | 48.74 ± 22.64 | 43 (35 - 59) | 81 / 117 | 7.51 ± 3.54 | 7 (5 - 9) |

*Key Takeaway:* Most business names are concise (3–4 words, ~24–25 characters). Addresses average 7–8 words (~46–57 characters), but exhibit a right-skewed tail up to 269 characters due to multi-tier Indian addresses (plot, sector, colony, district, state) and French postal formats.

---

### 5. Duplicate & Near-Duplicate Patterns

A full uniqueness audit was executed across all normalized strings (lowercased, whitespace-stripped):

| File | Total Records | Unique Normalized Names | Name Duplicate Rate (%) | Max Name Freq | Unique Normalized Addresses | Addr Duplicate Rate (%) | Unique (Name, Addr) Pairs | Pair Duplicate Rate (%) | Max Pair Freq |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **train_s1** | 2,206,821 | 1,538,804 | 38.34% | 253 | 2,130,606 | 5.27% | **2,206,821** | **0.00%** | **1** |
| **train_s2** | 5,034,616 | 4,287,470 | 20.51% | 397 | 4,300,742 | 20.11% | 4,997,966 | 1.43% | 5 |
| **train_s3** | 5,285,603 | 4,579,083 | 18.76% | 421 | 4,632,184 | 16.29% | 5,261,572 | 0.90% | 4 |
| **test_s1** | 1,732,544 | 1,238,244 | 36.06% | 205 | 1,674,549 | 5.29% | **1,732,544** | **0.00%** | **1** |
| **test_s2** | 4,887,273 | 4,207,077 | 19.40% | 302 | 4,182,274 | 21.52% | 4,855,261 | 1.29% | 5 |
| **test_s3** | 5,082,316 | 4,446,091 | 17.78% | 387 | 4,450,856 | 17.78% | 5,060,460 | 0.85% | 4 |

#### Empirical Observations:
1. **Deduplication Verification:** In both `train_source1` and `test_source1`, the number of unique `(business_name, business_address)` pairs equals the exact total row count (2,206,821 and 1,732,544 respectively). Duplicate pair rate is exactly **0.00%**. Source 1 contains zero duplicate business entities.
2. **Homonym Business Names:** In Source 1, 38.3% of records share a business name with at least one other record (e.g. common franchise names, trade styles, "General Store", "Sai Enterprises", "Subway"). The maximum name frequency reaches 253 in train S1 and 205 in test S1. Name matching alone without address disambiguation will cause false merges.
3. **Internal Duplication in S2/S3:** In Source 2 and Source 3, ~0.85% to 1.43% of records possess identical `(business_name, business_address)` strings with different `entity_id`s (max frequency 4–5). Multiple distinct records in S2/S3 represent the exact same entity and match the same S1 parent.

---

### 6. Ground-Truth Match Characteristics & Singleton Rate

Analysis of all 2,206,821 training ground-truth rows:

| Metric | Ground-Truth Value |
| :--- | :--- |
| **Total Source 1 Records** | 2,206,821 |
| **Total Ground Truth Matches** | **7,638,365** |
| **Source 2 Matches** | 3,693,619 (48.36%) |
| **Source 3 Matches** | 3,944,746 (51.64%) |
| **Entities with BOTH S2 and S3 Matches** | **1,776,047 (80.48%)** |
| **Entities with ONLY S2 Matches** | 143,029 (6.48%) |
| **Entities with ONLY S3 Matches** | 164,498 (7.45%) |
| **Singletons (0 Matches)** | **123,247 (5.5848%)** |
| **Duplicate IDs in Match Lists** | **0** |
| **Invalid ID Prefixes** | **0** |

#### Match Count Distribution per Source 1 Entity:
- **Mean:** 3.4613 matches
- **Standard Deviation:** 1.7053
- **Min:** 0 | **25th percentile:** 2.0 | **Median (50th):** 3.0 | **75th percentile:** 5.0
- **90th percentile:** 6.0 | **95th percentile:** 6.0 | **99th percentile:** 8.0 | **Max:** 11 matches

| Matches per S1 Entity | Count of S1 Entities | Percentage (%) | Cumulative (%) |
| :---: | :---: | :---: | :---: |
| **0 (Singletons)** | 123,247 | 5.58% | 5.58% |
| **1** | 119,157 | 5.40% | 10.98% |
| **2** | 375,212 | 17.00% | 27.98% |
| **3** | **530,841** | **24.05%** | 52.03% |
| **4** | 484,115 | 21.94% | 73.97% |
| **5** | 321,957 | 14.59% | 88.56% |
| **6** | 164,868 | 7.47% | 96.03% |
| **7** | 63,968 | 2.90% | 98.93% |
| **8** | 18,680 | 0.85% | 99.78% |
| **9** | 4,205 | 0.19% | 99.97% |
| **10** | 534 | 0.02% | 99.99% |
| **11** | 37 | 0.00% | 100.00% |

#### Singleton Rate Deep Dive:
- **US Singletons:** 73,896 out of 1,323,633 (**5.5828%**)
- **India Singletons:** 49,351 out of 883,188 (**5.5878%**)
- **Length Comparison:** 
  - Singleton Name Length: Mean 24.00, Median 24.0 (Non-singleton: Mean 24.04, Median 24.0)
  - Singleton Address Length: Mean 52.07, Median 41.0 (Non-singleton: Mean 52.07, Median 41.0)
- **Impact on Scoring:** Because macro $F_{0.5}$ gives an individual score of 1.0 to singletons correctly predicted as empty, and 0.0 to false merges on singletons, an over-eager matching threshold will heavily degrade the macro average score.

---

### 7. Real Noise Patterns & Variation Examples from Ground Truth

Actual matched pairs extracted from ground truth illustrate the complex noise profile:

#### A. Name Variations
1. **Legal Suffixes & Prefixes:**
   - *Moved to prefix inside brackets:* `Dick Regional Armada Corp` $\leftrightarrow$ `[Corp] Dick Regional Armada` (`S1-145361722` vs `S3-96572514`)
   - *Moved to prefix:* `Swastik Om Solutions LLP` $\leftrightarrow$ `[LLP] Swastik Om Solutions` (`S1-727602285` vs `S2-138660620`)
   - *Prefix without brackets:* `Crystal Staffing Solutions LLC` $\leftrightarrow$ `LLC Crystal Sttfrifng Solutions` (`S1-546142636` vs `S2-392804085`)
   - *Punctuation variations:* `Obsidian, LLC` $\leftrightarrow$ `Obsidian,-LLC` $\leftrightarrow$ `Obsidian, [[LLC]]` (`S1-274126313` vs `S2-736616474` vs `S3-461175723`)
   - *Suffix omission:* `Maure Williams Colombier Inc` $\leftrightarrow$ `Maure Williams Colombier` (`S1-965667` vs `S2-743505751`)

2. **Typos & Character Corruptions:**
   - *Internal char substitution:* `Dahlia Power Reliable Scientific LLC` $\leftrightarrow$ `Dahlia Ponr Reliable Scientific LLC` (`S1-343815751` vs `S3-878454467`)
   - *Transposition typo:* `Payne Enterprises` $\leftrightarrow$ `Payne Enterpires` (`S1-102811957` vs `S2-553508714`)
   - *Severe typo:* `Orellana Investments LLC` $\leftrightarrow$ `LLC Orellana Invsmbens` (`S1-730934468` vs `S3-352439310`)
   - *Word doubling:* `Orellana Investments LLC` $\leftrightarrow$ `Orellana Investments Investments Llc` (`S1-730934468` vs `S2-356983532`)

3. **Domain Names as Business Names:**
   - *Exact domain match:* `Maure Williams Colombier Inc` $\leftrightarrow$ `maurewilliamscolombier.com` (`S1-965667` vs `S3-11291185`)
   - *Domain match:* `Summit Health LLC` $\leftrightarrow$ `summithealth.com` (`S1-840162906` vs `S3-799520471`)
   - *Domain match:* `True Factory Ltd` $\leftrightarrow$ `truefactory.com` (`S1-525304403` vs `S2-293401164`)
   - *Domain with country token:* `Systel Buildstructure (India) Private Limited` $\leftrightarrow$ `systelbuildstructureindia.com` (`S1-9962387` vs `S3-160760047`)

4. **Scripts, Accents & Transliteration:**
   - *Latin to Tamil script:* `Raj Investments LLP` $\leftrightarrow$ `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` (`S1-55344266` vs `S2-249013014`)
   - *Latin to Devanagari script:* `Ss Food Private Limited` $\leftrightarrow$ `एसएस फूड प्राइवेट लिमिटेड` (`S1-656753428` vs `S2-153058913`)
   - *Accented characters:* `Payne Enterprises` $\leftrightarrow$ `Payne Énterprises` (`S1-102811957` vs `S2-478959098`); `Lumay Boral` $\leftrightarrow$ `Lumay Bóral` (`S1-18727616` vs `S3-187831601`)

#### B. Address Variations
1. **Street & Locality Abbreviations:**
   - `85 Wayne Avenue, Ticonderoga, NY` $\leftrightarrow$ `Wayne Ave, Ticonderoga Townshiip, New York` (`S1-965667` vs `S3-11291185`)
   - `3315 Fremont Street, Peoria, IL` $\leftrightarrow$ `3315 FREMONT ST, PEORIA, IL` $\leftrightarrow$ `Fremont St, Peoria, Illinois` (`S1-102811957` vs `S2-478959098` vs `S3-449308785`)
   - `8706 Kentucky Derby Drive, Waxhaw, NC` $\leftrightarrow$ `8706 KENTUCKY DERBY DR, WAXHAW, NC` (`S1-546142636` vs `S2-487600131`)

2. **Landmarks & Indian Plot/Sector Additions:**
   - `Y.L.Randive House, A.S.Road, Thane, Maharashtra` $\leftrightarrow$ `Plot ##311 Y.l.randive House, Thane, MH` (`S1-829232083` vs `S3-919114016`)
   - `D-127/4 Sangam Vihar, Delhi, New Delhi, Delhi` $\leftrightarrow$ `Plot 134 D-127/4 Sangam Vihar, Delhi, New Delhi, DL` (`S1-817863581` vs `S3-359228059`)
   - `Flat No-D-107, Panda Residency, Bhubaneswar, Khordha, Orissa` $\leftrightarrow$ `PLOT 961 FLAT NO-D-107, PANDA RESIDENCY, BHUBANESWAR, KHORDHA, ଓଡ଼ିଶା` (`S1-30038696` vs `S2-789253940`)
   - `Sco No-19 Sec-56, Gurugram, Gurgaon, Haryana` $\leftrightarrow$ `GURGAON, Haryana, PLOT 555 SCO NO-19 SEC-56, GURUGRAM, GURGAON` (`S1-520706871` vs `S2-905169888`)

3. **Floor / Unit References:**
   - `8082 Armiger Drive, Fl 0, Pasadena, MD` $\leftrightarrow$ `Pasadena, 8082b Armiger Drive, Maryland, Floor 0` (`S1-793447671` vs `S3-325254367`)
   - `605 Berkeley Way, Fl 1, Medford, OR` $\leftrightarrow$ `605 Berkeley Way, Floor 1, Medford, Oregon` (`S1-842582339` vs `S3-517179359`)

4. **Missing Addresses in Ground Truth Matches:**
   - `Maure Williams Colombier Inc` at `85 Wayne Avenue, Ticonderoga, NY` matches:
     - `S2-681193310` (`Maure Wilblims Colombier Inc`, address: `<EMPTY>`)
     - `S2-743505751` (`Maure Williams Colombier`, address: `<EMPTY>`)
     - `S3-860443364` (`Maure Williams Inc Center`, address: `<EMPTY>`)
   - `Obsidian, LLC` at `3907 Hamilton Road, Deer Park, WA` matches:
     - `S2-680265918` (`obsidian, llc`, address: `<EMPTY>`)
     - `S3-461175723` (`Obsidian, [[LLC]]`, address: `<EMPTY>`)

---

### 8. Pairwise Scale & Reduction Ratio Analysis

The scale of all possible comparisons without blocking:

| Split | Source 1 Entities ($N_{S1}$) | Candidate Target Pool ($N_{S2} + N_{S3}$) | Full Cartesian Pairs ($N_{S1} \times (N_{S2} + N_{S3})$) | With Exact Country Partition | Reduction Ratio from Country Partition |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 2,206,821 | 10,320,219 | **$2.277 \times 10^{13}$** (22.77 Trillion) | **$8.868 \times 10^{12}$** (8.87 Trillion) | 61.06% |
| **Test** | 1,732,544 | 9,969,589 | **$1.727 \times 10^{13}$** (17.27 Trillion) | **$6.724 \times 10^{12}$** (6.72 Trillion) | 61.06% |

#### Test Set Country Breakdown:
- **India:** $809,986 \times (2,312,565 + 2,405,000) = 809,986 \times 4,717,565 = \mathbf{3.821 \times 10^{12}}$ pairs
- **US:** $663,106 \times (1,871,330 + 1,945,701) = 663,106 \times 3,817,031 = \mathbf{2.531 \times 10^{12}}$ pairs
- **France:** $259,452 \times (703,378 + 731,615) = 259,452 \times 1,434,993 = \mathbf{3.723 \times 10^{11}}$ pairs
- **Sum of Country Partition:** $\mathbf{6.724 \times 10^{12}}$ pairs

#### Necessary Candidate Reduction Ratio:
- To score candidate pairs with an ML model in reasonable runtime (e.g. inference over 20–50 candidates per S1 entity), the candidate set must contain approximately:
  $$\text{Target Candidate Pairs} \approx 1,732,544 \times 30 \approx 5.2 \times 10^7 \text{ pairs (52 million)}$$
- Required Reduction Ratio:
  $$\text{RR} = 1 - \frac{5.2 \times 10^7}{1.727 \times 10^{13}} = \mathbf{99.9997\%}$$
- Candidate blocking must be multi-pass to achieve $\ge 99.999\%$ reduction while retaining $\ge 95\%$ ground-truth recall.

---

### 9. Recommended Candidate Blocking Strategies

Based **strictly on observed patterns** in the provided data (and not assuming external knowledge), we recommend the following 4 candidate blocking strategies:

#### Strategy 1: Country-Partitioned Multi-Pass Token & Core Inverted Index (Recommended Primary)
- **Design:** Partition data by exact `country`. Strip legal prefixes/suffixes (`inc`, `llc`, `ltd`, `pvt`, `corp`, `[corp]`, `[llp]`), strip URLs/domain extensions (`.com`, `www.`), and index records across three complementary keys:
  1. *Pass A (Normalized First Significant Token):* Matches entities where the primary brand name is intact.
  2. *Pass B (Compressed Alphanumeric Core):* Strips spaces and punctuation to form a condensed 8–10 character stem (e.g. `maurewilliamscolombier` matches `maurewilliamscolombier.com`; `zanderblue` matches `zblue.com`).
  3. *Pass C (Locality / Postal Code + High-IDF Token):* For entities with non-empty addresses, match on `(country, postal_code/state, token)`.
- **Why it fits the data:** Accommodates domain names, word swaps, legal suffix reordering, and missing S2/S3 addresses.

#### Strategy 2: Disjunctive MinHash / LSH on Character 3-Grams
- **Design:** Within each country partition, compute character 3-gram shingles over the normalized name (e.g. `dahlia power reliable` $\rightarrow$ `dah`, `ahl`, `hli`, `lia`...). Apply MinHash with Locality Sensitive Hashing (LSH) using ~64–128 hash functions banded for Jaccard threshold $\ge 0.5$.
- **Why it fits the data:** The ground truth contains pervasive single-char and double-char typos (`Dahlia Ponr`, `Payne Enterpires`, `Invsmbens`), accented characters (`Énterprises`), and transliterations. MinHash LSH retrieves near-duplicate names regardless of where the typo occurs.

#### Strategy 3: Multi-Script Transliteration & Phonetic Normalization Blocking
- **Design:** 
  1. Detect non-ASCII character blocks (Devanagari: `\u0900-\u097F`, Tamil: `\u0980-\u09FF`, Odia: `\u0B00-\u0B7F`, Latin accents: `\u00C0-\u00FF`).
  2. Normalize Unicode accents (NFKD decomposition).
  3. Convert native Indic script tokens into Latin phonetic equivalents or use character n-gram MinHash over phonetic hash representations.
- **Why it fits the data:** Indian entities in S2 and S3 frequently use Tamil or Hindi scripts for both name and address (`ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்`, `एसएस फूड प्राइवेट लिमिटेड`). Standard ASCII-only blocking keys completely fail to link these entities.

#### Strategy 4: Hybrid Name-Address Dual-Layer Blocking with Fallback
- **Design:**
  - *Layer 1 (Dual Match):* For the 97% of records having addresses, require at least one shared high-IDF name token AND at least one shared address token (street number, PIN/postal code, or city/state).
  - *Layer 2 (Address-Free Fallback):* For the ~3% of S2/S3 records with empty addresses, fall back to high-stringency name-only matching (exact normalized core or Jaccard $\ge 0.75$).
- **Why it fits the data:** Prevents candidate explosion caused by homonym franchise names (e.g., 253 instances of identical names in S1) while ensuring zero recall loss on empty-address records.

---

### 10. Separation of Observed Facts vs. Hypotheses

| Topic | Empirically Observed Fact | Hypothesis / Model Implication |
| :--- | :--- | :--- |
| **Country Isolation** | 0 out of 7,638,365 ground-truth matches cross countries. | France in the test set will also be strictly self-contained; test pairs across countries can be blocked 100% with zero loss. |
| **Address Missingness** | 0% missing in S1; ~3% missing in S2/S3. Matches exist in ground truth for empty addresses. | Address similarity must either be imputed, masked, or handled via a dual-path model (with vs. without address features). |
| **Singleton Distribution** | Singletons account for 5.58% in US and 5.59% in India; name and address lengths are identical to non-singletons. | Singletons are randomly unlinked business coverage gaps. Predicting "no match" when max candidate score is below threshold will be critical for macro $F_{0.5}$. |
| **Multi-Matches** | Mode is 3 matches (24.1%); 80.5% of S1 entities match both S2 and S3; max is 11 matches. | Final prediction must NOT enforce a 1-to-1 match constraint. Each candidate in the final model must be classified independently or with dynamic thresholding. |
| **Non-ASCII & Scripts** | Indic scripts (Tamil, Hindi, Odia) and French accents appear in S2/S3 ground truth. | Multilingual token embeddings or phonetic transliteration preprocessing will capture matches missed by pure English string matching. |
