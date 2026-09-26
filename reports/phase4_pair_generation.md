# Amazon ML Challenge 2026: Business Entity Resolution
## Phase 4: Training Pair Generation & Negative Sampling Report

**Date:** September 2026  
**Execution Runtime:** 372.44s (~6.2 min)  
**Peak Memory (RSS):** 2611.98 MB (Measured via `psutil`)  
**Random Seed:** 42 (Deterministic execution)  
**Frozen Candidate Pipeline:** `Block A + Block B(10k) + Block C(12) + Block D + Block E(T=6) + Block G6(rare_pair, df<=2500)`

---

### 1. Executive Summary: Core Metrics

| Metric Domain | Metric Name | Value |
| :--- | :--- | :--- |
| **Queries** | Total Source 1 Queries Evaluated | **25,000** |
| **Queries** | Train S1 Entities (80.0%) | **20,000** |
| **Queries** | Validation S1 Entities (20.0%) | **5,000** |
| **Queries** | Train/Val Entity Overlap | **0 (Strictly Disjoint)** |
| **Ground Truth** | Authoritative Ground-Truth Links | **86,570** |
| **Ground Truth** | Positives Captured by Blocking Pipeline | **84,966** (98.15%) |
| **Ground Truth** | Positives Missed by Blocking Pipeline | **1,604** (1.85%) |
| **Candidates** | Raw Candidate Pairs Evaluated across 25k Queries | **115,013,048** (~4,600/S1) |
| **Candidates** | Raw Negative Candidate Pairs before Sampling | **114,928,082** |
| **Memory** | Initial RSS $\to$ Final RSS | 81.2 MB $\to$ 2612.0 MB |
| **Memory** | Peak Process RSS (Empirical) | **2612.0 MB** (< 15 GB ceiling) |

---

### 2. Negative-to-Positive Ratio Benchmarks

Comparison of generated training pair datasets across sampling ratios:

| Ratio Config | Total Pairs | Positives | Negatives | Realized Ratio | Train Pairs (Pos / Neg) | Val Pairs (Pos / Neg) | Hard Neg % | Medium Neg % | Background Neg % | Parquet Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`ratio_1_5`** | **517,195** | 84,966 | 432,229 | **1 : 5.09** | 413,983 (68,014 / 345,969) | 103,212 (16,952 / 86,260) | **59.43%** | 7.74% | 32.84% | **8.31 MB** |
| **`ratio_1_10`** | **947,310** | 84,966 | 862,344 | **1 : 10.15** | 758,256 (68,014 / 690,242) | 189,054 (16,952 / 172,102) | **57.23%** | 7.78% | 34.99% | **14.05 MB** |
| **`ratio_1_20`** | **1,800,777** | 84,966 | 1,715,811 | **1 : 20.19** | 1,441,266 (68,014 / 1,373,252) | 359,511 (16,952 / 342,559) | **53.91%** | 7.8% | 38.29% | **26.14 MB** |

---

### 3. Population Stratification & Distribution Audit

The train and validation splits were created using stratified sampling across the actual country distribution, match cardinality, singleton presence, and script characteristics without artificial rebalancing:

| Dimension | Total Queries | Train Split (80%) | Val Split (20%) | Population Proportion Preserved |
| :--- | :--- | :--- | :--- | :--- |
| **US Entities** | 15,000 (60.0%) | 12,000 (60.0%) | 3,000 (60.0%) | Exact 60.0% preserved |
| **India Entities** | 10,000 (40.0%) | 8,000 (40.0%) | 2,000 (40.0%) | Exact 40.0% preserved |
| **Singletons (0 Matches)** | 1,396 (5.6%) | 1,116 (5.6%) | 280 (5.6%) | Exact 5.6% preserved |
| **Cross-Script Queries** | 3,075 (23.7%) | 2,460 (23.7%) | 615 (23.7%) | Exact 23.7% preserved |

---

### 4. Memory Footprint by Execution Stage (Empirical `psutil` Measurement)

| Stage | Stage Description | RSS at Stage (MB) |
| :--- | :--- | :--- |
| `Initial State` | Memory checkpoint | **81.21 MB** |
| `After Stage 1 (Queries & GT Loaded)` | Memory checkpoint | **127.00 MB** |
| `After Stage 2 (Target IDs Mapped)` | Memory checkpoint | **1443.69 MB** |
| `After Stage 3 (Block G6 Available)` | Memory checkpoint | **1477.80 MB** |
| `After Stage 4 (Train/Val Split Completed)` | Memory checkpoint | **1484.01 MB** |
| `After Stage 5 (Streaming & Sampling Done)` | Memory checkpoint | **2414.13 MB** |
| `After Stage 6 (Parquet Datasets Saved)` | Memory checkpoint | **2566.88 MB** |
| `Final State` | Memory checkpoint | **2611.98 MB** |

---

### 5. Validation & Quality Assurance Verification

- [x] **Zero Data Leakage:** `len(set(train_s1_eids) & set(val_s1_eids)) == 0`. Every S1 entity resides strictly in Train or Validation.
- [x] **Ground Truth Fidelity:** 100% of rows labeled `label=1` verified to exist in `train_ground_truth.tsv`.
- [x] **Negative Label Purity:** 100% of rows labeled `label=0` verified to NOT exist in `train_ground_truth.tsv`.
- [x] **Dataset Integrity:** Original TSV files and `student_resource/utils/validate_submission.py` unaltered.
- [x] **Traceability:** Candidate integer IDs and original string entity IDs (`S2-*`, `S3-*`) are 100% traceable.
- [x] **Artifact Persisted:** Generated Parquet files are stored in `phase4/data/` with snappy compression.