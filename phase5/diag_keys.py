import time, os, sys, re
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from phase2.blocking import EntityNormalizer

norm = EntityNormalizer()
t0 = time.time()

c_freq = Counter()
d_post_freq = Counter()
d_reg_freq = Counter()

int_id = 0
for path in ['student_resource/dataset/train/train_source2.tsv', 'student_resource/dataset/train/train_source3.tsv']:
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip('\r\n').split('\t')
            if len(parts) >= 4:
                eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                norm_n = norm.normalize_name(name)
                core12 = norm.extract_compressed_core(norm_n, prefix_len=12)
                if core12:
                    c_freq[f"{country}_{core12}"] += 1
                
                toks = norm.extract_tokens(norm_n)
                if toks:
                    first_tok = toks[0]
                    sig = norm.extract_address_signals(addr, country)
                    if not sig['is_empty']:
                        if sig['postal_code']:
                            d_post_freq[f"{country}_{sig['postal_code']}_{first_tok}"] += 1
                        if sig['region']:
                            d_reg_freq[f"{country}_{sig['region']}_{first_tok}"] += 1
            int_id += 1

print(f"Scanned {int_id:,} records in {time.time()-t0:.2f}s")
print(f"Unique Block C keys: {len(c_freq):,}")
print(f"Unique Block D Postal keys: {len(d_post_freq):,}")
print(f"Unique Block D Region keys: {len(d_reg_freq):,}")

# Top keys
print("\nTop 15 Block C keys:")
for k, v in c_freq.most_common(15):
    print(f"  {k}: {v:,}")

print("\nTop 15 Block D Region keys:")
for k, v in d_reg_freq.most_common(15):
    print(f"  {k}: {v:,}")

print("\nTop 15 Block D Postal keys:")
for k, v in d_post_freq.most_common(15):
    print(f"  {k}: {v:,}")

# Distribution percentiles
def report_percentiles(name, counter):
    vals = np.array(list(counter.values()))
    print(f"\n--- Distribution for {name} ({len(vals):,} keys) ---")
    print(f"  Min: {np.min(vals)}")
    print(f"  Median: {np.median(vals)}")
    print(f"  Mean: {np.mean(vals):.2f}")
    print(f"  P90: {np.percentile(vals, 90):.1f}")
    print(f"  P95: {np.percentile(vals, 95):.1f}")
    print(f"  P99: {np.percentile(vals, 99):.1f}")
    print(f"  P99.5: {np.percentile(vals, 99.5):.1f}")
    print(f"  P99.9: {np.percentile(vals, 99.9):.1f}")
    print(f"  Max: {np.max(vals):,}")
    for th in [100, 250, 500, 1000, 2500, 5000, 10000]:
        cnt = int((vals > th).sum())
        print(f"  Keys > {th}: {cnt:,} ({cnt/len(vals)*100:.3f}%)")

report_percentiles("Block C (Core 12)", c_freq)
report_percentiles("Block D (Postal + First Token)", d_post_freq)
report_percentiles("Block D (Region + First Token)", d_reg_freq)
d_combined = Counter(d_post_freq)
d_combined.update(d_reg_freq)
report_percentiles("Block D Combined", d_combined)
