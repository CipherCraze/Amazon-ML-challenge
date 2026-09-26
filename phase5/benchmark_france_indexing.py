import os
import sys
import time
import array
from collections import defaultdict, Counter
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase2"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase3"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase4"))

from phase2.blocking import EntityNormalizer
from phase3.address_blocking import AddressNormalizer

normalizer = EntityNormalizer()
norm_addr = AddressNormalizer()

t0 = time.time()
print("Reading France targets...")
target_eids = []
target_names = []
target_addrs = []

for p in ['student_resource/dataset/test/test_source2.tsv', 'student_resource/dataset/test/test_source3.tsv']:
    with open(p, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip('\r\n').split('\t')
            if len(parts) >= 4 and parts[3] == 'France':
                target_eids.append(parts[0])
                target_names.append(parts[1])
                target_addrs.append(parts[2])

print(f"Loaded {len(target_eids):,} France targets in {time.time()-t0:.2f}s | RSS: {psutil.Process().memory_info().rss/(1024*1024):.2f} MB")

t0_idx = time.time()
print("Building 6 blocking indexes for France...")
idx_a = defaultdict(lambda: array.array('I'))
idx_b = defaultdict(lambda: array.array('I'))
tok_freq = Counter()
idx_c = defaultdict(lambda: array.array('I'))
idx_d_post = defaultdict(lambda: array.array('I'))
idx_d_reg = defaultdict(lambda: array.array('I'))
idx_e = defaultdict(lambda: array.array('I'))
ng_freq = Counter()
idx_g6 = defaultdict(lambda: array.array('I'))
g6_freq = Counter()

for tid in range(len(target_eids)):
    name = target_names[tid]
    addr = target_addrs[tid]

    # Block A & B & C & D & E
    norm_n = normalizer.normalize_name(name)
    idx_a[norm_n].append(tid)

    toks = normalizer.extract_tokens(norm_n)
    for t in set(toks):
        idx_b[t].append(tid)
        tok_freq[t] += 1

    core12 = normalizer.extract_compressed_core(norm_n, prefix_len=12)
    if core12:
        idx_c[core12].append(tid)

    if toks:
        first_tok = toks[0]
        sig = normalizer.extract_address_signals(addr, "France")
        if not sig["is_empty"]:
            if sig["postal_code"]:
                idx_d_post[f"{sig['postal_code']}_{first_tok}"].append(tid)
            if sig["region"]:
                idx_d_reg[f"{sig['region']}_{first_tok}"].append(tid)

    ngrams = set(normalizer.extract_char_ngrams(norm_n, n=3))
    for ng in ngrams:
        idx_e[ng].append(tid)
        ng_freq[ng] += 1

    # Block G6
    clean_a = norm_addr.clean_address(addr)
    toks_a = norm_addr.extract_tokens(clean_a, min_len=2, filter_generic=True)
    spec_toks = [t for t in toks_a if any(c.isdigit() for c in t) or len(t) >= 4]
    if len(spec_toks) >= 2:
        for i_t in range(min(len(spec_toks), 5)):
            for j_t in range(i_t + 1, min(len(spec_toks), 5)):
                t1, t2 = sorted([spec_toks[i_t], spec_toks[j_t]])
                k = f"{t1}_{t2}"
                idx_g6[k].append(tid)
                g6_freq[k] += 1

print(f"Raw indexes built in {time.time()-t0_idx:.2f}s | RSS: {psutil.Process().memory_info().rss/(1024*1024):.2f} MB")

# Pruning
for t in list(idx_b.keys()):
    if tok_freq[t] > 10000:
        del idx_b[t]
del tok_freq

for ng in list(idx_e.keys()):
    if ng_freq[ng] > 10000:
        del idx_e[ng]
del ng_freq

for k in list(idx_g6.keys()):
    if g6_freq[k] > 2500:
        del idx_g6[k]
del g6_freq

print(f"Pruned indexes | Current RSS: {psutil.Process().memory_info().rss/(1024*1024):.2f} MB")
