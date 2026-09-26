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

# Load 100,000 targets for quick test
print("Loading subset of France targets...")
target_eids = []
target_names = []
target_addrs = []
with open('student_resource/dataset/test/test_source2.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.strip('\r\n').split('\t')
        if len(parts) >= 4 and parts[3] == 'France':
            target_eids.append(parts[0])
            target_names.append(parts[1])
            target_addrs.append(parts[2])
            if len(target_eids) >= 100000:
                break

idx_a = defaultdict(lambda: array.array('I'))
idx_b = defaultdict(lambda: array.array('I'))
idx_c = defaultdict(lambda: array.array('I'))
idx_d_post = defaultdict(lambda: array.array('I'))
idx_d_reg = defaultdict(lambda: array.array('I'))
idx_e = defaultdict(lambda: array.array('I'))
idx_g6 = defaultdict(lambda: array.array('I'))

for tid in range(len(target_eids)):
    name, addr = target_names[tid], target_addrs[tid]
    norm_n = normalizer.normalize_name(name)
    idx_a[norm_n].append(tid)
    toks = normalizer.extract_tokens(norm_n)
    for t in set(toks):
        idx_b[t].append(tid)
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
    for ng in set(normalizer.extract_char_ngrams(norm_n, n=3)):
        idx_e[ng].append(tid)

# Read 5,000 France S1 queries
s1_queries = []
with open('student_resource/dataset/test/test_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.strip('\r\n').split('\t')
        if len(parts) >= 4 and parts[3] == 'France':
            s1_queries.append((parts[0], parts[1], parts[2]))
            if len(s1_queries) >= 5000:
                break

print(f"Querying {len(s1_queries)} queries...")
t0_q = time.time()
cand_pair_count = 0
for s1_eid, name, addr in s1_queries:
    norm_q = normalizer.normalize_name(name)
    toks_q = set(normalizer.extract_tokens(norm_q))
    core_q = normalizer.extract_compressed_core(norm_q, prefix_len=12)
    
    cand_bits = defaultdict(int)
    # A
    if norm_q in idx_a:
        for tid in idx_a[norm_q]:
            cand_bits[tid] |= 1
    # B
    for t in toks_q:
        if t in idx_b:
            for tid in idx_b[t]:
                cand_bits[tid] |= 2
    # C
    if core_q in idx_c:
        for tid in idx_c[core_q]:
            cand_bits[tid] |= 4
    # E
    ng_counts = Counter()
    for ng in set(normalizer.extract_char_ngrams(norm_q, n=3)):
        if ng in idx_e:
            for tid in idx_e[ng]:
                ng_counts[tid] += 1
    for tid, cnt in ng_counts.items():
        if cnt >= 6:
            cand_bits[tid] |= 16

    cand_pair_count += len(cand_bits)

q_time = time.time() - t0_q
print(f"Queried {len(s1_queries)} queries in {q_time:.2f}s ({len(s1_queries)/q_time:.0f} queries/sec)")
print(f"Total candidate pairs generated: {cand_pair_count:,} (Avg {cand_pair_count/len(s1_queries):.1f}/S1)")
