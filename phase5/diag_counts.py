import sys, os
sys.path.insert(0, '.')
from phase2.blocking import EntityNormalizer
from phase3.address_blocking import AddressNormalizer
import array
from collections import defaultdict, Counter
import numpy as np

normalizer = EntityNormalizer()
norm_addr = AddressNormalizer()

target_eids, target_names, target_addrs = [], [], []
with open('student_resource/dataset/test/test_source2.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.strip('\r\n').split('\t')
        if len(parts) >= 4 and parts[3].strip() == 'France':
            target_eids.append(parts[0])
            target_names.append(parts[1])
            target_addrs.append(parts[2])

print(f'Targets: {len(target_eids):,}')

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
    name, addr = target_names[tid], target_addrs[tid]
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
        sig = normalizer.extract_address_signals(addr, 'France')
        if not sig['is_empty']:
            if sig['postal_code']:
                idx_d_post[f"{sig['postal_code']}_{first_tok}"].append(tid)
            if sig['region']:
                idx_d_reg[f"{sig['region']}_{first_tok}"].append(tid)
    for ng in set(normalizer.extract_char_ngrams(norm_n, n=3)):
        idx_e[ng].append(tid)
        ng_freq[ng] += 1
    clean_a = norm_addr.clean_address(addr)
    toks_a = norm_addr.extract_tokens(clean_a, min_len=2, filter_generic=True)
    spec_toks = [t for t in toks_a if any(c.isdigit() for c in t) or len(t) >= 4]
    if len(spec_toks) >= 2:
        for i_t in range(min(len(spec_toks), 5)):
            for j_t in range(i_t + 1, min(len(spec_toks), 5)):
                t1, t2 = sorted([spec_toks[i_t], spec_toks[j_t]])
                idx_g6[f"{t1}_{t2}"].append(tid)
                g6_freq[f"{t1}_{t2}"] += 1

# Prune
for t in list(idx_b.keys()):
    if tok_freq[t] > 10000: del idx_b[t]
for ng in list(idx_e.keys()):
    if ng_freq[ng] > 10000: del idx_e[ng]
for k in list(idx_g6.keys()):
    if g6_freq[k] > 2500: del idx_g6[k]

print('Indexes built.')

s1_queries = []
with open('student_resource/dataset/test/test_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.strip('\r\n').split('\t')
        if len(parts) >= 4 and parts[3].strip() == 'France':
            s1_queries.append((parts[0], parts[1], parts[2]))
            if len(s1_queries) >= 1000: break

counts = Counter()
cands_per_query = []
for q_eid, q_name, q_addr in s1_queries:
    norm_q = normalizer.normalize_name(q_name)
    tok_list_q = normalizer.extract_tokens(norm_q)
    core_q = normalizer.extract_compressed_core(norm_q, prefix_len=12)
    cand_bits = defaultdict(int)
    if norm_q in idx_a:
        for tid in idx_a[norm_q]: cand_bits[tid] |= 1; counts['A'] += 1
    for t in set(tok_list_q):
        if t in idx_b:
            for tid in idx_b[t]: cand_bits[tid] |= 2; counts['B'] += 1
    if core_q in idx_c:
        for tid in idx_c[core_q]: cand_bits[tid] |= 4; counts['C'] += 1
    if tok_list_q:
        first_tok_q = tok_list_q[0]
        sig_q = normalizer.extract_address_signals(q_addr, 'France')
        if not sig_q['is_empty']:
            if sig_q['postal_code']:
                k = f"{sig_q['postal_code']}_{first_tok_q}"
                if k in idx_d_post:
                    for tid in idx_d_post[k]: cand_bits[tid] |= 8; counts['D_post'] += 1
            if sig_q['region']:
                k = f"{sig_q['region']}_{first_tok_q}"
                if k in idx_d_reg:
                    for tid in idx_d_reg[k]: cand_bits[tid] |= 8; counts['D_reg'] += 1
    ngrams_q = set(normalizer.extract_char_ngrams(norm_q, n=3))
    p_views = [np.frombuffer(idx_e[ng], dtype=np.uint32) for ng in ngrams_q if ng in idx_e and len(idx_e[ng]) > 0]
    if p_views:
        u, cnts = np.unique(np.concatenate(p_views), return_counts=True)
        for tid in u[cnts >= 6]: cand_bits[int(tid)] |= 16; counts['E'] += 1
    cands_per_query.append(len(cand_bits))

print('Block candidate pair contributions for 1,000 queries:', dict(counts))
print(f'Total candidate pairs: {sum(cands_per_query):,} (Avg {np.mean(cands_per_query):.1f}/query, Max {max(cands_per_query)})')
