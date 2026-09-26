import os
import sys
import json
import re
from collections import defaultdict, Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase2"))

from address_blocking import AddressNormalizer, ADDRESS_GENERIC_WORDS

def main():
    print("Testing address token distributions on 25k queries...", flush=True)
    with open("phase2/stratified_sample_25k.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    addr_norm = AddressNormalizer()
    
    # Token frequency counter across 25k queries
    tok_freq = Counter()
    postal_freq = Counter()
    pair_freq = Counter()

    for q in queries.values():
        c = q["country"]
        clean = addr_norm.clean_address(q["business_address"])
        post = addr_norm.extract_postal_code(q["business_address"], c)
        if post:
            postal_freq[(c, post)] += 1
        toks = addr_norm.extract_tokens(clean, min_len=2, filter_generic=True)
        for t in toks:
            tok_freq[(c, t)] += 1
        pairs = addr_norm.extract_token_pairs(clean, max_pairs=10, filter_generic=True)
        for p in pairs:
            pair_freq[(c, p[0], p[1])] += 1

    print(f"Total distinct postal codes in queries: {len(postal_freq):,}")
    print(f"Total distinct address tokens in queries: {len(tok_freq):,}")
    print(f"Total distinct address token pairs in queries: {len(pair_freq):,}")

    print("\nMost common postal codes in queries:")
    for (c, p), cnt in postal_freq.most_common(10):
        print(f"  {c} Postal '{p}': {cnt} queries")

    print("\nMost common address tokens in queries:")
    for (c, t), cnt in tok_freq.most_common(10):
        print(f"  {c} Token '{t}': {cnt} queries")

    print("\nMost common token pairs in queries:")
    for (c, t1, t2), cnt in pair_freq.most_common(10):
        print(f"  {c} Pair '{t1} + {t2}': {cnt} queries")

if __name__ == "__main__":
    main()
