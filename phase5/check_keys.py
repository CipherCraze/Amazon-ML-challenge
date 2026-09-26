import os
import sys
import json
import time
from collections import defaultdict, Counter
import array
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from phase2.blocking import EntityNormalizer
from phase3.address_blocking import AddressNormalizer

normalizer = EntityNormalizer()
norm_addr = AddressNormalizer()

t0 = time.time()
with open("phase2/stratified_sample_25k.json", "r", encoding="utf-8") as f:
    dev_queries = json.load(f)

needed_d_post = set()
needed_d_reg = set()
for q in dev_queries.values():
    c = q["country"]
    norm_q = normalizer.normalize_name(q["business_name"])
    toks_q = normalizer.extract_tokens(norm_q)
    if toks_q:
        first_tok_q = toks_q[0]
        sig_q = normalizer.extract_address_signals(q["business_address"], c)
        if not sig_q["is_empty"]:
            if sig_q["postal_code"]:
                needed_d_post.add(f"{c}_{sig_q['postal_code']}_{first_tok_q}")
            if sig_q["region"]:
                needed_d_reg.add(f"{c}_{sig_q['region']}_{first_tok_q}")

print(f"Needed D post keys: {len(needed_d_post):,}, reg keys: {len(needed_d_reg):,}")
