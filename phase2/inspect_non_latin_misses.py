import os
import sys
import json
import re
from collections import Counter
import numpy as np

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

NON_LATIN_REGEX = re.compile(r'[\u0900-\u0D7F\u0E00-\u0E7F\u4E00-\u9FFF\u0600-\u06FF\u0400-\u04FF]')
PIN_REGEX = re.compile(r'\b[1-9][0-9]{5}\b')

def main():
    print("Loading queries...", flush=True)
    with open("phase2/stratified_sample_25k.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    # Load candidate arrays once
    print("Loading candidate arrays...", flush=True)
    cand_dir = "phase2/candidates"
    data_a = np.load(os.path.join(cand_dir, "cands_block_a.npz"))
    data_b = np.load(os.path.join(cand_dir, "cands_block_b.npz"))
    data_c = np.load(os.path.join(cand_dir, "cands_block_c.npz"))
    data_d = np.load(os.path.join(cand_dir, "cands_block_d.npz"))
    data_e = np.load(os.path.join(cand_dir, "cands_block_e.npz"))

    # Extract arrays into memory once
    blocks = [
        (data_a["offsets"], data_a["candidates"]),
        (data_b["offsets"], data_b["candidates"]),
        (data_c["offsets"], data_c["candidates"]),
        (data_d["offsets"], data_d["candidates"]),
        (data_e["offsets"], data_e["candidates"]),
    ]

    query_keys = list(queries.keys())
    all_needed_gt = {m for q in queries.values() for m in q["gt_matches"]}

    print("Scanning target records...", flush=True)
    gt_meta = {}
    int_id = 0
    s2_path = "student_resource/dataset/train/train_source2.tsv"
    s3_path = "student_resource/dataset/train/train_source3.tsv"

    for path in [s2_path, s3_path]:
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip("\r\n").split("\t")
                if len(parts) >= 4:
                    eid = parts[0]
                    if eid in all_needed_gt:
                        gt_meta[eid] = {
                            "int_id": int_id,
                            "eid": eid,
                            "name": parts[1],
                            "address": parts[2],
                            "country": parts[3]
                        }
                int_id += 1

    # Map query GT to int_ids
    for q in queries.values():
        q["gt_ints"] = {gt_meta[m]["int_id"]: m for m in q["gt_matches"] if m in gt_meta}

    print("Finding non-Latin misses in final union...", flush=True)
    non_latin_misses = []
    
    for i, eid in enumerate(query_keys):
        q = queries[eid]
        cands = set()
        for off, arr in blocks:
            start = off[i]
            end = off[i+1]
            if start < end:
                cands.update(arr[start:end])

        for gt_int, gt_eid in q["gt_ints"].items():
            if gt_int not in cands:
                target = gt_meta[gt_eid]
                t_name = target["name"]
                if NON_LATIN_REGEX.search(t_name):
                    non_latin_misses.append({
                        "s1_id": eid,
                        "s1_name": q["business_name"],
                        "s1_addr": q["business_address"],
                        "s1_country": q["country"],
                        "target_id": gt_eid,
                        "target_name": t_name,
                        "target_addr": target["address"],
                        "target_country": target["country"]
                    })

    print(f"Total non-Latin misses identified: {len(non_latin_misses)}", flush=True)

    # Analyze address signals of non-Latin misses
    s1_has_pin = 0
    target_has_pin = 0
    pin_match = 0
    target_addr_latin = 0
    addr_token_overlap = 0

    script_counts = Counter()

    for m in non_latin_misses:
        t_name = m["target_name"]
        # Detect Unicode block
        for ch in t_name:
            code = ord(ch)
            if 0x0900 <= code <= 0x097F:
                script_counts["Devanagari"] += 1
                break
            elif 0x0C00 <= code <= 0x0C7F:
                script_counts["Telugu"] += 1
                break
            elif 0x0B80 <= code <= 0x0BFF:
                script_counts["Tamil"] += 1
                break
            elif 0x0C80 <= code <= 0x0CFF:
                script_counts["Kannada"] += 1
                break
            elif 0x0D00 <= code <= 0x0D7F:
                script_counts["Malayalam"] += 1
                break
            elif 0x0A80 <= code <= 0x0AFF:
                script_counts["Gujarati"] += 1
                break
            elif 0x0980 <= code <= 0x09FF:
                script_counts["Bengali"] += 1
                break

        s1_pins = set(PIN_REGEX.findall(m["s1_addr"]))
        t_pins = set(PIN_REGEX.findall(m["target_addr"]))
        if s1_pins: s1_has_pin += 1
        if t_pins: target_has_pin += 1
        if s1_pins and t_pins and (s1_pins & t_pins):
            pin_match += 1
        if not NON_LATIN_REGEX.search(m["target_addr"]):
            target_addr_latin += 1

        s1_toks = set(re.findall(r'\b\w+\b', m["s1_addr"].lower()))
        t_toks = set(re.findall(r'\b\w+\b', m["target_addr"].lower()))
        if len(s1_toks & t_toks) >= 2:
            addr_token_overlap += 1

    print("\nScript Breakdown of Misses:", flush=True)
    for s_name, cnt in script_counts.most_common():
        print(f"  {s_name}: {cnt} ({cnt/len(non_latin_misses)*100:.1f}%)", flush=True)

    print("\nNon-Latin Misses Address Analysis:", flush=True)
    print(f"  S1 address has PIN: {s1_has_pin}/{len(non_latin_misses)} ({s1_has_pin/len(non_latin_misses)*100:.1f}%)", flush=True)
    print(f"  Target address has PIN: {target_has_pin}/{len(non_latin_misses)} ({target_has_pin/len(non_latin_misses)*100:.1f}%)", flush=True)
    print(f"  Exact PIN match: {pin_match}/{len(non_latin_misses)} ({pin_match/len(non_latin_misses)*100:.1f}%)", flush=True)
    print(f"  Target address is written in Latin: {target_addr_latin}/{len(non_latin_misses)} ({target_addr_latin/len(non_latin_misses)*100:.1f}%)", flush=True)
    print(f"  Address shared tokens >= 2: {addr_token_overlap}/{len(non_latin_misses)} ({addr_token_overlap/len(non_latin_misses)*100:.1f}%)", flush=True)

    print("\nSample Misses:", flush=True)
    for ex in non_latin_misses[:6]:
        print(f"  S1: '{ex['s1_name']}' | Addr: '{ex['s1_addr']}'", flush=True)
        print(f"  Target: '{ex['target_name']}' | Addr: '{ex['target_addr']}' ({ex['target_id']})", flush=True)
        print("  ---", flush=True)

if __name__ == "__main__":
    main()
