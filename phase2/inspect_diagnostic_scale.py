import os
import json
import numpy as np

def main():
    with open("phase2/stratified_sample_25k.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"Total queries: {len(queries)}")

    cand_dir = "phase2/candidates"
    data_a = np.load(os.path.join(cand_dir, "cands_block_a.npz"))
    data_b = np.load(os.path.join(cand_dir, "cands_block_b.npz"))
    data_c = np.load(os.path.join(cand_dir, "cands_block_c.npz"))
    data_d = np.load(os.path.join(cand_dir, "cands_block_d.npz"))
    data_e = np.load(os.path.join(cand_dir, "cands_block_e.npz"))

    print("Candidate array sizes:")
    print("  Block A:", len(data_a["candidates"]))
    print("  Block B:", len(data_b["candidates"]))
    print("  Block C:", len(data_c["candidates"]))
    print("  Block D:", len(data_d["candidates"]))
    print("  Block E:", len(data_e["candidates"]))

if __name__ == "__main__":
    main()
