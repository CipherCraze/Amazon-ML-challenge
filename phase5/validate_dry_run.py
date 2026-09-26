import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("VALIDATING DRY-RUN OUTPUTS AGAINST OFFICIAL SUBMISSION RULES")
print("=" * 80)

cand_path = "output/dry_run/candidate_pairs.tsv"
match_path = "output/dry_run/matching_results.tsv"

assert os.path.exists(cand_path), f"Missing {cand_path}"
assert os.path.exists(match_path), f"Missing {match_path}"

# 1. Header checks
with open(cand_path, "r", encoding="utf-8") as f:
    cand_header = f.readline().strip("\r\n").split("\t")
assert cand_header == ["source1_entity_id", "candidate_entity_ids"], f"Invalid candidate header: {cand_header}"
print("✓ Candidate header valid: ['source1_entity_id', 'candidate_entity_ids']")

with open(match_path, "r", encoding="utf-8") as f:
    match_header = f.readline().strip("\r\n").split("\t")
assert match_header == ["source1_entity_id", "matched_entity_ids"], f"Invalid match header: {match_header}"
print("✓ Match header valid: ['source1_entity_id', 'matched_entity_ids']")

# 2. Row count and deduplication check
cand_dict = {}
cand_s1_list = []
with open(cand_path, "r", encoding="utf-8") as f:
    next(f)
    for line_num, line in enumerate(f, start=2):
        parts = line.strip("\r\n").split("\t")
        assert len(parts) in (1, 2), f"Line {line_num} malformed: {len(parts)} columns"
        s1 = parts[0]
        cands = parts[1].split(",") if len(parts) == 2 and parts[1].strip() else []
        assert s1 not in cand_dict, f"Duplicate S1 entity in candidate_pairs: {s1}"
        assert len(cands) == len(set(cands)), f"Duplicate candidate IDs for {s1}"
        assert s1 not in cands, f"Self-match (S1 in candidates) for {s1}"
        cand_dict[s1] = set(cands)
        cand_s1_list.append(s1)

print(f"✓ Candidate file valid: {len(cand_dict):,} unique S1 entities, no duplicate IDs, no self-matches")

match_dict = {}
match_s1_list = []
total_matches = 0
with open(match_path, "r", encoding="utf-8") as f:
    next(f)
    for line_num, line in enumerate(f, start=2):
        parts = line.strip("\r\n").split("\t")
        assert len(parts) in (1, 2), f"Line {line_num} malformed: {len(parts)} columns"
        s1 = parts[0]
        matches = parts[1].split(",") if len(parts) == 2 and parts[1].strip() else []
        assert s1 not in match_dict, f"Duplicate S1 entity in matching_results: {s1}"
        assert len(matches) == len(set(matches)), f"Duplicate match IDs for {s1}"
        assert s1 not in matches, f"Self-match (S1 in matches) for {s1}"
        match_dict[s1] = set(matches)
        match_s1_list.append(s1)
        total_matches += len(matches)

print(f"✓ Matching file valid: {len(match_dict):,} unique S1 entities, {total_matches:,} total matches")

# 3. Exact S1 alignment
assert cand_s1_list == match_s1_list, "S1 order mismatch between candidate_pairs and matching_results!"
print("✓ Exact 1:1 entity alignment and ordering verified between candidate_pairs and matching_results")

# 4. Strict subset check: matches must be subset of candidates
violations = 0
for s1, matches in match_dict.items():
    cands = cand_dict[s1]
    if not matches.issubset(cands):
        violations += 1
        print(f"ERROR: Matches for {s1} not a subset of candidates! Excess: {matches - cands}")
        if violations >= 5: break

assert violations == 0, f"Found {violations} subset violations!"
print("✓ Strict subset check passed: 100% of final matches are subsets of the candidate sets!")

# 5. Check target ID validity (must be S2 or S3 test targets)
print("\nVerifying target ID validity against test_source2 and test_source3...")
valid_target_ids = set()
for src in ["test_source2.tsv", "test_source3.tsv"]:
    with open(f"student_resource/dataset/test/{src}", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            valid_target_ids.add(line.split("\t", 1)[0].strip())

print(f"Loaded {len(valid_target_ids):,} valid target IDs from test sources.")

invalid_cands = 0
for s1, cands in cand_dict.items():
    diff = cands - valid_target_ids
    if diff:
        invalid_cands += len(diff)
assert invalid_cands == 0, f"Found {invalid_cands} fabricated/invalid candidate IDs!"
print("✓ 100% of candidate IDs exist in test_source2 or test_source3 (zero fabricated IDs)")

invalid_matches = 0
for s1, matches in match_dict.items():
    diff = matches - valid_target_ids
    if diff:
        invalid_matches += len(diff)
assert invalid_matches == 0, f"Found {invalid_matches} fabricated/invalid matched IDs!"
print("✓ 100% of matched IDs exist in test_source2 or test_source3 (zero fabricated matches)")

print("\n" + "=" * 80)
print("ALL SUBMISSION VALIDATION CHECKS PASSED WITH ZERO ERRORS!")
print("=" * 80)
