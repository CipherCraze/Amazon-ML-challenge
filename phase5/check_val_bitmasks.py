import pandas as pd
from collections import Counter

df_feat = pd.read_parquet('phase4/data/pair_features_ratio_1_10.parquet')
val_mask = (df_feat['split'] == 'validation')
df_pairs = pd.read_parquet('phase4/data/training_pairs_ratio_1_10.parquet')[val_mask]

bitmasks = df_pairs['block_bitmask'].values
labels = df_feat.loc[val_mask, 'label'].values

print("Bitmasks with only D (bitmask == 8):")
only_d = (bitmasks == 8)
print(f"Total only D: {only_d.sum():,} (Pos: {(only_d & (labels==1)).sum():,}, Neg: {(only_d & (labels==0)).sum():,})")

print("\nBitmasks with D (bitmask & 8 != 0):")
has_d = ((bitmasks & 8) != 0)
print(f"Total has D: {has_d.sum():,} (Pos: {(has_d & (labels==1)).sum():,}, Neg: {(has_d & (labels==0)).sum():,})")

print("\nBitmask counts with labels:")
for bm, cnt in Counter(bitmasks).most_common(20):
    pos = ((bitmasks == bm) & (labels == 1)).sum()
    neg = ((bitmasks == bm) & (labels == 0)).sum()
    blocks = []
    if bm & 1: blocks.append('A')
    if bm & 2: blocks.append('B')
    if bm & 4: blocks.append('C')
    if bm & 8: blocks.append('D')
    if bm & 16: blocks.append('E')
    if bm & 32: blocks.append('G6')
    print(f"  Bitmask {bm:2d} ({'+'.join(blocks):<12}): Total={cnt:6,d}, Pos={pos:5,d}, Neg={neg:6,d}")
