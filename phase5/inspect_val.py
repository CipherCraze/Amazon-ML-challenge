import pandas as pd
import numpy as np

df_feat = pd.read_parquet('phase4/data/pair_features_ratio_1_10.parquet')
val_mask = (df_feat['split'] == 'validation')
df_val = df_feat[val_mask]

df_pairs = pd.read_parquet('phase4/data/training_pairs_ratio_1_10.parquet')
df_pairs_val = df_pairs[val_mask]

print(f"Total validation pairs in phase4: {len(df_val):,}")
print(f"Unique validation queries: {df_val['s1_eid'].nunique():,}")
print(f"Label counts in val:\n{df_val['label'].value_counts()}")
print("Block bitmask counts:")
for bit, name in [(1,'A'), (2,'B'), (4,'C'), (8,'D'), (16,'E'), (32,'G6')]:
    hit = (df_pairs_val['block_bitmask'] & bit) > 0
    print(f"  Hit {name}: {hit.sum():,} ({hit.mean()*100:.2f}%)")
