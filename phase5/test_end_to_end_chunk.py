import os
import sys
import time
import array
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import psutil

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase2"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase3"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase4"))

from phase2.blocking import EntityNormalizer
from phase3.address_blocking import AddressNormalizer
from phase4.extract_pair_features import precompute_entity, compute_pair_features, FEATURE_NAMES
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

print("Testing end-to-end chunk pipeline on France...", flush=True)

# 1. Train quick models on ratio 1:10 sample
print("Loading training data...", flush=True)
df_train = pd.read_parquet("phase4/data/pair_features_ratio_1_10.parquet")
train_mask = (df_train["split"] == "train")
X_tr = df_train.loc[train_mask, FEATURE_NAMES].values.astype(np.float32)
y_tr = df_train.loc[train_mask, "label"].values.astype(int)

print(f"Training LightGBM on {len(X_tr):,} pairs...", flush=True)
lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
lgb.fit(X_tr, y_tr)

print(f"Training XGBoost on {len(X_tr):,} pairs...", flush=True)
xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=-1)
xgb.fit(X_tr, y_tr)

del df_train, X_tr, y_tr
import gc; gc.collect()
print("Models trained.", flush=True)
