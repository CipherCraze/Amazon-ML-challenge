import os
import sys
import time
import json
import gc
import numpy as np
import pandas as pd
import psutil

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

FEATURE_NAMES = [
    # Group A: Name Features (12)
    "name_exact_match", "name_alphanumeric_match", "name_token_jaccard", "name_token_dice",
    "name_token_overlap", "name_jaro_winkler", "name_normalized_levenshtein", "name_lcs_similarity",
    "name_char_3gram_jaccard", "name_compressed_core_sim", "name_length_diff", "name_token_count_diff",
    # Group B: Address Features (12)
    "addr_exact_match", "addr_token_jaccard", "addr_token_overlap", "addr_char_3gram_jaccard",
    "addr_normalized_levenshtein", "addr_numeric_exact_match", "addr_numeric_shared_count",
    "addr_numeric_conflict", "addr_postal_code_match", "addr_rare_token_shared_count",
    "addr_length_ratio", "addr_token_count_diff",
    # Group C: Cross-Script / Interaction Features (6)
    "is_cross_script", "cross_script_x_addr_overlap", "cross_script_x_numeric_match",
    "country_code", "target_source", "primary_brand_match",
    # Group D: Blocking / Provenance Features (6)
    "block_hit_count", "hit_block_a", "hit_block_b", "hit_block_c", "hit_block_d", "hit_block_g6"
]

def get_rss_mb():
    """Empirically measure current RSS using psutil."""
    return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)

def compute_pairwise_f05(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def compute_official_macro_f05(val_meta_df: pd.DataFrame, y_pred: np.ndarray, val_gt_dict: dict) -> dict:
    """
    Computes the official competition metric:
    Macro-average F_0.5 across all Source 1 entities in the evaluation set.
    Singletons:
      - Empty prediction list -> 1.0
      - Any match prediction -> 0.0
    Non-singletons:
      - Empty prediction list -> 0.0
      - Precision = |Pred ∩ GT| / |Pred|
      - Recall = |Pred ∩ GT| / |GT|
      - F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    # Group predictions by s1_eid
    val_meta = val_meta_df.copy()
    val_meta["pred"] = y_pred

    pred_grouped = {}
    pred_pos = val_meta[val_meta["pred"] == 1]
    for s1, group in pred_pos.groupby("s1_eid"):
        pred_grouped[s1] = set(group["cand_eid"].tolist())

    s1_scores = []
    singleton_scores = []
    non_singleton_scores = []

    for s1, gt_set in val_gt_dict.items():
        pred_set = pred_grouped.get(s1, set())

        if len(gt_set) == 0:
            # Singleton entity
            sc = 1.0 if len(pred_set) == 0 else 0.0
            s1_scores.append(sc)
            singleton_scores.append(sc)
        else:
            # Non-singleton entity
            if len(pred_set) == 0:
                sc = 0.0
            else:
                tp = len(pred_set & gt_set)
                p = tp / len(pred_set)
                r = tp / len(gt_set)
                if p + r == 0.0:
                    sc = 0.0
                else:
                    sc = (1.25 * p * r) / (0.25 * p + r)
            s1_scores.append(sc)
            non_singleton_scores.append(sc)

    macro_f05 = float(np.mean(s1_scores))
    singleton_acc = float(np.mean(singleton_scores)) if singleton_scores else 0.0
    non_singleton_f05 = float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0

    return {
        "macro_f05": round(macro_f05, 4),
        "singleton_accuracy": round(singleton_acc, 4),
        "non_singleton_macro_f05": round(non_singleton_f05, 4),
        "total_evaluated_s1": len(s1_scores),
        "singleton_count": len(singleton_scores),
        "non_singleton_count": len(non_singleton_scores)
    }

def evaluate_predictions_sweep(y_true: np.ndarray, y_prob: np.ndarray, val_meta_df: pd.DataFrame, val_gt_dict: dict, num_val_s1: int = 5000) -> dict:
    roc_auc = round(float(roc_auc_score(y_true, y_prob)), 4)
    pr_auc = round(float(average_precision_score(y_true, y_prob)), 4)

    # Threshold grid from 0.05 to 0.95 with finer increments around 0.50 to 0.85
    thresh_coarse_1 = np.linspace(0.05, 0.45, 9)
    thresh_fine = np.linspace(0.50, 0.85, 15)
    thresh_coarse_2 = np.linspace(0.88, 0.95, 4)
    thresholds = np.unique(np.round(np.concatenate([thresh_coarse_1, thresh_fine, thresh_coarse_2]), 3))

    best_macro_f05 = -1.0
    best_tau = 0.50
    best_metrics = {}
    thresh_curve = []

    for tau in thresholds:
        y_pred = (y_prob >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f05_pair = compute_pairwise_f05(p, r)
        f1_pair = f1_score(y_true, y_pred, zero_division=0)
        fp_per_s1 = fp / num_val_s1

        # Macro F0.5
        macro_res = compute_official_macro_f05(val_meta_df, y_pred, val_gt_dict)

        cur_m = {
            "threshold": round(float(tau), 3),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "pairwise_f05": round(float(f05_pair), 4),
            "f1": round(float(f1_pair), 4),
            "macro_f05": macro_res["macro_f05"],
            "singleton_acc": macro_res["singleton_accuracy"],
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "pred_positives": int(tp + fp),
            "fp_per_s1": round(float(fp_per_s1), 4)
        }
        thresh_curve.append(cur_m)

        if macro_res["macro_f05"] > best_macro_f05:
            best_macro_f05 = macro_res["macro_f05"]
            best_tau = tau
            best_metrics = cur_m

    # Default at 0.50
    default_metrics = min(thresh_curve, key=lambda m: abs(m["threshold"] - 0.50))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "optimal_tau": round(float(best_tau), 3),
        "optimal_metrics": best_metrics,
        "default_metrics": default_metrics,
        "threshold_curve": thresh_curve
    }

def main():
    start_time = time.time()
    print("=" * 80, flush=True)
    print("PHASE 4D: ENSEMBLE EVALUATION & PROBABILITY CALIBRATION", flush=True)
    print("=" * 80, flush=True)

    peak_rss = get_rss_mb()
    print(f"Initial RSS: {peak_rss:.2f} MB", flush=True)

    data_dir = "phase4/data"
    feat_path = os.path.join(data_dir, "pair_features_ratio_1_10.parquet")
    gt_path = "student_resource/dataset/train/train_ground_truth.tsv"

    print(f"Loading {feat_path}...", flush=True)
    t0 = time.time()
    df = pd.read_parquet(feat_path)
    print(f"Loaded {len(df):,} pairs in {time.time()-t0:.2f}s | RSS: {get_rss_mb():.2f} MB", flush=True)

    # Load Ground Truth for validation entities
    print(f"Loading Ground Truth from {gt_path}...", flush=True)
    gt_df = pd.read_csv(gt_path, sep="\t")
    val_s1_set = set(df[df["split"] == "validation"]["s1_eid"].unique())
    print(f"Total validation S1 entities: {len(val_s1_set):,}", flush=True)

    val_gt = gt_df[gt_df["source1_entity_id"].isin(val_s1_set)]
    val_gt_dict = {}
    for _, row in val_gt.iterrows():
        s1 = row["source1_entity_id"]
        raw = row["matched_entity_ids"]
        if pd.isna(raw) or not str(raw).strip():
            val_gt_dict[s1] = set()
        else:
            val_gt_dict[s1] = set(m.strip() for m in str(raw).split(",") if m.strip())

    train_mask = (df["split"] == "train")
    val_mask = (df["split"] == "validation")

    X_train = df.loc[train_mask, FEATURE_NAMES].values.astype(np.float32)
    y_train = df.loc[train_mask, "label"].values.astype(int)

    X_val = df.loc[val_mask, FEATURE_NAMES].values.astype(np.float32)
    y_val = df.loc[val_mask, "label"].values.astype(int)

    val_meta_df = df.loc[val_mask, ["s1_eid", "cand_eid", "label", "country", "is_singleton", "is_cross_script"]].reset_index(drop=True)

    print(f"Train set: {X_train.shape[0]:,} pairs | Val set: {X_val.shape[0]:,} pairs", flush=True)

    # Train base models
    print("\n--- Training Base Models on Ratio 1:10 ---", flush=True)

    # 1. Logistic Regression
    print("1. Training Logistic Regression...", flush=True)
    t0_lr = time.time()
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    lr.fit(X_tr_sc, y_train)
    lr_train_time = time.time() - t0_lr
    X_v_sc = scaler.transform(X_val)
    prob_lr = lr.predict_proba(X_v_sc)[:, 1]
    print(f"   Done in {lr_train_time:.2f}s | Val prob range: [{prob_lr.min():.4f}, {prob_lr.max():.4f}]", flush=True)

    # 2. LightGBM
    print("2. Training LightGBM...", flush=True)
    t0_lgb = time.time()
    lgb = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
    )
    lgb.fit(X_train, y_train)
    lgb_train_time = time.time() - t0_lgb
    prob_lgb = lgb.predict_proba(X_val)[:, 1]
    print(f"   Done in {lgb_train_time:.2f}s | Val prob range: [{prob_lgb.min():.4f}, {prob_lgb.max():.4f}]", flush=True)

    # 3. XGBoost
    print("3. Training XGBoost...", flush=True)
    t0_xgb = time.time()
    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    xgb_train_time = time.time() - t0_xgb
    prob_xgb = xgb.predict_proba(X_val)[:, 1]
    print(f"   Done in {xgb_train_time:.2f}s | Val prob range: [{prob_xgb.min():.4f}, {prob_xgb.max():.4f}]", flush=True)

    # 4. CatBoost
    print("4. Training CatBoost...", flush=True)
    t0_cat = time.time()
    cat = CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=6, random_seed=42, thread_count=-1, verbose=0
    )
    cat.fit(X_train, y_train)
    cat_train_time = time.time() - t0_cat
    prob_cat = cat.predict_proba(X_val)[:, 1]
    print(f"   Done in {cat_train_time:.2f}s | Val prob range: [{prob_cat.min():.4f}, {prob_cat.max():.4f}]", flush=True)

    # Free heavy feature matrices
    del X_train, y_train, X_tr_sc, X_v_sc, df
    gc.collect()

    # Persist validation model predictions
    val_pred_df = val_meta_df.copy()
    val_pred_df["prob_lr"] = prob_lr.astype(np.float32)
    val_pred_df["prob_lgb"] = prob_lgb.astype(np.float32)
    val_pred_df["prob_xgb"] = prob_xgb.astype(np.float32)
    val_pred_df["prob_cat"] = prob_cat.astype(np.float32)

    val_pred_path = os.path.join(data_dir, "validation_model_predictions.parquet")
    val_pred_df.to_parquet(val_pred_path, index=False)
    print(f"\nPersisted {len(val_pred_df):,} validation predictions to {val_pred_path} ({os.path.getsize(val_pred_path)/(1024*1024):.2f} MB)", flush=True)

    # Define candidate models and ensembles
    candidate_configs = {
        # Single Models
        "Logistic_Regression": prob_lr,
        "LightGBM": prob_lgb,
        "XGBoost": prob_xgb,
        "CatBoost": prob_cat,

        # 2-Model Ensembles: LightGBM + XGBoost
        "Ens_LGB_50_XGB_50": 0.50 * prob_lgb + 0.50 * prob_xgb,
        "Ens_LGB_60_XGB_40": 0.60 * prob_lgb + 0.40 * prob_xgb,
        "Ens_LGB_70_XGB_30": 0.70 * prob_lgb + 0.30 * prob_xgb,

        # 2-Model Ensembles: LightGBM + CatBoost
        "Ens_LGB_50_CAT_50": 0.50 * prob_lgb + 0.50 * prob_cat,
        "Ens_LGB_60_CAT_40": 0.60 * prob_lgb + 0.40 * prob_cat,
        "Ens_LGB_70_CAT_30": 0.70 * prob_lgb + 0.30 * prob_cat,

        # 2-Model Ensembles: XGBoost + CatBoost
        "Ens_XGB_50_CAT_50": 0.50 * prob_xgb + 0.50 * prob_cat,
        "Ens_XGB_60_CAT_40": 0.60 * prob_xgb + 0.40 * prob_cat,
        "Ens_XGB_70_CAT_30": 0.70 * prob_xgb + 0.30 * prob_cat,

        # 3-Model Ensembles: LightGBM + XGBoost + CatBoost
        "Ens_LGB_40_XGB_30_CAT_30": 0.40 * prob_lgb + 0.30 * prob_xgb + 0.30 * prob_cat,
        "Ens_LGB_50_XGB_25_CAT_25": 0.50 * prob_lgb + 0.25 * prob_xgb + 0.25 * prob_cat,
        "Ens_LGB_25_XGB_50_CAT_25": 0.25 * prob_lgb + 0.50 * prob_xgb + 0.25 * prob_cat,
        "Ens_LGB_25_XGB_25_CAT_50": 0.25 * prob_lgb + 0.25 * prob_xgb + 0.50 * prob_cat,
        "Ens_LGB_34_XGB_33_CAT_33": (prob_lgb + prob_xgb + prob_cat) / 3.0,
    }

    print("\n--- Evaluating Models and Ensembles ---", flush=True)
    results = {}
    for name, p_vec in candidate_configs.items():
        t_eval = time.time()
        ev = evaluate_predictions_sweep(y_val, p_vec, val_meta_df, val_gt_dict)
        ev["eval_time_sec"] = round(time.time() - t_eval, 3)
        results[name] = ev
        opt = ev["optimal_metrics"]
        print(f"[{name:28s}] PR-AUC: {ev['pr_auc']:.4f} | ROC-AUC: {ev['roc_auc']:.4f} | Optimal tau: {ev['optimal_tau']:.3f} -> Macro F0.5: {opt['macro_f05']:.4f} (Pair F0.5: {opt['pairwise_f05']:.4f}, P: {opt['precision']:.4f}, R: {opt['recall']:.4f}, FP/S1: {opt['fp_per_s1']:.4f})", flush=True)

    # Save summary json
    out_json = "phase4/data/ensemble_evaluation_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ensemble evaluation results to {out_json}", flush=True)
    print(f"Total time: {time.time()-start_time:.2f}s | Peak RSS: {get_rss_mb():.2f} MB", flush=True)

if __name__ == "__main__":
    main()
