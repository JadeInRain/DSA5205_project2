# -*- coding: utf-8 -*-
"""
step5_models.py
===============
Step 5: Model fitting + probability calibration (no leakage)

What this script does
---------------------
- Reads preprocessed features from Step 2&3: `data/prepared/btc_4h_preprocessed.csv`
- Reads time splits from Step 4: `data/splits/rolling_folds.csv`
- For each rolling fold and for each target (y_up, y_dn):
  1) Fit models on **train** only (no future info).
  2) Compute raw scores on **val**.
  3) Fit a **Platt scaler (logistic on raw scores)** on **val** to turn scores into calibrated probabilities.
  4) Save **calibrated probabilities** for the validation set (per-fold) to CSV.

Models included (all from scikit-learn)
---------------------------------------
- Baseline (linear): LogisticRegression (L2, class_weight='balanced')
- Linear margin:     LinearSVC (class_weight='balanced') + Platt scaling
- Tree ensemble:     GradientBoostingClassifier (shallow) + Platt scaling

Notes
-----
- Threshold search / trading rules are deferred to Step 6.
- We output out-of-fold **validation** probabilities for threshold tuning; models themselves are not persisted.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression as PlattScaler  # for calibration

# ========================
# User-editable constants
# ========================
INPUT_PREP_CSV = os.path.join("data", "prepared", "btc_4h_preprocessed.csv")
FOLDS_CSV      = os.path.join("data", "splits", "rolling_folds.csv")
OUT_DIR        = os.path.join("data", "predictions")

TIMESTAMP_CANDIDATES = ["timestamp", "open_time", "time", "datetime"]

# Feature exclusion rules (must match Step 2&3)
PRICE_COLS = ["open", "high", "low", "close", "volume"]
LABEL_COLS = ["y_up", "y_dn", "r_future_7d"]
EXCLUDE_ALSO = ["future_close", "next_open"]

# Models config
LOGREG_CFG = dict(C=1.0, penalty="l2", solver="liblinear",
                  class_weight="balanced", max_iter=2000, random_state=42)
LSVC_CFG   = dict(C=1.0, class_weight="balanced", random_state=42)
# IMPORTANT: random_state only in the config, not duplicated at call time
GBC_CFG    = dict(n_estimators=200, learning_rate=0.05, max_depth=2,
                  subsample=0.8, random_state=42)

RANDOM_STATE = 42

# ========================
# Utilities
# ========================
def _find_ts_col(df: pd.DataFrame) -> str:
    cols = [c.lower() for c in df.columns]
    for c in TIMESTAMP_CANDIDATES:
        if c in cols:
            return c
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            continue
    return df.columns[0]

def _load_data() -> Tuple[pd.DataFrame, str]:
    if not os.path.exists(INPUT_PREP_CSV):
        raise FileNotFoundError(f"Not found: {INPUT_PREP_CSV}")
    df = pd.read_csv(INPUT_PREP_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    ts_col = _find_ts_col(df)
    try:
        df[ts_col] = pd.to_datetime(df[ts_col])
    except Exception:
        pass
    df = df.sort_values(ts_col).reset_index(drop=True)
    return df, ts_col

def _load_folds() -> pd.DataFrame:
    if not os.path.exists(FOLDS_CSV):
        raise FileNotFoundError(f"Not found: {FOLDS_CSV}. Did you run Step 4?")
    folds = pd.read_csv(FOLDS_CSV)
    folds.columns = [c.strip().lower() for c in folds.columns]
    return folds

def _select_features(df: pd.DataFrame, ts_col: str) -> List[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = set([ts_col] + PRICE_COLS + LABEL_COLS + EXCLUDE_ALSO)
    exclude |= {c for c in num_cols if c.startswith("future_") or c.startswith("label_")}
    features = [c for c in num_cols if c not in exclude]
    if not features:
        raise RuntimeError("No numeric feature columns left after exclusion. "
                           "Check your input columns or exclusion rules.")
    return features

def _fit_models(X_tr, y_tr):
    # Baseline: Logistic Regression
    m1 = LogisticRegression(**LOGREG_CFG)
    m1.fit(X_tr, y_tr)

    # Linear SVM
    m2 = LinearSVC(**LSVC_CFG)
    m2.fit(X_tr, y_tr)

    # Gradient Boosting (shallow)
    m3 = GradientBoostingClassifier(**GBC_CFG)
    m3.fit(X_tr, y_tr)

    return m1, m2, m3

def _score_raw(model, X):
    # Logistic / GBC have predict_proba; LSVC uses decision_function
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        return proba
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X)
        return score
    else:
        # fallback
        return model.predict(X)

def _platt_calibrate(val_scores, y_val):
    # Fit logistic on raw scores -> calibrated probability
    scaler = PlattScaler(C=1.0, solver="liblinear", max_iter=1000)
    y = y_val.astype(int).reshape(-1)
    s = np.asarray(val_scores).reshape(-1, 1)
    scaler.fit(s, y)
    return scaler

def _eval_calibrated(p, y):
    y = y.astype(int).reshape(-1)
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = np.nan
    try:
        ap = average_precision_score(y, p)
    except Exception:
        ap = np.nan
    try:
        brier = brier_score_loss(y, p)
    except Exception:
        brier = np.nan
    return dict(auc=auc, ap=ap, brier=brier)

def run_step5():
    df, ts_col = _load_data()
    folds = _load_folds()
    features = _select_features(df, ts_col)

    os.makedirs(OUT_DIR, exist_ok=True)
    all_val_outputs = []

    fold_ids = sorted(folds["fold_id"].unique().tolist())
    for fid in fold_ids:
        ftrain = folds[(folds["fold_id"]==fid) & (folds["segment"]=="train")].iloc[0]
        fval   = folds[(folds["fold_id"]==fid) & (folds["segment"]=="val")].iloc[0]

        tr_idx = slice(int(ftrain["start_idx"]), int(ftrain["end_idx"])+1)
        va_idx = slice(int(fval["start_idx"]), int(fval["end_idx"])+1)

        X_tr = df.loc[tr_idx, features].values
        X_va = df.loc[va_idx, features].values

        y_up_tr = df.loc[tr_idx, "y_up"].astype("Int64")
        y_up_va = df.loc[va_idx, "y_up"].astype("Int64")
        y_dn_tr = df.loc[tr_idx, "y_dn"].astype("Int64")
        y_dn_va = df.loc[va_idx, "y_dn"].astype("Int64")

        # time-respecting NaN filtering per split
        tr_mask = np.isfinite(X_tr).all(axis=1) & y_up_tr.notna().values & y_dn_tr.notna().values
        va_mask = np.isfinite(X_va).all(axis=1) & y_up_va.notna().values & y_dn_va.notna().values

        X_tr2 = X_tr[tr_mask]
        X_va2 = X_va[va_mask]
        y_up_tr2 = y_up_tr[tr_mask].astype(int).values
        y_up_va2 = y_up_va[va_mask].astype(int).values
        y_dn_tr2 = y_dn_tr[tr_mask].astype(int).values
        y_dn_va2 = y_dn_va[va_mask].astype(int).values

        # Guard: skip fold if not enough samples after filtering
        if len(X_tr2) < 100 or len(X_va2) < 20:
            print(f"[Fold {fid}] skipped due to insufficient samples after filtering.")
            continue

        # UP task
        up_m1, up_m2, up_m3 = _fit_models(X_tr2, y_up_tr2)
        up_s1 = _score_raw(up_m1, X_va2)
        up_s2 = _score_raw(up_m2, X_va2)
        up_s3 = _score_raw(up_m3, X_va2)

        up_platt1 = _platt_calibrate(up_s1, y_up_va2)
        up_platt2 = _platt_calibrate(up_s2, y_up_va2)
        up_platt3 = _platt_calibrate(up_s3, y_up_va2)

        up_p1 = up_platt1.predict_proba(np.asarray(up_s1).reshape(-1,1))[:,1]
        up_p2 = up_platt2.predict_proba(np.asarray(up_s2).reshape(-1,1))[:,1]
        up_p3 = up_platt3.predict_proba(np.asarray(up_s3).reshape(-1,1))[:,1]

        # DOWN task
        dn_m1, dn_m2, dn_m3 = _fit_models(X_tr2, y_dn_tr2)
        dn_s1 = _score_raw(dn_m1, X_va2)
        dn_s2 = _score_raw(dn_m2, X_va2)
        dn_s3 = _score_raw(dn_m3, X_va2)

        dn_platt1 = _platt_calibrate(dn_s1, y_dn_va2)
        dn_platt2 = _platt_calibrate(dn_s2, y_dn_va2)
        dn_platt3 = _platt_calibrate(dn_s3, y_dn_va2)

        dn_p1 = dn_platt1.predict_proba(np.asarray(dn_s1).reshape(-1,1))[:,1]
        dn_p2 = dn_platt2.predict_proba(np.asarray(dn_s2).reshape(-1,1))[:,1]
        dn_p3 = dn_platt3.predict_proba(np.asarray(dn_s3).reshape(-1,1))[:,1]

        # Eval (val set)
        up_eval1 = _eval_calibrated(up_p1, y_up_va2)
        up_eval2 = _eval_calibrated(up_p2, y_up_va2)
        up_eval3 = _eval_calibrated(up_p3, y_up_va2)
        dn_eval1 = _eval_calibrated(dn_p1, y_dn_va2)
        dn_eval2 = _eval_calibrated(dn_p2, y_dn_va2)
        dn_eval3 = _eval_calibrated(dn_p3, y_dn_va2)

        # Assemble validation output
        va_df = df.loc[va_idx, [ts_col, "y_up", "y_dn"]].copy()
        va_df = va_df.loc[va_mask].reset_index(drop=True)
        va_df.rename(columns={ts_col: "time"}, inplace=True)

        out = pd.DataFrame({
            "fold_id": fid,
            "time": va_df["time"],
            "y_up": va_df["y_up"].astype(int),
            "y_dn": va_df["y_dn"].astype(int),
            "p_up_logreg": up_p1,
            "p_up_linsvc": up_p2,
            "p_up_gb":     up_p3,
            "p_dn_logreg": dn_p1,
            "p_dn_linsvc": dn_p2,
            "p_dn_gb":     dn_p3,
        })

        os.makedirs(OUT_DIR, exist_ok=True)
        out.to_csv(os.path.join(OUT_DIR, f"val_fold_{fid}.csv"), index=False)

        # for overview
        out["up_auc_logreg"] = up_eval1["auc"]
        out["up_auc_linsvc"] = up_eval2["auc"]
        out["up_auc_gb"]     = up_eval3["auc"]
        out["dn_auc_logreg"] = dn_eval1["auc"]
        out["dn_auc_linsvc"] = dn_eval2["auc"]
        out["dn_auc_gb"]     = dn_eval3["auc"]
        all_val_outputs.append(out)

        print(f"[Fold {fid}] UP AUC: logreg={up_eval1['auc']:.3f} | linsvc={up_eval2['auc']:.3f} | gb={up_eval3['auc']:.3f}")
        print(f"[Fold {fid}] DN AUC: logreg={dn_eval1['auc']:.3f} | linsvc={dn_eval2['auc']:.3f} | gb={dn_eval3['auc']:.3f}")

    if all_val_outputs:
        cat = pd.concat(all_val_outputs, axis=0, ignore_index=True)
        cat.to_csv(os.path.join(OUT_DIR, "val_all_folds.csv"), index=False)

    print("="*72)
    print("[Step 5] Finished: per-fold validation probabilities saved to data/predictions/")
    print("Next: Step 6 — threshold selection & conflict resolution using these probabilities.")
    print("="*72)

if __name__ == "__main__":
    run_step5()
