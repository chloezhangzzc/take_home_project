"""
Train tuned LightGBM and XGBoost, pick a single validation threshold, and (optionally) blend.
Outputs:
  - models/obj1/: saved preprocessors and models
  - reports/objective1/: OOF metrics, threshold diagnostics
"""

import argparse, os, yaml, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from lightgbm import LGBMClassifier
import xgboost as xgb

from common.io_utils import set_global_seed, read_data, ensure_dir, stratified_splits
from common.preprocessing import make_tree_preprocessor
from common.metrics import pr_auc_weighted, roc_auc_weighted, best_f1_threshold_from_pr, eval_at_threshold


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    set_global_seed(cfg["data"]["random_state"])

    # --- Load data ---
    df = read_data(cfg["data"]["path"])
    y_col = cfg["data"]["target_col"]
    w_col = cfg["data"]["weight_col"]
    assert y_col in df.columns, f"Missing target column: {y_col}"
    assert w_col in df.columns, f"Missing weight column: {w_col}"

    # Map/clean target if needed before splitting (you already did in notebook)
    # df[y_col] = map_label_to_binary_census(df[y_col])  # if required

    # --- Split ---
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_splits(
        df, y_col, cfg["data"]["test_size"], cfg["data"]["val_size"], cfg["data"]["random_state"]
    )
    w_train = df.loc[X_train.index, w_col]
    w_val   = df.loc[X_val.index,   w_col]
    w_test  = df.loc[X_test.index,  w_col]

    # --- Build preprocessor ---
    # You can explicitly pass columns from cfg["columns"] or infer:
    numeric_cols = cfg["columns"]["numeric"] or X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object","category"]).columns.tolist()
    # Split cats into low/high cardinality (simple heuristic, or set in config)
    low_card = cfg["columns"]["low_card_cats"] or [c for c in cat_cols if X_train[c].nunique() <= 20]
    high_card = [c for c in cat_cols if c not in low_card]

    pre_tree = make_tree_preprocessor(numeric_cols, low_card, high_card)

    # --- LightGBM small grid ---
    base_lgbm = LGBMClassifier(
        objective="binary", n_estimators=2000, learning_rate=0.05,
        subsample=1.0, colsample_bytree=1.0, random_state=cfg["data"]["random_state"]
    )
    # scale_pos_weight determined per fold from weighted class ratio:
    grid = cfg["lightgbm"]["grid"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["data"]["random_state"])

    def lgbm_oof_and_best():
        best = {"f1": -1, "params": None, "thr": None, "ap": None}
        for num_leaves in grid["num_leaves"]:
            for min_child in grid["min_child_samples"]:
                for reg_lambda in grid["reg_lambda"]:
                    for mult in grid["scale_pos_weight_mult"]:
                        oof = np.zeros(len(X_train) + len(X_val), float)
                        idx_all = np.concatenate([X_train.index.values, X_val.index.values])

                        for tr_idx, va_idx in skf.split(X_train.append(X_val), y_train.append(y_val)):
                            tr_ids = idx_all[tr_idx]; va_ids = idx_all[va_idx]
                            X_tr = df.loc[tr_ids, X_train.columns]; y_tr = df.loc[tr_ids, y_col]; w_tr = df.loc[tr_ids, w_col]
                            X_va = df.loc[va_ids, X_train.columns]; y_va = df.loc[va_ids, y_col]; w_va = df.loc[va_ids, w_col]

                            spw = float(w_tr[y_tr==0].sum() / max(w_tr[y_tr==1].sum(), 1e-12)) * float(mult)
                            est = clone(base_lgbm).set_params(num_leaves=num_leaves, min_child_samples=min_child, reg_lambda=reg_lambda, scale_pos_weight=spw)

                            # Fit fold-specific preprocessor to avoid leakage
                            pipe = clone(pre_tree)
                            pipe.fit(X_tr, y_tr)
                            X_tr_t = pipe.transform(X_tr); X_va_t = pipe.transform(X_va)

                            est.fit(X_tr_t, y_tr, sample_weight=w_tr,
                                    eval_set=[(X_va_t, y_va)], eval_metric="average_precision",
                                    callbacks=[])

                            oof[va_idx] = est.predict_proba(X_va_t)[:,1]

                        # Pick single global threshold on OOF
                        y_trainval = y_train.append(y_val)
                        w_trainval = w_train.append(w_val)
                        thr, f1 = best_f1_threshold_from_pr(y_trainval, oof, w_trainval)
                        ap = pr_auc_weighted(y_trainval, oof, w_trainval)
                        if f1 > best["f1"]:
                            best.update({"f1": f1, "thr": thr, "params": dict(num_leaves=num_leaves, min_child_samples=min_child, reg_lambda=reg_lambda, scale_pos_weight="per-fold"), "ap": ap})
        return best

    lgbm_best = lgbm_oof_and_best()

    # --- XGBoost random search (compact; mirrors your notebook) ---
    base_params = dict(objective="binary:logistic", eval_metric="aucpr", tree_method="hist", seed=cfg["data"]["random_state"])
    space = cfg["xgboost"]["param_space"]
    num_trials = cfg["xgboost"]["random_trials"]
    esr = cfg["xgboost"]["early_stopping_rounds"]; nround = cfg["xgboost"]["num_boost_round"]

    def sample_params():
        import random
        return {k: random.choice(v) for k, v in space.items()}

    def xgb_oof_and_best():
        best = {"f1": -1, "params": None, "thr": None}
        X_trainval = pd.concat([X_train, X_val]); y_trainval = pd.concat([y_train, y_val]); w_trainval = pd.concat([w_train, w_val])
        for _ in range(num_trials):
            trial = sample_params()
            oof = np.zeros(len(X_trainval))
            for tr_idx, va_idx in skf.split(X_trainval, y_trainval):
                X_tr, X_va = X_trainval.iloc[tr_idx], X_trainval.iloc[va_idx]
                y_tr, y_va = y_trainval.iloc[tr_idx], y_trainval.iloc[va_idx]
                w_tr, w_va = w_trainval.iloc[tr_idx], w_trainval.iloc[va_idx]

                spw = float(w_tr[y_tr==0].sum()) / max(float(w_tr[y_tr==1].sum()), 1e-12)
                params = dict(base_params, **trial, scale_pos_weight=spw)

                pipe = clone(pre_tree).fit(X_tr, y_tr)
                Dtr = xgb.DMatrix(pipe.transform(X_tr), label=y_tr.values, weight=w_tr.values)
                Dva = xgb.DMatrix(pipe.transform(X_va), label=y_va.values, weight=w_va.values)
                bst = xgb.train(params, Dtr, nround, evals=[(Dva,"valid")], early_stopping_rounds=esr, verbose_eval=False)
                best_n = (getattr(bst,"best_iteration", None) or (nround-1)) + 1
                oof[va_idx] = bst.predict(Dva, iteration_range=(0, best_n))
            thr, f1 = best_f1_threshold_from_pr(y_trainval, oof, w_trainval)
            if f1 > best["f1"]:
                best = {"f1": float(f1), "params": trial, "thr": float(thr)}
        return best

    xgb_best = xgb_oof_and_best()

    # --- Refit best single models on TRAIN+VAL and test once ---
    X_trainval = pd.concat([X_train, X_val]); y_trainval = pd.concat([y_train, y_val]); w_trainval = pd.concat([w_train, w_val])

    # LightGBM final
    spw_w = float(w_trainval[y_trainval==0].sum()) / max(float(w_trainval[y_trainval==1].sum()), 1e-12)
    lgbm_final = clone(base_lgbm).set_params(**{k:v for k,v in lgbm_best["params"].items() if k in ["num_leaves","min_child_samples","reg_lambda"]})
    pipe_lgbm = clone(pre_tree).fit(X_trainval, y_trainval)
    X_trv_t = pipe_lgbm.transform(X_trainval); X_te_t = pipe_lgbm.transform(X_test)
    lgbm_final.set_params(scale_pos_weight=spw_w)
    lgbm_final.fit(X_trv_t, y_trainval, sample_weight=w_trainval)
    prob_lgbm = lgbm_final.predict_proba(X_te_t)[:,1]

    # XGB final
    Dtrv = xgb.DMatrix(X_trv_t, label=y_trainval.values, weight=w_trainval.values)
    Dte  = xgb.DMatrix(X_te_t, label=y_test.values, weight=w_test.values)
    params = dict(base_params, **xgb_best["params"], scale_pos_weight=spw_w)
    xgb_final = xgb.train(params, Dtrv, num_boost_round=cfg["xgboost"]["num_boost_round"], verbose_eval=False)
    prob_xgb = xgb_final.predict(Dte)

    # Blend
    if cfg["blend"]["enabled"]:
        alpha = float(cfg["blend"]["alpha"])
        prob_blend = alpha * prob_lgbm + (1.0 - alpha) * prob_xgb
        thr = lgbm_best["thr"] if cfg["blend"]["pick_threshold_by"] == "f1_weighted" else 0.5
        m_blend = eval_at_threshold(y_test, prob_blend, w_test, thr)
    else:
        prob_blend, m_blend = None, None

    # Save artifacts
    ensure_dir("models/obj1"); ensure_dir("reports/objective1")
    joblib.dump(pipe_lgbm, "models/obj1/preprocessor.joblib")
    joblib.dump(lgbm_final, "models/obj1/lgbm.joblib")
    joblib.dump(xgb_final,  "models/obj1/xgb.model")

    # Write a small metrics CSV
    rows = []
    rows.append(dict(model="LightGBM", ap=pr_auc_weighted(y_test, prob_lgbm, w_test), roc=roc_auc_weighted(y_test, prob_lgbm, w_test)))
    rows.append(dict(model="XGBoost", ap=pr_auc_weighted(y_test, prob_xgb,  w_test), roc=roc_auc_weighted(y_test, prob_xgb,  w_test)))
    if prob_blend is not None:
        rows.append(dict(model="LGBM+XGB", ap=pr_auc_weighted(y_test, prob_blend, w_test), roc=roc_auc_weighted(y_test, prob_blend, w_test)))
    pd.DataFrame(rows).to_csv("reports/objective1/test_summary.csv", index=False)

    print("Done. Artifacts saved under models/obj1 and reports/objective1.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)