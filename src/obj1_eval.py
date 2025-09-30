"""
Load saved preprocessor + models and evaluate on TEST again (or on a supplied file).
This script is handy for reviewers to verify metrics without retraining.
"""

import argparse, yaml, joblib
import pandas as pd
from common.io_utils import read_data
from common.metrics import pr_auc_weighted, roc_auc_weighted, best_f1_threshold_from_pr, eval_at_threshold


def main(cfg_path, model_dir):
    cfg = yaml.safe_load(open(cfg_path))
    df = read_data(cfg["data"]["path"])
    y = df[cfg["data"]["target_col"]]
    w = df[cfg["data"]["weight_col"]]
    X = df.drop(columns=[cfg["data"]["target_col"]])

    pre = joblib.load(f"{model_dir}/preprocessor.joblib")
    Xt = pre.transform(X)

    # Example: evaluate LightGBM and XGB if present
    import os
    if os.path.exists(f"{model_dir}/lgbm.joblib"):
        lgbm = joblib.load(f"{model_dir}/lgbm.joblib")
        prob = lgbm.predict_proba(Xt)[:,1]
        print("[LGBM] AP:", pr_auc_weighted(y, prob, w), "ROC:", roc_auc_weighted(y, prob, w))

    if os.path.exists(f"{model_dir}/xgb.model"):
        import xgboost as xgb
        d = xgb.DMatrix(Xt, label=y.values, weight=w.values)
        xgbm = xgb.Booster()
        xgbm.load_model(f"{model_dir}/xgb.model")
        prob = xgbm.predict(d)
        print("[XGB ] AP:", pr_auc_weighted(y, prob, w), "ROC:", roc_auc_weighted(y, prob, w))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model_dir", required=True)
    args = ap.parse_args()
    main(args.config, args.model_dir)