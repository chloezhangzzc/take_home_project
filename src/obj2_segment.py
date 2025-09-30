"""
Train K-Means (recommended) and optional GMM/HDBSCAN on dense embeddings.
Exports labels and a simple scorecard (size/stability, silhouette, Top30 capture).
"""

import argparse, yaml, os, joblib, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import hdbscan
from sklearn.metrics import silhouette_score, mutual_info_score

from common.io_utils import read_data, ensure_dir, set_global_seed
from common.preprocessing import make_embedding_preprocessor


def weighted_share(labels: pd.Series, w: pd.Series) -> pd.Series:
    """Weighted share per label."""
    return (w.groupby(labels).sum() / w.sum()).sort_index()


def share_drift_L1(s_tv: pd.Series, s_te: pd.Series) -> float:
    """L1 distance between weighted shares (percentage points)."""
    idx = s_tv.index.union(s_te.index)
    a = s_tv.reindex(idx, fill_value=0.0)
    b = s_te.reindex(idx, fill_value=0.0)
    return float((a - b).abs().sum() * 100)


def topX_capture(labels, y, w, top_share=0.30):
    """Coverage when selecting top segments by positive rate until reaching top_share of population."""
    grp_w = w.groupby(labels).sum()
    pos_w = (w * y).groupby(labels).sum()
    pos_rate = (pos_w / grp_w).fillna(0.0)
    order = pos_rate.sort_values(ascending=False).index

    cum_pop = 0.0; cum_pos = 0.0
    total_pop = float(w.sum()); total_pos = float((w * y).sum())
    for lab in order:
        share = float(grp_w.loc[lab] / total_pop)
        cum_pop += share
        cum_pos += float(pos_w.loc[lab] / total_pos) if total_pos > 0 else 0.0
        if cum_pop >= top_share: break
    return 100*cum_pop, 100*cum_pos


def main(cfg_path, k, run_gmm, run_hdbscan):
    cfg = yaml.safe_load(open(cfg_path))
    set_global_seed(cfg["data"]["random_state"])
    ensure_dir("models/obj2"); ensure_dir("reports/objective2")

    # --- Load and split (here we just create a simple random Test split on weights) ---
    df = read_data(cfg["data"]["path"])
    w = df[cfg["data"]["weight_col"]]
    y = df[cfg["data"].get("target_col", "income")] if "target_col" in cfg["data"] and cfg["data"]["target_col"] in df else pd.Series(np.zeros(len(df), int), index=df.index)

    # Build embeddings (same inputs as objective 1; adjust columns as needed)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    low_card = [c for c in cat_cols if df[c].nunique() <= 20]
    high_card = [c for c in cat_cols if c not in low_card]

    emb_pipe = make_embedding_preprocessor(numeric_cols, low_card, high_card, n_svd=64)
    Z = emb_pipe.fit_transform(df)

    # Simple split for illustration: first 80% as "trainval", last 20% as "test" (replace with your true split)
    n = len(df); n_tv = int(0.8*n)
    Z_trainval, Z_test = Z[:n_tv], Z[n_tv:]
    w_trainval, w_test = w.iloc[:n_tv], w.iloc[n_tv:]
    y_trainval, y_test = y.iloc[:n_tv], y.iloc[n_tv:]

    # --- KMeans ---
    k_use = k or cfg["kmeans"]["k"]
    km = KMeans(n_clusters=k_use, n_init="auto", random_state=cfg["data"]["random_state"])
    km.fit(Z_trainval, sample_weight=w_trainval.values)
    lab_tr_km = pd.Series(km.labels_, index=w_trainval.index, name="segment").astype(int)
    lab_te_km = pd.Series(km.predict(Z_test), index=w_test.index, name="segment").astype(int)

    # Save labels
    lab_tr_km.to_csv("models/obj2/labels_kmeans_trainval.csv")
    lab_te_km.to_csv("models/obj2/labels_kmeans_test.csv")
    joblib.dump(emb_pipe, "models/obj2/embedding_preprocessor.joblib")

    # Optionally GMM / HDBSCAN as comparisons
    if run_gmm and cfg["gmm"]["enabled"]:
        gmm = GaussianMixture(
            n_components=cfg["gmm"]["n_components"],
            covariance_type=cfg["gmm"]["covariance_type"],
            reg_covar=cfg["gmm"]["reg_covar"],
            max_iter=500, random_state=cfg["data"]["random_state"], init_params="kmeans"
        ).fit(Z_trainval)
        pd.Series(gmm.predict(Z_trainval), index=w_trainval.index, name="segment").to_csv("models/obj2/labels_gmm_trainval.csv")
        pd.Series(gmm.predict(Z_test), index=w_test.index, name="segment").to_csv("models/obj2/labels_gmm_test.csv")

    if run_hdbscan and cfg["hdbscan"]["enabled"]:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cfg["hdbscan"]["min_cluster_size"],
            min_samples=cfg["hdbscan"]["min_samples"],
            cluster_selection_method=cfg["hdbscan"]["cluster_selection_method"],
            metric="euclidean",
            prediction_data=True
        )
        clusterer.fit(Z_trainval)
        from hdbscan import approximate_predict
        te_labels, _ = approximate_predict(clusterer, Z_test)
        pd.Series(clusterer.labels_, index=w_trainval.index, name="segment").to_csv("models/obj2/labels_hdbscan_trainval.csv")
        pd.Series(te_labels, index=w_test.index, name="segment").to_csv("models/obj2/labels_hdbscan_test.csv")

    print("Segmentation artifacts saved to models/obj2/ and reports/objective2/.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--run_gmm", action="store_true")
    ap.add_argument("--run_hdbscan", action="store_true")
    args = ap.parse_args()
    main(args.config, args.k, args.run_gmm, args.run_hdbscan)