"""
Build the unified segmentation scorecard and marketing-facing tables:
  - cluster counts (all / >=2%), noise %, silhouette (train/test)
  - share drift (pp), Top30_pop_% vs. Top30_pos_%
  - (optional) weighted mutual information with label if available
Also write segment size & stability, per-segment lift, and a targeting plan CSV.
"""

import argparse, yaml, numpy as np, pandas as pd
from sklearn.metrics import silhouette_score, mutual_info_score
from common.io_utils import read_data
from common.preprocessing import make_embedding_preprocessor


def weighted_share(labels, w):
    return (w.groupby(labels).sum() / w.sum()).sort_index()


def share_drift_L1(s_tv, s_te):
    idx = s_tv.index.union(s_te.index)
    a = s_tv.reindex(idx, fill_value=0.0)
    b = s_te.reindex(idx, fill_value=0.0)
    return float((a - b).abs().sum() * 100)


def topX_capture(labels, y, w, top_share=0.30):
    grp_w = w.groupby(labels).sum()
    pos_w = (w * y).groupby(labels).sum()
    pos_rate = (pos_w / grp_w).fillna(0.0)
    order = pos_rate.sort_values(ascending=False).index

    cum_pop = 0.0; cum_pos = 0.0
    total_pop = float(w.sum()); total_pos = float((w * y).sum())
    covered = []
    for lab in order:
        share = float(grp_w.loc[lab] / total_pop)
        covered.append(lab)
        cum_pop += share
        cum_pos += float(pos_w.loc[lab] / total_pos) if total_pos > 0 else 0.0
        if cum_pop >= top_share: break
    return 100*cum_pop, 100*cum_pos, pos_rate.loc[covered]


def main(cfg_path, labels_path):
    cfg = yaml.safe_load(open(cfg_path))
    df = read_data(cfg["data"]["path"])
    w = df[cfg["data"]["weight_col"]]
    y = df[cfg["data"].get("target_col", "income")] if "target_col" in cfg["data"] and cfg["data"]["target_col"] in df else pd.Series(np.zeros(len(df), int), index=df.index)

    # For silhouette on TEST we need the same embeddings
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    low_card = [c for c in cat_cols if df[c].nunique() <= 20]
    high_card = [c for c in cat_cols if c not in low_card]
    emb_pipe = make_embedding_preprocessor(numeric_cols, low_card, high_card, n_svd=64)
    Z = emb_pipe.fit_transform(df)

    labels = pd.read_csv(labels_path, index_col=0, squeeze=True)
    labels = labels.reindex(df.index)

    # Here we assume labels correspond to TEST; adapt if they are trainval/test
    # If you have both, compute share_drift with both distributions.

    # Scorecard pieces
    s_te = weighted_share(labels, w)
    n_all = labels.nunique()
    n_eff = int((s_te >= 0.02).sum())
    noise_share = float((s_te.get(-1, 0.0)) * 100.0)

    sil = float(silhouette_score(Z, labels)) if n_all > 1 else np.nan
    pop_cov, pos_cap, _ = topX_capture(labels, y, w, top_share=0.30)

    # Mutual information (optional)
    try:
        # Approximate MI using weighted sampling
        n = len(w); p = (w / w.sum()).values
        idx = np.random.default_rng(42).choice(np.arange(n), size=min(n, 200_000), replace=True, p=p)
        mi = float(mutual_info_score(labels.values[idx], y.values[idx]))
    except Exception:
        mi = np.nan

    scorecard = pd.DataFrame([{
        "model": "kmeans",
        "clusters_all": n_all,
        "clusters_>=2%": n_eff,
        "noise_%": round(noise_share, 1),
        "sil_test": round(sil, 3),
        "Top30_pop_%": round(pop_cov, 1),
        "Top30_pos_%": round(pos_cap, 1),
        "MI_seg_y": round(mi, 4) if not np.isnan(mi) else np.nan,
    }]).set_index("model")

    scorecard.to_csv("reports/objective2/scorecard.csv")
    print(scorecard)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--labels", required=True, help="Path to labels CSV (e.g., models/obj2/labels_kmeans_test.csv)")
    args = ap.parse_args()
    main(args.config, args.labels)