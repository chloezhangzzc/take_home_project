"""
Preprocessing builders for tree models, neural nets, and clustering/embeddings.
Keep all transformers here so train/test/serve share the exact same logic.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from .feature_builder import FeatureBuilder

# Optional: your TargetMeanEncoder1D implementation (from notebook)
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class TargetMeanEncoder1D(BaseEstimator, TransformerMixin):
    """Smoothed target mean encoder for high-cardinality categoricals."""
    def __init__(self, m=50):
        self.m = m
    def fit(self, X, y):
        s = self._to_series(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        self.global_mean_ = float(y.mean())
        g = pd.DataFrame({"cat": s, "y": y}).groupby("cat")["y"]
        means, counts = g.mean(), g.count()
        self.mapping_ = ((means * counts + self.m * self.global_mean_) / (counts + self.m)).to_dict()
        return self
    def transform(self, X):
        s = self._to_series(X)
        arr = s.map(self.mapping_).fillna(self.global_mean_)
        return np.asarray(arr, dtype=float).reshape(-1, 1)
    @staticmethod
    def _to_series(X):
        if isinstance(X, pd.Series): return X
        if hasattr(X, "shape") and len(X.shape) == 2: return pd.Series(X[:,0])
        return pd.Series(X)


def make_tree_preprocessor(numeric_cols, low_card_cats, high_card_cats):
    """
    Build a ColumnTransformer suitable for tree models:
      - numerics: median impute (no scaling required for trees)
      - low-card categoricals: one-hot
      - high-card categoricals: target mean encoding
    """
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
    tgt_blocks = [(f"tgt_{c}", TargetMeanEncoder1D(m=50), [c]) for c in high_card_cats]

    ct = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("ohe", ohe, low_card_cats),
            *tgt_blocks,
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    pipe = Pipeline([("feat", FeatureBuilder()), ("prep", ct)])
    return pipe


def make_embedding_preprocessor(numeric_cols, low_card_cats, high_card_cats, n_svd=64):
    """
    Build a dimensionality-reduction pipeline for clustering:
      - same imputing/encoding as trees
      - followed by TruncatedSVD to get dense embeddings Z
    """
    base = make_tree_preprocessor(numeric_cols, low_card_cats, high_card_cats)
    emb = Pipeline([
        ("base", base),
        ("svd", TruncatedSVD(n_components=n_svd, random_state=42))
    ])
    return emb