"""
Reusable feature engineering components used by both objectives.
Keep this deterministic and side-effect free so train/serve parity is guaranteed.
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Applies conservative, explainable transformations:
      - Normalize survey tokens ("Not in universe", "Unknown") to a single "Missing"
      - Map state -> region (then drop state)
      - Drop low-value/redundant columns (year, weight, parents' birthplace)
      - Create robust binary + log channels for sparse monetary fields
      - Add simple flags (full_year_worker / no_work / hourly_worker / has_any_capital / is_married / is_union / is_self_employed)
      - Create age / weeks buckets, and (optionally) a cross feature age x education
    """

    def __init__(self):
        self.drop_cols_ = [
            "year", "weight",
            "state_of_previous_residence",
            "country_of_birth_father", "country_of_birth_mother",
        ]
        self.household_pair_ = [
            "detailed_household_and_family_stat",
            "detailed_household_summary_in_household",
        ]
        self.missing_tokens_ = ["Not in universe", "?", " Unknown", "unknown", "  ?"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # 1) Normalize missing-like tokens for categorical columns
        obj_cols = df.select_dtypes(include=["object", "category"]).columns
        for c in obj_cols:
            df[c] = (
                df[c].astype(str).str.strip()
                  .replace(self.missing_tokens_, "Missing")
            )

        # 2) If region is missing but state exists, map state -> Census region (then drop state later)
        if "region_of_previous_residence" not in df.columns and "state_of_previous_residence" in df.columns:
            df["region_of_previous_residence"] = df["state_of_previous_residence"].map(self._state_to_region)

        # 3) Drop redundant/low-value columns
        drop_exist = [c for c in self.drop_cols_ if c in df.columns]
        df = df.drop(columns=drop_exist, errors="ignore")

        # 4) If both household columns exist, keep one to reduce redundancy
        both = [c for c in self.household_pair_ if c in df.columns]
        if len(both) == 2:
            df = df.drop(columns=[both[1]])

        # 5) Derived numeric channels
        if "wage_per_hour" in df.columns:
            df["hourly_worker"] = (df["wage_per_hour"] > 0).astype(int)
            df["log_wage_per_hour"] = np.where(df["wage_per_hour"] > 0, np.log1p(df["wage_per_hour"]), 0.0)

        for col in ["capital_gains", "capital_losses", "dividends_from_stocks"]:
            if col in df.columns:
                df[f"has_{col}"] = (df[col] > 0).astype(int)
                df[f"log_{col}"] = np.where(df[col] > 0, np.log1p(df[col]), 0.0)

        if "weeks_worked_in_year" in df.columns:
            df["full_year_worker"] = (df["weeks_worked_in_year"] >= 50).astype(int)
            df["no_work"] = (df["weeks_worked_in_year"] == 0).astype(int)

        # 6) Simple mobility flag (combine multiple survey fields where present)
        df = self._add_is_mover(df)

        # 7) Buckets and crosses
        if "age" in df.columns:
            df["age_bucket"] = pd.cut(
                df["age"].astype(float),
                bins=[0, 17, 24, 34, 44, 54, 64, 120],
                labels=["u18","18-24","25-34","35-44","45-54","55-64","65+"],
                include_lowest=True
            ).astype("category")

        if "weeks_worked_in_year" in df.columns:
            w = df["weeks_worked_in_year"].astype(float)
            df["weeks_bucket"] = pd.cut(w, bins=[-1,0,26,51,100], labels=["0","1-26","27-51","52"]).astype("category")

        if "age_bucket" in df.columns and "education" in df.columns:
            df["age_edu"] = (df["age_bucket"].astype(str) + "|" + df["education"].astype(str)).astype("category")

        # 8) Convenience flags
        if "marital_stat" in df.columns:
            m = df["marital_stat"].astype(str)
            df["is_married"] = m.str.contains("Married", case=False, na=False).astype(int)

        if "member_of_a_labor_union" in df.columns:
            u = df["member_of_a_labor_union"].astype(str).str.strip().str.lower()
            df["is_union"] = u.eq("yes").astype(int)

        if "class_of_worker" in df.columns:
            cw = df["class_of_worker"].astype(str).str.lower()
            df["is_self_employed"] = cw.str.contains("self-employed", na=False).astype(int)

        # 9) Capital presence and net
        has_cols = []
        for c in ["capital_gains","capital_losses","dividends_from_stocks"]:
            if c in df.columns:
                has_cols.append((df[c] > 0).astype(int))
        if has_cols:
            df["has_any_capital"] = pd.concat(has_cols, axis=1).max(axis=1).astype(int)

        if "capital_gains" in df.columns and "capital_losses" in df.columns:
            net = (df["capital_gains"].fillna(0) - df["capital_losses"].fillna(0))
            df["cap_net"] = net
            df["log_cap_net_pos"] = np.where(net > 0, np.log1p(net), 0.0)

        return df

    # --- helpers ---

    @staticmethod
    def _state_to_region(s: str) -> str:
        """Map state name to one of four Census regions; return 'Other'/'Missing' for outliers."""
        s = str(s)
        if s.lower() in {"missing","not in universe","?","unknown","  ?"}:
            return "Missing"
        northeast = {"Connecticut","Maine","Massachusetts","New Hampshire","Rhode Island","Vermont","New Jersey","New York","Pennsylvania"}
        midwest   = {"Illinois","Indiana","Michigan","Ohio","Wisconsin","Iowa","Kansas","Minnesota","Missouri","Nebraska","North Dakota","South Dakota"}
        south     = {"Delaware","District of Columbia","Florida","Georgia","Maryland","North Carolina","South Carolina","Virginia","West Virginia","Alabama","Kentucky","Mississippi","Tennessee","Arkansas","Louisiana","Oklahoma","Texas"}
        west      = {"Arizona","Colorado","Idaho","Montana","Nevada","New Mexico","Utah","Wyoming","Alaska","California","Hawaii","Oregon","Washington"}
        if s in northeast: return "Northeast"
        if s in midwest:   return "Midwest"
        if s in south:     return "South"
        if s in west:      return "West"
        if "puerto" in s.lower() or "guam" in s.lower() or "outlying" in s.lower():
            return "Other"
        return "Other"

    @staticmethod
    def _add_is_mover(df: pd.DataFrame) -> pd.DataFrame:
        """Create a simple 'is_mover' flag by fusing several migration survey fields if present."""
        move_signals = []

        if "live_in_this_house_1_year_ago" in df.columns:
            s = df["live_in_this_house_1_year_ago"].astype(str)
            move_signals.append(s.map({"Yes": 0, "No": 1, "Not in universe under 1 year old": np.nan, "Missing": np.nan}))

        if "migration_code-change_in_msa" in df.columns:
            s = df["migration_code-change_in_msa"].astype(str)
            move_signals.append(s.map({
                "Nonmover": 0, "MSA to MSA": 1, "NonMSA to nonMSA": 1, "MSA to nonMSA": 1, "NonMSA to MSA": 1,
                "Abroad to MSA": 1, "Abroad to nonMSA": 1, "Not identifiable": np.nan, "Not in universe": np.nan, "Missing": np.nan
            }))

        if "migration_code-change_in_reg" in df.columns:
            s = df["migration_code-change_in_reg"].astype(str)
            move_signals.append(s.map({
                "Nonmover": 0, "Same county": 0, "Different region": 1, "Different county same state": 1,
                "Different division same region": 1, "Different state same division": 1, "Abroad": 1,
                "Not in universe": np.nan, "Missing": np.nan
            }))

        if "migration_code-move_within_reg" in df.columns:
            s = df["migration_code-move_within_reg"].astype(str)
            move_signals.append(s.map({
                "Nonmover": 0, "Same county": 0, "Different county same state": 1,
                "Different state in South": 1, "Different state in Northeast": 1,
                "Different state in Midwest": 1, "Different state in West": 1, "Abroad": 1,
                "Not in universe": np.nan, "Missing": np.nan
            }))

        if move_signals:
            sig = pd.concat(move_signals, axis=1)
            df["is_mover"] = sig.apply(lambda r: 1 if (r == 1).any() else (0 if (r == 0).any() else np.nan), axis=1).astype(float)
        return df