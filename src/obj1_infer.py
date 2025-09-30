"""
Batch scoring script: load the production preprocessor + model and write scores.
"""

import argparse, joblib, pandas as pd
from common.io_utils import read_data

def main(model_dir, input_path, output_path):
    pre = joblib.load(f"{model_dir}/preprocessor.joblib")
    model = joblib.load(f"{model_dir}/lgbm.joblib")   # or your blend logic
    df = read_data(input_path)
    proba = model.predict_proba(pre.transform(df))[:,1]
    pd.DataFrame({"score": proba}, index=df.index).to_csv(output_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()
    main(args.model_dir, args.input_path, args.output_path)