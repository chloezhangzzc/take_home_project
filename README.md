# Income Classification & Market Segmentation

This repository contains two deliverables built on the same tabular dataset (~40 demographic/employment variables plus a binary income label).

- **Objective 1 — Income Classification**: predict whether annual income is `≤ $50K` vs. `> $50K`.
- **Objective 2 — Market Segmentation**: create customer segments suitable for targeting and messaging (K-Means recommended; GMM/HDBSCAN for comparison).

The code supports **fully reproducible** runs (fixed seeds, immutable splits, fold‑internal preprocessing), and can be executed either from **scripts ** or directly from the **Jupyter notebooks**.

---

## 0) Repository Structure

```
.
├── configs/
│   ├── obj1.yaml            # paths, columns (target/weights), model flags for Objective 1
│   └── obj2.yaml            # paths, SVD/clustering settings for Objective 2
├── data/
│   ├── raw/                 # place your source CSV here (e.g., adult.csv)
│   
├── models/
│   ├── obj1/                # saved classifier(s) + preprocessing
│   └── obj2/                # saved clusterers + labels
├── notebooks/
│   ├── notebook_1.ipynb     # Objective 1 (EDA, FE, modeling)
│   └── notebook_2.ipynb     # Objective 2 (EDA, segmentation)
├
│   
├── src/
│   ├── common/              # shared preprocessing, encoders, utils
│   ├── obj1_train.py        # train/tune LightGBM & XGBoost, optional blend
│   ├── obj1_eval.py         # load artifacts, evaluate on test
│   ├── obj2_segment.py      # fit K-Means (and optional GMM/HDBSCAN)
│   └── obj2_eval.py         # build scorecards/lift tables from labels
├── requirements.txt
└── README.md
```



---

## 1) Environment Setup

### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Optional (MLP only):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Build tools (only if needed):**
- macOS: `brew install libomp` (for LightGBM); `brew install cmake`
- Ubuntu/Debian: `sudo apt-get install build-essential`
- Windows: Install “Build Tools for Visual Studio” (C++).

Deactivate any time with `deactivate`.



## 2) Data & Config

1. Place your CSV under `data/raw/` (e.g., `data/raw/adult.csv`).  
2. Edit `configs/obj1.yaml` and `configs/obj2.yaml` with your paths/columns.


## 3) Reproduce **Objective 1** (Classification)

### Run from scripts (recommended)
**Train + tune + export artifacts**
```bash
python -m src.obj1_train --config configs/obj1.yaml
```

**Evaluate saved model(s) on the fixed test set**
```bash
python -m src.obj1_eval --config configs/obj1.yaml --model_dir models/obj1
```

**Outputs**
- `models/obj1/`: preprocessing transformer, tuned LightGBM/XGBoost, optional blend
- `reports/objective1/test_summary.csv`: PR‑AUC, ROC‑AUC, precision/recall/F1 at selected threshold(s)
- Optional PR/ROC curve images if enabled in config

### Run from the notebook
Open `notebooks/notebook_1.ipynb` and run all cells. Exported artifacts and reports will land in the same folders.



## 4) Reproduce **Objective 2** (Segmentation)

### Run from scripts (recommended)

**K‑Means only (k=10)**
```bash
python -m src.obj2_segment --config configs/obj2.yaml --k 10
```

**With GMM and HDBSCAN comparisons**
```bash
python -m src.obj2_segment --config configs/obj2.yaml --k 10 --run_gmm --run_hdbscan
```

**Build scorecards & marketing tables**
```bash
python -m src.obj2_eval --config configs/obj2.yaml \
  --labels models/obj2/labels_kmeans_test.csv
```

**Outputs**
- `models/obj2/labels_*_{trainval,test}.csv` – segment assignments
- `reports/objective2/scorecard.csv` – clusters, silhouette, Top‑30% capture, MI
- `reports/objective2/size_stability.csv` – TrainVal vs Test shares + drift (pp)
- `reports/objective2/lift_by_segment.csv` – positive rate & lift by segment
- `reports/objective2/targeting_plan.csv` – ordered by lift with cumulative coverage

### Run from the notebook
Open `notebooks/notebook_2.ipynb` and run all cells.



## 5) Determinism & Governance

- **Fixed seeds** for all randomized steps (splits/initialization).
- **Immutable splits**: Train/Validation/Test split is created once and reused.
- **Fold‑internal preprocessing** for CV to avoid leakage.
- **Survey weights** used in training and metrics where applicable.
- **Model cards** (short text files in `models/obj1/`) capture parameters & thresholds.
- **Drift monitoring**: segment share drift and outcome incidence can be tracked over time using the exported tables.



## 6) Troubleshooting

- `lightgbm` on macOS → `brew install libomp`, then `pip install lightgbm`.
- Windows PowerShell activation policy → run as Admin:  
  `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`
- `hdbscan` build errors → ensure a C/C++ compiler (see Prerequisites).
- If your label is not 0/1, normalize it first (e.g., map `<=50K` to 0, `>50K` to 1).


