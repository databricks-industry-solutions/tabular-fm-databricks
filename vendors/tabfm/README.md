# TabFM (Google)

[TabFM](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/)
is a zero-shot tabular foundation model from Google Research. It is **self-hosted**:
model weights are downloaded from Hugging Face (`google/tabfm-1.0.0-pytorch`) and
inference runs locally in a single forward pass (in-context learning — the training
rows are passed as context, no gradient training).

## ⚠️ License — non-commercial

The TabFM **model weights** are released under the **TabFM Non-Commercial License v1.0**
(Hugging Face license id `tabfm-non-commercial-v1.0`). Use in this repository is for
**evaluation and comparison only** — the license restricts the weights to
non-commercial, non-production use (testing / evaluation / academic research).
**Commercial or production use requires a separate commercial license from Google.**
Note the source code at [`google-research/tabfm`](https://github.com/google-research/tabfm)
is Apache-2.0, but the *weights* (what these notebooks download and run) are not — the
non-commercial terms are what apply here. This is the key difference from the other
vendors and must be surfaced to any customer evaluating it. See the full license on the
[model card](https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE).

## At a glance

| | |
|---|---|
| **Consumption** | Self-hosted weights (Hugging Face) — runs locally, single forward pass |
| **Compute on Databricks** | **GPU recommended**, specially with Databricks AI Runtime for instant GPU access; CPU works but is slow |
| **Auth** | None for open weights; standard Hugging Face access to download the model |
| **Tasks covered** | **Classification and Regression only** — no outlier detection or forecasting |
| **License** | Weights: **TabFM Non-Commercial License v1.0** ⚠️ (source code is Apache-2.0) |
| **Data limits** | Benchmarked ~700–150k samples; ≤10 classes for classification |

> A managed **BigQuery `AI.PREDICT`** path for TabFM was announced by Google but is
> BigQuery-native (not Databricks) and not GA at time of writing. This vendor therefore
> stays self-hosted on Databricks.

## Setup

1. Attach to a **serverless GPU** (AI runtime) with environment AI v6 or above. CPU is supported but significantly slower and provisioning all-purpose cluster with GPU might have longer wait times.
2. Run the shared data prep first:
   [`shared/notebooks/00_data_preparation.ipynb`](../../shared/notebooks/00_data_preparation.ipynb).
3. Open the notebooks below. The first cell `%pip install tabfm[pytorch]` pulls the
   package; weights download from Hugging Face on first `load()`.

## Notebooks

| Notebook | Task |
|----------|------|
| `notebooks/01_classification.ipynb` | Binary & multi-class classification |
| `notebooks/02_regression.ipynb` | Regression |

Native API (two-step load, then sklearn-style):
```python
from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0
model = tabfm_v1_0_0.load(model_type="classification")
clf = TabFMClassifier(model=model)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)
```

Both notebooks import the shared helpers from [`common/evaluation.py`](../../common/evaluation.py)
for the split, metrics, sklearn baselines, and writing results to the shared
`vendor_benchmark_results` Delta table that the comparison notebook reads.

## Path note

The notebooks add the repo-level `common/` directory to `sys.path` via a relative
`REPO_ROOT` (three levels up from `vendors/tabfm/notebooks/`). If you check the repo
out at a non-standard workspace path, adjust `REPO_ROOT` in the Configuration cell.
