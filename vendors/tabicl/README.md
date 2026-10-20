# TabICLv2 (INRIA / soda-inria)

[TabICLv2](https://arxiv.org/abs/2602.11139) is an in-context-learning tabular
foundation model from the soda-inria group. Like TabFM it is **self-hosted**: the
`tabicl` package downloads pretrained weights from Hugging Face `jingang/TabICL` and runs inference
locally in a single forward pass without needing training.

- TabICLv1 Paper (02/2025): https://arxiv.org/abs/2502.05564
- TabICLv2 Paper (02/2026): https://arxiv.org/abs/2602.11139
- Code: https://github.com/soda-inria/tabicl
- Docs: https://tabicl.readthedocs.io/

## ✅ License: Can be used commercially

Both the inference code ([`soda-inria/tabicl`](https://github.com/soda-inria/tabicl))
and the [model weights](https://huggingface.co/jingang/TabICL) — the official soda-inria
v2 checkpoints — are released under the [BSD 3-Clause License](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/bsd-3-clause.md),
so they can be used commercially. (The `tabicl` package bundles Prior Labs-derived code
under Apache-2.0 in `src/tabicl/forecast`.)

## At a glance

| | |
|---|---|
| **Consumption** | Self-hosted weights (Hugging Face) — runs locally, single forward pass |
| **Compute on Databricks** | **GPU recommended**, specially with Databricks AI Runtime for instant GPU access; CPU works but is slow |
| **Auth** | None; downloads checkpoint from Hugging Face on first use |
| **Tasks covered** | Classification and Regression (see below re: forecast/shap extras) |
| **License** | **BSD 3-Clause** (code + weights) — commercial use OK |
| **Training data** | Only on syntehtic generated datasets. Datasets between 300 and 60k samples |
| **Data limits** | Benchmarked performance on datasets of \<600k rows, with more than 10 classes. Tested scalability up to 1M rows and 500 features on a 50GB GPU.|

## Setup

1. Attach to a **serverless GPU** (AI runtime) with environment AI v6 or above. CPU is supported but significantly slower and provisioning all-purpose cluster with GPU might have longer wait times.
2. Run the shared data prep first:
   [`shared/notebooks/00_data_preparation.ipynb`](../../shared/notebooks/00_data_preparation.ipynb).
3. Open the notebooks below.

## Notebooks

| Notebook | Task |
|----------|------|
| `notebooks/01_classification.ipynb` | Binary & multi-class classification |
| `notebooks/02_regression.ipynb` | Regression |
| `notebooks/03_inference_on_serving_endpoint.ipynb` | Serve the model through a GPU serving endpoint invoked from a notebook with CPU only|

Native API (sklearn-style, single-step):
```python
from tabicl import TabICLClassifier, TabICLRegressor
clf = TabICLClassifier()          # downloads checkpoint on first use
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)
```
Optional extras: `pip install tabicl[forecast]` (time series), `tabicl[shap]`,
`tabicl[finetune]`.

## Path note

The notebooks add the repo-level `common/` directory to `sys.path` via a relative
`REPO_ROOT` (three levels up from `vendors/tabfm/notebooks/`). If you check the repo
out at a non-standard workspace path, adjust `REPO_ROOT` in the Configuration cell.
