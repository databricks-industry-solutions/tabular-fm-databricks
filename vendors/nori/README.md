# Nori (Synthefy)

[Nori](https://github.com/Synthefy/synthefy-nori) is a tabular foundation model
for regression via in-context learning. Given labeled context rows, it predicts new
rows in a single forward pass without task-specific training or fine-tuning. Nori is
trained entirely on synthetic data and is **self-hosted**: the `synthefy-nori` package
downloads the public checkpoint from Hugging Face and runs inference inside the
Databricks notebook.

## License: commercial use permitted

The [`synthefy-nori`](https://pypi.org/project/synthefy-nori/) package and the
[`Synthefy/Nori`](https://huggingface.co/Synthefy/Nori) model weights are released
under the **Apache License 2.0**, which permits commercial use. The package includes
its own `NOTICE` and third-party attribution files.

## At a glance

| | |
|---|---|
| **Consumption** | Self-hosted weights (Hugging Face), running locally in a single forward pass |
| **Compute on Databricks** | **GPU recommended**, specially with Databricks AI Runtime for instant GPU access; CPU works but is slow |
| **Auth** | None; the public checkpoint downloads on first prediction |
| **Tasks covered** | **Regression** with point estimates and prediction intervals |
| **License** | **Apache License 2.0** (code and weights), commercial use permitted |
| **Model variants** | `nori-6m` (base, approximately 6M parameters) and `nori-30m` (larger, stronger variant) |
| **Intended scale** | Small-to-medium tabular regression; context-cache memory grows with the number of rows |

## Setup

1. Provision a GPU cluster (or GPU serverless where available). CPU works for the
   small demonstration tables, but a GPU is recommended for interactive inference.
2. Run the shared data preparation notebook first:
   [`shared/notebooks/00_data_preparation.ipynb`](../../shared/notebooks/00_data_preparation.ipynb).
3. Open the notebook below. Its first cell installs the pinned `synthefy-nori`
   package; the base checkpoint downloads from Hugging Face on the first prediction.

## Notebook

| Notebook | Task |
|----------|------|
| `notebooks/02_regression.ipynb` | Price-elasticity and supplier-lead-time regression, including 80% prediction intervals |

Native API (scikit-learn style):

```python
from synthefy_nori import NoriRegressor

# text_columns=[] enables Nori's numeric + categorical DataFrame preprocessing;
# it does not load a text-embedding model.
reg = NoriRegressor(model="nori-6m", text_columns=[])
reg.fit(X_train, y_train)  # stores context; no gradient training
y_pred = reg.predict(X_test)
q10, q90 = reg.predict(
    X_test,
    output_type="quantiles",
    quantiles=[0.1, 0.9],
)
```

Use `model="nori-30m"` to select the larger checkpoint. Both checkpoints use the
same estimator API.

The notebook imports shared helpers from
[`common/evaluation.py`](../../common/evaluation.py) for the deterministic split,
metrics, sklearn baselines, and writes to the shared `vendor_benchmark_results`
Delta table consumed by the vendor-comparison notebook.

## Path note

The notebook adds the repository-level `common/` directory to `sys.path` through
a relative `REPO_ROOT` (three levels up from `vendors/nori/notebooks/`). If you
check the repository out at a non-standard workspace path, adjust `REPO_ROOT` in
the Configuration cell.
