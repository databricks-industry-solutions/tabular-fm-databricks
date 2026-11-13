# KumoTabular (Nvidia)

[KumoTabular](https://github.com/NVIDIA/structured-data-models) is a zero-shot
tabular foundation model from Nvidia. It is **self-hosted**: the
`structured-data-models` (`sdm`) package downloads pretrained weights from
Hugging Face and runs inference locally in a single forward pass — no
task-specific training or fine-tuning required.

- Code: https://github.com/NVIDIA/structured-data-models
- License: https://openmdw.ai/license/1-1/
- HuggingFace model: https://huggingface.co/nvidia/Kumo-Tabular 
- Article: https://huggingface.co/blog/nvidia/kumo-tabular 

## ✅ License: Can be used commercially

Kumo is released under a
[commercially permissive license](https://openmdw.ai/license/1-1/) (OpenMDW
License 1.1). See the repository for full terms and attribution requirements.

## At a glance

| | |
|---|---|
| **Consumption** | Self-hosted weights (Hugging Face) — runs locally, single forward pass |
| **Compute on Databricks** | **GPU recommended**, specially with Databricks AI Runtime for instant GPU access; CPU works but is slow |
| **Auth** | None; public checkpoint downloads on first use |
| **Tasks covered** | Classification, Regression |
| **License** | **OpenMDW License 1.1** — commercial use permitted |
| **Training data** | Only on synthetic data
| **Sizes** | Comes in three different sizes Small, Medium and Large|

## Setup

1. Attach to a **serverless GPU** (AI runtime) with environment AI v6 or above. CPU is supported but significantly slower and provisioning all-purpose cluster with GPU might have longer wait times.
2. Run the shared data prep first:
   [`shared/notebooks/00_data_preparation.ipynb`](../../shared/notebooks/00_data_preparation.ipynb).
3. For `notebooks/03_inference_on_serving_endpoint.ipynb`, attach to **Serverless CPU** with environment v5
   instead. Only the serving endpoint uses a GPU. The notebook also needs a Lakebase database instance.

## Notebooks

| Notebook | Task |
|----------|------|
| `notebooks/01_classification.ipynb` | Binary & multi-class classification (supplier delay risk, material shortage, OTIF risk) |
| `notebooks/02_regression.ipynb` | Regression (price elasticity, supplier lead time) |
| `notebooks/03_inference_on_serving_endpoint.ipynb` | Serve all five tasks from one GPU serving endpoint, with the context of each task in Lakebase |


The notebooks import shared helpers from
[`common/evaluation.py`](../../common/evaluation.py) for the deterministic
split, metrics, sklearn baselines, and writes to the shared
`vendor_benchmark_results` Delta table consumed by the vendor-comparison
notebook.

## Serving with the context in Lakebase

`notebooks/03_inference_on_serving_endpoint.ipynb` serves KumoTabular from one GPU serving endpoint for many
tasks. It runs on Serverless CPU. Only the endpoint uses a GPU.

- The context rows of each task are written to a Delta table with a version number and synced to Lakebase.
- A request names its task. The endpoint fetches the task's context from Lakebase on first use, fits it once,
  and keeps the fitted context on the GPU.
- A new context version needs a write to the Delta table and a sync. No model is logged and the endpoint is
  not updated.

The notebook needs a Lakebase database instance, and the `databricks_superuser` role on it for the user who
creates the endpoint. Its last section lists what to expect for the first request of a task and the limits.

## Path note

The notebooks add the repo-level `common/` directory to `sys.path` via a
relative `REPO_ROOT` (three levels up from `vendors/kumo/notebooks/`). If you
check the repo out at a non-standard workspace path, adjust `REPO_ROOT` in the
Configuration cell.
