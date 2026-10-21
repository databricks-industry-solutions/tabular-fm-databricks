# Tabular Foundation Models on Databricks

[![Databricks](https://img.shields.io/badge/Databricks-Solution_Accelerator-FF3621?style=for-the-badge&logo=databricks)](https://databricks.com)
[![Unity Catalog](https://img.shields.io/badge/Unity_Catalog-Enabled-00A1C9?style=for-the-badge)](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)
[![Serverless](https://img.shields.io/badge/Serverless-Compute-00C851?style=for-the-badge)](https://docs.databricks.com/en/compute/serverless.html)

A solution accelerator demonstrating how to use **tabular foundation models** (TFMs) on
Databricks. Tabular foundation models are pretrained models that make accurate predictions
on tabular data with little-to-no task-specific training — eliminating much of the model
training, hyperparameter tuning, and per-use-case maintenance overhead of traditional ML.

> **Multi-vendor scope.** This repository hosts examples across multiple tabular
> foundation model vendors behind a shared evaluation harness. See
> [Supported vendors](#supported-vendors) for what's included, or
> [Adding a vendor](#adding-a-vendor) to contribute another.

> **Fine-tuning.** We also demonstrate how to fine-tune tabular foundation models
> using the Databricks AI Runtime (Serverless GPU). See the
> [`air_finetuning/`](air_finetuning/) folder for the fine-tuning notebooks.

## Overview

### The Challenge: Enterprise-Scale Predictive Analytics

Across industries, enterprises rely on large-scale tabular prediction to drive critical
business decisions — from forecasting demand to assessing risk to optimizing operations.
These workloads often span interconnected business processes, each requiring its own
models, features, and continuous maintenance. At enterprise scale organizations may
maintain **thousands or even millions of models** across products, SKUs, regions, and
customers — substantial, costly operational overhead.

### The Solution: Foundation Models for Tabular Data

**A pretrained tabular foundation model:**
- Works **out of the box** on tabular data
- Requires **minimal preprocessing**
- Eliminates most **model training, hyperparameter tuning, and experimentation**
- Delivers performance **comparable to, or better than**, many carefully tuned models
- Often provides **built-in uncertainty quantification** (prediction intervals, not just point estimates)

## Features

| Feature | Description |
|---------|-------------|
| **Classification** | Binary and multi-class classification with probability estimates |
| **Regression** | Continuous value prediction with uncertainty quantification |
| **Outlier Detection** | Anomaly scoring using semi-supervised learning |
| **Time Series Forecasting** | Lag-based forecasting |
| **Unity Catalog Integration** | Read/write data from Delta tables |
| **Realistic Datasets** | Synthetic retail/CPG data with business-relevant features |

## Supported vendors

| Vendor | Consumption | Compute on Databricks | Tasks | License |
|--------|-------------|-----------------------|-------|---------|
| **[TabPFN](vendors/tabpfn/)** (Prior Labs) | Hosted API & Self-hosted weights | Serverless **CPU** | Classification · Regression · Outlier · Forecasting | ⚠️ Weights **TABPFN-3 Non-Commercial** (Code Prior Labs License; API is commercial) |
| **[TabFM](vendors/tabfm/)** (Google) | Self-hosted weights (HF) | Serverless **GPU** (AI Runtime) Recommended; CPU supported| Classification · Regression | ⚠️ Weights **TabFM Non-Commercial License v1.0** (code Apache-2.0) |
| **[TabICL](vendors/tabicl/)** (soda-inria) | Self-hosted weights (HF) | Serverless **GPU** (AI Runtime) Recommended; CPU supported | Classification · Regression | **BSD 3-Clause** |
| **[Nori](vendors/nori/)** (Synthefy) | Self-hosted weights (HF) | Serverless **GPU** (AI Runtime) Recommended; CPU supported | Regression | **Apache-2.0** (code and weights) |
| **[KumoTabular](vendors/kumo/)** (Nvidia) | Self-hosted weights (HF) | Serverless **GPU** (AI Runtime) Recommended; CPU supported | Classification · Regression | **OpenMDW License 1.1** (commercial use permitted) |

Each vendor lives in `vendors/<vendor>/` with its own README (setup, auth, license,
compute), `requirements.txt`, and task notebooks. Pick the folder for the vendor you
care about — the notebooks use each vendor's **native** API so they're realistic and
copy-pasteable — or run several and compare.

## Project Structure

```
tabular-fm-databricks/
├── shared/
│   ├── notebooks/
│   │   ├── 00_data_preparation.ipynb    # Generate retail/CPG datasets (run first)
│   │   └── 05_vendor_comparison.ipynb   # Aggregate + compare all vendors
│   └── scripts/
│       ├── util.py                       # Synthetic data generators
│       └── cleanup.sh                    # Resource cleanup
├── common/
│   └── evaluation.py                     # Shared split, metrics, baselines, results logging
│   └── config.py                         # Configuration file to adapt to your Unity Catalog structure
├── vendors/
│   ├── tabpfn/    (README + requirements + notebooks 01–04)
│   ├── tabfm/     (README + requirements + notebooks 01–02)
│   ├── tabicl/    (README + requirements + notebooks 01–02)
│   ├── nori/      (README + requirements + regression notebook 02)
│   └── kumo/      (README + requirements + notebooks 01–02)
├── air_finetuning/                        # Fine-tuning notebooks using AI Runtime (Serverless GPU)
├── databricks.yml                        # Databricks Asset Bundle configuration
├── requirements.txt                      # Base/shared dependencies
├── CONTRIBUTING.md · LICENSE.md · NOTICE.md · SECURITY.md
```

**How it fits together.** `shared/` generates one set of Delta tables that every vendor
reads — so comparisons are apples-to-apples. `common/evaluation.py` provides the
vendor-agnostic pieces (identical train/test split, sklearn baselines, metrics) and
writes results to a single `vendor_benchmark_results` Delta table. Each vendor notebook
runs its native model on that shared data and appends its results; the comparison
notebook reads them back.

## Prerequisites

1. Databricks Workspace with Unity Catalog enabled
2. Each vendor has specific credentials/compute requirements. See each vendor's README. For instance TabPFN needs an API token while the the self-hosted models (TabFM, TabICL, Nori, KumoTabular) need no credentials and recommend a GPU, ideally Serverless GPU (AI Runtime)
3. Databricks CLI (optional, for local development)

## Getting Started

### Zero-shot inference (no training)

1. **Generate the shared data** — run [`shared/notebooks/00_data_preparation.ipynb`](shared/notebooks/00_data_preparation.ipynb). This creates the Delta tables all vendors use. Make sure to update [`common/config.py`](common/config.py) with your catalog configuration.
2. **Pick a vendor** — open `vendors/<vendor>/` and follow its README for setup, then run its `01`/`02`… notebooks.
3. **Compare** (optional) — after running one or more vendors, run [`shared/notebooks/05_vendor_comparison.ipynb`](shared/notebooks/05_vendor_comparison.ipynb) for a side-by-side ranking against the shared baselines.

### Fine-tuning with AI Runtime

The [`air_finetuning/`](air_finetuning/) folder is self-contained with everything needed to explore fine-tuning tabular foundation models on Databricks Serverless GPU (AI Runtime) — including data preparation, training, and inference. See its own README for details.

## Compute Requirements

All self-hosted models (TabFM, TabICL, Nori, KumoTabular) can run on **Serverless
CPU**, but inference will be slow. We recommend using a **GPU** for practical
performance — the best experience is with **Serverless GPU (AI Runtime)** thanks to
its instant provisioning. Models can also run on a dedicated **all-purpose GPU
cluster**. **TabPFN** is the
exception: it calls a hosted API, so Serverless CPU is sufficient.

Because every vendor writes to the same results table, you can run them on different
clusters and still compare them afterward. For **TabICL** there is an example on how
to serve the model through a GPU serving endpoint, which allows pipelines to run on
CPU and offload inference to the dedicated GPU endpoint.

## Adding a vendor

New vendors are self-contained under `vendors/<vendor>/`. The quickest path is to copy an
existing vendor (TabFM is a good template for a self-hosted model) and swap the model
layer, keeping the shared data, `common/evaluation.py` helpers, and results-logging
contract intact. See [`vendors/tabicl/README.md`](vendors/tabicl/README.md) for a
worked-through contributor checklist.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and CLA information.

## License

This project is provided subject to the [Databricks License](LICENSE.md).

---

© 2025 Databricks, Inc. All rights reserved.
