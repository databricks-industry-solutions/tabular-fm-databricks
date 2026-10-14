# Fine-tuning Tabular Foundation Models with AI Runtime

This folder contains two end-to-end examples of **fine-tuning KumoTabular Small** on Databricks using Serverless GPU (AI Runtime). It covers both **classification** and **regression**, end-to-end, from data generation through fine-tuning to inference on a held-out test set.

## Datasets

We use two OpenML datasets that resemble real-world enterprise workloads: large enough that GPU memory becomes a bottleneck, and wide enough to stress feature handling.

| Task | Dataset | Samples | Features |
| --- | --- | --- | --- |
| Classification | BNP Paribas Cardif Claims Management (did=46625) | ~114 k | 133 |
| Regression | Allstate Claims Severity (did=45046) | ~188 k | 125 |

## Notebooks

| Notebook | What it does | Compute |
| --- | --- | --- |
| `01_data_preparation` | Downloads datasets from OpenML, splits 80/20 into train/score, saves to Unity Catalog | Serverless CPU |
| `02a_classification_finetune` | Fine-tunes KumoTabular Small on the classification dataset | **H100** (AI Runtime) |
| `02b_regression_finetune` | Fine-tunes KumoTabular Small on the regression dataset | **H100** (AI Runtime) |
| `03a_classification_inference` | Runs inference on the held-out test set — pretrained vs fine-tuned | A10 (AI Runtime) |
| `03b_regression_inference` | Runs inference on the held-out test set — pretrained vs fine-tuned | A10 (AI Runtime) |

The `02` notebooks fine-tune the model and persist it in two ways: as a raw Volume checkpoint and as an MLflow-registered model in Unity Catalog (recommended for improved governance and lifecycle). The `03` notebooks load both versions plus the pretrained baseline and compare them on the score set.

## Why fine-tune on Databricks?

Fine-tuning a tabular foundation model is a deep-learning workload. Running it on Databricks gives you several concrete advantages over doing it on a standalone GPU box or a generic cloud VM:

* **Instant H100 Serverless GPU access.** No cluster provisioning, no queue. You quickly get an H100 GPU in seconds through AI Runtime, fine-tune, and release it when done. No idle cost, no capacity planning.

* **Full model lifecycle with MLflow and Unity Catalog.** Every fine-tuning run is logged as an MLflow experiment with parameters, metrics, and the model artifact. The model is registered in Unity Catalog with versioning and `@champion`/`@challenger` aliases.

* **Governance through Unity Catalog Model Registry.** Fine-tuned models are registered as versioned assets in Unity Catalog, inheriting the same permission model and audit trail you already use for tables and volumes. Every model version carries full lineage back to its training run, dataset, and parameters, giving compliance and review teams a single place to verify what was trained, by whom, and on which data.

* **Proper experiment tracking.** Training loss curves, evaluation metrics, and GPU utilization are all captured in MLflow. Compare runs, roll back to a previous version, or audit who trained what and when, without spreadsheets or manual bookkeeping.

* **Genie Code with deep-learning skills.** Databricks' AI assistant includes skills specifically designed for GPU and deep-learning workloads: debugging CUDA errors, reviewing training loops, suggesting hyperparameter changes, helping you interpret loss curves and optimize GPU usage. 

* **Closer path from experimentation to production.** The same notebook that fine-tunes or runs inference can be scheduled as a Lakeflow Job. There is no separate "export-to-production" step; inference runs in the same environment where you developed, reading from and writing to Unity Catalog tables.

## Best practices for fine-tuning

Fine-tuning does not always outperform a pretrained model (this is the case with the datasets and setup used here), and it requires an H100 compared to just an A10 for zero-shot inference. In many scenarios, switching to a larger, more expressive zero-shot model is enough. Fine-tuning also gives up most of the lower operational overhead and faster time-to-market that make zero-shot tabular foundation models attractive in the first place. Evaluate whether the accuracy gain justifies the added cost and complexity before committing to a fine-tuning workflow.

### First steps into fine-tuning: tips

The tips below are distilled from our own experience and from the reference implementations in [TabICLv2](https://github.com/soda-inria/tabicl/blob/main/tutorials/finetune_classifier.py) and [NVIDIA Structured Data Models](https://github.com/NVIDIA/structured-data-models) (KumoTabular). Each takes a different approach; fine-tuning is a highly tunable process. These steps are meant as a starting point, not a recipe.

1. **Stabilize the early-stopping metric.** Make sure your train/validation split leaves enough validation samples. Then increase `FT_EVAL_CONTEXT_SIZE` (the number of context samples used when computing validation metrics) until the metric is stable across evaluations; aim for around 10 k samples. If that causes GPU OOM, reduce `FT_NUM_ESTIMATORS_VAL` to 1 or 2 to lower memory pressure while keeping the metric reliable.

2. **Size the training batch to avoid OOM without starving the gradient.** The main memory bottleneck is the fine-tuning step itself, due to the backward pass. Try to get `FT_CONTEXT_SIZE + FT_QUERY_SIZE` (the total number of samples used for gradient computation per step) close to 10 k. The exact ceiling depends on your dataset's width and the GPU, but the goal is to keep it small enough to avoid OOM while large enough to produce stable gradient updates.

3. **If you cannot reach ~10 k, reduce estimators and feature coverage.** Lower `FT_NUM_ESTIMATORS_TRAIN` to 1 and bring `FT_FEATURE_PERCENTAGE_TRAIN` closer to 0.5 (especially useful when the dataset has many features). This frees enough memory to keep `FT_CONTEXT_SIZE + FT_QUERY_SIZE` at a reasonable level.

The MLflow side panel and the system metrics logged during each fine-tuning run (GPU memory, utilization) give you real-time visibility to find the right balance between these parameters. Genie Code can further assist by reading those metrics and suggesting size adjustments, helping you iteratively squeeze the most out of the available GPU memory.
