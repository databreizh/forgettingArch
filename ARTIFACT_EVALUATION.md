# Artifact Evaluation

This document describes how to evaluate the artifact accompanying the paper:

**Collaborative Forgetting in Multi-Party Dataflows**  
(Submitted to VLDB 2026)

The goal of this artifact is to demonstrate the correctness, reproducibility,
and practical relevance of the proposed Collaborative Forgetting Graph (CFG)
model and its propagation algorithms.

---

## 1. Artifact Overview

The artifact provides:

- A reference implementation of the **Collaborative Forgetting Graph (CFG)**.
- Multiple forgetting propagation algorithms:
  - Naive cascading deletion
  - Greedy minimal propagation
  - Cost-aware forgetting
  - Cluster-based propagation
- Synthetic and semi-realistic experimental workflows.
- Scripts to reproduce all figures reported in the paper.

The artifact supports the following evaluation dimensions:
- **Scalability**
- **Forgetting cost**
- **Comparison against naive deletion**
- **Behavior on multi-party workflows**

---

## 2. Requirements

- Python **3.9 or later**
- Tested on Linux, macOS, and Windows
- No GPU required

All dependencies are listed in `requirements.txt`.

---

## 3. Installation

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
````

Expected outcome:

* All dependencies install without errors.
* No additional system libraries are required.

---

## 4. Sanity Check (Core Imports)

Verify that the CFG core and propagation algorithms can be imported:

```bash
python - <<EOF
from cfg.core.graph import CollaborativeForgettingGraph
from cfg.propagation.greedy import GreedyMinimalPropagation
from cfg.propagation.cost_aware import CostAwarePropagation
from cfg.propagation.cluster_based import ClusterBasedPropagation
print("CFG core imports OK")
EOF
```

Expected output:

```
CFG core imports OK
```

---

## 5. Reproducing Synthetic Experiments

To reproduce the synthetic benchmarks reported in the paper:

```bash
python -m experiments.run_all_experiments
```

This command:

* Generates synthetic CFGs of increasing size.
* Runs all forgetting algorithms with fixed random seeds.
* Produces per-run JSON summaries.

Expected output directory:

```
experiments_output_camera/
  ├── n500/
  ├── n1000/
  ├── ...
```

Each subdirectory contains files of the form:

```
*_summary.json
```

Expected runtime: a few minutes on a standard laptop.

---

## 6. Aggregating Results

Aggregate all experiment summaries into a single CSV file:

```bash
python -m experiments.aggregate_summaries \
    experiments_output_camera \
    --output experiments/summary_all.csv
```

Expected output:

```
experiments/summary_all.csv
```

This file contains:

* Runtime metrics
* Deletion and recomputation sizes
* Raw and normalized forgetting costs
* Algorithm identifiers and graph sizes

---

## 7. Generating Plot Tables

Extract exactly the values used for plotting the paper figures:

```bash
python -m experiments.make_plot_tables \
    experiments/summary_all.csv \
    --out results/plot_table_used_for_figures.csv
```

Expected output:

```
results/plot_table_used_for_figures.csv
```

This CSV is the direct input used for figure generation.

---

## 8. Reproducing Synthetic Figures

Generate the figures reported in the synthetic evaluation:

```bash
python -m experiments.plot_results \
    experiments/summary_all.csv \
    --outdir figures
```

Expected figures:

```
figures/
  ├── runtime_vs_size.png
  ├── runtime_vs_size.pdf
  ├── normalized_cost_vs_size.png
  ├── normalized_cost_vs_size.pdf
  ├── relative_cost_vs_size.png
  ├── relative_cost_vs_size.pdf
```

Minor numerical variations are expected due to randomness, but
all qualitative trends must match the paper.

---

## 9. Reproducing Semi-Realistic Workflows

To evaluate forgetting on pseudo-real multi-party pipelines:

```bash
python -m experiments.run_real_flows
```

This evaluates:

* A mutualized crowdsourcing pipeline
* A hybrid ML pipeline with shared representations

Outputs are written to:

```
experiments_real/
  ├── crowd/
  └── ml/
```

---

## 10. Reproducing Real-Flow Figures

Generate figures for semi-realistic workflows:

```bash
python -m experiments.plot_real_flows
```

Expected figures:

```
figures/
  ├── real_flows_deleted_size_bar.png
  ├── real_flows_runtime_bar_log.png
  ├── real_flows_weighted_cost_bar_log.png
```

Log scales are used where appropriate to handle scale differences
between workflows.

---

## 11. Limitations

* Exact minimal forgetting is computed only for very small graphs
  (≤ 26 nodes), due to NP-hardness.
* Semi-realistic workflows are representative but not extracted
  from proprietary production traces.
* The artifact focuses on static DAGs and does not yet support
  streaming or incremental forgetting.

---

## ✉️ Contact

For questions, please open an issue or contact the authors via the VLDB
submission system.

