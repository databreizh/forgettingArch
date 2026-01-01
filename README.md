# Collaborative Forgetting in Multi-Party Dataflows

This repository contains the reference implementation and experimental artifacts
for the research paper:

**“Collaborative Forgetting in Multi-Party Dataflows”**  
(*Submitted to VLDB 2026*)

The goal of this project is to provide a principled framework and practical
algorithms for handling **data deletion requests** (e.g., GDPR Right to be
Forgotten) in **multi-party data pipelines**, where data items are shared,
transformed, aggregated, and reused across heterogeneous actors.

In such settings, deleting a single contribution may invalidate large parts of
the pipeline. Collaborative Forgetting makes these effects explicit and
computable.

---

## 💡 Key Concepts

At the core of this project is the **Collaborative Forgetting Graph (CFG)**, a
directed acyclic graph modeling provenance, ownership, and dependency semantics.

- **Nodes** represent data items, transformations, composite artifacts, or models.
- **Edges** encode dependency semantics and propagation rules.

The framework supports:

- **Strong dependencies**  
  Mandatory propagation: if a parent node is deleted, all strongly dependent
descendants must be invalidated.

- **Weak / aggregated dependencies**  
  Resilience-aware propagation: deleting a single input may not invalidate an
aggregation or downstream artifact.

- **Cost-aware forgetting**  
  Forgetting decisions balance **deletion cost** against **recomputation cost**
in order to approximate *Minimal Consistent Forgetting (MCF)*.

We study and compare several propagation strategies:
- **Naive cascading deletion** (baseline)
- **Greedy minimal propagation**
- **Cost-aware propagation**
- **Cluster-based propagation**

---

## 📁 Repository Structure

```text
forgettingArch/
├── cfg/                          # Core CFG framework
│   ├── core/                     # Graph, Node, Edge definitions
│   ├── propagation/              # Forgetting algorithms
│   │   ├── naive.py
│   │   ├── greedy.py
│   │   ├── cost_aware.py
│   │   └── cluster_based.py
│   ├── synthetic/                # Synthetic CFG generators
│   ├── real_flows.py             # Pseudo-realistic CFGs (crowd & ML)
│   └── utils/                    # Summaries, costs, helpers
│
├── experiments/                  # Reproducibility scripts
│   ├── run_all_experiments.py    # Synthetic experiments (main entry point)
│   ├── run_real_flows.py         # Semi-realistic pipelines
│   ├── aggregate_summaries.py    # Merge JSON summaries → CSV
│   ├── make_plot_tables.py       # Tables used for figures
│   └── plot_results.py           # Paper-ready figures (B/W + PDF)
│
├── experiments_output_camera/    # Precomputed camera-ready results
├── results/                      # Aggregated CSV tables
├── figures/                      # Final figures (PNG + PDF)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

**Requirements**
- Python ≥ 3.9

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ✅ Quick Sanity Check

Verify that the core framework imports correctly:

```bash
python -c "import cfg; print('CFG import OK')"
```

---

## 🚀 Reproducing the Experiments

### 1️⃣ Synthetic CFG Experiments

Run the full synthetic benchmark suite used in the paper:

```bash
python -m experiments.run_all_experiments
```

---

### 2️⃣ Aggregate Results

```bash
python -m experiments.aggregate_summaries \
  experiments_output_camera \
  --output experiments/summary_all.csv
```

---

### 3️⃣ Generate Tables Used for Figures

```bash
python -m experiments.make_plot_tables \
  experiments/summary_all.csv \
  --out results/plot_table_used_for_figures.csv
```

---

### 4️⃣ Generate Paper Figures

```bash
python -m experiments.plot_results \
  experiments/summary_all.csv \
  --outdir figures
```

---

## 🌍 Semi-Realistic Workflows

```bash
python -m experiments.run_real_flows
python -m experiments.plot_real_flows
```

---

## 📊 Metrics Reported

- deleted_size
- raw_cost / weighted_cost
- relative_cost_vs_naive
- runtime_sec
- exact solution (for small graphs)

---

## 📝 Citation

```bibtex
@article{cfg2026vldb,
  title     = {Collaborative Forgetting in Multi-Party Dataflows},
  author    = {Anonymous Authors},
  journal   = {Proceedings of the VLDB Endowment (PVLDB)},
  year      = {2026},
  note      = {Under review}
}
```

---

## ✉️ Contact

For questions, please open an issue or contact the authors via the VLDB
submission system.

