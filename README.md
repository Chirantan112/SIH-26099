# SIH-26099 — CPSE Material Code Harmonization

This repository starts with **Module 1 only**: a reproducible, synthetic CPSE material-master dataset and labelled evaluation pairs for SIH 2026 Problem Statement 26099, *AI-Driven Standardization and Harmonization of Material Codes Across CPSEs*.

No frontend, API, database, ERP integration, normalization, matching, embeddings, LLMs, or machine-learning dependencies are included.

## Quick start

Requires Python 3.10+ and no third-party packages.

```powershell
python src/generate_dataset.py
python -m unittest discover -s tests -v
```

Generated artifacts:

- `data/demo/material_master.csv` — synthetic legacy material records from CPCL, IOCL, and NTPC.
- `data/evaluation/material_pairs.csv` — known `SAME` and `DIFFERENT` record pairs.

Each material-master row has the common fields `cpse`, `legacy_material_code`, `raw_description`, `category`, and `ground_truth_material_id`; it also includes category-specific technical fields.

- A **canonical material** is one complete technical specification represented by a `ground_truth_material_id`.
- A **legacy material code** is a CPSE-local code. Multiple legacy codes may refer to the same canonical material, including deliberate same-CPSE duplicate cases.
- **Ground truth** is the known canonical-material identity used for development and evaluation.
- An evaluation pair labelled **SAME** has identical ground truth; **DIFFERENT** has distinct ground truth.

This is a small synthetic development/evaluation fixture, not real CPSE procurement data and not a representation of actual CPSE material-master statistics.

See [docs/dataset_design.md](docs/dataset_design.md) for the schema and assumptions.
