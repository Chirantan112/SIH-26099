# SIH-26099 — CPSE Material Code Harmonization

## Project Overview

SIH 2026 Problem Statement 26099 focuses on standardizing and harmonizing material descriptions and material codes used across Central Public Sector Enterprises (CPSEs). This repository contains an offline, deterministic prototype for that workflow, including a synthetic CPSE development/evaluation dataset.

The dataset is **not real CPSE procurement data**. It is a small, reproducible fixture designed to exercise normalization, technical attribute extraction, record linkage, catalog mapping, evaluation, and the demonstration interface. It does not represent actual CPSE material-master statistics or production performance.

## Architecture

```text
Raw Material Description
↓
LEGO #2 — Normalization
↓
LEGO #3 — Attribute Extraction
↓
LEGO #4 — Record Linkage
↓
LEGO #5 — Catalog Mapping
↓
LEGO #6 — Evaluation
↓
LEGO #7 — Demo Pipeline
↓
LEGO #8 — Streamlit Dashboard
```

- **LEGO #1 — Synthetic CPSE Material Dataset:** Generates the deterministic synthetic material master and labelled evaluation-pair fixtures.
- **LEGO #2 — Normalization:** Expands controlled abbreviations, standardizes units and formatting, and records the transformations applied.
- **LEGO #3 — Attribute Extraction:** Parses normalized descriptions into structured, category-specific technical attributes.
- **LEGO #4 — Record Linkage:** Compares extracted attributes using deterministic similarity scoring and returns `SAME`, `DIFFERENT`, or `UNCERTAIN` with audit evidence.
- **LEGO #5 — Catalog Mapping:** Resolves legacy records against a canonical catalog individually or in batches, returning `MATCHED`, `UNCERTAIN`, or `NEW_CANDIDATE`.
- **LEGO #6 — Evaluation:** Computes classification counts and metrics from the synthetic labelled pair fixture.
- **LEGO #7 — Demo Pipeline:** Connects the normalization, extraction, and catalog-mapping stages into one deterministic offline API for the demonstration.
- **LEGO #8 — Streamlit Dashboard:** Provides an interactive UI for entering descriptions and viewing normalized text, extracted attributes, decisions, scores, and candidate evidence.

## LEGO Modules

| LEGO | File | Purpose |
|---|---|---|
| #1 | `src/generate_dataset.py` | Generate the synthetic CPSE material master and evaluation pairs. |
| #2 | `src/normalization.py` | Deterministically normalize material descriptions. |
| #3 | `src/attribute_extraction.py` | Extract structured technical material attributes. |
| #4 | `src/record_linkage.py` | Perform deterministic pairwise linkage and similarity scoring. |
| #5 | `src/catalog_mapping.py` | Perform batch catalog mapping and entity resolution. |
| #6 | `src/evaluation.py` | Compute evaluation counts and classification metrics. |
| #7 | `src/demo_pipeline.py` | Run one raw description through the offline demonstration pipeline. |
| #8 | `app.py` | Run the Streamlit interactive demonstration dashboard. |

## Current Capabilities

- Deterministic description normalization
- Structured technical attribute extraction
- Pairwise `SAME` / `DIFFERENT` / `UNCERTAIN` linkage decisions
- `MATCHED` / `UNCERTAIN` / `NEW_CANDIDATE` catalog-mapping decisions
- Batch catalog mapping
- Deterministic similarity scoring
- Explainable candidate evidence and audit notes
- Evaluation metrics
- Streamlit interactive demonstration

LLM-based or AI semantic matching is **not currently implemented**. The prototype is deterministic, offline, and does not require external APIs.

## Evaluation Results

Verified results on the synthetic evaluation fixture:

```text
TP = 21
TN = 10
FP = 0
FN = 6

Accuracy  = 0.8378
Precision = 1.0000
Recall    = 0.7778
F1        = 0.8750
```

These metrics come from the synthetic evaluation fixture and **must not be interpreted as real CPSE production performance**.

The repository has **78 automated tests passing**.

## Quick Start

The dataset generator and core pipeline use the Python standard library. To regenerate the synthetic fixtures and run the automated tests:

```powershell
python src/generate_dataset.py
python -m unittest discover -s tests -v
```

Generated artifacts include:

- `data/demo/material_master.csv` — synthetic legacy material records.
- `data/evaluation/material_pairs.csv` — labelled `SAME` and `DIFFERENT` record pairs.

See [docs/dataset_design.md](docs/dataset_design.md) for the dataset schema and assumptions.

## Streamlit Demo

Install Streamlit in the active Python environment if it is not already available, then start the dashboard with:

```powershell
python -m streamlit run app.py
```

The dashboard accepts a raw material description and displays the normalized description, extracted technical attributes, catalog decision, similarity score, and explainable candidate evidence.
