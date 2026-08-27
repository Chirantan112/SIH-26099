# SIH-26099 — CPSE Material Code Harmonization

## SIH 2026 Problem Statement 26099

**AI-Driven Standardization and Harmonization of Material Codes Across CPSEs**

This project standardizes and harmonizes material descriptions and material codes used across Central Public Sector Enterprises (CPSEs). It combines an explainable deterministic technical matching pipeline with optional AI advisory components while keeping the final material-mapping decision reproducible and authoritative.

The repository contains a **synthetic CPSE development/evaluation dataset**. It is not real CPSE procurement or material-master data and must not be interpreted as production statistics or production performance.

## Final Architecture

The deterministic path is the authority. Optional Local NLP and Gemini 2.5 Flash provide separate advisory candidate suggestions and interpretation only.

```text
Raw Material Description
        |
        v
Deterministic Normalization
        |
        v
Technical Attribute Extraction
        |
        v
Deterministic Record Linkage + Catalog Mapping
        |
        +------------------------+
        |                        |
        v                        v
 Optional Local NLP         Optional Gemini 2.5 Flash
 Advisory Retrieval         LLM Advisory Interpretation
        |                        |
        +------------+-----------+
                     |
                     v
          AUTHORITATIVE FINAL DECISION
```

`run_hybrid_pipeline()` orchestrates the unchanged deterministic LEGO #2-#5 flow and optional AI diagnostics. AI output is bounded and validated against the supplied catalog; it never changes the authoritative mapping result.

## Decision Authority

| Component | Role | Authority |
|---|---|---|
| Normalization | Canonicalizes descriptions, terminology, units, punctuation, and formatting | Deterministic |
| Attribute extraction | Extracts explicit technical material attributes | Deterministic |
| Record linkage | Computes deterministic technical similarity/evidence | **Authoritative** |
| Catalog mapping | Produces `MATCHED`, `UNCERTAIN`, or `NEW_CANDIDATE` | **Authoritative** |
| Local NLP | Optional local embedding retrieval | Advisory only |
| Gemini 2.5 Flash | Optional LLM candidate interpretation | Advisory only |
| `run_hybrid_pipeline()` | Coordinates deterministic result plus advisory diagnostics | Deterministic result remains authoritative |

**AI suggestions never override, replace, filter, or feed back into the deterministic final decision.**

## Judge-Facing Streamlit Dashboard

`app.py` provides a responsive dashboard for demonstrating the complete workflow.

The interface is designed for desktop, laptop, tablet, and mobile use and presents:

- a premium dark enterprise/industrial visual theme;
- a persistent sidebar with a functional **Hybrid AI / Deterministic Only** mode selector;
- explicit **AUTHORITATIVE** vs **ADVISORY** roles;
- a material-analysis workspace with quick demo scenarios;
- the pipeline **INPUT -> NORMALIZE -> EXTRACT -> MATCH -> AI ADVISE -> DECIDE**;
- a prominent authoritative decision card;
- normalization and technical-attribute evidence;
- deterministic candidate evidence;
- separate Local NLP and Gemini 2.5 Flash advisory sections;
- graceful empty-input, unavailable-service, fallback, and no-candidate states;
- stale-result protection when the analysis mode changes after a result has been produced.

The dashboard does not use fake confidence values, fake analytics, or fabricated AI behavior.

## AI Advisory Components

### Local NLP

Local NLP is implemented as an optional local embedding retrieval adapter. When its optional embedding dependencies are available, it produces advisory catalog candidates from the supplied catalog.

It is:

- optional;
- local/offline when the required optional dependencies are available;
- lazily initialized;
- advisory only;
- never used to override deterministic mapping.

The cloud deployment environment must install the required Local NLP dependencies before claiming that Local NLP is available in the deployed app.

### Gemini 2.5 Flash

Gemini uses:

```text
gemini-2.5-flash
```

Its role is limited to returning advisory catalog candidates and short reasons. It does not produce the authoritative mapping decision, confidence, probability, or deterministic decision state.

Configure the credential using:

```text
GEMINI_API_KEY
```

Never place the credential in source code, tests, README files, commits, or repository configuration.

If Gemini is unavailable, missing credentials, or fails, the application safely continues with deterministic matching.

## Demo Example

Representative input:

```text
CS GATE VLV 50MM FLG CL150
```

The application normalizes the legacy description, extracts explicit technical attributes, performs deterministic catalog mapping, and displays the authoritative result. When AI Advisory is enabled, Local NLP and Gemini 2.5 Flash may provide additional candidates and reasons without changing that deterministic result.

Available demonstration states:

- `MATCHED`
- `UNCERTAIN`
- `NEW CANDIDATE`

## Repository Structure

```text
SIH-26099/
├── app.py
├── README.md
├── data/
│   ├── demo/
│   └── evaluation/
├── docs/
│   └── dataset_design.md
├── src/
│   ├── ai_retrieval.py
│   ├── attribute_extraction.py
│   ├── catalog_mapping.py
│   ├── demo_pipeline.py
│   ├── evaluation.py
│   ├── gemini_llm.py
│   ├── generate_dataset.py
│   ├── hybrid_pipeline.py
│   ├── llm_interpretation.py
│   ├── local_embedding_retrieval.py
│   ├── normalization.py
│   └── record_linkage.py
└── tests/
    └── ...
```

## Run Locally

Create/use a Python virtual environment and install the project's required dependencies for the environment you are testing.

Run the deterministic regression suite:

```powershell
python -m unittest discover -s tests -v
```

Launch the dashboard:

```powershell
python -m streamlit run app.py
```

## Streamlit Community Cloud

For a reproducible cloud deployment, the repository must include the dependency configuration required by the selected environment, including the packages required for optional Local NLP and Gemini functionality.

Configure the Gemini credential in the deployment's Secrets configuration:

```toml
GEMINI_API_KEY = "your-key"
```

Never commit this value to GitHub.

The laptop's `.venv` is **not** part of the repository and must not be committed. Cloud environments create their own runtime from the repository dependency configuration.

## Safety and Decision Boundary

The project intentionally separates deterministic engineering logic from AI assistance:

- **DETERMINISTIC ENGINE = AUTHORITATIVE**
- **LOCAL NLP = OPTIONAL / ADVISORY**
- **GEMINI 2.5 FLASH = OPTIONAL / LLM / ADVISORY**

AI candidates must refer to canonical IDs already present in the supplied catalog. Invalid, malformed, duplicate, or unknown candidates are rejected by the validation/orchestration layer.

When AI Advisory is disabled, Local NLP and Gemini are not invoked. Deterministic processing continues normally.

Missing or conflicting technical information is handled conservatively by the deterministic matching logic.

## Project Principles

1. Deterministic technical matching comes first.
2. AI is optional and advisory.
3. AI cannot override deterministic decisions.
4. Candidate IDs are validated against the supplied catalog.
5. Missing or conflicting technical information is handled conservatively.
6. The core harmonization flow remains usable without external AI services.
7. Credentials belong in environment/deployment secrets, never in the repository.
8. The dashboard should remain understandable and usable across device sizes.
