# SIH-26099 — CPSE Material Code Harmonization

## SIH 2026 Problem Statement 26099

**AI-Driven Standardization and Harmonization of Material Codes Across CPSEs**

This project addresses the standardization and harmonization of material descriptions and material codes used across Central Public Sector Enterprises (CPSEs). It combines a deterministic technical matching pipeline with optional AI advisory components while keeping the final material-mapping decision explainable, reproducible, and authoritative.

The repository includes a **synthetic CPSE development/evaluation dataset**. It is not real CPSE procurement or material-master data and must not be interpreted as production statistics or production performance.

## Final Architecture

The core decision path is deterministic. Optional AI components provide additional candidate suggestions and interpretation, but they never participate in or override the authoritative mapping decision.

```text
Raw Material Description
        |
        v
Deterministic Normalization
        |
        v
Attribute Extraction / Local NLP
        |
        v
Deterministic Record Linkage + Catalog Mapping
        |
        +----------------------+
        |                      |
        v                      v
 Optional Local NLP       Optional Gemini 2.5 Flash
 Advisory Retrieval       Advisory Interpretation
        |                      |
        +----------+-----------+
                   |
                   v
        AUTHORITATIVE FINAL DECISION
```

`run_hybrid_pipeline()` is the orchestration layer around the unchanged deterministic LEGO #2-#5 flow. The deterministic mapping is performed independently of AI suggestions; AI output is bounded, validated against the supplied catalog, and retained only as advisory information.

### Decision authority

| Component | Role | Authority |
|---|---|---|
| Normalization | Canonicalizes material descriptions | Deterministic |
| Attribute extraction | Extracts technical material attributes | Deterministic |
| Record linkage | Computes deterministic technical similarity/evidence | **Authoritative** |
| Catalog mapping | Produces `MATCHED`, `UNCERTAIN`, or `NEW_CANDIDATE` | **Authoritative** |
| Local NLP | Optional local embedding retrieval | Advisory only |
| Gemini 2.5 Flash | Optional candidate interpretation | Advisory only |
| `run_hybrid_pipeline()` | Orchestrates deterministic result plus optional AI diagnostics | Deterministic result remains authoritative |

**AI suggestions never override, replace, filter, or feed back into the deterministic final decision.** AI is not required for the core system, and the deterministic pipeline can operate without external AI services.

Missing or conflicting technical information is handled conservatively by the deterministic matching logic. The resulting decision and candidate evidence remain explainable and reproducible.

## What the System Does

The system takes a legacy material description and:

1. Normalizes controlled terminology, abbreviations, units, punctuation, and formatting.
2. Extracts structured technical attributes.
3. Performs deterministic record linkage and catalog mapping.
4. Produces an authoritative `MATCHED`, `UNCERTAIN`, or `NEW_CANDIDATE` result with deterministic scoring and candidate evidence.
5. Optionally obtains Local NLP and Gemini advisory candidates when **AI Advisory** is enabled.
6. Validates advisory candidates against the supplied canonical catalog without allowing them to alter the deterministic result.

When **AI Advisory is disabled**, Local NLP retrieval and Gemini are not called. Deterministic processing continues normally.

## Streamlit Judge-Facing Dashboard

`app.py` provides the interactive Streamlit dashboard for demonstrating the complete workflow.

The dashboard presents:

- a judge-facing CPSE Material Harmonization interface;
- the pipeline from **INPUT -> NORMALIZE -> EXTRACT -> MATCH -> AI ADVISE -> DECIDE**;
- a prominent **Authoritative Decision** section;
- canonical material ID and deterministic score;
- normalized description and normalization transformations;
- extracted technical attributes;
- deterministic candidate evidence;
- separate **Local NLP** and **Gemini LLM** advisory sections;
- Local NLP and Gemini availability/fallback status;
- an **AI Advisory ON/OFF** control.

The dashboard makes the decision boundary explicit:

> **AI suggestions are advisory only. They cannot override the deterministic decision.**

The existing demonstration outcomes remain available:

- `MATCHED`
- `UNCERTAIN`
- `NEW CANDIDATE`

## AI Advisory Components

### Local NLP

Local NLP is implemented as an **optional local embedding retrieval adapter**. When its optional local embedding dependencies are available, it can produce advisory catalog candidates from the supplied catalog.

It is:

- optional;
- local/offline when the required optional dependency is available;
- lazily initialized;
- advisory only;
- never used to override deterministic mapping.

For Streamlit Community Cloud, the repository currently has **no root `requirements.txt`**. Therefore, do not claim that Local NLP is deployable in the cloud environment as-is. Deployment dependency configuration is still required before claiming that the deployed cloud environment provides Local NLP.

### Gemini 2.5 Flash

Gemini is an optional LLM interpretation adapter using:

```text
gemini-2.5-flash
```

Its role is limited to returning advisory catalog candidates and short reasons. It does not produce the authoritative mapping decision, confidence, probability, or deterministic decision state.

Configure the Gemini credential through the environment/deployment secret:

```text
GEMINI_API_KEY
```

**Never place the API key or any other credential in source code, tests, README files, commits, or repository configuration.**

If Gemini is unavailable, missing credentials, or fails, the application falls back safely to deterministic matching. The core system remains operational.

## Safety / Decision Authority

The project intentionally uses a conservative boundary between deterministic technical matching and AI assistance.

- **DETERMINISTIC ENGINE = AUTHORITATIVE**
- **LOCAL NLP = OPTIONAL / ADVISORY**
- **GEMINI LLM = OPTIONAL / ADVISORY**

AI candidates must refer to canonical IDs already present in the supplied catalog. Invalid, malformed, duplicate, or unknown candidates are rejected by the adapter/orchestration validation layer.

AI output cannot create a `MappingResult`, change an existing `MappingResult`, call the deterministic mapper, or override a deterministic technical conflict. The deterministic LEGO #2-#5 result is always the final decision.

This design also means that external AI availability is not a prerequisite for the core harmonization workflow.

## Demo Example

A representative dashboard input is:

```text
CS GATE VLV 50MM FLG CL150
```

The deterministic pipeline normalizes the description, extracts the technical attributes, and performs catalog mapping. The resulting deterministic decision is **authoritative**. If AI Advisory is enabled, Local NLP and/or Gemini may provide additional catalog candidates and reasons, but those suggestions remain advisory and cannot change the deterministic result.

## Repository Structure

The important current repository files are:

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

Key responsibilities:

| File | Responsibility |
|---|---|
| `app.py` | Streamlit judge-facing dashboard and UI orchestration |
| `src/normalization.py` | Deterministic description normalization |
| `src/attribute_extraction.py` | Structured technical attribute extraction |
| `src/record_linkage.py` | Deterministic pairwise linkage and similarity evidence |
| `src/catalog_mapping.py` | Deterministic catalog mapping and decision generation |
| `src/hybrid_pipeline.py` | Deterministic pipeline plus optional advisory orchestration |
| `src/local_embedding_retrieval.py` | Optional Local NLP advisory retrieval |
| `src/gemini_llm.py` | Optional Gemini advisory interpretation |
| `src/ai_retrieval.py` | Advisory candidate/status interfaces |
| `src/llm_interpretation.py` | LLM advisory interface |
| `src/demo_pipeline.py` | Demonstration catalog/pipeline support |
| `src/evaluation.py` | Synthetic evaluation calculations |
| `src/generate_dataset.py` | Synthetic CPSE dataset generation |
| `tests/` | Automated regression and component tests |

## Testing

The repository's currently verified local regression result is:

```powershell
python -m unittest discover -s tests -v
```

```text
141 tests passing
```

This is a **local verification result**. It is not a claim that 141 tests execute automatically in Streamlit Community Cloud.

No coverage percentage is claimed here.

## Run Locally

The deterministic test suite can be run with:

```powershell
python -m unittest discover -s tests -v
```

To launch the Streamlit dashboard:

```powershell
python -m streamlit run app.py
```

The repository currently has **no root `requirements.txt`**. Do not infer or assume a complete deployment dependency set from this README.

## Streamlit Community Cloud Deployment

The dashboard can be deployed from the GitHub repository using Streamlit Community Cloud. Before claiming a cloud deployment provides optional AI functionality, configure the dependencies required by the selected deployment environment.

For Gemini:

1. Deploy the Streamlit app from the GitHub repository.
2. Open the application's deployment **Secrets** configuration.
3. Add `GEMINI_API_KEY` as a deployment secret.
4. Never commit the credential to GitHub or place it in README/source files.

Gemini remains optional: if the credential or service is unavailable, deterministic matching continues to operate.

For **Local NLP**, the repository does not currently provide a root `requirements.txt`, so the cloud deployment dependency configuration must be established before claiming that Local NLP is available in the deployed environment.

## Synthetic Dataset and Evaluation

The repository contains a small synthetic material master and labelled evaluation fixture used for deterministic development and testing. These fixtures are reproducible development artifacts, not real CPSE procurement data.

The evaluation implementation remains separate from the authoritative runtime decision path. Synthetic evaluation numbers should not be interpreted as production CPSE performance.

## Project Principles

1. **Deterministic technical matching comes first.**
2. **AI is optional and advisory.**
3. **AI cannot override deterministic decisions.**
4. **Catalog IDs are validated against the supplied catalog.**
5. **Missing or conflicting technical information is handled conservatively.**
6. **The core system remains usable without external AI services.**
7. **Credentials belong in environment/deployment secrets, never in the repository.**
