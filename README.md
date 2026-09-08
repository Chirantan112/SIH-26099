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
              Human Review when needed
                     |
                     v
          AUTHORITATIVE FINAL DECISION
```

`run_hybrid_pipeline()` orchestrates the deterministic matching flow and optional AI diagnostics. AI output is bounded and validated against the supplied catalog; it never changes the authoritative mapping result.

### Decision boundary

**DETERMINISTIC ENGINE = AUTHORITATIVE**  
**LOCAL NLP = OPTIONAL / ADVISORY**  
**GEMINI = OPTIONAL / ADVISORY**  
**HUMAN REVIEW = ESCALATION FOR INSUFFICIENT EVIDENCE**

## Judge-Facing Dashboard

`app.py` provides a responsive dashboard for demonstrating the complete workflow. It presents:

- a premium dark enterprise/industrial visual theme;
- Hybrid AI / Deterministic Only mode selection;
- explicit AUTHORITATIVE vs ADVISORY roles;
- quick demo scenarios for MATCHED, UNCERTAIN and NEW CANDIDATE states;
- the six-stage INPUT -> NORMALIZE -> EXTRACT -> MATCH -> AI ADVISE -> DECIDE pipeline;
- authoritative decision and technical evidence;
- separate Local NLP and Gemini advisory sections;
- cross-model advisory consensus with independent technical validation;
- graceful unavailable-service and deterministic-fallback states;
- responsive evidence tables and mobile-friendly presentation.

The dashboard does not use fake confidence values, fake analytics, or fabricated AI behavior.

## Evaluation & Benchmark Evidence

The repository includes a reproducible synthetic stress benchmark at `scripts/benchmark.py`. It generates controlled CPSE-style description variants from the canonical demo catalog and reports:

- exact canonical mapping rate;
- UNCERTAIN rate;
- wrong mapping rate;
- runtime and throughput;
- distinct canonical-pair safety outcomes.

The benchmark is deliberately labelled as synthetic. It is a regression/stress test, not evidence of production CPSE accuracy.

Run it with:

```powershell
python scripts/benchmark.py
python scripts/benchmark.py --variants-per-material 50
```

The benchmark exits non-zero if a generated equivalent description maps to the wrong canonical material. See [`docs/benchmark_protocol.md`](docs/benchmark_protocol.md) for the evidence boundary and extension plan.

## Continuous Verification

GitHub Actions runs the deterministic test suite and synthetic benchmark on Python 3.11 and 3.12 for pushes to the readiness branch and pull requests targeting `main`.

Local verification:

```powershell
python -m unittest discover -s tests -v
python scripts/benchmark.py
```

Optional AI dependencies are intentionally not required for the deterministic CI job. This keeps core verification independent of Gemini credentials, model downloads, or network services.

## AI Advisory Components

### Local NLP

Local NLP is an optional local embedding retrieval adapter. When its optional embedding dependencies are available, it produces advisory catalog candidates from the supplied catalog.

It is lazy, bounded, fault-tolerant, and never used to override deterministic mapping.

### Gemini 2.5 Flash

Gemini uses `gemini-2.5-flash` for advisory candidate interpretation and technical evidence. It does not produce the authoritative mapping decision, confidence, probability, or deterministic decision state.

Configure credentials using `GEMINI_API_KEY`. Never commit the credential.

If Gemini is unavailable, missing credentials, or fails, the application safely continues with deterministic matching.

## Technical References

The following references document the established technical foundations used by CATALYST. They are references for the engineering approaches, not evidence of validation on official CPSE production data.

### Semantic Retrieval / Embeddings

- [Sentence Transformers — Semantic Search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html) — background on embedding-based semantic retrieval and candidate search.

### LLM Advisory Interpretation

- [Google AI for Developers — Gemini API Models](https://ai.google.dev/gemini-api/docs/models) — official Gemini API model documentation.
- [Google AI for Developers — Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash) — official documentation for the model used by the advisory component.

### Project Evidence

- [SIH-26099 repository](https://github.com/Chirantan112/SIH-26099) — implementation, tests, synthetic evaluation data, benchmark protocol, and dashboard.
- [`docs/dataset_design.md`](docs/dataset_design.md) — synthetic CPSE-style dataset design and evidence boundary.
- [`docs/benchmark_protocol.md`](docs/benchmark_protocol.md) — reproducible benchmark methodology and limitations.

## Human Review & Auditability

The governance layer records analysis and reviewer actions in an append-only, session-friendly audit trail. Review actions are constrained to `APPROVE`, `REJECT`, or `REVIEW`, with timestamp, deterministic decision, selected candidate and reviewer note available for export by the UI.

The prototype deliberately keeps governance storage independent from the matching engine so a production deployment can replace in-memory storage with an approved database without changing the decision contract.

## Real-World Integration Boundary

The current demo source is the repository's validated CSV material master. An explicit `SAPMaterialCatalogSource` integration boundary exists, but live SAP/ERP connectivity is **not claimed or configured** because CPSE endpoint contracts and credentials are not available in this prototype.

This distinction is intentional: the repository demonstrates the harmonization engine and integration contract without pretending that a live enterprise connector exists.

## Demo Example

Representative input:

```text
CS GATE VLV 50MM FLG CL150
```

The application normalizes the legacy description, extracts explicit technical attributes, performs deterministic catalog mapping, and displays the authoritative result. When AI Advisory is enabled, Local NLP and Gemini may provide additional candidates and reasons without changing that deterministic result.

Available demonstration states:

- `MATCHED`
- `UNCERTAIN`
- `NEW CANDIDATE`

## Repository Structure

```text
SIH-26099/
├── .github/workflows/ci.yml
├── app.py
├── README.md
├── data/
│   ├── demo/
│   └── evaluation/
├── docs/
│   ├── benchmark_protocol.md
│   └── dataset_design.md
├── scripts/
│   └── benchmark.py
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
    ├── test_benchmark.py
    └── ...
```

## Run Locally

Create/use a Python virtual environment and install the project's required dependencies.

Run deterministic verification:

```powershell
python -m unittest discover -s tests -v
python scripts/benchmark.py
```

Launch the dashboard:

```powershell
python -m streamlit run app.py
```

## Streamlit Community Cloud

Configure the Gemini credential in the deployment's Secrets configuration:

```toml
GEMINI_API_KEY = "your-key"
```

Never commit this value to GitHub. The laptop's `.venv` is not part of the repository.

## Safety and Decision Boundary

The project intentionally separates deterministic engineering logic from AI assistance:

- deterministic normalization and attribute extraction are reproducible;
- deterministic record linkage and catalog mapping are authoritative;
- AI candidates must refer to canonical IDs already present in the supplied catalog;
- invalid, malformed, duplicate, or unknown AI candidates are rejected;
- missing or conflicting technical information is handled conservatively;
- AI failures do not break deterministic processing;
- credentials belong in environment/deployment secrets, never in the repository.

## Project Principles

1. Deterministic technical matching comes first.
2. AI is optional and advisory.
3. AI cannot override deterministic decisions.
4. Candidate IDs are validated against the supplied catalog.
5. Missing or conflicting technical information is handled conservatively.
6. Core harmonization remains usable without external AI services.
7. Benchmarks are reproducible and their evidence boundary is explicit.
8. Production CPSE integration is not claimed without an approved interface contract.
9. Human review is available for unresolved evidence.
10. Credentials never belong in source code or repository configuration.
