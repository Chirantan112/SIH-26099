# Technical References

These references document the core engineering approaches used by CATALYST. They are included to distinguish established technical foundations from the project's own implementation and synthetic evaluation evidence.

## Semantic Retrieval / Embeddings

- [Sentence Transformers — Semantic Search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html) — background on embedding-based semantic retrieval and candidate search.

## LLM Advisory Interpretation

- [Google AI for Developers — Gemini API Models](https://ai.google.dev/gemini-api/docs/models) — official model documentation for the Gemini API.
- [Google AI for Developers — Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash) — official documentation for the stable `gemini-2.5-flash` model used by the advisory component.

## Project-Specific Evidence

- [SIH-26099 repository](https://github.com/Chirantan112/SIH-26099) — implementation, tests, synthetic evaluation data, benchmark protocol, and dashboard.
- [`docs/dataset_design.md`](dataset_design.md) — synthetic CPSE-style dataset design and evidence boundary.
- [`docs/benchmark_protocol.md`](benchmark_protocol.md) — reproducible benchmark methodology and limitations.

> The external references document general technical foundations. They do not constitute evidence that CATALYST has been validated on official CPSE production data. The repository's dataset and benchmark are explicitly synthetic.
