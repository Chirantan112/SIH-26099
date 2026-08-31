# Operations Demo

The **Operations & Evidence** Streamlit page adds judge-facing operational workflows without changing the existing harmonization or LLM implementation.

## What it adds

- **Single Evidence**: shows authoritative deterministic mapping evidence and, when explicitly enabled, the existing Local NLP/Gemini advisory results.
- **Batch Processing**: processes pasted descriptions or a CSV through the deterministic pipeline only. It does not construct or call Gemini/Local NLP.
- **Review Queue**: surfaces `UNCERTAIN` and `NEW_CANDIDATE` batch results and records `APPROVE`, `REJECT`, or `REVIEW` actions in the existing session-friendly audit contract.
- **Audit Trail**: displays analysis/review events for the current session.
- **Benchmark**: runs the existing synthetic benchmark from the UI and clearly labels its evidence boundary.
- **Export**: downloads deterministic batch results as CSV.

## Demo sequence

1. Run the default valve example with AI advisory **off** to show the deterministic authority.
2. Enable AI advisory for a single record to demonstrate that AI evidence is separate and cannot override the final mapping.
3. Paste several matched, uncertain, and new-candidate descriptions into Batch Processing.
4. Process the batch and open the Review Queue.
5. Record a reviewer action and show the Audit Trail.
6. Run the synthetic benchmark and use the measured output in the SIH presentation.

## LLM safety boundary

The page intentionally keeps LLM use narrow:

- AI is opt-in for **single-record** evidence review.
- Batch processing never constructs or invokes Local NLP or Gemini.
- The existing `run_hybrid_pipeline()` remains responsible for AI orchestration.
- No prompt, model, credential, candidate-validation, or deterministic decision logic is changed by this feature.
- AI remains advisory; the deterministic mapping remains authoritative.
