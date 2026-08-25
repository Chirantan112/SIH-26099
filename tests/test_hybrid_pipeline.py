"""Tests for LEGO #9A's dependency-free, non-authoritative AI architecture."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import unittest

from src.ai_retrieval import AdapterStatus, CandidateSuggestion, UnavailableRetrievalAdapter
from src.attribute_extraction import MaterialAttributes, extract_attributes
from src.catalog_mapping import CatalogRecord, LegacyRecord, map_records
from src.hybrid_pipeline import MAX_AI_CANDIDATE_SUGGESTIONS, run_hybrid_pipeline
from src.llm_interpretation import UnavailableLLMAdapter


CATALOG = (
    CatalogRecord("VAL-001", extract_attributes("CS GATE VLV 50MM FLG CL150").attributes),
    CatalogRecord("VAL-002", extract_attributes("CS GATE VLV 50MM FLG CL300").attributes),
)


@dataclass(frozen=True)
class RetrievalStub:
    suggestions: tuple[object, ...] = ()
    available: bool = True

    def status(self) -> AdapterStatus:
        return AdapterStatus("local_nlp", self.available, "Retrieval stub status.")

    def retrieve(self, *_args: object) -> tuple[object, ...]:
        return self.suggestions


@dataclass(frozen=True)
class LLMStub:
    suggestions: tuple[object, ...] = ()
    available: bool = True

    def status(self) -> AdapterStatus:
        return AdapterStatus("llm", self.available, "LLM stub status.")

    def interpret(self, *_args: object) -> tuple[object, ...]:
        return self.suggestions


class FailingRetrievalAdapter:
    def status(self) -> AdapterStatus:
        return AdapterStatus("local_nlp", True, "Ready before simulated failure.")

    def retrieve(self, *_args: object) -> tuple[CandidateSuggestion, ...]:
        raise RuntimeError("simulated retrieval failure")


class HybridPipelineTests(unittest.TestCase):
    def test_both_ai_adapters_unavailable_uses_deterministic_fallback(self):
        result = run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150", CATALOG)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.mapping_result.decision, "MATCHED")
        self.assertEqual(result.ai_candidate_suggestions, ())
        self.assertEqual([status.available for status in result.ai_statuses], [False, False])

    def test_nlp_unavailable_llm_available_continues_safely(self):
        suggestion = CandidateSuggestion("VAL-001", 0.8, "llm", "Advisory suggestion")
        result = run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150", CATALOG, llm_adapter=LLMStub((suggestion,)))
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.ai_candidate_suggestions, (suggestion,))
        self.assertFalse(result.ai_statuses[0].available)
        self.assertTrue(result.ai_statuses[1].available)

    def test_llm_unavailable_nlp_available_continues_safely(self):
        suggestion = CandidateSuggestion("VAL-001", 0.8, "local_nlp", "Advisory suggestion")
        result = run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150", CATALOG, retrieval_adapter=RetrievalStub((suggestion,)))
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.ai_candidate_suggestions, (suggestion,))
        self.assertTrue(result.ai_statuses[0].available)
        self.assertFalse(result.ai_statuses[1].available)

    def test_both_adapters_succeed_without_authoritative_decision(self):
        nlp = CandidateSuggestion("VAL-001", 0.8, "local_nlp", "Semantic similarity")
        llm = CandidateSuggestion("VAL-002", 0.7, "llm", "Interpretation hint")
        result = run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150", CATALOG, retrieval_adapter=RetrievalStub((nlp,)), llm_adapter=LLMStub((llm,)))
        self.assertFalse(result.fallback_used)
        self.assertEqual(result.ai_candidate_suggestions, (nlp, llm))
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-001")
        self.assertIn("advisory only", result.explanation)

    def test_ai_adapter_exception_is_exposed_and_non_fatal(self):
        result = run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150", CATALOG, retrieval_adapter=FailingRetrievalAdapter())
        self.assertEqual(result.mapping_result.decision, "MATCHED")
        self.assertFalse(result.ai_statuses[0].available)
        self.assertIn("simulated retrieval failure", result.ai_statuses[0].detail)

    def test_malformed_ai_candidates_are_ignored(self):
        malformed = (
            CandidateSuggestion("NOT-IN-CATALOG", 0.9, "local_nlp", "Unknown ID"),
            CandidateSuggestion("VAL-001", float("nan"), "local_nlp", "Invalid score"),
            CandidateSuggestion("VAL-001", 0.9, "", "Missing source"),
            "not a suggestion",
        )
        result = run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150", CATALOG, retrieval_adapter=RetrievalStub(malformed))
        self.assertEqual(result.ai_candidate_suggestions, ())
        self.assertEqual(result.mapping_result.decision, "MATCHED")

    def test_ai_suggestion_cannot_override_technical_conflict(self):
        suggestion = CandidateSuggestion("VAL-001", 1.0, "local_nlp", "Claims a match")
        result = run_hybrid_pipeline("CS GATE VLV 50MM FLG CL300", CATALOG, retrieval_adapter=RetrievalStub((suggestion,)))
        self.assertEqual(result.mapping_result.decision, "MATCHED")
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-002")
        self.assertEqual(result.ai_candidate_suggestions, (suggestion,))

    def test_missing_technical_information_remains_uncertain(self):
        suggestion = CandidateSuggestion("VAL-001", 1.0, "llm", "Suggests omitted connection")
        result = run_hybrid_pipeline("GATE VLV CS 50MM CL150", CATALOG, llm_adapter=LLMStub((suggestion,)))
        self.assertEqual(result.mapping_result.decision, "UNCERTAIN")
        self.assertIsNone(result.mapping_result.canonical_material_id)

    def test_deterministic_fallback_matches_lego_five_exactly(self):
        raw_description = "CS GATE VLV 50MM FLG CL150"
        expected = map_records((LegacyRecord("CHECK", raw_description),), CATALOG)[0]
        actual = run_hybrid_pipeline(raw_description, CATALOG, "CHECK").mapping_result
        self.assertEqual(actual, expected)

    def test_repeated_unavailable_calls_are_deterministic(self):
        first = run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150", CATALOG)
        second = run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150", CATALOG)
        self.assertEqual(first, second)

    def test_availability_status_and_bounded_suggestions_are_exposed(self):
        suggestions = tuple(CandidateSuggestion("VAL-001", float(index), "local_nlp", "Hint") for index in range(10))
        result = run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150", CATALOG, retrieval_adapter=RetrievalStub(suggestions))
        self.assertEqual(result.ai_statuses[0].component, "local_nlp")
        self.assertLessEqual(len(result.ai_candidate_suggestions), MAX_AI_CANDIDATE_SUGGESTIONS)

    def test_unavailable_adapters_require_no_external_model_or_api(self):
        retrieval = UnavailableRetrievalAdapter()
        llm = UnavailableLLMAdapter()
        self.assertEqual(retrieval.retrieve(None, "", MaterialAttributes(), CATALOG), ())
        self.assertEqual(llm.interpret(None, "", MaterialAttributes(), CATALOG), ())


if __name__ == "__main__":
    unittest.main()
