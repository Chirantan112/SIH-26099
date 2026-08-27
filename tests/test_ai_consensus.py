"""Tests for advisory AI technical consensus and failure isolation."""

from __future__ import annotations

import unittest

from src.ai_retrieval import AdapterStatus, CandidateSuggestion
from src.attribute_extraction import extract_attributes
from src.catalog_mapping import CatalogRecord
from src.hybrid_pipeline import run_hybrid_pipeline


CATALOG = (
    CatalogRecord("VAL-001", extract_attributes("CS GATE VLV 50MM FLG CL150").attributes),
    CatalogRecord("VAL-002", extract_attributes("CS GATE VLV 50MM FLG CL300").attributes),
)


class RetrievalStub:
    def __init__(self, suggestions=(), available=True):
        self.suggestions = tuple(suggestions)
        self.available = available

    def status(self):
        return AdapterStatus("local_nlp", self.available, "test local NLP")

    def retrieve(self, *_args):
        return self.suggestions


class GeminiStub:
    def __init__(self, suggestions=(), available=True):
        self.suggestions = tuple(suggestions)
        self.available = available

    def status(self):
        return AdapterStatus("llm", self.available, "test Gemini")

    def interpret(self, *_args):
        return self.suggestions


def evidence(source, candidate, compatible, matching=(), conflicts=(), missing=()):
    return CandidateSuggestion(
        candidate,
        0.9,
        source,
        "test evidence",
        tuple(matching),
        tuple(conflicts),
        tuple(missing),
        compatible,
    )


class AIConsensusTests(unittest.TestCase):
    def test_equivalent_candidate_is_matched_when_both_ai_components_agree(self):
        local = evidence("local_embedding", "VAL-001", True, ("category", "valve_type", "material", "size_mm", "pressure_class", "connection"))
        gemini = evidence("gemini", "VAL-001", True, ("category", "valve_type", "material", "size_mm", "pressure_class", "connection"))
        result = run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150", CATALOG, retrieval_adapter=RetrievalStub((local,)), llm_adapter=GeminiStub((gemini,)))
        self.assertEqual(result.ai_consensus.conclusion, "MATCHED")
        self.assertEqual(result.ai_consensus.canonical_material_id, "VAL-001")
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-001")

    def test_critical_conflict_is_not_treated_as_match(self):
        local = evidence("local_embedding", "VAL-002", False, ("category", "valve_type", "material", "size_mm", "connection"), ("pressure_class",))
        gemini = evidence("gemini", "VAL-002", False, ("category", "valve_type", "material", "size_mm", "connection"), ("pressure_class",))
        result = run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150", CATALOG, retrieval_adapter=RetrievalStub((local,)), llm_adapter=GeminiStub((gemini,)))
        self.assertIn(result.ai_consensus.conclusion, {"UNCERTAIN", "NEW_CANDIDATE"})
        self.assertNotEqual(result.ai_consensus.conclusion, "MATCHED")
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-001")

    def test_no_technically_compatible_candidate_is_new_candidate(self):
        local = evidence("local_embedding", "VAL-001", False, conflicts=("pressure_class",))
        gemini = evidence("gemini", "VAL-001", False, conflicts=("pressure_class",))
        result = run_hybrid_pipeline("CS GATE VLV 50MM FLG CL300", CATALOG, retrieval_adapter=RetrievalStub((local,)), llm_adapter=GeminiStub((gemini,)))
        self.assertEqual(result.ai_consensus.conclusion, "NEW_CANDIDATE")
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-002")

    def test_local_nlp_unavailable_does_not_break_deterministic_pipeline(self):
        gemini = evidence("gemini", "VAL-001", True, ("category", "valve_type", "material", "size_mm", "pressure_class", "connection"))
        result = run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150", CATALOG, retrieval_adapter=RetrievalStub(available=False), llm_adapter=GeminiStub((gemini,)))
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-001")
        self.assertEqual(result.ai_consensus.conclusion, "MATCHED")

    def test_gemini_unavailable_does_not_break_deterministic_pipeline(self):
        local = evidence("local_embedding", "VAL-001", True, ("category", "valve_type", "material", "size_mm", "pressure_class", "connection"))
        result = run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150", CATALOG, retrieval_adapter=RetrievalStub((local,)), llm_adapter=GeminiStub(available=False))
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-001")
        self.assertEqual(result.ai_consensus.conclusion, "MATCHED")

    def test_ai_disagreement_becomes_uncertain(self):
        local = evidence("local_embedding", "VAL-001", True)
        gemini = evidence("gemini", "VAL-002", True)
        result = run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150", CATALOG, retrieval_adapter=RetrievalStub((local,)), llm_adapter=GeminiStub((gemini,)))
        self.assertEqual(result.ai_consensus.conclusion, "UNCERTAIN")
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-001")

    def test_unknown_gemini_id_is_rejected(self):
        gemini = evidence("gemini", "NOT-IN-CATALOG", True)
        result = run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150", CATALOG, retrieval_adapter=RetrievalStub(available=False), llm_adapter=GeminiStub((gemini,)))
        self.assertEqual(result.ai_candidate_suggestions, ())
        self.assertEqual(result.ai_consensus.conclusion, "NEW_CANDIDATE")
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-001")

    def test_both_ai_components_unavailable_produce_unavailable_consensus(self):
        result = run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150", CATALOG, retrieval_adapter=RetrievalStub(available=False), llm_adapter=GeminiStub(available=False))
        self.assertEqual(result.ai_consensus.conclusion, "UNAVAILABLE")
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-001")


if __name__ == "__main__":
    unittest.main()
