"""Focused regression tests for single-advisor AI consensus behavior."""

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


def evidence(source, candidate, compatible, conflicts=()):
    return CandidateSuggestion(
        candidate,
        0.9,
        source,
        "test evidence",
        ("category", "valve_type", "material"),
        tuple(conflicts),
        (),
        compatible,
    )


class SingleAdvisorConsensusTests(unittest.TestCase):
    def test_both_unavailable_is_unavailable(self):
        result = run_hybrid_pipeline(
            "CS GATE VLV 50MM FLG CL150",
            CATALOG,
            retrieval_adapter=RetrievalStub(available=False),
            llm_adapter=GeminiStub(available=False),
        )
        self.assertEqual(result.ai_consensus.conclusion, "UNAVAILABLE")
        self.assertIsNone(result.ai_consensus.canonical_material_id)
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-001")

    def test_only_local_compatible_is_matched(self):
        local = evidence("local_embedding", "VAL-001", True)
        result = run_hybrid_pipeline(
            "CS GATE VLV 50MM FLG CL150",
            CATALOG,
            retrieval_adapter=RetrievalStub((local,)),
            llm_adapter=GeminiStub(available=False),
        )
        self.assertEqual(result.ai_consensus.conclusion, "MATCHED")
        self.assertEqual(result.ai_consensus.canonical_material_id, "VAL-001")
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-001")

    def test_only_local_explicit_conflict_is_new_candidate(self):
        local = evidence("local_embedding", "VAL-001", False, conflicts=("pressure_class",))
        result = run_hybrid_pipeline(
            "CS GATE VLV 50MM FLG CL600",
            CATALOG,
            retrieval_adapter=RetrievalStub((local,)),
            llm_adapter=GeminiStub(available=False),
        )
        self.assertEqual(result.ai_consensus.conclusion, "NEW_CANDIDATE")
        self.assertIsNone(result.ai_consensus.canonical_material_id)

    def test_only_gemini_compatible_is_matched(self):
        gemini = evidence("gemini", "VAL-001", True)
        result = run_hybrid_pipeline(
            "CS GATE VLV 50MM FLG CL150",
            CATALOG,
            retrieval_adapter=RetrievalStub(available=False),
            llm_adapter=GeminiStub((gemini,)),
        )
        self.assertEqual(result.ai_consensus.conclusion, "MATCHED")
        self.assertEqual(result.ai_consensus.canonical_material_id, "VAL-001")
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-001")

    def test_only_gemini_explicit_conflict_is_new_candidate(self):
        gemini = evidence("gemini", "VAL-001", False, conflicts=("pressure_class",))
        result = run_hybrid_pipeline(
            "CS GATE VLV 50MM FLG CL600",
            CATALOG,
            retrieval_adapter=RetrievalStub(available=False),
            llm_adapter=GeminiStub((gemini,)),
        )
        self.assertEqual(result.ai_consensus.conclusion, "NEW_CANDIDATE")
        self.assertIsNone(result.ai_consensus.canonical_material_id)

    def test_single_advisor_incomplete_evidence_is_uncertain(self):
        local = evidence("local_embedding", "VAL-001", None)
        result = run_hybrid_pipeline(
            "CS GATE VLV 50MM FLG",
            CATALOG,
            retrieval_adapter=RetrievalStub((local,)),
            llm_adapter=GeminiStub(available=False),
        )
        self.assertEqual(result.ai_consensus.conclusion, "UNCERTAIN")
        self.assertIsNone(result.ai_consensus.canonical_material_id)


if __name__ == "__main__":
    unittest.main()
