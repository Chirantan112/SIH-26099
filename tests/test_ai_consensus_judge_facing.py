from dataclasses import dataclass
import unittest

from src.ai_retrieval import AdapterStatus, CandidateSuggestion
from src.attribute_extraction import extract_attributes
from src.catalog_mapping import CatalogRecord
from src.hybrid_pipeline import run_hybrid_pipeline


CATALOG = (
    CatalogRecord("VAL-001", extract_attributes("CS GATE VLV 50MM FLG CL150").attributes),
    CatalogRecord("VAL-002", extract_attributes("CS GATE VLV 50MM FLG CL300").attributes),
)


@dataclass(frozen=True)
class RetrievalStub:
    suggestions: tuple[CandidateSuggestion, ...] = ()
    available: bool = True

    def status(self) -> AdapterStatus:
        return AdapterStatus("local_nlp", self.available, "stub")

    def retrieve(self, *_args):
        return self.suggestions


@dataclass(frozen=True)
class GeminiStub:
    suggestions: tuple[CandidateSuggestion, ...] = ()
    available: bool = True

    def status(self) -> AdapterStatus:
        return AdapterStatus("llm", self.available, "stub")

    def interpret(self, *_args):
        return self.suggestions


class AIConsensusJudgeFacingTests(unittest.TestCase):
    def test_both_sources_agree_only_after_independent_technical_validation(self):
        local = CandidateSuggestion("VAL-001", 0.91, "local_embedding", "semantic evidence")
        gemini = CandidateSuggestion("VAL-001", 0.88, "gemini", "LLM evidence")
        result = run_hybrid_pipeline(
            "CS GATE VLV 50MM FLG CL150",
            CATALOG,
            retrieval_adapter=RetrievalStub((local,)),
            llm_adapter=GeminiStub((gemini,)),
        )
        self.assertEqual(result.ai_consensus.conclusion, "MATCHED")
        self.assertEqual(result.ai_consensus.canonical_material_id, "VAL-001")
        self.assertIn("independent", result.ai_consensus.reason)
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-001")

    def test_ai_candidate_with_technical_conflict_cannot_create_match(self):
        local = CandidateSuggestion("VAL-001", 1.0, "local_embedding", "claims match")
        gemini = CandidateSuggestion("VAL-001", 1.0, "gemini", "claims match")
        result = run_hybrid_pipeline(
            "CS GATE VLV 50MM FLG CL300",
            CATALOG,
            retrieval_adapter=RetrievalStub((local,)),
            llm_adapter=GeminiStub((gemini,)),
        )
        self.assertEqual(result.ai_consensus.conclusion, "UNCERTAIN")
        self.assertIsNone(result.ai_consensus.canonical_material_id)
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-002")

    def test_single_gemini_candidate_is_truthful_uncertain_not_fake_consensus(self):
        gemini = CandidateSuggestion("VAL-001", 1.0, "gemini", "candidate only")
        result = run_hybrid_pipeline(
            "CS GATE VLV 50MM FLG CL150",
            CATALOG,
            llm_adapter=GeminiStub((gemini,)),
        )
        self.assertEqual(result.ai_consensus.conclusion, "UNCERTAIN")
        self.assertEqual(result.ai_consensus.canonical_material_id, "VAL-001")
        self.assertIn("single advisory source", result.ai_consensus.reason)

    def test_different_ai_candidates_are_uncertain(self):
        local = CandidateSuggestion("VAL-001", 0.9, "local_embedding", "local")
        gemini = CandidateSuggestion("VAL-002", 0.9, "gemini", "gemini")
        result = run_hybrid_pipeline(
            "CS GATE VLV 50MM FLG CL150",
            CATALOG,
            retrieval_adapter=RetrievalStub((local,)),
            llm_adapter=GeminiStub((gemini,)),
        )
        self.assertEqual(result.ai_consensus.conclusion, "UNCERTAIN")
        self.assertIsNone(result.ai_consensus.canonical_material_id)

    def test_empty_ai_evidence_never_becomes_new_candidate(self):
        result = run_hybrid_pipeline(
            "CS GATE VLV 50MM FLG CL150",
            CATALOG,
            retrieval_adapter=RetrievalStub(()),
            llm_adapter=GeminiStub(()),
        )
        self.assertEqual(result.ai_consensus.conclusion, "UNCERTAIN")
        self.assertIsNone(result.ai_consensus.canonical_material_id)
        self.assertIn("Empty evidence", result.ai_consensus.reason)

    def test_both_ai_services_unavailable_is_explicit(self):
        result = run_hybrid_pipeline(
            "CS GATE VLV 50MM FLG CL150",
            CATALOG,
            retrieval_adapter=RetrievalStub((), available=False),
            llm_adapter=GeminiStub((), available=False),
        )
        self.assertEqual(result.ai_consensus.conclusion, "UNAVAILABLE")
        self.assertIsNone(result.ai_consensus.canonical_material_id)


if __name__ == "__main__":
    unittest.main()
