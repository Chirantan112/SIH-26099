"""Tests for advisory AI technical consensus and failure isolation."""

from __future__ import annotations

import os
import unittest

from src.ai_retrieval import AdapterStatus, CandidateSuggestion
from src.attribute_extraction import MaterialAttributes, extract_attributes
from src.catalog_mapping import CatalogRecord
from src.gemini_llm import GeminiLLMAdapter
from src.hybrid_pipeline import run_hybrid_pipeline
from src.local_embedding_retrieval import LocalEmbeddingRetrievalAdapter

CATALOG = (
    CatalogRecord("VAL-001", extract_attributes("CS GATE VLV 50MM FLG CL150").attributes),
    CatalogRecord("VAL-002", extract_attributes("CS GATE VLV 50MM FLG CL300").attributes),
)

class RetrievalStub:
    def __init__(self, suggestions=(), available=True): self.suggestions, self.available = tuple(suggestions), available
    def status(self): return AdapterStatus("local_nlp", self.available, "test local NLP")
    def retrieve(self, *_args): return self.suggestions

class GeminiStub:
    def __init__(self, suggestions=(), available=True): self.suggestions, self.available = tuple(suggestions), available
    def status(self): return AdapterStatus("llm", self.available, "test Gemini")
    def interpret(self, *_args): return self.suggestions

def evidence(source, candidate, compatible, matching=(), conflicts=(), missing=()):
    return CandidateSuggestion(candidate, 0.9, source, "test evidence", tuple(matching), tuple(conflicts), tuple(missing), compatible)

class AIConsensusTests(unittest.TestCase):
    def test_equivalent_candidate_is_matched_when_both_ai_components_agree(self):
        fields=("category","valve_type","material","size_mm","pressure_class","connection")
        local=evidence("local_embedding","VAL-001",True,fields); gemini=evidence("gemini","VAL-001",True,fields)
        result=run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150",CATALOG,retrieval_adapter=RetrievalStub((local,)),llm_adapter=GeminiStub((gemini,)))
        self.assertEqual(result.ai_consensus.conclusion,"MATCHED"); self.assertEqual(result.ai_consensus.canonical_material_id,"VAL-001"); self.assertEqual(result.mapping_result.canonical_material_id,"VAL-001")
    def test_ai_disagreement_becomes_uncertain(self):
        local=evidence("local_embedding","VAL-001",True); gemini=evidence("gemini","VAL-002",True)
        result=run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150",CATALOG,retrieval_adapter=RetrievalStub((local,)),llm_adapter=GeminiStub((gemini,)))
        self.assertEqual(result.ai_consensus.conclusion,"UNCERTAIN"); self.assertIsNone(result.ai_consensus.canonical_material_id); self.assertEqual(result.mapping_result.canonical_material_id,"VAL-001")
    def test_missing_critical_attribute_is_unresolved_not_new_candidate(self):
        local=evidence("local_embedding","VAL-001",None,matching=("category","valve_type","material","size_mm","connection"),missing=("pressure_class",)); gemini=evidence("gemini","VAL-001",None,matching=("category","valve_type","material","size_mm","connection"),missing=("pressure_class",))
        result=run_hybrid_pipeline("CS GATE VLV 50MM FLG",CATALOG,retrieval_adapter=RetrievalStub((local,)),llm_adapter=GeminiStub((gemini,)))
        self.assertEqual(result.ai_consensus.conclusion,"UNCERTAIN"); self.assertNotEqual(result.ai_consensus.conclusion,"NEW_CANDIDATE"); self.assertEqual(result.mapping_result.decision,"UNCERTAIN")
    def test_explicit_critical_conflict_from_all_supplied_candidates_is_new_candidate(self):
        local_one=evidence("local_embedding","VAL-001",False,conflicts=("pressure_class",)); local_two=evidence("local_embedding","VAL-002",False,conflicts=("pressure_class",)); gemini_one=evidence("gemini","VAL-001",False,conflicts=("pressure_class",)); gemini_two=evidence("gemini","VAL-002",False,conflicts=("pressure_class",))
        result=run_hybrid_pipeline("CS GATE VLV 50MM FLG CL600",CATALOG,retrieval_adapter=RetrievalStub((local_one,local_two)),llm_adapter=GeminiStub((gemini_one,gemini_two)))
        self.assertEqual(result.ai_consensus.conclusion,"NEW_CANDIDATE")
    def test_no_candidates_with_available_ai_is_uncertain_when_no_negative_evidence_exists(self):
        result=run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150",CATALOG,retrieval_adapter=RetrievalStub(()),llm_adapter=GeminiStub(())); self.assertEqual(result.ai_consensus.conclusion,"UNCERTAIN"); self.assertEqual(result.mapping_result.canonical_material_id,"VAL-001")
    def test_one_ai_unavailable_and_other_incomplete_is_uncertain(self):
        local=evidence("local_embedding","VAL-001",None,missing=("pressure_class",)); result=run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150",CATALOG,retrieval_adapter=RetrievalStub((local,)),llm_adapter=GeminiStub(available=False)); self.assertEqual(result.ai_consensus.conclusion,"UNCERTAIN"); self.assertEqual(result.mapping_result.canonical_material_id,"VAL-001")
    def test_one_ai_unavailable_and_other_compatible_is_matched(self):
        local=evidence("local_embedding","VAL-001",True); result=run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150",CATALOG,retrieval_adapter=RetrievalStub((local,)),llm_adapter=GeminiStub(available=False)); self.assertEqual(result.ai_consensus.conclusion,"MATCHED"); self.assertEqual(result.ai_consensus.canonical_material_id,"VAL-001"); self.assertEqual(result.mapping_result.canonical_material_id,"VAL-001")
    def test_both_ai_components_unavailable_produce_unavailable_consensus(self):
        result=run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150",CATALOG,retrieval_adapter=RetrievalStub(available=False),llm_adapter=GeminiStub(available=False)); self.assertEqual(result.ai_consensus.conclusion,"UNAVAILABLE"); self.assertEqual(result.mapping_result.canonical_material_id,"VAL-001")
    def test_unknown_gemini_id_is_rejected(self):
        gemini=evidence("gemini","NOT-IN-CATALOG",True); result=run_hybrid_pipeline("CS GATE VLV 50MM FLG CL150",CATALOG,retrieval_adapter=RetrievalStub(available=False),llm_adapter=GeminiStub((gemini,))); self.assertEqual(result.ai_candidate_suggestions,()); self.assertEqual(result.ai_consensus.conclusion,"UNCERTAIN"); self.assertEqual(result.mapping_result.canonical_material_id,"VAL-001")
    def test_malformed_gemini_response_fails_safely(self):
        class Response: status="completed"; output_text="not-json"
        class FakeInteractions:
            def create(self, **_kwargs): return Response()
        class FakeClient: interactions=FakeInteractions()
        previous=os.environ.get("GEMINI_API_KEY"); os.environ["GEMINI_API_KEY"]="test-key"
        try:
            adapter=GeminiLLMAdapter(client_factory=lambda _key: FakeClient()); suggestions=adapter.interpret(None,"valve gate carbon steel",MaterialAttributes(category="Valve"),CATALOG); self.assertEqual(suggestions,()); self.assertFalse(adapter.status().available)
        finally:
            if previous is None: os.environ.pop("GEMINI_API_KEY",None)
            else: os.environ["GEMINI_API_KEY"]=previous
    def test_local_technical_evidence_exact_agreement_is_true(self):
        left=extract_attributes("CS GATE VLV 50MM FLG CL150").attributes; right=extract_attributes("CS GATE VLV 50MM FLG CL150").attributes; matching,conflicts,missing,compatible=LocalEmbeddingRetrievalAdapter._technical_evidence(left,right); self.assertTrue(compatible); self.assertIn("pressure_class",matching); self.assertEqual(conflicts,()); self.assertEqual(missing,())
    def test_local_technical_evidence_critical_conflict_is_false(self):
        left=extract_attributes("CS GATE VLV 50MM FLG CL150").attributes; right=extract_attributes("CS GATE VLV 50MM FLG CL300").attributes; _matching,conflicts,_missing,compatible=LocalEmbeddingRetrievalAdapter._technical_evidence(left,right); self.assertFalse(compatible); self.assertIn("pressure_class",conflicts)
    def test_local_technical_evidence_missing_critical_attribute_is_none(self):
        left=extract_attributes("CS GATE VLV 50MM FLG").attributes; right=extract_attributes("CS GATE VLV 50MM FLG CL150").attributes; _matching,_conflicts,missing,compatible=LocalEmbeddingRetrievalAdapter._technical_evidence(left,right); self.assertIsNone(compatible); self.assertIn("pressure_class",missing)
    def test_local_technical_evidence_different_category_is_false(self):
        left=MaterialAttributes(category="Valve"); right=MaterialAttributes(category="Pipe"); _matching,conflicts,_missing,compatible=LocalEmbeddingRetrievalAdapter._technical_evidence(left,right); self.assertFalse(compatible); self.assertIn("category",conflicts)
    def test_local_technical_evidence_without_usable_category_is_none(self):
        left=MaterialAttributes(); right=MaterialAttributes(); _matching,conflicts,missing,compatible=LocalEmbeddingRetrievalAdapter._technical_evidence(left,right); self.assertIsNone(compatible); self.assertEqual(conflicts,()); self.assertEqual(missing,())
    def test_realistic_dataset_description_is_matched(self):
        result=run_hybrid_pipeline("50 mm Carbon Steel gate valve, class 150 FLANGED",CATALOG,retrieval_adapter=RetrievalStub(()),llm_adapter=GeminiStub(())); self.assertEqual(result.mapping_result.decision,"MATCHED"); self.assertEqual(result.mapping_result.canonical_material_id,"VAL-001")
    def test_realistic_partial_description_is_uncertain(self):
        result=run_hybrid_pipeline("CS GATE VALVE 50 MM FLG",CATALOG,retrieval_adapter=RetrievalStub(()),llm_adapter=GeminiStub(())); self.assertEqual(result.mapping_result.decision,"UNCERTAIN")
    def test_realistic_unsupported_specification_is_new_candidate_deterministically(self):
        result=run_hybrid_pipeline("CS GATE VALVE 50 MM 600# FLG",CATALOG,retrieval_adapter=RetrievalStub(()),llm_adapter=GeminiStub(())); self.assertEqual(result.mapping_result.decision,"NEW_CANDIDATE")

if __name__ == "__main__": unittest.main()
