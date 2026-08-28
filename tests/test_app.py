"""Focused tests for the judge-facing Streamlit presentation layer."""

from __future__ import annotations

import unittest
from decimal import Decimal

from app import advisory_rows, analyze_material, attribute_rows, candidate_rows
from src.ai_retrieval import AdapterStatus, CandidateSuggestion
from src.attribute_extraction import MaterialAttributes, extract_attributes
from src.catalog_mapping import CatalogRecord
from src.demo_pipeline import CandidateEvidence


CATALOG = (
    CatalogRecord("VAL-001", extract_attributes("CS GATE VLV 50MM FLG CL150").attributes),
    CatalogRecord("VAL-002", extract_attributes("CS GATE VLV 50MM FLG CL300").attributes),
)


class FakeAdapter:
    def __init__(self, component: str, suggestions=(), available: bool = True):
        self.component = component
        self.suggestions = tuple(suggestions)
        self.available = available
        self.calls = 0

    def status(self):
        return AdapterStatus(self.component, self.available, "fake adapter")

    def retrieve(self, *args):
        self.calls += 1
        return self.suggestions

    def interpret(self, *args):
        self.calls += 1
        return self.suggestions


class AppPresentationTests(unittest.TestCase):
    def test_attribute_rows_render_missing_values_and_decimals_cleanly(self):
        attributes = MaterialAttributes(category="Valve", material="carbon steel", size_mm=Decimal("50"), pressure_class=150)
        rows = attribute_rows(attributes)
        values = {row["Attribute"]: row["Extracted value"] for row in rows}
        self.assertEqual(values["Category"], "Valve")
        self.assertEqual(values["Size (mm)"], "50")
        self.assertEqual(values["Connection"], "Not provided")

    def test_candidate_rows_keep_only_ui_safe_evidence_fields(self):
        rows = candidate_rows((CandidateEvidence("VAL-001", "SAME", 1.0, "All required attributes match."),))
        self.assertEqual(rows, [{"Canonical material ID": "VAL-001", "Decision": "SAME", "Score": 1.0, "Explanation": "All required attributes match."}])

    def test_advisory_rows_are_ui_safe_and_use_explicit_score_label(self):
        rows = advisory_rows(
            (CandidateSuggestion("VAL-001", 1.0, "gemini", "Advisory only."),),
            "Gemini Compatibility Score",
        )
        self.assertEqual(rows, [{"Canonical material ID": "VAL-001", "Source": "gemini", "Gemini Compatibility Score": 1.0, "Reason": "Advisory only."}])
        self.assertNotIn("Advisory Rank", rows[0])

    def test_missing_attributes_render_cleanly(self):
        rows = attribute_rows(MaterialAttributes(category="Pipe", material="carbon steel"))
        values = {row["Attribute"]: row["Extracted value"] for row in rows}
        self.assertEqual(values["Outside diameter (mm)"], "Not provided")
        self.assertEqual(values["Thickness (mm)"], "Not provided")

    def test_decimal_attributes_render_without_python_repr(self):
        rows = attribute_rows(MaterialAttributes(size_mm=Decimal("50.00"), dimensions=(Decimal("10"), Decimal("20"))))
        values = {row["Attribute"]: row["Extracted value"] for row in rows}
        self.assertEqual(values["Size (mm)"], "50.00")
        self.assertEqual(values["Dimensions (mm)"], "10 × 20")

    def test_deterministic_decision_remains_authoritative_when_ai_succeeds(self):
        llm = FakeAdapter("llm", (CandidateSuggestion("VAL-002", 1.0, "gemini", "Advisory conflict."),))
        retrieval = FakeAdapter("local_nlp")
        result = analyze_material("CS GATE VLV 50MM FLG CL150", CATALOG, True, retrieval, llm)
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-001")
        self.assertEqual(result.mapping_result.decision, "MATCHED")
        self.assertEqual(result.ai_candidate_suggestions[0].canonical_material_id, "VAL-002")

    def test_ai_suggestions_are_displayed_separately_from_deterministic_evidence(self):
        llm = FakeAdapter("llm", (CandidateSuggestion("VAL-001", 1.0, "gemini", "Advisory."),))
        result = analyze_material("CS GATE VLV 50MM FLG CL150", CATALOG, True, FakeAdapter("local_nlp"), llm)
        rows = advisory_rows(result.ai_candidate_suggestions, "Gemini Compatibility Score")
        self.assertEqual(rows[0]["Source"], "gemini")
        self.assertEqual(rows[0]["Gemini Compatibility Score"], 1.0)
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-001")

    def test_ai_suggestions_cannot_replace_deterministic_canonical_id(self):
        llm = FakeAdapter("llm", (CandidateSuggestion("VAL-001", 1.0, "gemini", "Suggestion only."),))
        result = analyze_material("CS GATE VLV 50MM FLG CL300", CATALOG, True, FakeAdapter("local_nlp"), llm)
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-002")
        self.assertEqual(result.ai_candidate_suggestions[0].canonical_material_id, "VAL-001")

    def test_ai_unavailable_state_preserves_deterministic_result(self):
        llm = FakeAdapter("llm", available=False)
        retrieval = FakeAdapter("local_nlp", available=False)
        result = analyze_material("CS GATE VLV 50MM FLG CL150", CATALOG, True, retrieval, llm)
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-001")
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.ai_candidate_suggestions, ())

    def test_ai_advisory_disabled_does_not_call_adapters(self):
        retrieval = FakeAdapter("local_nlp")
        llm = FakeAdapter("llm")
        result = analyze_material("CS GATE VLV 50MM FLG CL150", CATALOG, False, retrieval, llm)
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-001")
        self.assertEqual(retrieval.calls, 0)
        self.assertEqual(llm.calls, 0)

    def test_ai_advisory_enabled_uses_existing_hybrid_pipeline(self):
        suggestion = CandidateSuggestion("VAL-001", 1.0, "gemini", "Matches valve specification.")
        retrieval = FakeAdapter("local_nlp")
        llm = FakeAdapter("llm", (suggestion,))
        result = analyze_material("CS GATE VLV 50MM FLG CL150", CATALOG, True, retrieval, llm)
        self.assertEqual(retrieval.calls, 1)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(result.ai_candidate_suggestions, (suggestion,))

    def test_suggestions_remain_bounded(self):
        suggestions = tuple(CandidateSuggestion(f"VAL-{i:03d}", 1.0 - i / 10, "gemini", "Advisory") for i in range(1, 8))
        catalog = tuple(CatalogRecord(f"VAL-{i:03d}", CATALOG[0].attributes) for i in range(1, 8))
        result = analyze_material("CS GATE VLV 50MM FLG CL150", catalog, True, FakeAdapter("local_nlp"), FakeAdapter("llm", suggestions))
        self.assertLessEqual(len(result.ai_candidate_suggestions), 5)


if __name__ == "__main__":
    unittest.main()
