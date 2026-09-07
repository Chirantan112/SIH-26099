"""Focused diagnostics for the optional Gemini advisory adapter."""

from __future__ import annotations

import os
from pathlib import Path
import unittest

from src.ai_retrieval import CandidateSuggestion
from src.attribute_extraction import extract_attributes
from src.catalog_mapping import CatalogRecord
from src.gemini_llm import GeminiLLMAdapter


CATALOG = (
    CatalogRecord("VAL-001", extract_attributes("CS GATE VLV 50MM FLG CL150").attributes),
    CatalogRecord("VAL-002", extract_attributes("CS GATE VLV 50MM FLG CL300").attributes),
)


class FakeResponse:
    def __init__(self, output_text: str, status: str = "completed"):
        self.output_text = output_text
        self.status = status


class _FakeInteractions:
    def __init__(self, response: FakeResponse | None = None, error: Exception | None = None):
        self.response = response
        self.error = error

    def create(self, **_kwargs):
        if self.error is not None:
            raise self.error
        return self.response


class GeminiLiveDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.previous_key = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = "placeholder"

    def tearDown(self):
        if self.previous_key is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = self.previous_key

    def _adapter(self, payload: str | None, error: Exception | None = None, status: str = "completed"):
        response = FakeResponse(payload, status=status) if payload is not None else None
        interactions = _FakeInteractions(response, error)
        adapter = GeminiLLMAdapter(client_factory=lambda _key: type("FakeClient", (), {"interactions": interactions})())
        return adapter

    def test_api_failure_is_not_reported_as_zero_candidates(self):
        adapter = self._adapter(None, error=RuntimeError("request failed"))
        self.assertEqual(adapter.interpret("desc", "desc", object(), CATALOG), ())
        self.assertFalse(adapter.status().available)
        self.assertIn("Gemini interpretation failed", adapter.status().detail)
        self.assertNotIn("zero candidates", adapter.status().detail.lower())

    def test_invalid_score_unknown_id_duplicate_and_malformed_candidates_are_diagnosed(self):
        payload = '{"candidates":[{"canonical_material_id":"NOT-IN-CATALOG","compatibility_score":0.9,"reason":"bad"},{"canonical_material_id":"VAL-001","compatibility_score":"bad","reason":"bad"},{"canonical_material_id":"VAL-001","compatibility_score":0.8,"reason":"good"},{"canonical_material_id":"VAL-001","compatibility_score":0.7,"reason":"duplicate"}]}'
        adapter = self._adapter(payload)
        result = adapter.interpret("desc", "desc", object(), CATALOG)
        self.assertEqual([item.canonical_material_id for item in result], ["VAL-001"])
        self.assertEqual(result[0].score, 0.8)

    def test_missing_score_is_rejected_without_positional_fallback(self):
        adapter = self._adapter('{"candidates":[{"canonical_material_id":"VAL-001","reason":"no score"}]}')
        self.assertEqual(adapter.interpret("desc", "desc", object(), CATALOG), ())

    def test_real_compatibility_score_is_preserved_exactly(self):
        adapter = self._adapter('{"candidates":[{"canonical_material_id":"VAL-001","compatibility_score":0.93,"reason":"exact"}]}')
        result = adapter.interpret("desc", "desc", object(), CATALOG)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].score, 0.93)
        self.assertIsInstance(result[0], CandidateSuggestion)

    def test_streamlit_uses_explicit_advisory_score_labels(self):
        app_source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
        self.assertIn('"Cosine Similarity"', app_source)
        self.assertIn('"Gemini Compatibility · Advisory"', app_source)
        self.assertNotIn('"Advisory Rank"', app_source)

    def test_successful_empty_response_is_distinguishable_from_failure(self):
        adapter = self._adapter('{"candidates":[]}')
        self.assertEqual(adapter.interpret("desc", "desc", object(), CATALOG), ())
        self.assertTrue(adapter.status().available)


if __name__ == "__main__":
    unittest.main()
