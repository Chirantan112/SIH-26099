"""Dependency-free tests for the optional LEGO #10 Gemini adapter."""

from __future__ import annotations

import inspect
import os
import unittest
from unittest.mock import patch

from src.ai_retrieval import AdapterStatus, CandidateSuggestion
from src.attribute_extraction import extract_attributes
from src.catalog_mapping import CatalogRecord, LegacyRecord, map_records
from src.gemini_llm import DEFAULT_GEMINI_MODEL, GeminiLLMAdapter
from src.hybrid_pipeline import run_hybrid_pipeline

CATALOG = (
    CatalogRecord("VAL-001", extract_attributes("CS GATE VLV 50MM FLG CL150").attributes),
    CatalogRecord("VAL-002", extract_attributes("CS GATE VLV 50MM FLG CL300").attributes),
)

class FakeResponse:
    def __init__(self, output_text: str, status: str = "completed"):
        self.output_text = output_text
        self.status = status

class FakeInteractions:
    def __init__(self, response=None, error=None): self.response, self.error, self.calls = response, error, []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None: raise self.error
        return self.response

class FakeClient:
    def __init__(self, interactions): self.interactions = interactions

class GeminiLLMAdapterTests(unittest.TestCase):
    def setUp(self):
        self.previous_key = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = "placeholder"
    def tearDown(self):
        if self.previous_key is None: os.environ.pop("GEMINI_API_KEY", None)
        else: os.environ["GEMINI_API_KEY"] = self.previous_key
    def _adapter(self, payload, factory_calls=None, error=None, status="completed"):
        interactions = FakeInteractions(FakeResponse(payload, status=status) if payload is not None else None, error=error)
        def factory(api_key):
            if factory_calls is not None: factory_calls.append(api_key)
            return FakeClient(interactions)
        return GeminiLLMAdapter(client_factory=factory), interactions
    def test_missing_api_key_is_unavailable(self):
        os.environ.pop("GEMINI_API_KEY", None)
        adapter = GeminiLLMAdapter(client_factory=lambda _key: None)
        self.assertFalse(adapter.status().available)
        self.assertIn("API key", adapter.status().detail)
        self.assertEqual(adapter.interpret("x", "x", object(), CATALOG), ())
    def test_missing_sdk_is_unavailable(self):
        with patch("src.gemini_llm.find_spec", return_value=None): status = GeminiLLMAdapter().status()
        self.assertFalse(status.available); self.assertIn("google-genai", status.detail)
    def test_status_makes_no_network_or_client_request(self):
        calls = []; adapter = GeminiLLMAdapter(client_factory=lambda key: calls.append(key))
        self.assertTrue(adapter.status().available); self.assertEqual(calls, [])
    def test_client_initialization_is_lazy(self):
        calls = []; adapter, interactions = self._adapter('{"candidates": []}', factory_calls=calls)
        self.assertEqual(calls, []); adapter.status(); self.assertEqual(calls, [])
        adapter.interpret("desc", "desc", object(), CATALOG)
        self.assertEqual(calls, ["placeholder"]); self.assertEqual(len(interactions.calls), 1)
    def test_successful_structured_response(self):
        adapter, interactions = self._adapter('{"candidates":[{"canonical_material_id":"VAL-001","reason":"Matching valve specification."}]}')
        result = adapter.interpret("desc", "desc", object(), CATALOG)
        self.assertEqual(len(result), 1); self.assertEqual(result[0].canonical_material_id, "VAL-001")
        self.assertEqual(result[0].source, "gemini"); self.assertEqual(result[0].score, 1.0)
        self.assertIn("NON-PROBABILISTIC", result[0].explanation)
        self.assertEqual(len(interactions.calls), 1)
        call = interactions.calls[0]
        self.assertEqual(call["model"], DEFAULT_GEMINI_MODEL)
        response_format = call["response_format"]
        self.assertIsInstance(response_format, dict)
        self.assertEqual(response_format["type"], "text")
        self.assertEqual(response_format["mime_type"], "application/json")
        schema = response_format["schema"]
        self.assertIsInstance(schema, dict)
        self.assertEqual(schema["type"], "object")
        self.assertIn("candidates", schema["properties"])
        candidate_schema = schema["properties"]["candidates"]["items"]
        self.assertIn("canonical_material_id", candidate_schema["properties"])
        self.assertIn("reason", candidate_schema["properties"])
    def test_successful_fenced_json_response_shape(self):
        payload = """```json
{
  "candidates": [
    {
      "canonical_material_id": "VAL-001",
      "reason": "It is a gate valve."
    }
  ]
}
```"""
        adapter, _ = self._adapter(payload)
        result = adapter.interpret("CS GATE VLV 50MM FLG CL150", "CS GATE VLV 50MM FLG CL150", object(), CATALOG)
        self.assertEqual([(item.canonical_material_id, item.explanation.split(" [Advisory")[0]) for item in result], [("VAL-001", "It is a gate valve.")])
    def test_completed_markdown_bullet_list_is_rejected(self):
        adapter, _ = self._adapter("* VAL-001: General category match.\n* VAL-002: Matches gate valve type.")
        self.assertEqual(adapter.interpret("desc", "desc", object(), CATALOG), ())
    def test_completed_ordinary_prose_is_rejected(self):
        adapter, _ = self._adapter("VAL-001 looks like the best match because it is a gate valve.")
        self.assertEqual(adapter.interpret("desc", "desc", object(), CATALOG), ())
    def test_failed_status_is_rejected_and_unavailable(self):
        adapter, _ = self._adapter('{"candidates":[]}', status="failed")
        self.assertEqual(adapter.interpret("desc", "desc", object(), CATALOG), ())
        self.assertFalse(adapter.status().available); self.assertIn("status='failed'", adapter.status().detail)
    def test_incomplete_status_is_rejected_and_unavailable(self):
        adapter, _ = self._adapter('{"candidates":[]}', status="incomplete")
        self.assertEqual(adapter.interpret("desc", "desc", object(), CATALOG), ())
        self.assertFalse(adapter.status().available); self.assertIn("status='incomplete'", adapter.status().detail)
    def test_unknown_catalog_ids_are_rejected(self):
        adapter, _ = self._adapter('{"candidates":[{"canonical_material_id":"NOT-IN-CATALOG","reason":"Nope"},{"canonical_material_id":"VAL-001","reason":"Known"}]}')
        self.assertEqual([x.canonical_material_id for x in adapter.interpret("desc", "desc", object(), CATALOG)], ["VAL-001"])
    def test_exact_catalog_id_is_accepted(self):
        adapter, _ = self._adapter('{"candidates":[{"canonical_material_id":"VAL-001","reason":"Known exact ID"}]}')
        result = adapter.interpret("desc", "desc", object(), CATALOG)
        self.assertEqual([x.canonical_material_id for x in result], ["VAL-001"])
    def test_malformed_response_is_rejected_safely(self):
        adapter, _ = self._adapter("not-json")
        self.assertEqual(adapter.interpret("desc", "desc", object(), CATALOG), ()); self.assertFalse(adapter.status().available)
    def test_duplicate_ids_are_removed(self):
        adapter, _ = self._adapter('{"candidates":[{"canonical_material_id":"VAL-001","reason":"first"},{"canonical_material_id":"VAL-001","reason":"duplicate"},{"canonical_material_id":"VAL-002","reason":"second"}]}')
        result = adapter.interpret("desc", "desc", object(), CATALOG)
        self.assertEqual([x.canonical_material_id for x in result], ["VAL-001", "VAL-002"]); self.assertEqual([x.score for x in result], [1.0, 0.8])
    def test_maximum_five_candidates(self):
        catalog = tuple(CatalogRecord(f"VAL-{i:03d}", CATALOG[0].attributes) for i in range(1, 9))
        payload = '{"candidates":[' + ','.join(f'{{"canonical_material_id":"VAL-{i:03d}","reason":"reason {i}"}}' for i in range(1, 9)) + ']}'
        adapter, _ = self._adapter(payload); result = adapter.interpret("desc", "desc", object(), catalog)
        self.assertEqual(len(result), 5); self.assertEqual([x.score for x in result], [1.0, 0.8, 0.6, 0.4, 0.2])
    def test_api_exception_is_handled_safely(self):
        adapter, _ = self._adapter(None, error=TimeoutError("quota timeout"))
        self.assertEqual(adapter.interpret("desc", "desc", object(), CATALOG), ()); self.assertFalse(adapter.status().available)
    def test_empty_and_none_description_are_safe(self):
        calls = []; adapter, interactions = self._adapter('{"candidates": []}', factory_calls=calls)
        self.assertEqual(adapter.interpret("", "", object(), CATALOG), ()); self.assertEqual(adapter.interpret(None, "", object(), CATALOG), ())
        self.assertEqual(calls, []); self.assertEqual(interactions.calls, [])
    def test_rank_derived_scores_are_deterministic_and_not_probabilities(self):
        payload = '{"candidates":[{"canonical_material_id":"VAL-002","reason":"second"},{"canonical_material_id":"VAL-001","reason":"first"}]}'
        first, _ = self._adapter(payload); second, _ = self._adapter(payload)
        a = first.interpret("desc", "desc", object(), CATALOG); b = second.interpret("desc", "desc", object(), CATALOG)
        self.assertEqual(a, b); self.assertEqual([x.score for x in a], [1.0, 0.8]); self.assertTrue(all("NON-PROBABILISTIC" in x.explanation for x in a))
    def test_repeated_mocked_response_is_deterministic(self):
        adapter, interactions = self._adapter('{"candidates":[{"canonical_material_id":"VAL-002","reason":"B"},{"canonical_material_id":"VAL-001","reason":"A"}]}')
        self.assertEqual(adapter.interpret("desc", "desc", object(), CATALOG), adapter.interpret("desc", "desc", object(), CATALOG)); self.assertEqual(len(interactions.calls), 2)
    def test_mapping_unchanged_when_gemini_succeeds(self):
        adapter, _ = self._adapter('{"candidates":[{"canonical_material_id":"VAL-002","reason":"Advisory only"}]}')
        raw = "CS GATE VLV 50MM FLG CL150"; expected = map_records((LegacyRecord("CHECK", raw),), CATALOG)[0]
        actual = run_hybrid_pipeline(raw, CATALOG, "CHECK", llm_adapter=adapter).mapping_result
        self.assertEqual(actual, expected); self.assertEqual(actual.canonical_material_id, "VAL-001")
    def test_mapping_unchanged_when_gemini_fails(self):
        adapter, _ = self._adapter(None, error=RuntimeError("service unavailable")); raw = "CS GATE VLV 50MM FLG CL150"
        self.assertEqual(run_hybrid_pipeline(raw, CATALOG, "CHECK", llm_adapter=adapter).mapping_result, map_records((LegacyRecord("CHECK", raw),), CATALOG)[0])
    def test_technical_conflicts_cannot_be_overridden(self):
        adapter, _ = self._adapter('{"candidates":[{"canonical_material_id":"VAL-001","reason":"Gemini suggestion"}]}')
        result = run_hybrid_pipeline("CS GATE VLV 50MM FLG CL300", CATALOG, "CHECK", llm_adapter=adapter)
        self.assertEqual(result.mapping_result.canonical_material_id, "VAL-002"); self.assertEqual(result.mapping_result.decision, "MATCHED")
        self.assertEqual(result.ai_candidate_suggestions[0].canonical_material_id, "VAL-001")
    def test_no_api_key_value_is_present_in_source(self):
        source = inspect.getsource(GeminiLLMAdapter); self.assertNotIn("placeholder", source); self.assertIn("GEMINI_API_KEY", source)
    def test_no_authoritative_mapping_result_is_returned(self):
        adapter, _ = self._adapter('{"candidates":[{"canonical_material_id":"VAL-001","reason":"hint"}]}')
        result = adapter.interpret("desc", "desc", object(), CATALOG)
        self.assertIsInstance(result, tuple); self.assertTrue(all(isinstance(x, CandidateSuggestion) for x in result)); self.assertFalse(any(hasattr(x, "decision") for x in result))
    def test_status_uses_environment_key_without_client(self):
        adapter = GeminiLLMAdapter(client_factory=lambda _key: (_ for _ in ()).throw(AssertionError("client should not initialize")))
        self.assertTrue(adapter.status().available)

if __name__ == "__main__": unittest.main()
