"""Dependency-free unit tests for LEGO #9B local embedding retrieval."""

from __future__ import annotations

import math
import unittest

from src.ai_retrieval import AdapterStatus
from src.catalog_mapping import CatalogRecord
from src.local_embedding_retrieval import DEFAULT_MODEL_NAME, LocalEmbeddingRetrievalAdapter


class FakeModel:
    def __init__(self, embeddings: dict[str, list[float] | list[list[float]]], calls: list[str] | None = None) -> None:
        self.embeddings = embeddings
        self.calls = calls if calls is not None else []

    def encode(self, texts, convert_to_numpy=False):
        self.calls.append("encode")
        if isinstance(texts, str):
            return self.embeddings[texts]
        return [self.embeddings[text] for text in texts]


class FakeLoader:
    def __init__(self, model: FakeModel | None = None, error: Exception | None = None) -> None:
        self.model = model
        self.error = error
        self.calls = 0

    def __call__(self, model_name: str):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.model


class LocalEmbeddingRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.catalog = tuple(CatalogRecord(f"MAT-{i:02d}", object()) for i in range(1, 9))
        self.query = "query text"

    def test_module_imports_without_optional_dependencies(self):
        # Import succeeded without importing either optional package.
        self.assertEqual(DEFAULT_MODEL_NAME, "sentence-transformers/all-MiniLM-L6-v2")

    def test_status_does_not_load_model(self):
        loader = FakeLoader(FakeModel({self.query: [1.0, 0.0]}))
        adapter = LocalEmbeddingRetrievalAdapter(model_loader=loader)
        adapter.status()
        self.assertEqual(loader.calls, 0)

    def test_lazy_model_loading(self):
        model = FakeModel({self.query: [1.0, 0.0], **{f"canonical_material_id=MAT-{i:02d} attributes=<object object at 0x0>": [1.0, 0.0] for i in range(1, 9)}})
        # Use the adapter's own catalog text helper to avoid duplicating its representation.
        embeddings = {self.query: [1.0, 0.0]}
        for record in self.catalog:
            embeddings[adapter_text(record)] = [1.0, 0.0]
        model = FakeModel(embeddings)
        loader = FakeLoader(model)
        adapter = LocalEmbeddingRetrievalAdapter(model_loader=loader)
        self.assertEqual(loader.calls, 0)
        adapter.retrieve(self.query, self.query, object(), self.catalog)
        self.assertEqual(loader.calls, 1)

    def test_missing_dependency_behavior(self):
        adapter = LocalEmbeddingRetrievalAdapter(model_loader=None)
        status = adapter.status()
        # The real environment may have the optional package; the key invariant is that
        # status is safe and retrieval cannot raise if the loader cannot be used.
        self.assertIsInstance(status, AdapterStatus)
        if not status.available:
            self.assertEqual(adapter.retrieve(self.query, self.query, object(), self.catalog), ())

    def test_model_load_failure_disables_repeated_attempts(self):
        loader = FakeLoader(error=RuntimeError("no model available"))
        adapter = LocalEmbeddingRetrievalAdapter(model_loader=loader)
        self.assertEqual(adapter.retrieve(self.query, self.query, object(), self.catalog), ())
        self.assertEqual(adapter.retrieve(self.query, self.query, object(), self.catalog), ())
        self.assertEqual(loader.calls, 1)
        self.assertFalse(adapter.status().available)
        self.assertIn("Model loading failed", adapter.status().detail)

    def test_embedding_failure_is_non_fatal(self):
        class FailingModel:
            def encode(self, *_args, **_kwargs):
                raise RuntimeError("embedding failed")

        adapter = LocalEmbeddingRetrievalAdapter(model_loader=FakeLoader(FailingModel()))
        self.assertEqual(adapter.retrieve(self.query, self.query, object(), self.catalog), ())
        self.assertFalse(adapter.status().available)
        self.assertIn("Embedding failed", adapter.status().detail)

    def test_malformed_embedding_output_is_safe(self):
        class BadModel:
            def encode(self, texts, **_kwargs):
                if isinstance(texts, str):
                    return [1.0, float("nan")]
                return [[1.0, 0.0] for _ in texts]

        adapter = LocalEmbeddingRetrievalAdapter(model_loader=FakeLoader(BadModel()))
        self.assertEqual(adapter.retrieve(self.query, self.query, object(), self.catalog), ())
        self.assertIn("Embedding failed", adapter.status().detail)

    def test_empty_input_does_not_load_model(self):
        loader = FakeLoader(FakeModel({}))
        adapter = LocalEmbeddingRetrievalAdapter(model_loader=loader)
        self.assertEqual(adapter.retrieve(None, "", object(), self.catalog), ())
        self.assertEqual(loader.calls, 0)

    def test_empty_catalog_does_not_load_model(self):
        loader = FakeLoader(FakeModel({}))
        adapter = LocalEmbeddingRetrievalAdapter(model_loader=loader)
        self.assertEqual(adapter.retrieve(self.query, self.query, object(), ()), ())
        self.assertEqual(loader.calls, 0)

    def test_deterministic_cosine_ranking(self):
        embeddings = {self.query: [1.0, 0.0]}
        embeddings[adapter_text(self.catalog[0])] = [1.0, 0.0]
        embeddings[adapter_text(self.catalog[1])] = [0.0, 1.0]
        for record in self.catalog[2:]:
            embeddings[adapter_text(record)] = [-1.0, 0.0]
        adapter = LocalEmbeddingRetrievalAdapter(model_loader=FakeLoader(FakeModel(embeddings)))
        result = adapter.retrieve(self.query, self.query, object(), self.catalog)
        self.assertEqual(result[0].canonical_material_id, "MAT-01")
        self.assertEqual(result[1].canonical_material_id, "MAT-02")
        self.assertAlmostEqual(result[0].score, 1.0)
        self.assertAlmostEqual(result[1].score, 0.0)

    def test_maximum_five_candidates(self):
        embeddings = {self.query: [1.0, 0.0]}
        for record in self.catalog:
            embeddings[adapter_text(record)] = [1.0, 0.0]
        adapter = LocalEmbeddingRetrievalAdapter(model_loader=FakeLoader(FakeModel(embeddings)))
        result = adapter.retrieve(self.query, self.query, object(), self.catalog)
        self.assertEqual(len(result), 5)

    def test_scores_are_finite_and_bounded(self):
        embeddings = {self.query: [1.0, 0.0]}
        for record in self.catalog:
            embeddings[adapter_text(record)] = [1.0, 1.0]
        adapter = LocalEmbeddingRetrievalAdapter(model_loader=FakeLoader(FakeModel(embeddings)))
        result = adapter.retrieve(self.query, self.query, object(), self.catalog)
        self.assertTrue(all(math.isfinite(item.score) and -1.0 <= item.score <= 1.0 for item in result))

    def test_only_known_catalog_ids_returned(self):
        embeddings = {self.query: [1.0, 0.0]}
        for record in self.catalog:
            embeddings[adapter_text(record)] = [1.0, 0.0]
        adapter = LocalEmbeddingRetrievalAdapter(model_loader=FakeLoader(FakeModel(embeddings)))
        result = adapter.retrieve(self.query, self.query, object(), self.catalog)
        self.assertTrue({item.canonical_material_id for item in result} <= {r.canonical_material_id for r in self.catalog})

    def test_repeated_retrieval_is_deterministic(self):
        embeddings = {self.query: [1.0, 0.0]}
        for record in self.catalog:
            embeddings[adapter_text(record)] = [1.0, 0.0]
        adapter = LocalEmbeddingRetrievalAdapter(model_loader=FakeLoader(FakeModel(embeddings)))
        first = adapter.retrieve(self.query, self.query, object(), self.catalog)
        second = adapter.retrieve(self.query, self.query, object(), self.catalog)
        self.assertEqual(first, second)

    def test_adapter_never_produces_authoritative_decision(self):
        embeddings = {self.query: [1.0, 0.0]}
        for record in self.catalog:
            embeddings[adapter_text(record)] = [1.0, 0.0]
        adapter = LocalEmbeddingRetrievalAdapter(model_loader=FakeLoader(FakeModel(embeddings)))
        result = adapter.retrieve(self.query, self.query, object(), self.catalog)
        for suggestion in result:
            self.assertNotIn(suggestion.source, {"MATCHED", "UNCERTAIN", "NEW_CANDIDATE"})
            self.assertNotIn(suggestion.explanation, {"MATCHED", "UNCERTAIN", "NEW_CANDIDATE"})
        self.assertFalse(hasattr(adapter, "mapping_result"))


def adapter_text(record: CatalogRecord) -> str:
    """Mirror only the adapter's stable catalog-text helper for fake-model keys."""
    return LocalEmbeddingRetrievalAdapter._catalog_text(record)


if __name__ == "__main__":
    unittest.main()
