"""Optional local embedding retrieval adapter for LEGO #9B.

The adapter is deliberately isolated from deterministic normalization, attribute
extraction, record linkage, and catalog mapping. It produces advisory candidate
suggestions only; LEGO #4/#5 remains authoritative for every mapping decision.

Optional NLP dependencies are imported only when ``retrieve`` needs them. No
model is loaded by module import or by ``status``.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from importlib.util import find_spec
from math import isfinite, sqrt
from typing import Any, Callable

from src.ai_retrieval import AdapterStatus, CandidateSuggestion, RetrievalAdapter

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CANDIDATES = 5


class LocalEmbeddingRetrievalAdapter(RetrievalAdapter):
    """Lazy, fault-tolerant sentence-embedding candidate retrieval.

    ``model_loader`` is an optional test seam. In production the default loader
    imports sentence-transformers lazily and constructs ``SentenceTransformer``
    only inside ``retrieve``. The adapter never decides MATCHED/UNCERTAIN/
    NEW_CANDIDATE and never mutates deterministic mapping results.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model_loader: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self._model_loader = model_loader
        self._model: Any | None = None
        self._load_attempted = False
        self._failure_detail: str | None = None
        self._embedding_failure_detail: str | None = None

    def status(self) -> AdapterStatus:
        """Report availability without importing or loading the embedding model."""
        if self._failure_detail is not None:
            return AdapterStatus("local_nlp", False, self._failure_detail)
        if self._embedding_failure_detail is not None:
            return AdapterStatus("local_nlp", False, self._embedding_failure_detail)
        if self._model is not None:
            return AdapterStatus("local_nlp", True, f"Model loaded: {self.model_name}")
        if self._model_loader is not None:
            return AdapterStatus("local_nlp", True, "Local embedding adapter is ready; model loading is lazy.")
        if find_spec("sentence_transformers") is None:
            return AdapterStatus("local_nlp", False, "sentence-transformers is not installed.")
        if find_spec("torch") is None:
            return AdapterStatus("local_nlp", False, "torch is not installed.")
        return AdapterStatus("local_nlp", True, "Optional local embedding dependencies are available; model loading is lazy.")

    def retrieve(
        self,
        raw_description: str | None,
        normalized_description: str,
        attributes: Any,
        catalog: tuple[Any, ...],
    ) -> tuple[CandidateSuggestion, ...]:
        """Return up to five cosine-similarity suggestions from the supplied catalog."""
        if not normalized_description or not catalog:
            return ()
        if self._failure_detail is not None:
            return ()

        if not self._ensure_model():
            return ()

        try:
            query_embedding = self._encode(normalized_description)
            catalog_texts = tuple(self._catalog_text(record) for record in catalog)
            catalog_embeddings = self._encode(catalog_texts)
            query_vector = self._validate_vector(query_embedding)
            vectors = self._validate_matrix(catalog_embeddings, len(catalog))
            scored = []
            for record, vector in zip(catalog, vectors):
                score = self._cosine_similarity(query_vector, vector)
                if isfinite(score):
                    scored.append((score, record.canonical_material_id))
            scored.sort(key=lambda item: (-item[0], item[1]))
            return tuple(
                CandidateSuggestion(
                    canonical_material_id=canonical_id,
                    score=score,
                    source="local_embedding",
                    explanation=f"Cosine similarity from {self.model_name}; advisory only.",
                )
                for score, canonical_id in scored[:MAX_CANDIDATES]
            )
        except Exception as error:
            self._embedding_failure_detail = f"Embedding failed: {error}"
            return ()

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True
        try:
            if self._model_loader is not None:
                loader = self._model_loader
            else:
                # Optional dependency imports are intentionally local and occur
                # only when retrieve() needs to load the model.
                from sentence_transformers import SentenceTransformer

                loader = SentenceTransformer
            self._model = loader(self.model_name)
            return True
        except Exception as error:
            self._failure_detail = f"Model loading failed: {error}"
            self._model = None
            return False

    def _encode(self, texts: str | tuple[str, ...]) -> Any:
        if isinstance(texts, str):
            return self._model.encode(texts, convert_to_numpy=False)
        return self._model.encode(list(texts), convert_to_numpy=False)

    @staticmethod
    def _validate_vector(value: Any) -> tuple[float, ...]:
        if isinstance(value, (str, bytes)):
            raise ValueError("Embedding is not a numeric vector")
        try:
            values = tuple(value)
        except TypeError as error:
            raise ValueError("Embedding is not iterable") from error
        if not values:
            raise ValueError("Embedding vector is empty")
        result = tuple(float(item) for item in values)
        if not all(isfinite(item) for item in result):
            raise ValueError("Embedding contains a non-finite value")
        return result

    @classmethod
    def _validate_matrix(cls, value: Any, expected_rows: int) -> tuple[tuple[float, ...], ...]:
        if isinstance(value, (str, bytes)):
            raise ValueError("Catalog embeddings are not a matrix")
        try:
            rows = tuple(value)
        except TypeError as error:
            raise ValueError("Catalog embeddings are not iterable") from error
        if len(rows) != expected_rows:
            raise ValueError("Catalog embedding count does not match catalog size")
        return tuple(cls._validate_vector(row) for row in rows)

    @staticmethod
    def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        if len(left) != len(right):
            raise ValueError("Embedding dimensions do not match")
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = sqrt(sum(a * a for a in left))
        right_norm = sqrt(sum(b * b for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            raise ValueError("Cannot calculate cosine similarity for a zero vector")
        score = dot / (left_norm * right_norm)
        return max(-1.0, min(1.0, float(score)))

    @staticmethod
    def _catalog_text(record: Any) -> str:
        """Build embedding input from supplied descriptive/technical attributes only."""
        attributes = getattr(record, "attributes", None)
        if is_dataclass(attributes):
            parts = []
            for field in fields(attributes):
                value = getattr(attributes, field.name)
                if value is not None and str(value).strip():
                    parts.append(f"{field.name}={value}")
            return " ".join(parts)
        return f"attributes={attributes!r}"
