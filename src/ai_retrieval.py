"""Dependency-free contracts for optional local semantic candidate retrieval.

LEGO #9A deliberately defines no model implementation.  Adapters conforming to
these contracts are advisory only; deterministic LEGO #4/#5 verification
remains responsible for every final mapping decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.attribute_extraction import MaterialAttributes
    from src.catalog_mapping import CatalogRecord


@dataclass(frozen=True)
class AdapterStatus:
    """UI-safe availability and diagnostic state for one optional adapter."""

    component: str
    available: bool
    detail: str


@dataclass(frozen=True)
class CandidateSuggestion:
    """Non-authoritative, primitive-only candidate suggestion from an AI adapter."""

    canonical_material_id: str
    score: float
    source: str
    explanation: str


class RetrievalAdapter(Protocol):
    """Protocol for an optional local embedding or semantic-search adapter."""

    def status(self) -> AdapterStatus:
        """Return current availability without loading a model."""

    def retrieve(
        self,
        raw_description: str | None,
        normalized_description: str,
        attributes: MaterialAttributes,
        catalog: tuple[CatalogRecord, ...],
    ) -> tuple[CandidateSuggestion, ...]:
        """Return advisory catalog suggestions; never a final mapping decision."""


@dataclass(frozen=True)
class UnavailableRetrievalAdapter:
    """Safe default when no local NLP implementation is configured."""

    detail: str = "Local NLP retrieval is not configured."

    def status(self) -> AdapterStatus:
        return AdapterStatus(component="local_nlp", available=False, detail=self.detail)

    def retrieve(
        self,
        raw_description: str | None,
        normalized_description: str,
        attributes: MaterialAttributes,
        catalog: tuple[CatalogRecord, ...],
    ) -> tuple[CandidateSuggestion, ...]:
        return ()
