"""Dependency-free contracts for optional LLM interpretation.

LLM output is limited to non-authoritative candidate suggestions.  This module
does not import an SDK or make network requests, so the offline pipeline stays
usable without credentials, internet access, or an LLM installation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from src.ai_retrieval import AdapterStatus, CandidateSuggestion

if TYPE_CHECKING:
    from src.attribute_extraction import MaterialAttributes
    from src.catalog_mapping import CatalogRecord


class LLMInterpretationAdapter(Protocol):
    """Protocol for optional advisory LLM candidate interpretation."""

    def status(self) -> AdapterStatus:
        """Return current availability without calling a provider."""

    def interpret(
        self,
        raw_description: str | None,
        normalized_description: str,
        attributes: MaterialAttributes,
        catalog: tuple[CatalogRecord, ...],
    ) -> tuple[CandidateSuggestion, ...]:
        """Return advisory suggestions only; final decisions are prohibited."""


@dataclass(frozen=True)
class UnavailableLLMAdapter:
    """Safe default when no LLM provider is configured."""

    detail: str = "Optional LLM interpretation is not configured."

    def status(self) -> AdapterStatus:
        return AdapterStatus(component="llm", available=False, detail=self.detail)

    def interpret(
        self,
        raw_description: str | None,
        normalized_description: str,
        attributes: MaterialAttributes,
        catalog: tuple[CatalogRecord, ...],
    ) -> tuple[CandidateSuggestion, ...]:
        return ()
