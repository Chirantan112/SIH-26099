"""Fault-tolerant, non-authoritative orchestration for LEGO #9A.

This module adds optional AI suggestions around the existing deterministic
pipeline. It never changes candidate ranking, linkage, or catalog-mapping
logic: LEGO #4/#5 remains the sole authority for final decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Callable, TypeVar

from src.ai_retrieval import AdapterStatus, CandidateSuggestion, RetrievalAdapter, UnavailableRetrievalAdapter
from src.attribute_extraction import MaterialAttributes, extract_attributes
from src.catalog_mapping import CatalogRecord, LegacyRecord, MappingResult, map_records
from src.llm_interpretation import LLMInterpretationAdapter, UnavailableLLMAdapter
from src.normalization import normalize_description


MAX_AI_CANDIDATE_SUGGESTIONS = 5
_SuggestionAdapter = TypeVar("_SuggestionAdapter")


@dataclass(frozen=True)
class AIConsensus:
    """Explicit advisory conclusion derived only from AI evidence."""

    conclusion: str  # MATCHED, UNCERTAIN, NEW_CANDIDATE, or UNAVAILABLE
    canonical_material_id: str | None
    reason: str
    local_candidates: tuple[str, ...]
    gemini_candidates: tuple[str, ...]


@dataclass(frozen=True)
class HybridResult:
    """Deterministic result plus bounded, non-authoritative AI diagnostics."""

    legacy_material_code: str
    normalized_description: str
    normalization_transformations: tuple[str, ...]
    attributes: MaterialAttributes
    mapping_result: MappingResult
    ai_statuses: tuple[AdapterStatus, ...]
    ai_candidate_suggestions: tuple[CandidateSuggestion, ...]
    ai_consensus: AIConsensus
    fallback_used: bool
    fallback_information: str
    explanation: str


def _safe_status(component: str, adapter: object) -> AdapterStatus:
    """Read optional status without allowing an adapter failure to stop mapping."""
    try:
        status = adapter.status()  # type: ignore[attr-defined]
    except Exception as error:
        return AdapterStatus(component=component, available=False, detail=f"Status unavailable: {error}")
    if not isinstance(status, AdapterStatus) or status.component != component:
        return AdapterStatus(component=component, available=False, detail="Adapter returned an invalid status.")
    return status


def _safe_suggestions(
    component: str,
    adapter: _SuggestionAdapter,
    operation: Callable[[_SuggestionAdapter], tuple[CandidateSuggestion, ...]],
    status: AdapterStatus,
) -> tuple[tuple[CandidateSuggestion, ...], AdapterStatus]:
    """Run one optional adapter and convert failures into a non-fatal status."""
    if not status.available:
        return (), status
    try:
        suggestions = operation(adapter)
    except Exception as error:
        return (), replace(status, available=False, detail=f"Adapter failed: {error}")
    if not isinstance(suggestions, tuple):
        return (), replace(status, available=False, detail="Adapter returned an invalid suggestion collection.")
    return suggestions, status


def _valid_suggestions(
    suggestions: tuple[CandidateSuggestion, ...],
    catalog: tuple[CatalogRecord, ...],
) -> tuple[CandidateSuggestion, ...]:
    """Accept only bounded, well-formed references to known catalog IDs."""
    known_ids = {record.canonical_material_id for record in catalog}
    valid: list[CandidateSuggestion] = []
    for suggestion in suggestions:
        if not isinstance(suggestion, CandidateSuggestion):
            continue
        if (
            suggestion.canonical_material_id not in known_ids
            or not isinstance(suggestion.score, float)
            or not isfinite(suggestion.score)
            or not isinstance(suggestion.source, str)
            or not suggestion.source.strip()
            or not isinstance(suggestion.explanation, str)
            or not isinstance(suggestion.matching_attributes, tuple)
            or not isinstance(suggestion.conflicting_attributes, tuple)
            or not isinstance(suggestion.missing_attributes, tuple)
            or suggestion.technical_compatible not in (True, False, None)
        ):
            continue
        valid.append(suggestion)

    return tuple(sorted(valid, key=lambda item: (-item.score, item.canonical_material_id, item.source)))[:MAX_AI_CANDIDATE_SUGGESTIONS]


def _source_candidates(suggestions: tuple[CandidateSuggestion, ...], source: str) -> tuple[CandidateSuggestion, ...]:
    return tuple(item for item in suggestions if item.source == source)


def _has_explicit_conflict(suggestions: tuple[CandidateSuggestion, ...]) -> bool:
    """Return True only when every supplied candidate is explicitly ruled out."""
    return bool(suggestions) and all(
        item.technical_compatible is False and bool(item.conflicting_attributes)
        for item in suggestions
    )


def _single_advisor_consensus(
    source_label: str,
    candidates: tuple[CandidateSuggestion, ...],
    local_ids: tuple[str, ...],
    gemini_ids: tuple[str, ...],
) -> AIConsensus:
    """Map one advisor's usable evidence to the requested advisory state."""
    compatible = tuple(item for item in candidates if item.technical_compatible is True)
    if compatible:
        candidate = compatible[0].canonical_material_id
        return AIConsensus(
            "MATCHED",
            candidate,
            f"Only {source_label} supplied usable advisory evidence; it reports a technically compatible candidate.",
            local_ids,
            gemini_ids,
        )
    if _has_explicit_conflict(candidates):
        return AIConsensus(
            "NEW_CANDIDATE",
            None,
            f"Only {source_label} supplied usable advisory evidence; it explicitly rules out every supplied candidate with technical conflicts.",
            local_ids,
            gemini_ids,
        )
    return AIConsensus(
        "UNCERTAIN",
        None,
        f"Only {source_label} supplied usable advisory evidence; its technical evidence is incomplete or unresolved.",
        local_ids,
        gemini_ids,
    )


def _ai_consensus(
    statuses: tuple[AdapterStatus, AdapterStatus],
    suggestions: tuple[CandidateSuggestion, ...],
) -> AIConsensus:
    """Derive conservative AI evidence without consulting deterministic mapping."""
    local_status, gemini_status = statuses
    local = _source_candidates(suggestions, "local_embedding")
    gemini = _source_candidates(suggestions, "gemini")
    local_compatible = tuple(item for item in local if item.technical_compatible is True)
    gemini_compatible = tuple(item for item in gemini if item.technical_compatible is True)
    local_ids = tuple(item.canonical_material_id for item in local)
    gemini_ids = tuple(item.canonical_material_id for item in gemini)

    if not local_status.available and not gemini_status.available:
        return AIConsensus(
            "UNAVAILABLE",
            None,
            "Both advisory AI components are unavailable; no AI conclusion was produced.",
            local_ids,
            gemini_ids,
        )

    # Two available components should independently support the same candidate
    # for a strong MATCHED consensus. Any unresolved or conflicting evidence is
    # conservative UNCERTAIN rather than a new-material claim.
    local_evidence_available = local_status.available and bool(local)
    gemini_evidence_available = gemini_status.available and bool(gemini)
    if local_evidence_available and gemini_evidence_available:
        if local_compatible and gemini_compatible:
            if local_compatible[0].canonical_material_id == gemini_compatible[0].canonical_material_id:
                candidate = local_compatible[0].canonical_material_id
                return AIConsensus(
                    "MATCHED",
                    candidate,
                    "Local NLP and Gemini independently identify the same technically compatible candidate with no reported critical conflict.",
                    local_ids,
                    gemini_ids,
                )
            return AIConsensus(
                "UNCERTAIN",
                None,
                "Local NLP and Gemini identify different technically compatible candidates; advisory evidence disagrees.",
                local_ids,
                gemini_ids,
            )

        if local_compatible or gemini_compatible:
            return AIConsensus(
                "UNCERTAIN",
                None,
                "One advisory component supports technical compatibility while the other does not provide matching compatible evidence.",
                local_ids,
                gemini_ids,
            )

        if _has_explicit_conflict(local) and _has_explicit_conflict(gemini):
            return AIConsensus(
                "NEW_CANDIDATE",
                None,
                "Both advisory components explicitly rule out every supplied candidate with technical conflicts.",
                local_ids,
                gemini_ids,
            )

        # Empty, unresolved, missing, or ambiguous evidence does not establish
        # that the catalog lacks a compatible material.
        return AIConsensus(
            "UNCERTAIN",
            None,
            "AI evidence is incomplete or unresolved; insufficient evidence for a new-candidate conclusion.",
            local_ids,
            gemini_ids,
        )

    if local_evidence_available:
        return _single_advisor_consensus("Local NLP", local, local_ids, gemini_ids)
    if gemini_evidence_available:
        return _single_advisor_consensus("Gemini", gemini, local_ids, gemini_ids)

    return AIConsensus(
        "UNCERTAIN",
        None,
        "AI services are available, but neither returned usable candidate evidence for this input.",
        local_ids,
        gemini_ids,
    )


def run_hybrid_pipeline(
    raw_description: str | None,
    catalog: tuple[CatalogRecord, ...],
    legacy_material_code: str = "HYBRID-INPUT",
    retrieval_adapter: RetrievalAdapter | None = None,
    llm_adapter: LLMInterpretationAdapter | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> HybridResult:
    """Run the existing deterministic flow with an optional execution-progress hook."""

    def _report(stage: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(stage)
        except Exception:
            pass

    retrieval = retrieval_adapter or UnavailableRetrievalAdapter()
    llm = llm_adapter or UnavailableLLMAdapter()
    _report("input")
    normalization = normalize_description(raw_description)
    _report("normalize")
    extraction = extract_attributes(raw_description)
    _report("extract")
    mapping_input = raw_description if normalization.normalized_text else ""

    # This is intentionally unchanged deterministic LEGO #2-#5 authority.
    mapping_result = map_records((LegacyRecord(legacy_material_code, mapping_input),), catalog)[0]
    _report("match")

    retrieval_status = _safe_status("local_nlp", retrieval)
    llm_status = _safe_status("llm", llm)
    retrieval_suggestions, retrieval_status = _safe_suggestions(
        "local_nlp",
        retrieval,
        lambda adapter: adapter.retrieve(raw_description, normalization.normalized_text, extraction.attributes, catalog),
        retrieval_status,
    )
    llm_suggestions, llm_status = _safe_suggestions(
        "llm",
        llm,
        lambda adapter: adapter.interpret(raw_description, normalization.normalized_text, extraction.attributes, catalog),
        llm_status,
    )
    _report("ai_advisory")
    suggestions = _valid_suggestions(retrieval_suggestions + llm_suggestions, catalog)
    statuses = (retrieval_status, llm_status)
    consensus = _ai_consensus(statuses, suggestions)
    fallback_used = not all(status.available for status in statuses)
    fallback_information = (
        "Deterministic LEGO #2-#5 mapping remains active; unavailable or failed AI adapters were skipped."
        if fallback_used
        else "Deterministic LEGO #2-#5 mapping remains authoritative; AI consensus is advisory evidence only."
    )
    result = HybridResult(
        legacy_material_code=legacy_material_code,
        normalized_description=normalization.normalized_text,
        normalization_transformations=normalization.transformations,
        attributes=extraction.attributes,
        mapping_result=mapping_result,
        ai_statuses=statuses,
        ai_candidate_suggestions=suggestions,
        ai_consensus=consensus,
        fallback_used=fallback_used,
        fallback_information=fallback_information,
        explanation="AI suggestions and consensus are advisory only; final mapping is the unchanged deterministic LEGO #4/#5 result.",
    )
    _report("decide")
    return result
