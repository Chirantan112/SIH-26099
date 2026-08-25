"""Batch catalog mapping and entity resolution layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from src.attribute_extraction import extract_attributes
from src.record_linkage import compare_records

if TYPE_CHECKING:
    from src.attribute_extraction import MaterialAttributes


@dataclass(frozen=True)
class LegacyRecord:
    """Represents a CPSE legacy material record to be matched."""

    legacy_material_code: str
    raw_description: str


@dataclass(frozen=True)
class CatalogRecord:
    """Represents a standardized canonical material catalog entry."""

    canonical_material_id: str
    attributes: MaterialAttributes


@dataclass(frozen=True)
class CandidateInfo:
    """Details about a compared candidate from the catalog."""

    canonical_material_id: str
    score: float
    decision: str  # "SAME", "DIFFERENT", or "UNCERTAIN"
    explanation: str


@dataclass(frozen=True)
class MappingResult:
    """Final result of a batch mapping operation for a single legacy record."""

    legacy_material_code: str
    canonical_material_id: str | None
    decision: str  # "MATCHED", "UNCERTAIN", or "NEW_CANDIDATE"
    score: float
    explanation: str
    all_candidates: tuple[CandidateInfo, ...]


def _sort_key(c: CandidateInfo) -> tuple[int, float, str]:
    """Deterministic sort key for ranking candidates.

    Sorts by:
    1. Decision priority: SAME (2) > UNCERTAIN (1) > DIFFERENT (0)
    2. Score descending (represented by -c.score)
    3. Canonical material ID alphabetically (for determinism)
    """
    decision_priority = 2 if c.decision == "SAME" else (1 if c.decision == "UNCERTAIN" else 0)
    return (-decision_priority, -c.score, c.canonical_material_id)


def map_records(
    legacy_records: Iterable[dict[str, str] | LegacyRecord],
    catalog_records: Iterable[CatalogRecord],
) -> tuple[MappingResult, ...]:
    """Map legacy CPSE material records against a canonical material catalog.

    Each legacy record is extracted and matched against every catalog record.
    The best compatible candidate is selected. If there are no compatible
    candidates (all are DIFFERENT), the decision is NEW_CANDIDATE. If there are
    multiple SAME candidates with equal top scores, or if the best candidate is
    UNCERTAIN, the decision is UNCERTAIN.
    """
    catalog_list = list(catalog_records)
    results: list[MappingResult] = []

    for legacy in legacy_records:
        if isinstance(legacy, dict):
            code = legacy.get("legacy_material_code", "")
            desc = legacy.get("raw_description", "")
        else:
            code = getattr(legacy, "legacy_material_code", "")
            desc = getattr(legacy, "raw_description", "")

        if not desc:
            results.append(
                MappingResult(
                    legacy_material_code=code,
                    canonical_material_id=None,
                    decision="NEW_CANDIDATE",
                    score=0.0,
                    explanation="Empty or missing description.",
                    all_candidates=(),
                )
            )
            continue

        # Extract attributes from legacy description
        extracted = extract_attributes(desc)

        # Compare against all catalog entries
        candidates: list[CandidateInfo] = []
        for cat_rec in catalog_list:
            linkage = compare_records(extracted.attributes, cat_rec.attributes)
            candidates.append(
                CandidateInfo(
                    canonical_material_id=cat_rec.canonical_material_id,
                    score=linkage.score,
                    decision=linkage.decision,
                    explanation="; ".join(linkage.notes),
                )
            )

        if not candidates:
            results.append(
                MappingResult(
                    legacy_material_code=code,
                    canonical_material_id=None,
                    decision="NEW_CANDIDATE",
                    score=0.0,
                    explanation="Canonical catalog is empty.",
                    all_candidates=(),
                )
            )
            continue

        # Deterministically rank candidates
        sorted_candidates = sorted(candidates, key=_sort_key)
        top_cand = sorted_candidates[0]

        decision: str
        canonical_material_id: str | None = None
        score: float = 0.0
        explanation: str

        if top_cand.decision == "SAME":
            # Check for ties among SAME candidates
            same_cands = [c for c in sorted_candidates if c.decision == "SAME"]
            if len(same_cands) > 1 and same_cands[1].score == top_cand.score:
                decision = "UNCERTAIN"
                score = top_cand.score
                explanation = (
                    f"Ambiguous match: multiple canonical materials ({top_cand.canonical_material_id}, "
                    f"{same_cands[1].canonical_material_id}) match with equal top score."
                )
            else:
                decision = "MATCHED"
                canonical_material_id = top_cand.canonical_material_id
                score = top_cand.score
                explanation = f"Matched to canonical material '{top_cand.canonical_material_id}'."
        elif top_cand.decision == "UNCERTAIN":
            decision = "UNCERTAIN"
            score = top_cand.score
            explanation = f"No clean match found. Best candidate is uncertain: '{top_cand.canonical_material_id}'."
        else:
            # All candidates are DIFFERENT (technical conflicts)
            decision = "NEW_CANDIDATE"
            score = 0.0
            explanation = "No compatible canonical materials found in the catalog (all had technical conflicts)."

        results.append(
            MappingResult(
                legacy_material_code=code,
                canonical_material_id=canonical_material_id,
                decision=decision,
                score=score,
                explanation=explanation,
                all_candidates=tuple(sorted_candidates),
            )
        )

    return tuple(results)
