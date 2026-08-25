"""Offline integration façade for the deterministic SIH material demo.

This module is intentionally framework-free.  It loads a validated reference
catalog once at application startup and exposes one function for running a
single raw description through the existing normalization, extraction, and
catalog-mapping LEGO modules.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict

from src.attribute_extraction import MaterialAttributes, extract_attributes
from src.catalog_mapping import CandidateInfo, CatalogRecord, LegacyRecord, MappingResult, map_records
from src.normalization import normalize_description

__all__ = [
    "CandidateEvidence",
    "DemoResult",
    "MAX_CANDIDATE_EVIDENCE",
    "load_demo_catalog",
    "run_demo",
]


CANONICAL_ID_COLUMN = "ground_truth_material_id"
REQUIRED_CATALOG_COLUMNS = frozenset(
    {
        CANONICAL_ID_COLUMN,
        "category",
        "valve_type",
        "material",
        "pressure_class",
        "end_connection",
        "size_mm",
        "bearing_family",
        "bearing_inner_diameter_mm",
        "bearing_outer_diameter_mm",
        "bearing_width_mm",
        "pipe_material",
        "pipe_od_mm",
        "pipe_thickness_mm",
        "pipe_schedule",
        "pipe_end",
    }
)
MAX_CANDIDATE_EVIDENCE = 5


@dataclass(frozen=True)
class CandidateEvidence:
    """Bounded, primitive-only candidate data safe for a future UI payload."""

    canonical_material_id: str
    decision: str
    score: float
    explanation: str


@dataclass(frozen=True)
class DemoResult:
    """Single deterministic demo run and the evidence needed to present it."""

    legacy_material_code: str
    original_raw_description: str | None
    normalized_description: str
    normalization_transformations: tuple[str, ...]
    attributes: MaterialAttributes
    mapping_result: MappingResult
    candidate_evidence: tuple[CandidateEvidence, ...]


def _required_value(row: dict[str, str], field: str, canonical_id: str) -> str:
    """Read one mandatory canonical field without repairing malformed values."""

    value = row[field]
    if not value or value != value.strip():
        raise ValueError(f"Canonical material '{canonical_id}' has a blank or malformed '{field}' value.")
    return value


def _decimal_value(row: dict[str, str], field: str, canonical_id: str) -> Decimal:
    value = _required_value(row, field, canonical_id)
    try:
        return Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"Canonical material '{canonical_id}' has an invalid decimal '{field}' value: {value!r}.") from error


def _int_value(row: dict[str, str], field: str, canonical_id: str) -> int:
    value = _required_value(row, field, canonical_id)
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"Canonical material '{canonical_id}' has an invalid integer '{field}' value: {value!r}.") from error


def _attributes_from_row(row: dict[str, str], canonical_id: str) -> MaterialAttributes:
    """Build complete category-specific catalog attributes from the published schema."""

    category = _required_value(row, "category", canonical_id)
    if category == "Valve":
        return MaterialAttributes(
            category=category,
            valve_type=_required_value(row, "valve_type", canonical_id).casefold(),
            material=_required_value(row, "material", canonical_id).casefold(),
            size_mm=_decimal_value(row, "size_mm", canonical_id),
            pressure_class=_int_value(row, "pressure_class", canonical_id),
            connection=_required_value(row, "end_connection", canonical_id).casefold(),
        )
    if category == "Bearing":
        return MaterialAttributes(
            category=category,
            bearing_family=_required_value(row, "bearing_family", canonical_id).casefold(),
            dimensions=(
                _decimal_value(row, "bearing_inner_diameter_mm", canonical_id),
                _decimal_value(row, "bearing_outer_diameter_mm", canonical_id),
                _decimal_value(row, "bearing_width_mm", canonical_id),
            ),
            # The catalog schema stores all bearing dimensions in millimetres.
            dimension_unit_present=True,
        )
    if category == "Pipe":
        return MaterialAttributes(
            category=category,
            material=_required_value(row, "pipe_material", canonical_id).casefold(),
            od_mm=_decimal_value(row, "pipe_od_mm", canonical_id),
            thickness_mm=_decimal_value(row, "pipe_thickness_mm", canonical_id),
            schedule=_int_value(row, "pipe_schedule", canonical_id),
            end=_required_value(row, "pipe_end", canonical_id).casefold(),
        )
    raise ValueError(f"Canonical material '{canonical_id}' has unsupported category {category!r}.")


def load_demo_catalog(path: Path) -> tuple[CatalogRecord, ...]:
    """Load a validated immutable catalog from ``material_master.csv``.

    The fixture's ``ground_truth_material_id`` column is used only as the ID of
    a preloaded static catalog record.  It is never supplied to live input or
    used as a shortcut for a mapping decision; ``map_records`` still compares
    only extracted technical attributes through LEGO #4 and #5.
    """

    records_by_id: Dict[str, MaterialAttributes] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        missing_columns = REQUIRED_CATALOG_COLUMNS - fieldnames
        if missing_columns:
            raise ValueError("Catalog CSV is missing required columns: " + ", ".join(sorted(missing_columns)))

        for row_number, row in enumerate(reader, start=2):
            canonical_id = row[CANONICAL_ID_COLUMN]
            if not canonical_id or canonical_id != canonical_id.strip():
                raise ValueError(f"Catalog row {row_number} has a blank or malformed canonical material ID.")
            attributes = _attributes_from_row(row, canonical_id)
            existing = records_by_id.get(canonical_id)
            if existing is not None and existing != attributes:
                raise ValueError(f"Canonical material '{canonical_id}' has conflicting specifications.")
            records_by_id[canonical_id] = attributes

    if not records_by_id:
        raise ValueError("Catalog CSV contains no canonical material records.")
    return tuple(
        CatalogRecord(canonical_material_id=canonical_id, attributes=records_by_id[canonical_id])
        for canonical_id in sorted(records_by_id)
    )


def _candidate_evidence(candidates: tuple[CandidateInfo, ...]) -> tuple[CandidateEvidence, ...]:
    """Convert existing deterministically ranked candidates to primitive UI evidence."""

    return tuple(
        CandidateEvidence(
            canonical_material_id=candidate.canonical_material_id,
            decision=candidate.decision,
            score=candidate.score,
            explanation=candidate.explanation,
        )
        for candidate in candidates[:MAX_CANDIDATE_EVIDENCE]
    )


def run_demo(
    raw_description: str | None,
    catalog: tuple[CatalogRecord, ...],
    legacy_material_code: str = "DEMO-INPUT",
) -> DemoResult:
    """Run one raw description through the existing deterministic pipeline."""

    normalization = normalize_description(raw_description)
    extraction = extract_attributes(raw_description)
    # Whitespace-only input is intentionally passed as empty so LEGO #5 returns
    # its existing safe NEW_CANDIDATE outcome for missing descriptions.
    mapping_input = raw_description if normalization.normalized_text else ""
    mapping_result = map_records(
        (LegacyRecord(legacy_material_code, mapping_input),),
        catalog,
    )[0]
    return DemoResult(
        legacy_material_code=legacy_material_code,
        original_raw_description=raw_description,
        normalized_description=normalization.normalized_text,
        normalization_transformations=normalization.transformations,
        attributes=extraction.attributes,
        mapping_result=mapping_result,
        candidate_evidence=_candidate_evidence(mapping_result.all_candidates),
    )
