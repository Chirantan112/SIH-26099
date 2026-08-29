"""Deterministic CPSE material-master onboarding and schema validation.

The onboarding layer converts common CPSE column names into the repository's
canonical input schema and reports data-quality issues before harmonization.
It is intentionally independent from the existing matching pipeline.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

CANONICAL_COLUMNS = (
    "cpse",
    "legacy_material_code",
    "raw_description",
)

_COLUMN_ALIASES = {
    "cpse": {
        "cpse", "company", "organization", "organisation", "cpse_name",
    },
    "legacy_material_code": {
        "legacy_material_code", "legacy_code", "material_code", "item_code",
        "item_number", "material_number", "material_no", "item_no", "code",
    },
    "raw_description": {
        "raw_description", "description", "material_description",
        "item_description", "material_desc", "item_desc", "short_description",
    },
}


@dataclass(frozen=True)
class ColumnMapping:
    """Deterministic source-to-canonical column mapping."""

    mapping: dict[str, str]
    unmapped_columns: tuple[str, ...]


@dataclass(frozen=True)
class DataQualityReport:
    """Quality summary produced before records enter harmonization."""

    total_rows: int
    valid_rows: int
    malformed_rows: int
    missing_cpse: int
    missing_codes: int
    missing_descriptions: int
    duplicate_codes: int
    unknown_columns: tuple[str, ...]
    canonical_columns_missing: tuple[str, ...]
    ready_for_harmonization: bool


@dataclass(frozen=True)
class OnboardingResult:
    """Mapped records plus deterministic schema and quality information."""

    records: tuple[dict[str, str], ...]
    column_mapping: ColumnMapping
    quality: DataQualityReport


def _normalize_column_name(name: str) -> str:
    return "_".join(name.strip().lower().replace("-", " ").split())


def map_columns(columns: Iterable[str]) -> ColumnMapping:
    """Map supported source column aliases to canonical names.

    Matching is case-insensitive and whitespace/hyphen tolerant. A source
    column is mapped at most once; ambiguous or unknown columns are reported.
    """
    normalized = tuple(columns)
    mapping: dict[str, str] = {}
    used_canonical: set[str] = set()
    unmapped: list[str] = []

    aliases = {
        canonical: {_normalize_column_name(alias) for alias in names}
        for canonical, names in _COLUMN_ALIASES.items()
    }
    for source in normalized:
        key = _normalize_column_name(source)
        matches = [canonical for canonical, names in aliases.items() if key in names]
        if len(matches) == 1 and matches[0] not in used_canonical:
            mapping[source] = matches[0]
            used_canonical.add(matches[0])
        else:
            unmapped.append(source)

    return ColumnMapping(mapping=mapping, unmapped_columns=tuple(unmapped))


def _mapped_rows(rows: Iterable[Mapping[str, object]], column_mapping: ColumnMapping) -> tuple[dict[str, str], ...]:
    reverse = column_mapping.mapping
    mapped: list[dict[str, str]] = []
    for row in rows:
        output = {column: "" for column in CANONICAL_COLUMNS}
        for source, canonical in reverse.items():
            value = row.get(source, "")
            output[canonical] = "" if value is None else str(value).strip()
        mapped.append(output)
    return tuple(mapped)


def validate_records(
    records: Iterable[Mapping[str, str]],
    column_mapping: ColumnMapping,
) -> DataQualityReport:
    """Validate canonical records without mutating them."""
    material_records = tuple(records)
    total = len(material_records)
    missing_columns = tuple(column for column in CANONICAL_COLUMNS if column not in column_mapping.mapping)
    missing_cpse = sum(not row.get("cpse", "") for row in material_records)
    missing_codes = sum(not row.get("legacy_material_code", "") for row in material_records)
    missing_descriptions = sum(not row.get("raw_description", "") for row in material_records)

    seen_codes: set[str] = set()
    duplicate_codes = 0
    for row in material_records:
        code = row.get("legacy_material_code", "")
        if code and code in seen_codes:
            duplicate_codes += 1
        elif code:
            seen_codes.add(code)

    malformed_rows = sum(
        1 for row in material_records
        if any(column not in row for column in CANONICAL_COLUMNS)
    )
    invalid_rows = {
        index
        for index, row in enumerate(material_records)
        if not row.get("legacy_material_code") or not row.get("raw_description") or any(
            column not in row for column in CANONICAL_COLUMNS
        )
    }
    valid_rows = total - len(invalid_rows)
    ready = (
        total > 0
        and not missing_columns
        and malformed_rows == 0
        and missing_codes == 0
        and missing_descriptions == 0
    )

    return DataQualityReport(
        total_rows=total,
        valid_rows=valid_rows,
        malformed_rows=malformed_rows,
        missing_cpse=missing_cpse,
        missing_codes=missing_codes,
        missing_descriptions=missing_descriptions,
        duplicate_codes=duplicate_codes,
        unknown_columns=column_mapping.unmapped_columns,
        canonical_columns_missing=missing_columns,
        ready_for_harmonization=ready,
    )


def onboard_rows(rows: Iterable[Mapping[str, object]]) -> OnboardingResult:
    """Map and validate in-memory rows from a CPSE material-master source."""
    rows_tuple = tuple(rows)
    if not rows_tuple:
        mapping = ColumnMapping({}, ())
        quality = validate_records((), mapping)
        return OnboardingResult((), mapping, quality)
    mapping = map_columns(rows_tuple[0].keys())
    records = _mapped_rows(rows_tuple, mapping)
    quality = validate_records(records, mapping)
    return OnboardingResult(records, mapping, quality)


def load_csv(path: Path) -> OnboardingResult:
    """Load a UTF-8 CSV and return mapped records plus quality information."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return onboard_rows(())
        return onboard_rows(reader)
