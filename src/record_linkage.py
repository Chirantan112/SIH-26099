"""Deterministic record linkage and similarity scoring for CPSE material attributes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.attribute_extraction import MaterialAttributes


@dataclass(frozen=True)
class LinkageResult:
    """Audit-ready result of a deterministic comparison between two material records."""

    decision: str  # "SAME", "DIFFERENT", or "UNCERTAIN"
    score: float  # Value between 0.0 and 1.0
    compared_fields: tuple[str, ...]  # Fields present in at least one record
    matching_fields: tuple[str, ...]  # Fields present and equal in both records
    conflicting_fields: tuple[str, ...]  # Fields present and different in both records
    unresolved_fields: tuple[str, ...]  # Fields present in one record and None in the other
    notes: tuple[str, ...]  # Explanations of decision-making steps and overrides


COMPARED_KEYS = (
    "category",
    "valve_type",
    "material",
    "size_mm",
    "pressure_class",
    "connection",
    "bearing_family",
    "dimensions",
    "dimension_unit_present",
    "od_mm",
    "thickness_mm",
    "schedule",
    "end",
)

REQUIRED_FIELDS = {
    "Valve": {"valve_type", "material", "size_mm", "pressure_class", "connection"},
    "Bearing": {"bearing_family", "dimensions", "dimension_unit_present"},
    "Pipe": {"material", "od_mm", "thickness_mm", "schedule", "end"},
}


def compare_records(a: MaterialAttributes, b: MaterialAttributes) -> LinkageResult:
    """Compare two MaterialAttributes records and determine equivalence.

    Any conflict in compared attributes immediately rules out a "SAME" match and
    results in "DIFFERENT", regardless of high similarity scores. If there are no
    conflicts, the decision is "SAME" if and only if all category-required fields
    are present and matching. Otherwise, the decision is "UNCERTAIN".
    """
    compared: list[str] = []
    matching: list[str] = []
    conflicting: list[str] = []
    unresolved: list[str] = []
    notes: list[str] = []

    # 1. Inspect all compared keys
    for key in COMPARED_KEYS:
        val_a = getattr(a, key, None)
        val_b = getattr(b, key, None)

        if val_a is not None or val_b is not None:
            compared.append(key)
            if val_a is not None and val_b is not None:
                if val_a == val_b:
                    matching.append(key)
                else:
                    conflicting.append(key)
            else:
                unresolved.append(key)

    # 2. Determine score: ratio of matching fields to compared fields
    if compared:
        score = len(matching) / len(compared)
    else:
        score = 0.0

    # 3. Decision Logic
    # 3a. Check for any hard technical conflict
    if conflicting:
        decision = "DIFFERENT"
        notes.append("Hard conflict detected in fields: " + ", ".join(sorted(conflicting)))
        if "category" in conflicting:
            notes.append(f"Category mismatch: {a.category} vs {b.category}")
    # 3b. Check if category is unresolved or missing
    elif a.category is None or b.category is None:
        decision = "UNCERTAIN"
        notes.append("Category is missing on one or both sides; cannot confirm match.")
        if unresolved:
            notes.append("Unresolved fields present: " + ", ".join(sorted(unresolved)))
    # 3c. Categories match (and are not None)
    else:
        category = a.category
        req_fields = REQUIRED_FIELDS.get(category, set())
        missing_reqs = req_fields - set(matching)

        if not missing_reqs:
            decision = "SAME"
            notes.append(f"All required fields for category '{category}' match.")
        else:
            decision = "UNCERTAIN"
            notes.append(
                f"Missing or unresolved required fields for '{category}': "
                + ", ".join(sorted(missing_reqs))
            )

    notes.append(f"Calculated similarity score: {score:.3f}")

    return LinkageResult(
        decision=decision,
        score=score,
        compared_fields=tuple(sorted(compared)),
        matching_fields=tuple(sorted(matching)),
        conflicting_fields=tuple(sorted(conflicting)),
        unresolved_fields=tuple(sorted(unresolved)),
        notes=tuple(notes),
    )
