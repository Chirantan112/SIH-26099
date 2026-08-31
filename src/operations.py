"""Operational helpers for the SIH-26099 demonstration workflow.

This module adds batch processing and review-queue presentation helpers without
changing the existing deterministic matcher or any LLM adapter. Batch mode is
intentionally deterministic-only so large runs never create unexpected LLM/API
traffic.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

from src.catalog_mapping import CatalogRecord, MappingResult
from src.demo_pipeline import DemoResult, run_demo


@dataclass(frozen=True)
class BatchResult:
    total: int
    matched: int
    uncertain: int
    new_candidate: int
    elapsed_seconds: float
    rows: tuple[DemoResult, ...]

    @property
    def throughput_per_second(self) -> float:
        return self.total / self.elapsed_seconds if self.elapsed_seconds else float("inf")

    @property
    def review_required(self) -> int:
        return self.uncertain + self.new_candidate


@dataclass(frozen=True)
class ReviewItem:
    input_description: str
    decision: str
    material_id: str | None
    reason: str


def process_batch(
    descriptions: Iterable[str],
    catalog: tuple[CatalogRecord, ...],
    code_prefix: str = "BATCH",
) -> BatchResult:
    """Process descriptions through the existing deterministic demo pipeline.

    No Local NLP or Gemini adapter is constructed or invoked here. This makes
    batch processing safe for demos, CI, and large synthetic runs.
    """
    values = tuple(descriptions)
    started = perf_counter()
    rows = tuple(
        run_demo(description, catalog, legacy_material_code=f"{code_prefix}-{index:06d}")
        for index, description in enumerate(values, start=1)
    )
    elapsed = perf_counter() - started
    counts = {"MATCHED": 0, "UNCERTAIN": 0, "NEW_CANDIDATE": 0}
    for row in rows:
        counts[row.mapping_result.decision] = counts.get(row.mapping_result.decision, 0) + 1
    return BatchResult(
        total=len(rows),
        matched=counts.get("MATCHED", 0),
        uncertain=counts.get("UNCERTAIN", 0),
        new_candidate=counts.get("NEW_CANDIDATE", 0),
        elapsed_seconds=elapsed,
        rows=rows,
    )


def review_item(result: DemoResult) -> ReviewItem:
    """Convert a deterministic result into a judge-friendly review item."""
    mapping: MappingResult = result.mapping_result
    return ReviewItem(
        input_description=result.original_raw_description or "",
        decision=mapping.decision,
        material_id=mapping.canonical_material_id,
        reason=mapping.explanation,
    )
