"""Robust evaluation for SIH-26099 harmonization decisions.

This evaluator complements the synthetic stress benchmark by constructing:
- positive SAME pairs from legacy records sharing a ground-truth canonical ID;
- hard-negative DIFFERENT pairs from different canonical IDs in the same category;
- Recall@1/@3/@5 evaluation for any advisory retrieval adapter.

The benchmark reports UNCERTAIN separately. An UNCERTAIN result is not counted as
an unsafe false match; it is an unresolved case that should be surfaced for review.
The dataset is repository-owned synthetic CPSE-style data and is not production
CPSE data.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai_retrieval import CandidateSuggestion, RetrievalAdapter
from src.attribute_extraction import MaterialAttributes, extract_attributes
from src.catalog_mapping import CatalogRecord
from src.record_linkage import compare_records
from src.demo_pipeline import load_demo_catalog


@dataclass(frozen=True)
class EvaluationPair:
    left_code: str
    right_code: str
    expected: str  # SAME or DIFFERENT
    category: str
    hardness: str


@dataclass(frozen=True)
class RobustEvaluation:
    pairs: int
    same_expected: int
    different_expected: int
    correct: int
    wrong: int
    uncertain: int
    precision: float
    recall: float
    f1: float
    resolved_accuracy: float
    uncertain_rate: float
    hard_negative_count: int
    hard_negative_rejections: int


def _record(code: str, description: str, category: str, ground_truth: str) -> tuple[str, MaterialAttributes]:
    return code, extract_attributes(description).attributes


def load_legacy_records(path: Path) -> tuple[tuple[str, MaterialAttributes, str], ...]:
    records: list[tuple[str, MaterialAttributes, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            attributes = extract_attributes(row["raw_description"]).attributes
            if attributes.category:
                records.append((row["legacy_material_code"], attributes, row["ground_truth_material_id"]))
    return tuple(records)


def _hardness(left: MaterialAttributes, right: MaterialAttributes) -> tuple[int, int]:
    result = compare_records(left, right)
    return (len(result.conflicting_fields), -len(result.matching_fields))


def build_evaluation_pairs(records: tuple[tuple[str, MaterialAttributes, str], ...], max_hard_negatives: int = 100) -> tuple[EvaluationPair, ...]:
    pairs: list[EvaluationPair] = []

    # Positives: records from different legacy codes that share the same
    # repository-owned ground-truth canonical material ID.
    grouped: dict[str, list[tuple[str, MaterialAttributes]]] = {}
    for code, attributes, ground_truth in records:
        grouped.setdefault(ground_truth, []).append((code, attributes))
    for ground_truth, items in sorted(grouped.items()):
        for (left_code, left), (right_code, right) in itertools.combinations(items, 2):
            if left.category == right.category:
                pairs.append(EvaluationPair(left_code, right_code, "SAME", str(left.category), "positive"))

    # Hard negatives: same-category records from different canonical IDs with
    # the fewest technical conflicts first. This intentionally targets near-miss
    # materials such as valve pressure-class or pipe thickness differences.
    negatives: list[tuple[tuple[int, int], EvaluationPair]] = []
    for (left_code, left, left_gt), (right_code, right, right_gt) in itertools.combinations(records, 2):
        if left_gt == right_gt or left.category != right.category or left.category is None:
            continue
        hardness = _hardness(left, right)
        negatives.append((hardness, EvaluationPair(left_code, right_code, "DIFFERENT", str(left.category), "hard-negative")))
    negatives.sort(key=lambda item: (item[0], item[1].left_code, item[1].right_code))
    pairs.extend(pair for _, pair in negatives[:max_hard_negatives])
    return tuple(pairs)


def evaluate_pairs(
    pairs: Iterable[EvaluationPair],
    records: tuple[tuple[str, MaterialAttributes, str], ...],
) -> RobustEvaluation:
    by_code = {code: attributes for code, attributes, _ in records}
    tp = tn = fp = fn = uncertain = 0
    hard_negative_count = hard_negative_rejections = 0

    for pair in pairs:
        result = compare_records(by_code[pair.left_code], by_code[pair.right_code])
        if pair.hardness == "hard-negative":
            hard_negative_count += 1
            if result.decision == "DIFFERENT":
                hard_negative_rejections += 1
        if result.decision == "UNCERTAIN":
            uncertain += 1
        elif pair.expected == "SAME" and result.decision == "SAME":
            tp += 1
        elif pair.expected == "DIFFERENT" and result.decision == "DIFFERENT":
            tn += 1
        elif pair.expected == "SAME":
            fn += 1
        else:
            fp += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    resolved = tp + tn + fp + fn
    return RobustEvaluation(
        pairs=tp + tn + fp + fn + uncertain,
        same_expected=tp + fn,
        different_expected=tn + fp,
        correct=tp + tn,
        wrong=fp + fn,
        uncertain=uncertain,
        precision=precision,
        recall=recall,
        f1=f1,
        resolved_accuracy=(tp + tn) / resolved if resolved else 0.0,
        uncertain_rate=uncertain / (resolved + uncertain) if resolved + uncertain else 0.0,
        hard_negative_count=hard_negative_count,
        hard_negative_rejections=hard_negative_rejections,
    )


def retrieval_recall_at_k(
    cases: Iterable[tuple[str, str, MaterialAttributes]],
    catalog: tuple[CatalogRecord, ...],
    adapter: RetrievalAdapter,
    k: int,
) -> float:
    cases = tuple(cases)
    if not cases:
        return 0.0
    hits = 0
    for expected_id, description, attributes in cases:
        normalized = extract_attributes(description).normalized_text
        suggestions = adapter.retrieve(description, normalized, attributes, catalog)
        ids = [suggestion.canonical_material_id for suggestion in suggestions[:k]]
        if expected_id in ids:
            hits += 1
    return hits / len(cases)


def _catalog_lookup(catalog: tuple[CatalogRecord, ...]) -> dict[str, CatalogRecord]:
    return {record.canonical_material_id: record for record in catalog}


def build_retrieval_cases(
    records: tuple[tuple[str, MaterialAttributes, str], ...],
    descriptions: dict[str, str],
) -> tuple[tuple[str, str, MaterialAttributes], ...]:
    cases = []
    for _, attributes, ground_truth in records:
        description = descriptions.get(ground_truth)
        if description and attributes.category:
            cases.append((ground_truth, description, attributes))
    return tuple(cases)


def print_report(result: RobustEvaluation, recall: dict[int, float] | None = None) -> None:
    print("SIH-26099 robust evaluation")
    print("Evidence boundary: repository-owned synthetic CPSE-style data only")
    print(f"Evaluation pairs: {result.pairs}")
    print(f"Expected SAME: {result.same_expected}")
    print(f"Expected DIFFERENT: {result.different_expected}")
    print(f"Correct resolved decisions: {result.correct}")
    print(f"Wrong resolved decisions: {result.wrong}")
    print(f"UNCERTAIN decisions: {result.uncertain}")
    print(f"Precision: {result.precision:.2%}")
    print(f"Recall: {result.recall:.2%}")
    print(f"F1: {result.f1:.2%}")
    print(f"Resolved accuracy: {result.resolved_accuracy:.2%}")
    print(f"UNCERTAIN rate: {result.uncertain_rate:.2%}")
    print(f"Hard negatives: {result.hard_negative_count}")
    print(f"Hard negatives correctly rejected: {result.hard_negative_rejections}")
    if recall:
        for k, value in sorted(recall.items()):
            print(f"Retrieval Recall@{k}: {value:.2%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-hard-negatives", type=int, default=100)
    args = parser.parse_args()
    if args.max_hard_negatives < 1:
        parser.error("--max-hard-negatives must be >= 1")

    data_path = ROOT / "data" / "demo" / "material_master.csv"
    records = load_legacy_records(data_path)
    pairs = build_evaluation_pairs(records, args.max_hard_negatives)
    result = evaluate_pairs(pairs, records)
    print_report(result)
    return 0 if result.wrong == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
