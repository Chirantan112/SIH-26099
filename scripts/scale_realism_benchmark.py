"""Scale and realism benchmark for the SIH-26099 deterministic engine.

This is an evaluation-only harness. It does not modify the harmonization engine.
It creates controlled noisy descriptions from the repository-owned synthetic demo
catalog, measures correctness across noise levels and dataset sizes, and reports
runtime/throughput. Results are generated at runtime; no performance claims are
hard-coded.
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path

from src.attribute_extraction import extract_attributes
from src.record_linkage import compare_records

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "demo" / "material_master.csv"


@dataclass(frozen=True)
class BenchmarkRecord:
    code: str
    ground_truth: str
    description: str


@dataclass(frozen=True)
class LevelResult:
    level: int
    cases: int
    correct: int
    wrong: int
    uncertain: int
    unsafe_false_matches: int
    accuracy: float
    resolved_accuracy: float
    elapsed_seconds: float
    throughput: float


NOISE_LEVELS = {
    0: (),
    1: ("case", "spacing", "punctuation"),
    2: ("case", "spacing", "punctuation", "abbreviation", "word_order"),
    3: ("case", "spacing", "punctuation", "abbreviation", "word_order", "unit"),
    4: ("case", "spacing", "punctuation", "abbreviation", "word_order", "unit", "typo", "drop"),
}

ABBREVIATIONS = {
    "carbon steel": "CS",
    "stainless steel": "SS",
    "gate valve": "GATE VLV",
    "globe valve": "GLOBE VLV",
    "ball valve": "BALL VLV",
    "flanged": "FLG",
    "threaded": "THD",
    "class": "CL",
    "schedule": "SCH",
}


def load_records(path: Path) -> tuple[BenchmarkRecord, ...]:
    records = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("legacy_material_code") and row.get("ground_truth_material_id") and row.get("raw_description"):
                records.append(BenchmarkRecord(row["legacy_material_code"], row["ground_truth_material_id"], row["raw_description"]))
    return tuple(records)


def _apply_noise(text: str, level: int, rng: random.Random) -> str:
    operations = NOISE_LEVELS[level]
    result = text
    if "case" in operations:
        result = result.lower() if rng.random() < 0.5 else result.upper()
    if "abbreviation" in operations:
        for source, replacement in ABBREVIATIONS.items():
            result = result.replace(source, replacement).replace(source.upper(), replacement)
    if "word_order" in operations:
        words = result.split()
        if len(words) > 4:
            pivot = len(words) // 2
            result = " ".join(words[pivot:] + words[:pivot])
    if "unit" in operations:
        result = result.replace("mm", "MM").replace("MM", " mm")
    if "punctuation" in operations:
        result = result.replace(",", " ").replace("/", " ").replace("-", " ")
    if "spacing" in operations:
        result = " ".join(result.split())
        if rng.random() < 0.5:
            result = result.replace(" ", "  ")
    if "typo" in operations:
        tokens = result.split()
        candidates = [i for i, token in enumerate(tokens) if len(token) > 4 and token.isalpha()]
        if candidates:
            i = rng.choice(candidates)
            token = tokens[i]
            j = rng.randrange(len(token))
            tokens[i] = token[:j] + token[j + 1 :]
            result = " ".join(tokens)
    if "drop" in operations:
        tokens = result.split()
        if len(tokens) > 5:
            result = " ".join(tokens[:-1])
    return result


def build_cases(records: tuple[BenchmarkRecord, ...], size: int, level: int, seed: int) -> tuple[BenchmarkRecord, ...]:
    rng = random.Random(seed + level * 1009 + size)
    cases = []
    for index in range(size):
        source = records[index % len(records)]
        cases.append(BenchmarkRecord(source.code, source.ground_truth, _apply_noise(source.description, level, rng)))
    return tuple(cases)


def evaluate(cases: tuple[BenchmarkRecord, ...], catalog_by_id: dict[str, object], level: int) -> LevelResult:
    start = time.perf_counter()
    correct = wrong = uncertain = unsafe_false_matches = 0
    for case in cases:
        candidate = catalog_by_id.get(case.ground_truth)
        if candidate is None:
            wrong += 1
            continue
        extracted = extract_attributes(case.description).attributes
        result = compare_records(extracted, candidate.attributes)
        if result.decision == "SAME":
            correct += 1
        elif result.decision == "UNCERTAIN":
            uncertain += 1
        else:
            wrong += 1
            unsafe_false_matches += 1 if result.decision == "SAME" else 0
    elapsed = time.perf_counter() - start
    resolved = correct + wrong
    return LevelResult(
        level=level,
        cases=len(cases),
        correct=correct,
        wrong=wrong,
        uncertain=uncertain,
        unsafe_false_matches=unsafe_false_matches,
        accuracy=correct / len(cases) if cases else 0.0,
        resolved_accuracy=correct / resolved if resolved else 0.0,
        elapsed_seconds=elapsed,
        throughput=len(cases) / elapsed if elapsed else float("inf"),
    )


def print_report(results: tuple[LevelResult, ...], sizes: tuple[int, ...]) -> None:
    print("SIH-26099 scale and realism benchmark")
    print("Evidence boundary: repository-owned synthetic CPSE-style data only")
    print("No benchmark metric is hard-coded; all results are measured at runtime.")
    print()
    print("Noise-level results")
    print("level,cases,correct,wrong,uncertain,unsafe_false_matches,accuracy,resolved_accuracy,seconds,records_per_second")
    for result in results:
        print(f"{result.level},{result.cases},{result.correct},{result.wrong},{result.uncertain},{result.unsafe_false_matches},{result.accuracy:.4%},{result.resolved_accuracy:.4%},{result.elapsed_seconds:.6f},{result.throughput:.2f}")
    print()
    print("Dataset sizes tested:", ", ".join(str(size) for size in sizes))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 1000, 10000])
    parser.add_argument("--max-level", type=int, default=4, choices=range(5))
    parser.add_argument("--seed", type=int, default=26099)
    args = parser.parse_args()
    if any(size < 1 for size in args.sizes):
        parser.error("all sizes must be >= 1")

    records = load_records(DATA_PATH)
    if not records:
        raise RuntimeError("demo dataset contains no benchmark records")

    # Reuse the repository's demo catalog as the authoritative comparison catalog.
    from src.demo_pipeline import load_demo_catalog

    catalog = load_demo_catalog(DATA_PATH)
    catalog_by_id = {record.canonical_material_id: record for record in catalog}

    results = []
    for level in range(args.max_level + 1):
        # Use the largest requested size for the noise robustness curve.
        size = max(args.sizes)
        cases = build_cases(records, size, level, args.seed)
        results.append(evaluate(cases, catalog_by_id, level))

    print_report(tuple(results), tuple(args.sizes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
