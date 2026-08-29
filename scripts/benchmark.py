"""Reproducible synthetic stress benchmark for SIH-26099.

The benchmark deliberately stays honest about its evidence boundary: it generates
CPSE-style description variants from the repository's canonical demo catalog.
It does not claim to represent production CPSE data.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --variants-per-material 50

The command reports:
- generated description count;
- deterministic exact-mapping rate;
- uncertain and wrong mapping rates;
- canonical-pair classification metrics;
- runtime and throughput.

All generated cases are deterministic; no random seed or external service is
required. This makes the benchmark suitable for local verification and CI.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.catalog_mapping import CatalogRecord, LegacyRecord, map_records
from src.demo_pipeline import load_demo_catalog
from src.record_linkage import compare_records


@dataclass(frozen=True)
class BenchmarkCase:
    expected_id: str
    description: str
    family: str


def _value(value: object) -> str:
    return str(value)


def _templates(record: CatalogRecord) -> tuple[str, ...]:
    a = record.attributes
    if a.category == "Valve":
        material = _value(a.material)
        valve = _value(a.valve_type)
        size = _value(a.size_mm)
        pressure = _value(a.pressure_class)
        connection = _value(a.connection)
        return (
            f"{material} {valve} valve {size} mm {connection} class {pressure}",
            f"{material} {valve} valve {size}MM {connection} CL{pressure}",
            f"{material} {valve} VLV {size}MM {connection} {pressure}#",
            f"{valve} valve {material} {connection} {size} mm class {pressure}",
            f"{material} {valve} VLV {size} MM {connection} CL-{pressure}",
            f"{material} {valve} valve {size}mm {connection} class-{pressure}",
            f"{material.upper()} {valve.upper()} VLV {size}MM FLG CL{pressure}",
            f"{material} {valve} valve; {size} mm; {connection}; class {pressure}",
        )
    if a.category == "Bearing":
        family = _value(a.bearing_family)
        d1, d2, width = a.dimensions or ("", "", "")
        unit = " mm" if a.dimension_unit_present else ""
        return (
            f"bearing {family} dimensions {d1} x {d2} x {width}{unit}",
            f"BRG {family} {d1}X{d2}X{width}{unit}",
            f"{family} bearing dimensions {d1} x {d2} x {width}{unit}",
            f"bearing {family}; dimensions {d1}x{d2}x{width}{unit}",
            f"BEARING {family.upper()} {d1} x {d2} x {width}{unit}",
        )
    if a.category == "Pipe":
        material = _value(a.material)
        od = _value(a.od_mm)
        thickness = _value(a.thickness_mm)
        schedule = _value(a.schedule)
        end = _value(a.end)
        return (
            f"pipe {material} OD {od} mm THK {thickness} mm SCH-{schedule} {end} end",
            f"{material} pipe {od}MM OD x {thickness}MM thickness schedule {schedule} {end}",
            f"PIPE {material.upper()} OD {od}MM THICKNESS {thickness}MM SCH{schedule} {end.upper()} END",
            f"pipe {material} {od} mm x {thickness} mm schedule {schedule} {end}",
            f"{material} pipe OD {od}mm / {thickness}mm SCH-{schedule} {end} end",
        )
    raise ValueError(f"Unsupported category: {a.category!r}")


def build_cases(catalog: tuple[CatalogRecord, ...], variants_per_material: int) -> tuple[BenchmarkCase, ...]:
    cases: list[BenchmarkCase] = []
    for record in catalog:
        templates = _templates(record)
        for index in range(variants_per_material):
            template = templates[index % len(templates)]
            cycle = index // len(templates)
            if cycle % 4 == 1:
                description = "  " + template.replace(" ", "  ") + "  "
                family = "spacing"
            elif cycle % 4 == 2:
                description = template.casefold()
                family = "case"
            elif cycle % 4 == 3:
                description = template.replace(",", ";")
                family = "punctuation"
            else:
                description = template
                family = "canonical-variant"
            cases.append(BenchmarkCase(record.canonical_material_id, description, family))
    return tuple(cases)


def evaluate_mapping(cases: tuple[BenchmarkCase, ...], catalog: tuple[CatalogRecord, ...]) -> dict[str, float | int]:
    start = time.perf_counter()
    results = map_records(
        (LegacyRecord(f"BENCH-{index:06d}", case.description) for index, case in enumerate(cases)),
        catalog,
    )
    elapsed = time.perf_counter() - start
    matched = uncertain = wrong = 0
    for case, result in zip(cases, results):
        if result.decision == "MATCHED" and result.canonical_material_id == case.expected_id:
            matched += 1
        elif result.decision == "UNCERTAIN":
            uncertain += 1
        else:
            wrong += 1
    total = len(cases)
    return {
        "cases": total,
        "matched": matched,
        "uncertain": uncertain,
        "wrong": wrong,
        "exact_mapping_rate": matched / total if total else 0.0,
        "uncertain_rate": uncertain / total if total else 0.0,
        "wrong_rate": wrong / total if total else 0.0,
        "seconds": elapsed,
        "throughput_per_second": total / elapsed if elapsed else float("inf"),
    }


def evaluate_pairs(catalog: tuple[CatalogRecord, ...]) -> dict[str, float | int]:
    tp = tn = fp = fn = 0
    same = different = uncertain = 0
    for left, right in itertools.combinations(catalog, 2):
        result = compare_records(left.attributes, right.attributes)
        if result.decision == "SAME":
            fp += 1
            same += 1
        elif result.decision == "DIFFERENT":
            tn += 1
            different += 1
        else:
            fn += 0
            uncertain += 1
            # For this all-distinct canonical pair benchmark, UNCERTAIN is not
            # a false negative; it is reported separately as unresolved.
    total = tp + tn + fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "pairs": len(tuple(itertools.combinations(catalog, 2))),
        "same_predictions": same,
        "different_predictions": different,
        "uncertain_predictions": uncertain,
        "false_positive_rate": fp / total if total else 0.0,
        "precision": precision,
        "accuracy_excluding_uncertain": accuracy,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants-per-material", type=int, default=40)
    args = parser.parse_args()
    if args.variants_per_material < 1:
        parser.error("--variants-per-material must be >= 1")

    catalog_path = ROOT / "data" / "demo" / "material_master.csv"
    catalog = load_demo_catalog(catalog_path)
    cases = build_cases(catalog, args.variants_per_material)
    mapping = evaluate_mapping(cases, catalog)
    pairs = evaluate_pairs(catalog)

    print("SIH-26099 synthetic stress benchmark")
    print("Evidence boundary: repository-owned synthetic CPSE-style data only")
    print(f"Canonical materials: {len(catalog)}")
    print(f"Generated descriptions: {mapping['cases']}")
    print(f"Variants/material: {args.variants_per_material}")
    print(f"Exact mapping rate: {mapping['exact_mapping_rate']:.2%}")
    print(f"Uncertain rate: {mapping['uncertain_rate']:.2%}")
    print(f"Wrong mapping rate: {mapping['wrong_rate']:.2%}")
    print(f"Runtime: {mapping['seconds']:.3f}s")
    print(f"Throughput: {mapping['throughput_per_second']:.1f} descriptions/s")
    print(f"Distinct canonical pairs: {pairs['pairs']}")
    print(f"Pair DIFFERENT predictions: {pairs['different_predictions']}")
    print(f"Pair UNCERTAIN predictions: {pairs['uncertain_predictions']}")
    print(f"Pair false-positive rate: {pairs['false_positive_rate']:.2%}")
    print(f"Pair accuracy excluding UNCERTAIN: {pairs['accuracy_excluding_uncertain']:.2%}")

    if mapping["wrong"]:
        print("\nBenchmark FAILED: at least one variant mapped to the wrong canonical material.")
        return 1
    print("\nBenchmark PASS: no generated variant produced a wrong canonical match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
