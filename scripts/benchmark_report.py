"""Generate a judge-facing markdown report from the existing synthetic benchmark.

The report contains only measured results from the repository-owned synthetic
catalog. It never calls an LLM or external service.

Usage:
    python scripts/benchmark_report.py --variants-per-material 40
    python scripts/benchmark_report.py --variants-per-material 40 --output docs/benchmark_results.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark import build_cases, evaluate_mapping, evaluate_pairs
from src.demo_pipeline import load_demo_catalog


def render_report(variants_per_material: int) -> str:
    catalog = load_demo_catalog(ROOT / "data" / "demo" / "material_master.csv")
    cases = build_cases(catalog, variants_per_material)
    mapping = evaluate_mapping(cases, catalog)
    pairs = evaluate_pairs(catalog)

    family_sections: list[str] = []
    for family in sorted({case.family for case in cases}):
        family_cases = tuple(case for case in cases if case.family == family)
        result = evaluate_mapping(family_cases, catalog)
        family_sections.append(
            f"| {family} | {result['cases']} | {result['exact_mapping_rate']:.2%} | "
            f"{result['uncertain_rate']:.2%} | {result['wrong_rate']:.2%} |"
        )

    return f"""# SIH-26099 Benchmark Results

> **Evidence boundary:** repository-owned synthetic CPSE-style data only. These measurements are regression/stress evidence, not production CPSE accuracy.

## Overall

| Metric | Measured result |
|---|---:|
| Canonical materials | {len(catalog)} |
| Generated descriptions | {mapping['cases']} |
| Variants per material | {variants_per_material} |
| Exact mapping rate | {mapping['exact_mapping_rate']:.2%} |
| UNCERTAIN rate | {mapping['uncertain_rate']:.2%} |
| Wrong mapping rate | {mapping['wrong_rate']:.2%} |
| Runtime | {mapping['seconds']:.3f}s |
| Throughput | {mapping['throughput_per_second']:.1f} descriptions/s |

## Variant-family breakdown

| Variant family | Cases | Exact mapping | UNCERTAIN | Wrong |
|---|---:|---:|---:|---:|
{chr(10).join(family_sections)}

## Canonical-pair safety

| Metric | Measured result |
|---|---:|
| Distinct canonical pairs | {pairs['pairs']} |
| DIFFERENT predictions | {pairs['different_predictions']} |
| UNCERTAIN predictions | {pairs['uncertain_predictions']} |
| Pair false-positive rate | {pairs['false_positive_rate']:.2%} |
| Pair accuracy excluding UNCERTAIN | {pairs['accuracy_excluding_uncertain']:.2%} |

## Interpretation

- A **wrong mapping** is treated as a benchmark failure for generated equivalent descriptions.
- `UNCERTAIN` is reported separately rather than being counted as a wrong match.
- Pair evaluation checks whether distinct canonical specifications are incorrectly treated as the same material.
- No Gemini, Local NLP, network call, or external credential is required to generate these measurements.

## Reproduce

```powershell
python scripts/benchmark_report.py --variants-per-material {variants_per_material}
```
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants-per-material", type=int, default=40)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.variants_per_material < 1:
        parser.error("--variants-per-material must be >= 1")
    report = render_report(args.variants_per_material)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
