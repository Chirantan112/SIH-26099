# Benchmark Protocol

## Purpose

The repository now includes a reproducible synthetic stress benchmark for the deterministic harmonization path. Its purpose is to expose normalization and matching regressions before an SIH demonstration.

## Evidence boundary

The benchmark is **not production CPSE data**. It derives controlled CPSE-style description variants from the repository-owned canonical demo catalog. No production accuracy, savings, or deployment performance should be inferred from the benchmark alone.

## Generated cases

For every canonical material, the benchmark creates multiple equivalent descriptions using controlled variations such as:

- case changes;
- repeated/irregular whitespace;
- punctuation changes;
- supported CPSE abbreviations;
- reordered descriptive wording;
- alternate schedule/class spellings;
- unit formatting variants;
- category-specific description templates.

The expected canonical material ID is known from the source catalog, so the benchmark can distinguish an exact match, an unresolved result, and a wrong match.

## Reported metrics

### Exact mapping rate

The percentage of generated equivalent descriptions that map to their expected canonical material ID.

### Uncertain rate

The percentage of generated descriptions for which the deterministic system refuses to select a canonical ID.

### Wrong mapping rate

The percentage of generated descriptions mapped to an incorrect canonical ID. The benchmark exits non-zero if this value is non-zero.

### Throughput

Generated descriptions processed per second on the machine running the benchmark. This is a local engineering measurement, not a production capacity guarantee.

### Canonical-pair safety check

Every pair of distinct canonical materials is compared through the deterministic record-linkage engine. The benchmark reports `DIFFERENT` and `UNCERTAIN` outcomes separately so unresolved evidence is not silently treated as a successful distinction.

## Running the benchmark

```powershell
python scripts/benchmark.py
python scripts/benchmark.py --variants-per-material 50
```

The default creates 40 variants per canonical material. CI runs the default workload on Python 3.11 and 3.12.

## Extending the benchmark

Future benchmark additions should prefer harder, explicitly labelled cases over simply increasing record count. Recommended additions include:

1. missing critical attributes;
2. contradictory technical specifications;
3. realistic cross-CPSE abbreviation dictionaries;
4. inch/mm conversion cases where the conversion rule is explicitly supported;
5. hard negative pairs with near-identical descriptions;
6. retrieval Recall@1/3/5 for Local NLP;
7. category-level precision, recall and F1;
8. persisted benchmark reports from a reviewed dataset version.

Any metric presented to SIH judges must be generated from a committed benchmark version and labelled according to its evidence boundary.
