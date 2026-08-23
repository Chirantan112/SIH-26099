# Dataset design

`src/generate_dataset.py` creates all dataset artifacts with `SEED = 26099`. It uses only Python's standard library and deterministically produces the same logical rows and ordering for a fixed source, runtime, and seed.

## Terms

- **Canonical material:** a complete technical specification, identified by `ground_truth_material_id`.
- **Legacy material code:** a CPSE-local code attached to a source-style record. Multiple codes can map to one canonical material.
- **Ground truth:** the known canonical-material identity used to label this synthetic fixture.
- **SAME:** both records map to one canonical material. **DIFFERENT:** the records map to distinct canonical materials.

## Master data schema

All records include:

- `cpse`, `legacy_material_code`, `raw_description`, `category`, `ground_truth_material_id`
- category-specific fields: `valve_type`, `material`, `pressure_class`, `end_connection`, `size_mm`; `bearing_family`, `bearing_inner_diameter_mm`, `bearing_outer_diameter_mm`, `bearing_width_mm`; and `pipe_material`, `pipe_od_mm`, `pipe_thickness_mm`, `pipe_schedule`, `pipe_end`

Blank fields are intentional where an attribute is not applicable or is omitted from a legacy description. Ground truth remains complete because it is generated from the canonical material specification.

## Evaluation pairs

The evaluation file links two master rows by their legacy material codes and supplies a `label` of `SAME` or `DIFFERENT`. It includes CPCL↔IOCL, CPCL↔NTPC, and IOCL↔NTPC positives, same-CPSE duplicate positives, and hard negatives differing in pressure class, valve size, pipe wall thickness/schedule, or bearing dimensions.

## Description variation

Description renderers intentionally vary abbreviations, synonyms, word order, capitalization, punctuation, spacing, units, and optional details. A deterministic subset introduces industrial variants such as `FLANGED`/`FLNGD`, `BEARING`/`BRG`, and `BALL`/`BLL`. The data also includes closely related but distinct canonical specifications.

## Limitations

This is synthetic data designed for development and evaluation. It is not drawn from CPSE systems, does not represent real CPSE procurement data or statistics, and cannot establish real-world matching accuracy. Its modest size and controlled variation make it suitable for transparent fixture-based tests, not production validation.

This module deliberately does not interpret, normalize, or match these descriptions.
