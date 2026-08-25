"""Evaluation utilities for the material code harmonization project.

This module loads the synthetic master dataset (``data/demo/material_master.csv``) and
the evaluation pair list (``data/evaluation/material_pairs.csv``) and computes a
deterministic set of classification metrics.

The implementation deliberately uses only the Python standard library and the
already‑implemented LEGO modules:

* ``src.attribute_extraction.extract_attributes`` – parses a raw description into a
  ``MaterialAttributes`` instance.
* ``src.record_linkage.compare_records`` – performs a deterministic, auditable
  comparison of two ``MaterialAttributes`` objects and returns a ``LinkageResult``
  with a decision of ``"SAME"``, ``"DIFFERENT"`` or ``"UNCERTAIN"``.

UNCERTAIN predictions are **treated conservatively**: they are counted as a
mistake regardless of the ground‑truth label.  Concretely, an UNCERTAIN prediction
contributes to ``FN`` when the true label is ``SAME`` and to ``FP`` when the true
label is ``DIFFERENT``.  This policy is explicit and documented in the
``EvaluationResult`` docstring.

All calculations are pure functions; no randomness, I/O side‑effects or external
services are involved, guaranteeing reproducibility.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

from src.attribute_extraction import extract_attributes
from src.record_linkage import compare_records

__all__ = ["EvaluationResult", "evaluate_pairs"]


@dataclass(frozen=True)
class EvaluationResult:
    """Container for evaluation counts and derived metrics.

    Attributes
    ----------
    tp : int
        True positives – predicted ``SAME`` and ground‑truth label ``SAME``.
    tn : int
        True negatives – predicted ``DIFFERENT`` and ground‑truth label ``DIFFERENT``.
    fp : int
        False positives – predicted ``SAME`` (or ``UNCERTAIN``) while the true
        label is ``DIFFERENT``.
    fn : int
        False negatives – predicted ``DIFFERENT`` (or ``UNCERTAIN``) while the
        true label is ``SAME``.
    decision_counts : dict[str, int]
        Number of predictions for each ``LinkageResult.decision`` value
        (``"SAME"``, ``"DIFFERENT"``, ``"UNCERTAIN"``).
    accuracy : float
        Overall accuracy = (tp + tn) / total_pairs.
    precision : float
        Precision = tp / (tp + fp) (0 when denominator is 0).
    recall : float
        Recall = tp / (tp + fn) (0 when denominator is 0).
    f1 : float
        F1‑score, harmonic mean of precision and recall (0 when both are 0).

    Notes
    -----
    The ``decision_counts`` dictionary is ordered alphabetically for deterministic
    output, but the order is not semantically important.
    """

    tp: int
    tn: int
    fp: int
    fn: int
    decision_counts: Dict[str, int] = field(default_factory=dict)
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    def __post_init__(self) -> None:  # pragma: no cover – dataclass is frozen
        # Ensure the dict is immutable for callers.
        object.__setattr__(self, "decision_counts", dict(self.decision_counts))

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return the four confusion‑matrix counts as a tuple.

        Useful for quick equality checks in tests.
        """

        return (self.tp, self.tn, self.fp, self.fn)


def _load_master(master_path: Path) -> Dict[str, str]:
    """Return a mapping ``legacy_material_code -> raw_description``.

    Parameters
    ----------
    master_path:
        Path to ``material_master.csv`` (the synthetic master file).
    """

    mapping: Dict[str, str] = {}
    with master_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get("legacy_material_code")
            desc = row.get("raw_description")
            if code and desc:
                mapping[code] = desc
    return mapping


def evaluate_pairs(
    pairs_path: Path,
    master_path: Path = Path(__file__).resolve().parents[1] / "data" / "demo" / "material_master.csv",
) -> EvaluationResult:
    """Evaluate the ``material_pairs.csv`` dataset against the deterministic system.

    The function loads the master file to obtain raw descriptions, extracts
    attribute objects, compares the two records with ``compare_records`` and
    aggregates a classic binary‑classification confusion matrix.

    Parameters
    ----------
    pairs_path:
        Path to the evaluation CSV (``data/evaluation/material_pairs.csv``).
    master_path:
        Path to the master CSV containing raw descriptions.  Defaults to the
        repository's built-in demo file but can be overridden for unit tests.

    Returns
    -------
    EvaluationResult
        A frozen dataclass containing counts, decision frequencies and derived
        metrics.
    """

    # Load master dictionary once – O(N) where N is number of unique legacy codes.
    master_map = _load_master(master_path)

    # Initialise counters.
    tp = tn = fp = fn = 0
    decision_counts: Dict[str, int] = {"SAME": 0, "DIFFERENT": 0, "UNCERTAIN": 0}
    total = 0

    with pairs_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip completely empty rows (possible trailing newline).
            if not any(row.values()):
                continue
            total += 1
            code_a = row["record_a_code"]
            code_b = row["record_b_code"]
            true_label = row["label"].strip().upper()  # "SAME" or "DIFFERENT"

            # Retrieve raw descriptions – if missing we fall back to an empty string
            # which will be classified as NEW_CANDIDATE downstream; the decision will
            # be "DIFFERENT" and the pair will be counted as a mismatch.
            raw_a = master_map.get(code_a, "")
            raw_b = master_map.get(code_b, "")

            # Extract attributes using the deterministic extractor.
            attrs_a = extract_attributes(raw_a).attributes
            attrs_b = extract_attributes(raw_b).attributes

            # Compare via LEGO #4 deterministic linkage.
            linkage = compare_records(attrs_a, attrs_b)
            pred = linkage.decision  # "SAME", "DIFFERENT" or "UNCERTAIN"
            decision_counts[pred] = decision_counts.get(pred, 0) + 1

            # Update confusion matrix using a conservative policy for UNCERTAIN.
            if pred == "SAME":
                if true_label == "SAME":
                    tp += 1
                else:
                    fp += 1
            elif pred == "DIFFERENT":
                if true_label == "DIFFERENT":
                    tn += 1
                else:
                    fn += 1
            else:  # UNCERTAIN – treat as a mistake for both classes.
                if true_label == "SAME":
                    fn += 1
                else:
                    fp += 1

    # Derive metrics safely – avoid division by zero.
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return EvaluationResult(
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        decision_counts=decision_counts,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )
