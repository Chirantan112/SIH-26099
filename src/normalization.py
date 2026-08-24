"""Safe deterministic text normalization for controlled CPSE descriptions."""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class NormalizationResult:
    original_text: str | None
    normalized_text: str
    transformations: tuple[str, ...]


TOKEN_REPLACEMENTS = {"CS": "carbon steel", "SS": "stainless steel", "MS": "mild steel", "CI": "cast iron", "VLV": "valve", "BRG": "bearing", "FLG": "flanged", "FLNGD": "flanged", "BLL": "ball", "SCH": "schedule", "CL": "class"}
VALVE_TYPES = ("gate", "globe", "ball", "check", "butterfly")
BEARING_FAMILIES = ("deep groove ball", "taper roller", "spherical roller", "needle roller")
PIPE_ENDS = ("plain", "bevelled", "threaded")
NUMBER_MM = r"(\d+(?:\.\d+)?)\s*mm\b"


def _changed(changes: list[str], rule: str, before: str, after: str) -> str:
    if before != after:
        changes.append(rule)
    return after


def _expand_tokens(text: str, changes: list[str]) -> str:
    updated = re.sub(r"\bss\s*316\b", "stainless steel 316", text)
    updated = re.sub(r"\bss\s*304\b", "stainless steel 304", updated)
    text = _changed(changes, "stainless-steel grade normalized", text, updated)
    for source, target in TOKEN_REPLACEMENTS.items():
        updated = re.sub(rf"\b{source.casefold()}\b", target, text)
        if updated != text:
            changes.append(f"token expanded: {source} -> {target}")
            text = updated
    return text


def _canonical_valve(text: str) -> str | None:
    if "valve" not in text:
        return None
    material = re.search(r"\b(carbon steel|stainless steel|mild steel|cast iron)\b", text)
    valve_type = next((v for v in VALVE_TYPES if re.search(rf"\b{v}\b", text)), None)
    size, pressure = re.search(NUMBER_MM, text), re.search(r"\bclass\s+(\d+)\b", text)
    connection = next((v for v in ("flanged", "threaded", "wafer") if re.search(rf"\b{v}\b", text)), None)
    if not all((material, valve_type, size, pressure, connection)):
        return None
    return f"valve {valve_type} {material.group(1)} size {size.group(1)} mm {connection} class {pressure.group(1)}"


def _canonical_bearing(text: str) -> str | None:
    if "bearing" not in text:
        return None
    family = next((v for v in BEARING_FAMILIES if v in text), None)
    dims = re.search(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)(\s*mm\b)?", text)
    if family is None or dims is None:
        return None
    unit = " mm" if dims.group(4) else ""
    return f"bearing {family} dimensions {dims.group(1)} x {dims.group(2)} x {dims.group(3)}{unit}"


def _canonical_pipe(text: str) -> str | None:
    if "pipe" not in text:
        return None
    material = re.search(r"\b(carbon steel|stainless steel(?:\s+(?:304|316))?|mild steel|cast iron)\b", text)
    schedule = re.search(r"\bschedule\s+(\d+)\b", text)
    if material is None or schedule is None:
        return None

    def labelled_value(labels: str) -> str | None:
        before = re.search(rf"\b(?:{labels})\s*{NUMBER_MM}", text)
        if before:
            return before.group(1)
        after = re.search(rf"{NUMBER_MM}\s*(?:{labels})\b", text)
        return after.group(1) if after else None

    od = labelled_value("od")
    thickness = labelled_value("thk|thickness")
    if od is None or thickness is None:
        # This controlled fixture form explicitly means "OD <value> x
        # thickness <value>".  It is deliberately limited to an OD-labelled
        # expression, rather than treating arbitrary two-number text as roles.
        od_first = re.search(rf"{NUMBER_MM}\s+od\s+x\s+{NUMBER_MM}", text)
        if od_first:
            od, thickness = od_first.group(1), od_first.group(2)
    if od is None or thickness is None:
        # The fixture's long pipe renderer is a documented ordered dimension
        # syntax: "pipe <material> <OD> mm x <thickness> mm".  Unknown text
        # does not enter this branch and therefore retains its original order.
        bare = re.search(rf"\bpipe\b.*?{NUMBER_MM}\s+x\s+{NUMBER_MM}", text)
        if bare:
            od, thickness = bare.group(1), bare.group(2)
    if od is None or thickness is None:
        return None

    pipe_end = next((v for v in PIPE_ENDS if re.search(rf"\b{v}\b", text)), None)
    result = f"pipe {material.group(1)} od {od} mm thk {thickness} mm schedule {schedule.group(1)}"
    return f"{result} end {pipe_end}" if pipe_end else result


def normalize_description(raw_text: str | None) -> NormalizationResult:
    """Normalize text only; unknown text retains its original token order.

    Unitless bearing tuples remain unitless. Millimetres are never inferred.
    """
    if raw_text is not None and not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string or None")
    if raw_text is None or not raw_text.strip():
        return NormalizationResult(raw_text, "", ("empty input",))
    changes: list[str] = []
    text = raw_text.casefold()
    _changed(changes, "case normalized", raw_text, text)
    text = _changed(changes, "list punctuation normalized", text, re.sub(r"[;,]+", " ", text))
    text = _changed(changes, "pressure class normalized", text, re.sub(r"\b(?:cl|class)\s*-?\s*(\d+)\b", r"class \1", text))
    text = _changed(changes, "pressure class normalized", text, re.sub(r"\b(\d+)\s*#(?!\w)", r"class \1", text))
    text = _changed(changes, "pipe schedule normalized", text, re.sub(r"\b(?:sch|schedule)-?\s*(\d+)\b", r"schedule \1", text))
    text = _changed(changes, "millimetre units normalized", text, re.sub(r"(?<=\d)\s*mm\b", " mm", text))
    text = _changed(changes, "dimension separators normalized", text, re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", " x ", text))
    text = _changed(changes, "numeric pipe slash normalized", text, re.sub(r"(\d+(?:\.\d+)?\s*mm)\s*/\s*(\d+(?:\.\d+)?\s*mm)", r"\1 x \2", text))
    text = _expand_tokens(text, changes)
    if "pipe" in text:
        text = _changed(changes, "pipe end abbreviation expanded", text, re.sub(r"\bpe\b", "plain", text))
        text = _changed(changes, "pipe end abbreviation expanded", text, re.sub(r"\bbe\b", "bevelled", text))
    text = _changed(changes, "connection abbreviation expanded", text, re.sub(r"\bthd\b", "threaded", text))
    text = _changed(changes, "connection abbreviation expanded", text, re.sub(r"\bwfr\b", "wafer", text))
    text = _changed(changes, "whitespace normalized", text, re.sub(r"\s+", " ", text).strip())
    canonical = _canonical_valve(text) or _canonical_bearing(text) or _canonical_pipe(text)
    if canonical is not None:
        text = _changed(changes, "known material format canonicalized", text, canonical)
    return NormalizationResult(raw_text, text, tuple(changes))
