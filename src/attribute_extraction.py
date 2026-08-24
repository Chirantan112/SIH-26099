"""Deterministic structured attribute extraction from LEGO #2 normalized text."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import re

from src.normalization import normalize_description


@dataclass(frozen=True)
class MaterialAttributes:
    """Category-specific technical values; unavailable values remain ``None``."""

    category: str | None = None
    valve_type: str | None = None
    material: str | None = None
    size_mm: Decimal | None = None
    pressure_class: int | None = None
    connection: str | None = None
    bearing_family: str | None = None
    dimensions: tuple[Decimal, Decimal, Decimal] | None = None
    dimension_unit_present: bool | None = None
    od_mm: Decimal | None = None
    thickness_mm: Decimal | None = None
    schedule: int | None = None
    end: str | None = None


@dataclass(frozen=True)
class ExtractionResult:
    """Source text, LEGO #2 output, extracted attributes, and audit notes."""

    original_text: str | None
    normalized_text: str
    attributes: MaterialAttributes
    extraction_notes: tuple[str, ...]


NUMBER = r"(\d+(?:\.\d+)?)"
MATERIAL = r"(carbon steel|stainless steel(?:\s+(?:304|316))?|mild steel|cast iron)"
VALVE_TYPES = ("gate", "globe", "ball", "check", "butterfly")
BEARING_FAMILIES = ("deep groove ball", "taper roller", "spherical roller", "needle roller")
CONNECTIONS = ("flanged", "threaded", "wafer")
PIPE_ENDS = ("plain", "bevelled", "threaded")


def _decimal_value(text: str, pattern: str) -> Decimal | None:
    match = re.search(pattern, text)
    return Decimal(match.group(1)) if match else None


def _word_value(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def _extract_valve(text: str) -> MaterialAttributes:
    valve_type = next((item for item in VALVE_TYPES if re.search(rf"^valve\s+{item}\b", text)), None)
    material = _word_value(text, rf"^valve\s+(?:{'|'.join(VALVE_TYPES)})\s+{MATERIAL}\b")
    connection = next((item for item in CONNECTIONS if re.search(rf"\b{item}\b", text)), None)
    return MaterialAttributes(
        category="Valve",
        valve_type=valve_type,
        material=material,
        size_mm=_decimal_value(text, rf"\bsize\s+{NUMBER}\s+mm\b"),
        pressure_class=(int(value) if (value := _word_value(text, r"\bclass\s+(\d+)\b")) else None),
        connection=connection,
    )


def _extract_bearing(text: str) -> MaterialAttributes:
    family = next((item for item in BEARING_FAMILIES if re.search(rf"^bearing\s+{re.escape(item)}\b", text)), None)
    dimensions = re.search(rf"\bdimensions\s+{NUMBER}\s+x\s+{NUMBER}\s+x\s+{NUMBER}(\s+mm\b)?", text)
    values = None
    unit_present = None
    if dimensions:
        values = tuple(Decimal(dimensions.group(index)) for index in (1, 2, 3))
        unit_present = bool(dimensions.group(4))
    return MaterialAttributes(category="Bearing", bearing_family=family, dimensions=values, dimension_unit_present=unit_present)


def _extract_pipe(text: str) -> MaterialAttributes:
    material = _word_value(text, rf"^pipe\s+{MATERIAL}\b")
    end = next((item for item in PIPE_ENDS if re.search(rf"\bend\s+{item}\b", text)), None)
    schedule = _word_value(text, r"\bschedule\s+(\d+)\b")
    return MaterialAttributes(
        category="Pipe",
        material=material,
        od_mm=_decimal_value(text, rf"\bod\s+{NUMBER}\s+mm\b"),
        thickness_mm=_decimal_value(text, rf"\bthk\s+{NUMBER}\s+mm\b"),
        schedule=int(schedule) if schedule else None,
        end=end,
    )


def extract_attributes(raw_text: str | None) -> ExtractionResult:
    """Extract only explicit, normalized LEGO #2 attributes from *raw_text*.

    This function does not compare records or infer omitted technical values.
    Unsupported and partial descriptions return an empty or partial immutable
    ``MaterialAttributes`` instance instead of raising for valid text input.
    """
    normalized = normalize_description(raw_text)
    text = normalized.normalized_text
    if text.startswith("valve "):
        attributes = _extract_valve(text)
    elif text.startswith("bearing "):
        attributes = _extract_bearing(text)
    elif text.startswith("pipe "):
        attributes = _extract_pipe(text)
    else:
        attributes = MaterialAttributes()
    notes = ["normalized with LEGO #2"]
    notes.append(f"category detected: {attributes.category}" if attributes.category else "category not detected")
    return ExtractionResult(raw_text, text, attributes, tuple(notes))
