"""Create deterministic synthetic CPSE material-master and pair-label datasets."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

SEED = 26099
CPSES = ("CPCL", "IOCL", "NTPC")
CATEGORIES = ("Valve", "Bearing", "Pipe")
ROOT = Path(__file__).resolve().parents[1]

MASTER_FIELDS = ["cpse", "legacy_material_code", "raw_description", "category", "ground_truth_material_id", "valve_type", "material", "pressure_class", "end_connection", "size_mm", "bearing_family", "bearing_inner_diameter_mm", "bearing_outer_diameter_mm", "bearing_width_mm", "pipe_material", "pipe_od_mm", "pipe_thickness_mm", "pipe_schedule", "pipe_end"]
PAIR_FIELDS = ["pair_id", "record_a_code", "record_b_code", "record_a_cpse", "record_b_cpse", "category", "label", "difference_reason", "record_a_ground_truth_material_id", "record_b_ground_truth_material_id"]

# Each item is a complete canonical material specification.
SPECS = [
    {"id":"VAL-001","category":"Valve","valve_type":"Gate","material":"Carbon Steel","pressure_class":"150","end_connection":"Flanged","size_mm":"50"},
    {"id":"VAL-002","category":"Valve","valve_type":"Gate","material":"Carbon Steel","pressure_class":"300","end_connection":"Flanged","size_mm":"50"},
    {"id":"VAL-003","category":"Valve","valve_type":"Gate","material":"Carbon Steel","pressure_class":"150","end_connection":"Flanged","size_mm":"80"},
    {"id":"VAL-004","category":"Valve","valve_type":"Globe","material":"Stainless Steel","pressure_class":"150","end_connection":"Flanged","size_mm":"50"},
    {"id":"VAL-005","category":"Valve","valve_type":"Ball","material":"Stainless Steel","pressure_class":"300","end_connection":"Flanged","size_mm":"25"},
    {"id":"VAL-006","category":"Valve","valve_type":"Check","material":"Carbon Steel","pressure_class":"150","end_connection":"Flanged","size_mm":"100"},
    {"id":"VAL-007","category":"Valve","valve_type":"Ball","material":"Stainless Steel","pressure_class":"150","end_connection":"Threaded","size_mm":"25"},
    {"id":"VAL-008","category":"Valve","valve_type":"Butterfly","material":"Cast Iron","pressure_class":"150","end_connection":"Wafer","size_mm":"150"},
    {"id":"BRG-001","category":"Bearing","bearing_family":"Deep Groove Ball","bearing_inner_diameter_mm":"20","bearing_outer_diameter_mm":"47","bearing_width_mm":"14"},
    {"id":"BRG-002","category":"Bearing","bearing_family":"Deep Groove Ball","bearing_inner_diameter_mm":"25","bearing_outer_diameter_mm":"52","bearing_width_mm":"15"},
    {"id":"BRG-003","category":"Bearing","bearing_family":"Taper Roller","bearing_inner_diameter_mm":"30","bearing_outer_diameter_mm":"62","bearing_width_mm":"17"},
    {"id":"BRG-004","category":"Bearing","bearing_family":"Deep Groove Ball","bearing_inner_diameter_mm":"20","bearing_outer_diameter_mm":"47","bearing_width_mm":"12"},
    {"id":"BRG-005","category":"Bearing","bearing_family":"Spherical Roller","bearing_inner_diameter_mm":"50","bearing_outer_diameter_mm":"110","bearing_width_mm":"27"},
    {"id":"BRG-006","category":"Bearing","bearing_family":"Needle Roller","bearing_inner_diameter_mm":"25","bearing_outer_diameter_mm":"32","bearing_width_mm":"20"},
    {"id":"PIP-001","category":"Pipe","pipe_material":"Carbon Steel","pipe_od_mm":"60.3","pipe_thickness_mm":"3.91","pipe_schedule":"40","pipe_end":"Plain"},
    {"id":"PIP-002","category":"Pipe","pipe_material":"Carbon Steel","pipe_od_mm":"60.3","pipe_thickness_mm":"5.54","pipe_schedule":"80","pipe_end":"Plain"},
    {"id":"PIP-003","category":"Pipe","pipe_material":"Stainless Steel 304","pipe_od_mm":"60.3","pipe_thickness_mm":"3.91","pipe_schedule":"40","pipe_end":"Bevelled"},
    {"id":"PIP-004","category":"Pipe","pipe_material":"Carbon Steel","pipe_od_mm":"88.9","pipe_thickness_mm":"5.49","pipe_schedule":"40","pipe_end":"Plain"},
    {"id":"PIP-005","category":"Pipe","pipe_material":"Carbon Steel","pipe_od_mm":"114.3","pipe_thickness_mm":"6.02","pipe_schedule":"40","pipe_end":"Bevelled"},
    {"id":"PIP-006","category":"Pipe","pipe_material":"Stainless Steel 316","pipe_od_mm":"33.4","pipe_thickness_mm":"3.38","pipe_schedule":"40","pipe_end":"Plain"},
    {"id":"PIP-007","category":"Pipe","pipe_material":"Carbon Steel","pipe_od_mm":"48.3","pipe_thickness_mm":"3.68","pipe_schedule":"40","pipe_end":"Threaded"},
]
DUPLICATES = (("VAL-001","CPCL"),("VAL-004","IOCL"),("BRG-001","IOCL"),("BRG-004","NTPC"),("PIP-001","NTPC"),("PIP-003","CPCL"))


def empty_row() -> Dict[str, str]:
    return {field: "" for field in MASTER_FIELDS}


def valve_description(spec: Dict[str, str], rng: random.Random) -> str:
    short = {"Carbon Steel":"CS", "Stainless Steel":"SS", "Cast Iron":"CI"}[spec["material"]]
    end_short = {"Flanged":"FLG", "Threaded":"THD", "Wafer":"WFR"}[spec["end_connection"]]
    end_long = {"Flanged":"FLANGED", "Threaded":"THREADED", "Wafer":"WAFER"}[spec["end_connection"]]
    return rng.choice([
        f"{spec['valve_type'].upper()} VLV {short} CL{spec['pressure_class']} {end_short} {spec['size_mm']}MM",
        f"{spec['size_mm']} mm {spec['material']} {spec['valve_type'].lower()} valve, class {spec['pressure_class']} {end_long}",
        f"{short} {spec['valve_type'].upper()} VALVE {spec['size_mm']} MM {spec['pressure_class']}# {end_short}",
        f"VALVE-{spec['valve_type'].upper()}; {short}; {end_long}; {spec['size_mm']} mm; CL {spec['pressure_class']}",
    ])


def bearing_description(spec: Dict[str, str], rng: random.Random) -> str:
    dims = "x".join((spec["bearing_inner_diameter_mm"], spec["bearing_outer_diameter_mm"], spec["bearing_width_mm"]))
    family = spec["bearing_family"]
    return rng.choice([
        f"{family.upper()} BEARING {dims} MM", f"{dims} mm {family.lower()} BRG",
        f"BRG {family.upper()} {dims.replace('x', ' X ')}", f"{family.upper().replace('BALL', 'BLL')} BRG, {dims}mm",
    ])


def pipe_description(spec: Dict[str, str], rng: random.Random) -> str:
    short = {"Carbon Steel":"CS", "Stainless Steel 304":"SS 304", "Stainless Steel 316":"SS316"}[spec["pipe_material"]]
    od, thk, sch, end = spec["pipe_od_mm"], spec["pipe_thickness_mm"], spec["pipe_schedule"], spec["pipe_end"]
    end_short = {"Plain":"PE", "Bevelled":"BE", "Threaded":"THD"}[end]
    return rng.choice([
        f"PIPE {short} OD {od} MM THK {thk} MM SCH {sch} {end.upper()} END",
        f"{spec['pipe_material']} pipe {od}mm x {thk} mm, schedule {sch}, {end.upper()}",
        f"{short} PIPE {od} MM OD X {thk}MM SCH-{sch}", f"SCH {sch}; {short}; PIPE {od}mm/{thk}mm; {end_short}",
    ])


def description(spec: Dict[str, str], rng: random.Random) -> str:
    if spec["category"] == "Valve": return valve_description(spec, rng)
    if spec["category"] == "Bearing": return bearing_description(spec, rng)
    return pipe_description(spec, rng)


def record_for(spec: Dict[str, str], code: str, cpse: str, rng: random.Random) -> Dict[str, str]:
    row = empty_row()
    row.update({key: value for key, value in spec.items() if key != "id"})
    row.update({"cpse":cpse, "legacy_material_code":code, "raw_description":description(spec, rng), "ground_truth_material_id":spec["id"]})
    return row


def create_records(seed: int = SEED) -> List[Dict[str, str]]:
    rng, rows = random.Random(seed), []
    by_id = {spec["id"]: spec for spec in SPECS}
    for index, spec in enumerate(SPECS, 1):
        for cpse_index, cpse in enumerate(CPSES, 1):
            rows.append(record_for(spec, f"{cpse[:2]}-{spec['category'][:3].upper()}-{index:03d}-{cpse_index}", cpse, rng))
    for number, (material_id, cpse) in enumerate(DUPLICATES, 1):
        spec = by_id[material_id]
        rows.append(record_for(spec, f"{cpse[:2]}-{spec['category'][:3].upper()}-DUP-{number:02d}", cpse, rng))
    return rows


def pair_row(number: int, left: Dict[str, str], right: Dict[str, str], label: str, reason: str) -> Dict[str, str]:
    return {"pair_id":f"PAIR-{number:03d}", "record_a_code":left["legacy_material_code"], "record_b_code":right["legacy_material_code"], "record_a_cpse":left["cpse"], "record_b_cpse":right["cpse"], "category":left["category"], "label":label, "difference_reason":reason, "record_a_ground_truth_material_id":left["ground_truth_material_id"], "record_b_ground_truth_material_id":right["ground_truth_material_id"]}


def create_pairs(records: List[Dict[str, str]], seed: int = SEED) -> List[Dict[str, str]]:
    by_id: Dict[str, List[Dict[str, str]]] = {}
    for row in records: by_id.setdefault(row["ground_truth_material_id"], []).append(row)
    def pick(material_id: str, cpse: str) -> Dict[str, str]: return next(row for row in by_id[material_id] if row["cpse"] == cpse)
    pairs: List[Dict[str, str]] = []
    combinations: Sequence[tuple[str, str]] = (("CPCL","IOCL"), ("CPCL","NTPC"), ("IOCL","NTPC"))
    for index, spec in enumerate(SPECS):
        left_cpse, right_cpse = combinations[index % 3]
        pairs.append(pair_row(len(pairs)+1, pick(spec["id"], left_cpse), pick(spec["id"], right_cpse), "SAME", "same canonical material"))
    for material_id, cpse in DUPLICATES:
        group = [row for row in by_id[material_id] if row["cpse"] == cpse]
        pairs.append(pair_row(len(pairs)+1, group[0], group[1], "SAME", "same material; same-CPSE legacy duplicate"))
    negatives = [
        ("VAL-001","VAL-002","NTPC","NTPC","pressure class differs"), ("VAL-001","VAL-003","CPCL","CPCL","nominal valve size differs"),
        ("VAL-005","VAL-007","IOCL","NTPC","pressure class and end connection differ"), ("VAL-006","VAL-008","NTPC","CPCL","valve type, material, size, and end connection differ"),
        ("PIP-001","PIP-002","NTPC","NTPC","pipe wall thickness and schedule differ"), ("PIP-001","PIP-004","IOCL","CPCL","pipe outside diameter differs"),
        ("PIP-003","PIP-006","CPCL","NTPC","pipe material, diameter, and thickness differ"), ("BRG-001","BRG-004","IOCL","IOCL","bearing width differs"),
        ("BRG-001","BRG-002","IOCL","CPCL","bearing dimensions differ"), ("BRG-003","BRG-005","NTPC","IOCL","bearing family and dimensions differ"),
    ]
    for left_id, right_id, left_cpse, right_cpse, reason in negatives:
        pairs.append(pair_row(len(pairs)+1, pick(left_id,left_cpse), pick(right_id,right_cpse), "DIFFERENT", reason))
    return pairs


def write_csv(path: Path, rows: Iterable[Dict[str, str]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def generate(output_root: Path = ROOT, seed: int = SEED) -> tuple[Path, Path]:
    records = create_records(seed); pairs = create_pairs(records, seed)
    master_path, pair_path = output_root / "data" / "demo" / "material_master.csv", output_root / "data" / "evaluation" / "material_pairs.csv"
    write_csv(master_path, records, MASTER_FIELDS); write_csv(pair_path, pairs, PAIR_FIELDS)
    return master_path, pair_path


if __name__ == "__main__":
    master, evaluation = generate()
    print(f"Generated {len(create_records())} material records: {master}")
    print(f"Generated {len(create_pairs(create_records()))} evaluation pairs: {evaluation}")
