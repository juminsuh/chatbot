"""Load data.json and expose lookup structures used across the pipeline."""
import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "map_data.json"

with open(DATA_PATH, encoding="utf-8") as f:
    _DATA = json.load(f)

VALUE_CATALOG: dict[str, dict] = _DATA["value_catalog"]

# stable ordering, 1:1 with each value's "number" field (1-27)
VALUE_IDS: list[str] = sorted(VALUE_CATALOG, key=lambda vid: VALUE_CATALOG[vid]["number"])


def get_value(value_id: str) -> dict:
    return VALUE_CATALOG[value_id]


def value_list_text() -> str:
    lines = []
    for vid in VALUE_IDS:
        v = VALUE_CATALOG[vid]
        lines.append(f"{v['number']}. {v['name_ko']} ({v['name_en']}) - {v['definition']}")
    return "\n".join(lines)
