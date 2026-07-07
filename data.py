"""Load data.json and expose lookup structures used across the pipeline."""
import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "map_data.json"

with open(DATA_PATH, encoding="utf-8") as f:
    _DATA = json.load(f)

AREA_LABELS: dict[str, str] = _DATA["area_labels"]
PATTERN_CATALOG: dict[str, dict] = _DATA["pattern_catalog"]
SELF_HELP_CATALOG: dict[str, dict] = _DATA["self_help_catalog"]

# stable ordering for the FAISS index: position i in the index <-> PATTERN_IDS[i]
PATTERN_IDS: list[str] = list(PATTERN_CATALOG.keys())
PATTERN_QUERY_TEXTS: list[str] = [PATTERN_CATALOG[pid]["pattern_query_text"] for pid in PATTERN_IDS]


def get_pattern(pattern_id: str) -> dict:
    return PATTERN_CATALOG[pattern_id]


def self_help_context(self_help_ids: list[str]) -> str:
    """user_facing_message + suggested_steps for the summary stage."""
    blocks = []
    for sid in self_help_ids:
        item = SELF_HELP_CATALOG[sid]
        steps = "\n".join(f"- {s}" for s in item["suggested_steps"])
        blocks.append(f"[{item['title']}]\n{item['user_facing_message']}\n{steps}")
    return "\n\n".join(blocks)
