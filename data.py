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


def get_self_help(self_help_id: str) -> dict:
    return SELF_HELP_CATALOG[self_help_id]


def self_help_candidates(self_help_ids: list[str]) -> list[dict]:
    return [
        {
            "self_help_id": sid,
            "title": SELF_HELP_CATALOG[sid]["title"],
            "intent": SELF_HELP_CATALOG[sid]["intent"],
            "user_facing_message": SELF_HELP_CATALOG[sid]["user_facing_message"],
            "suggested_steps": SELF_HELP_CATALOG[sid]["suggested_steps"],
        }
        for sid in self_help_ids
    ]


def self_help_context(self_help_ids: list[str]) -> str:
    blocks = []
    for c in self_help_candidates(self_help_ids):
        steps = "\n".join(f"- {s}" for s in c["suggested_steps"])
        blocks.append(f"[{c['title']}]\n의도: {c['intent']}\n메시지: {c['user_facing_message']}\n실천 방법: {steps}")
    return "\n\n".join(blocks)
