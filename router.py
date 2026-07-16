from langgraph.graph import END
from state import SessionState

STAGE_ENTRY_NODE = {"rapport": "rapport", "loop": "extract_and_detect", "done": "summary"}
ENTRY_MAP = {"rapport": "rapport", "extract_and_detect": "extract_and_detect", "summary": "summary"}
AFTER_RAPPORT_MAP = {END: END, "render_question": "render_question"}
AFTER_EXTRACT_MAP = {END: END, "render_question": "render_question"}
AFTER_GATE_MAP = {"consolidate_slots": "consolidate_slots", END: END}


def route_entry(state: SessionState) -> str:
    return STAGE_ENTRY_NODE[state["stage"]]


def route_after_rapport(state: SessionState) -> str:
    if state["stage"] == "rapport":
        return END
    return "render_question"


def route_after_extract(state: SessionState) -> str:
    if state["off_topic"]:
        return END
    return "render_question"


def route_after_gate(state: SessionState) -> str:
    if state["gate"]["coverage_ok"]:
        return "consolidate_slots"
    return END
