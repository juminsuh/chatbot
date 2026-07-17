from langgraph.graph import END
from state import SessionState

STAGE_ENTRY_NODE = {"rapport": "rapport", "loop": "render_question", "done": "summary"}
ENTRY_MAP = {"rapport": "rapport", "render_question": "render_question", "summary": "summary"}
AFTER_RAPPORT_MAP = {END: END, "render_question": "render_question"}
AFTER_GATE_MAP = {"consolidate_slots": "consolidate_slots", END: END}


def route_entry(state: SessionState) -> str:
    return STAGE_ENTRY_NODE[state["stage"]]


def route_after_rapport(state: SessionState) -> str:
    if state["stage"] == "rapport":
        return END
    return "render_question"


def route_after_gate(state: SessionState) -> str:
    if state["gate"]["coverage_ok"]:
        return "consolidate_slots"
    return END
