from langgraph.graph import END
from state import SessionState

STAGE_ENTRY_NODE = {
    "rapport": "rapport",
    "loop": "render_question",
    "values": "values",
    "done": "insight_report",
}
ENTRY_MAP = {
    "rapport": "rapport",
    "render_question": "render_question",
    "values": "values",
    "insight_report": "insight_report",
}
AFTER_RAPPORT_MAP = {END: END, "render_question": "render_question"}
AFTER_GATE_MAP = {"values": "values", END: END}
AFTER_VALUES_MAP = {"consolidate_slots": "consolidate_slots", END: END}


def route_entry(state: SessionState) -> str:
    return STAGE_ENTRY_NODE[state["stage"]]


def route_after_rapport(state: SessionState) -> str:
    if state["stage"] == "rapport":
        return END
    return "render_question"


def route_after_gate(state: SessionState) -> str:
    if state["gate"]["coverage_ok"]:
        return "values"
    return END


def route_after_values(state: SessionState) -> str:
    if len(state.get("selected_values") or []) == 5:
        return "consolidate_slots"
    return END
