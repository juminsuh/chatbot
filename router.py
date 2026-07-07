from langgraph.graph import END
from state import SessionState

STAGE_ENTRY_NODE = {"rapport": "rapport", "loop": "extract_and_detect", "done": "summary"}


def route_entry(state: SessionState) -> str:
    """Which node handles the incoming turn, based on the high-level stage."""
    return STAGE_ENTRY_NODE[state["stage"]]


def route_after_rapport(state: SessionState) -> str:
    if state["stage"] == "rapport":
        return END  # still on turn 0, awaiting the mood-check reply
    return "render_question"  # advanced to "loop" in the same invoke -> ask slot 1


def route_after_extract(state: SessionState) -> str:
    """Deterministic off-topic guard (pipeline step 4). gate_check no longer needs
    pattern_final -- early_summary_allowed/force_end only depend on slots/turn_count
    -- so pattern mapping doesn't need to run before it anymore."""
    if state["off_topic"]:
        return "offtopic_response"
    return "gate_check"


def route_after_gate(state: SessionState) -> str:
    """force_end bypasses the supervisor entirely (pipeline step 6). Either way,
    pattern mapping runs once here, right before it's actually needed."""
    if state["gate"]["force_end"]:
        return "pattern_mapping"
    return "supervisor"


def route_after_supervisor(state: SessionState) -> str:
    if state["supervisor_action"] == "ask_intake_question":
        return "render_question"
    return "pattern_mapping"  # recommend_self_help | end


ENTRY_MAP = {"rapport": "rapport", "extract_and_detect": "extract_and_detect", "summary": "summary"}
AFTER_RAPPORT_MAP = {END: END, "render_question": "render_question"}
AFTER_EXTRACT_MAP = {"offtopic_response": "offtopic_response", "gate_check": "gate_check"}
AFTER_GATE_MAP = {"supervisor": "supervisor", "pattern_mapping": "pattern_mapping"}
AFTER_SUPERVISOR_MAP = {"render_question": "render_question", "pattern_mapping": "pattern_mapping"}
