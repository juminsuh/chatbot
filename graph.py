from langgraph.graph import END, START, StateGraph
from state import SessionState
from nodes import (
    rapport_node,
    render_question_node,
    extract_and_detect_node,
    offtopic_response_node,
    pattern_mapping_node,
    gate_check_node,
    supervisor_node,
    self_help_node,
    summary_node,
)
from router import (
    route_entry,
    route_after_rapport,
    route_after_extract,
    route_after_gate,
    route_after_supervisor,
    ENTRY_MAP,
    AFTER_RAPPORT_MAP,
    AFTER_EXTRACT_MAP,
    AFTER_GATE_MAP,
    AFTER_SUPERVISOR_MAP,
)


def build_graph():
    g = StateGraph(SessionState)

    g.add_node("rapport", rapport_node)
    g.add_node("render_question", render_question_node)
    g.add_node("extract_and_detect", extract_and_detect_node)
    g.add_node("offtopic_response", offtopic_response_node)
    g.add_node("pattern_mapping", pattern_mapping_node)
    g.add_node("gate_check", gate_check_node)
    g.add_node("supervisor", supervisor_node)
    g.add_node("self_help", self_help_node)
    g.add_node("summary", summary_node)

    g.add_conditional_edges(START, route_entry, ENTRY_MAP)
    g.add_conditional_edges("rapport", route_after_rapport, AFTER_RAPPORT_MAP)
    g.add_edge("render_question", END)

    g.add_conditional_edges("extract_and_detect", route_after_extract, AFTER_EXTRACT_MAP)
    g.add_edge("offtopic_response", "render_question")

    g.add_conditional_edges("gate_check", route_after_gate, AFTER_GATE_MAP)
    g.add_conditional_edges("supervisor", route_after_supervisor, AFTER_SUPERVISOR_MAP)

    # both loop-exit paths (force_end bypass, supervisor recommend_self_help/end)
    # converge here -- pattern mapping now runs once, right before it's needed,
    # instead of tentatively every turn
    g.add_edge("pattern_mapping", "self_help")
    g.add_edge("self_help", "summary")
    g.add_edge("summary", END)

    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
