import time

from langgraph.graph import END, START, StateGraph
from debug import debug_print
from state import SessionState
from nodes import (
    rapport_node,
    render_question_node,
    values_node,
    gate_check_node,
    consolidate_slots_node,
    insight_report_node,
)
from router import (
    route_entry,
    route_after_rapport,
    route_after_gate,
    route_after_values,
    ENTRY_MAP,
    AFTER_RAPPORT_MAP,
    AFTER_GATE_MAP,
    AFTER_VALUES_MAP,
)


def _timed(name, fn):
    """Wraps a node function to log its total wall time -- includes any LLM
    calls inside it, which are separately broken down by _timed_llm_call in
    nodes.py, so the two logs together show node-level vs call-level cost."""
    def wrapped(state):
        start = time.perf_counter()
        result = fn(state)
        elapsed = time.perf_counter() - start
        debug_print(f"[TIMING DEBUG] node '{name}': {elapsed:.3f}s")
        return result
    return wrapped


def build_graph():
    g = StateGraph(SessionState)

    g.add_node("rapport", _timed("rapport", rapport_node))
    g.add_node("render_question", _timed("render_question", render_question_node))
    g.add_node("values", _timed("values", values_node))
    g.add_node("gate_check", _timed("gate_check", gate_check_node))
    g.add_node("consolidate_slots", _timed("consolidate_slots", consolidate_slots_node))
    g.add_node("insight_report", _timed("insight_report", insight_report_node))

    g.add_conditional_edges(START, route_entry, ENTRY_MAP)
    g.add_conditional_edges("rapport", route_after_rapport, AFTER_RAPPORT_MAP)

    g.add_edge("render_question", "gate_check")
    g.add_conditional_edges("gate_check", route_after_gate, AFTER_GATE_MAP)
    g.add_conditional_edges("values", route_after_values, AFTER_VALUES_MAP)
    g.add_edge("consolidate_slots", "insight_report")
    g.add_edge("insight_report", END)

    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
