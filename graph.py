import time

from langgraph.graph import END, START, StateGraph
from debug import debug_print
from state import SessionState
from nodes import (
    rapport_node,
    render_question_node,
    pattern_mapping_node,
    gate_check_node,
    consolidate_slots_node,
    self_help_node,
    summary_node,
)
from router import (
    route_entry,
    route_after_rapport,
    route_after_gate,
    ENTRY_MAP,
    AFTER_RAPPORT_MAP,
    AFTER_GATE_MAP,
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
    g.add_node("pattern_mapping", _timed("pattern_mapping", pattern_mapping_node))
    g.add_node("gate_check", _timed("gate_check", gate_check_node))
    g.add_node("consolidate_slots", _timed("consolidate_slots", consolidate_slots_node))
    g.add_node("self_help", _timed("self_help", self_help_node))
    g.add_node("summary", _timed("summary", summary_node))

    g.add_conditional_edges(START, route_entry, ENTRY_MAP)
    g.add_conditional_edges("rapport", route_after_rapport, AFTER_RAPPORT_MAP)

    g.add_edge("render_question", "gate_check")
    g.add_conditional_edges("gate_check", route_after_gate, AFTER_GATE_MAP)
    g.add_edge("consolidate_slots", "pattern_mapping")

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
