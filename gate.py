"""Coverage/gate computation. Pure function, no LLM or embedding calls."""
from state import GateState, REQUIRED_SLOTS, SessionState

FORCE_END_TURN_COUNT = 10


def compute_gate(state: SessionState) -> GateState:
    """early_summary_allowed only depends on coverage_ok, not on pattern-match
    confidence -- rerank_score is an LLM self-reported 0.0-1.0 (see
    prompts.PATTERN_RERANK_TASK), and it's not calibrated enough to gate a real
    decision on (observed hitting 1.0 with a matching dimension still empty, and
    clustering on round numbers turn to turn). self_help_node already takes
    whatever's in pattern_final unconditionally, so nothing downstream needs a
    confidence floor to behave safely."""
    slots = state["slots"]
    coverage_ok = all(slots[slot] for slot in REQUIRED_SLOTS)
    force_end = state["turn_count"] >= FORCE_END_TURN_COUNT

    return GateState(
        coverage_ok=coverage_ok,
        top1_score=state["gate"]["top1_score"],
        margin=state["gate"]["margin"],
        early_summary_allowed=coverage_ok,
        force_end=force_end,
    )
