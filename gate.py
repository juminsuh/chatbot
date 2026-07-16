"""Coverage/gate computation. Pure function, no LLM or embedding calls."""
from state import GateState, SLOT_ORDER, SessionState


def compute_gate(state: SessionState) -> GateState:
    slots = state["slots"]
    coverage_ok = all(slots[s] for s in SLOT_ORDER)

    return GateState(
        coverage_ok=coverage_ok,
    )
