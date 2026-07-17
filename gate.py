"""Coverage/gate computation. Pure function, no LLM or embedding calls."""
from state import GateState, SKIP_IF_FILLED_SLOTS, SLOT_ORDER, SessionState


def compute_gate(state: SessionState) -> GateState:
    slots = state["slots"]
    asked_slots = set(state["asked_slots"])
    # a slot only counts as covered by incidental content (never deliberately
    # asked) if it's one of the intentionally skip-if-filled slots -- for
    # everything else, opportunistic extraction from an unrelated turn must
    # not let the gate close before the slot was actually asked about
    coverage_ok = all(
        slots[s] and (s in SKIP_IF_FILLED_SLOTS or s in asked_slots)
        for s in SLOT_ORDER
    )

    return GateState(
        coverage_ok=coverage_ok,
    )
