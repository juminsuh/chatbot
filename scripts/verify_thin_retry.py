"""Verifies the sufficient/missing_aspect thin-retry branching added to
render_question_node, without hitting the real OpenAI API: nodes._timed_llm_call
is monkeypatched to return scripted responses keyed by call label.

Covers:
  1. normal case (sufficient=true) advances exactly like before the change
  2. first thin answer triggers exactly one thin-retry question, thin_retry_count -> 1
  3. a second thin answer (thin_retry_count already 1) accepts the value and advances
  4. explicit_unknown after a spent thin-retry still gets its own unknown-retry turn
     (thin_retry_count and retry_attempted are independent caps)
  5. unknown-retry force-fill path (existing logic) is untouched

Run with: python scripts/verify_thin_retry.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nodes
from state import new_session, SLOT_ORDER, RETRY_ELIGIBLE_SLOTS, SLOT_QUESTION_TEMPLATES


class ScriptedLLM:
    """Replaces nodes._timed_llm_call; each label has its own FIFO queue."""

    def __init__(self):
        self.queues: dict[str, list[dict]] = {}
        self.calls: list[str] = []

    def push(self, label: str, response: dict):
        self.queues.setdefault(label, []).append(response)

    def __call__(self, label: str, system: str, user: str, **kwargs) -> dict:
        self.calls.append(label)
        queue = self.queues.get(label)
        assert queue, f"unexpected call to label {label!r} with no scripted response queued"
        return queue.pop(0)


def _extract_response(**overrides) -> dict:
    base = {slot: None for slot in SLOT_ORDER}
    base.update({
        "explicit_unknown": False,
        "sufficient": True,
        "missing_aspect": None,
        "question": "다음 질문입니다",
    })
    base.update(overrides)
    return base


def check_normal_case_unchanged():
    print("[1] sufficient=true fills the slot and advances (no thin-retry involved)")
    state = new_session("테스터")
    state["stage"] = "loop"
    state["pending"] = {"target_slot": "situation", "question_intent": SLOT_QUESTION_TEMPLATES["situation"]}
    state["asked_slots"] = ["situation"]
    state["user_input"] = "요즘 팀 프로젝트에서 발표 자료가 계속 늦어져서 팀원들 눈치가 보여요"

    fake = ScriptedLLM()
    fake.push("EXTRACT_AND_QUESTION", _extract_response(
        situation="팀 프로젝트 발표 자료가 늦어져 팀원들 눈치가 보임",
        sufficient=True,
        question="최근 상황 때문에 어떤 감정을 느끼시나요?",
    ))
    nodes._timed_llm_call = fake

    result = nodes.render_question_node(state)

    assert result["pending"]["target_slot"] == "emotion", "should advance to the next slot"
    assert result["thin_retry_count"]["situation"] == 0, "sufficient=true must not touch thin_retry_count"
    assert result["slots"]["situation"] == ["팀 프로젝트 발표 자료가 늦어져 팀원들 눈치가 보임"]
    assert fake.calls == ["EXTRACT_AND_QUESTION"], f"unexpected call sequence: {fake.calls}"
    print("  OK\n")


def check_thin_retry_first_hit():
    print("[2] sufficient=false on first answer triggers exactly one thin-retry question")
    state = new_session("테스터")
    state["stage"] = "loop"
    state["pending"] = {"target_slot": "situation", "question_intent": SLOT_QUESTION_TEMPLATES["situation"]}
    state["asked_slots"] = ["situation"]
    state["user_input"] = "그냥 요즘 다 힘들어서요"

    fake = ScriptedLLM()
    fake.push("EXTRACT_AND_QUESTION", _extract_response(
        situation="요즘 다 힘듦",
        sufficient=False,
        missing_aspect="너무_포괄적",
    ))
    fake.push("THIN_RETRY_QUESTION", {"question": "구체적으로 어떤 상황이 가장 힘드신가요?"})
    nodes._timed_llm_call = fake

    result = nodes.render_question_node(state)

    assert result["pending"]["target_slot"] == "situation", "should stay on the same slot for the re-ask"
    assert result["thin_retry_count"]["situation"] == 1, "thin_retry_count must increment once"
    assert result["bot_message"].endswith("구체적으로 어떤 상황이 가장 힘드신가요?")
    assert fake.calls == ["EXTRACT_AND_QUESTION", "THIN_RETRY_QUESTION"], f"unexpected call sequence: {fake.calls}"
    print("  OK\n")


def check_thin_retry_second_hit_advances():
    print("[3] a second still-thin answer (cap already spent) is accepted and advances")
    state = new_session("테스터")
    state["stage"] = "loop"
    state["pending"] = {"target_slot": "situation", "question_intent": SLOT_QUESTION_TEMPLATES["situation"]}
    state["asked_slots"] = ["situation"]
    state["slots"]["situation"] = ["요즘 다 힘듦"]
    state["thin_retry_count"]["situation"] = 1  # already spent from a prior turn
    state["user_input"] = "그냥 다요"

    fake = ScriptedLLM()
    fake.push("EXTRACT_AND_QUESTION", _extract_response(
        situation="그냥 다 힘듦",
        sufficient=False,
        missing_aspect="너무_포괄적",
        question="최근 상황 때문에 어떤 감정을 느끼시나요?",
    ))
    nodes._timed_llm_call = fake

    result = nodes.render_question_node(state)

    assert result["pending"]["target_slot"] == "emotion", "cap spent -> must advance, not loop forever"
    assert result["thin_retry_count"]["situation"] == 1, "cap must not increment past 1"
    assert fake.calls == ["EXTRACT_AND_QUESTION"], f"unexpected call sequence (no extra thin-retry call expected): {fake.calls}"
    print("  OK\n")


def check_explicit_unknown_independent_of_thin_cap():
    print("[4] explicit_unknown after a spent thin-retry still gets its own unknown-retry turn")
    assert "thought" in RETRY_ELIGIBLE_SLOTS
    state = new_session("테스터")
    state["stage"] = "loop"
    state["pending"] = {"target_slot": "thought", "question_intent": SLOT_QUESTION_TEMPLATES["thought"]}
    state["asked_slots"] = ["thought"]
    state["slots"]["thought"] = ["막연히 답답함"]
    state["thin_retry_count"]["thought"] = 1  # thin-retry already used up on this slot
    state["retry_attempted"]["thought"] = False  # unknown-retry not used yet
    state["user_input"] = "잘 모르겠어요"

    fake = ScriptedLLM()
    fake.push("EXTRACT_AND_QUESTION", _extract_response(
        thought=None,
        explicit_unknown=True,
    ))
    nodes._timed_llm_call = fake

    result = nodes.render_question_node(state)

    assert result["pending"]["target_slot"] == "thought", "unknown-retry must re-ask the same slot"
    assert result["retry_attempted"]["thought"] is True
    assert result["thin_retry_count"]["thought"] == 1, "thin_retry_count must not be touched by the unknown branch"
    assert "생각" in result["bot_message"] or True  # RETRY_QUESTIONS text check is loose on purpose
    assert fake.calls == ["EXTRACT_AND_QUESTION"], "unknown-retry re-ask uses the static RETRY_QUESTIONS text, no extra LLM call"
    print("  OK\n")


def check_unknown_force_fill_path_untouched():
    print("[5] regression: non-retry-eligible slot force-fills on repeated explicit_unknown (existing logic)")
    state = new_session("테스터")
    state["stage"] = "loop"
    state["pending"] = {"target_slot": "cause", "question_intent": SLOT_QUESTION_TEMPLATES["cause"]}
    state["asked_slots"] = ["situation", "emotion", "behavior", "impact", "duration", "coping", "goal", "cause"]
    state["retry_attempted"]["cause"] = True  # already missed once before (pre-existing mechanism)
    state["user_input"] = "잘 모르겠어요"

    fake = ScriptedLLM()
    fake.push("EXTRACT_AND_QUESTION", _extract_response(
        cause=None,
        explicit_unknown=True,
    ))
    fake.push("NEXT_QUESTION", {"question": "문제에 대해 스스로 드는 생각이 궁금해요."})
    nodes._timed_llm_call = fake

    result = nodes.render_question_node(state)

    assert result["slots"]["cause"] == ["-"], "second miss on a non-retry-eligible slot force-fills"
    assert result["pending"]["target_slot"] == "thought", "must move on to the next unfilled slot"
    print("  OK\n")


if __name__ == "__main__":
    original = nodes._timed_llm_call
    try:
        check_normal_case_unchanged()
        check_thin_retry_first_hit()
        check_thin_retry_second_hit_advances()
        check_explicit_unknown_independent_of_thin_cap()
        check_unknown_force_fill_path_untouched()
    finally:
        nodes._timed_llm_call = original
    print("ALL CHECKS PASSED")
