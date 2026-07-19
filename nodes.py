import re
import time
from concurrent.futures import ThreadPoolExecutor

from data import VALUE_IDS, get_value, value_list_text
from debug import debug_print
from gate import compute_gate
from state import (
    SessionState,
    SLOT_ORDER,
    SLOT_QUESTION_TEMPLATES,
    SLOT_KOREAN_LABELS,
    EXPLICIT_UNKNOWN_VALUE,
    askable_slots,
)
from llm import call_openai_json
import prompts


def _timed_llm_call(label: str, system: str, user: str, **kwargs) -> dict:
    start = time.perf_counter()
    result = call_openai_json(system, user, **kwargs)
    elapsed = time.perf_counter() - start
    debug_print(f"[TIMING DEBUG] llm call '{label}': {elapsed:.3f}s")
    return result


def _log(state: SessionState, role: str, content: str) -> list[dict]:
    log = list(state["conversation_log"])
    log.append({"role": role, "content": content})
    return log


def _already_covered_label(slots) -> str:
    labels = [SLOT_KOREAN_LABELS[slot] for slot in SLOT_ORDER if slots[slot]]
    return ", ".join(labels) if labels else "(없음)"


def _reflect(bot_question: str, user_utterance: str) -> str:
    result = _timed_llm_call(
        "REFLECT",
        prompts.RESPONSE_SYSTEM,
        prompts.REFLECT_TASK.format(bot_question=bot_question, user_utterance=user_utterance),
    )
    return str(result.get("reflection", "")).strip()


def _generate_question(mode: str, **kwargs) -> str:
    if mode == "detail":
        task = prompts.DETAIL_QUESTION_TASK.format(**kwargs)
        system = prompts.DETAIL_QUESTION_SYSTEM
        label = "DETAIL_QUESTION"
    else:
        task = prompts.NEXT_QUESTION_TASK.format(**kwargs)
        system = prompts.QUESTION_PROMPT
        label = "NEXT_QUESTION"
    result = _timed_llm_call(label, system, task)
    return str(result.get("question", "")).strip()

_RAPPORT_STEPS = ["greeting", "mood", "how", "who"]


def _rapport_question(step: str, name: str) -> str:
    if step == "greeting":
        return prompts.RAPPORT_GREETING.format(name=name)
    if step == "mood":
        return prompts.RAPPORT_MOOD.format(name=name)
    if step == "how":
        return prompts.RAPPORT_HOW
    return prompts.RAPPORT_WHO


def rapport_node(state: SessionState) -> dict:
    """
    greeting -> mood -> how -> who -> loop
    """
    step = state.get("rapport_step", "greeting")
    user_input = state.get("user_input")

    if user_input is None:
        message = _rapport_question("greeting", state["name"])
        return {
            "bot_message": message,
            "rapport_step": "greeting",
            "conversation_log": _log(state, "bot", message),
        }

    next_index = _RAPPORT_STEPS.index(step) + 1
    next_step = _RAPPORT_STEPS[next_index] if next_index < len(_RAPPORT_STEPS) else None

    if step == "who":
        prefix_text = ""
    else:
        prefix_text = _reflect(state.get("bot_message") or "", user_input)
    prefix = f"{prefix_text} " if prefix_text else ""
    log = _log(state, "user", user_input)

    if next_step is None:
        closing = f"{prefix}{prompts.RAPPORT_CLOSING}".strip()
        return {
            "stage": "loop",
            "reflection_prefix": closing,
            "conversation_log": log,
        }

    message = f"{prefix}{_rapport_question(next_step, state['name'])}".strip()
    return {
        "bot_message": message,
        "rapport_step": next_step,
        "conversation_log": log + [{"role": "bot", "content": message}],
    }


def render_question_node(state: SessionState) -> dict:
    next_candidates = askable_slots(state["slots"], state["asked_slots"])
    if next_candidates:
        next_slot = next_candidates[0]
    else:
        next_slot = next(s for s in SLOT_ORDER if not state["slots"][s])
    next_question_intent = SLOT_QUESTION_TEMPLATES[next_slot]
    already_covered = _already_covered_label(state["slots"])

    slots = {slot: list(state["slots"][slot]) for slot in SLOT_ORDER}
    retry_count = dict(state.get("retry_count") or {s: 0 for s in SLOT_ORDER})
    pending = state.get("pending")

    speculative_question = None

    previous_question_intent = pending["question_intent"] if pending else "(없음)"
    target_slot_key = pending["target_slot"] if pending else "(없음)"

    user_utterance = (state.get("user_input") or "") if state["asked_slots"] else ""

    if state["asked_slots"]:
        extract_task = prompts.EXTRACT_TASK.format(
            previous_question_intent=previous_question_intent,
            target_slot_key=target_slot_key,
            user_utterance=user_utterance,
        )
        with ThreadPoolExecutor(max_workers=3) as executor:
            reflect_future = executor.submit(_reflect, state.get("bot_message") or "", user_utterance)
            extract_future = executor.submit(
                _timed_llm_call, "EXTRACT", prompts.QUESTION_PROMPT, extract_task
            )
            question_future = executor.submit(
                _generate_question, "next_slot", slot_goal=next_question_intent, already_asked=already_covered
            )
            reflection = reflect_future.result()
            result = extract_future.result()
            speculative_question = question_future.result() or next_question_intent
        turn_count = state["turn_count"] + 1
    else:
        speculative_question = _generate_question(
            "next_slot", slot_goal=next_question_intent, already_asked=already_covered
        ) or next_question_intent
        result = {}
        reflection = state.get("reflection_prefix") or ""
        turn_count = state["turn_count"]

    for slot in SLOT_ORDER:
        value = result.get(slot)
        if not value:
            continue
        text = str(value).strip()
        if text and text not in slots[slot]:
            slots[slot].append(text)
            
    needs_detail_question = False
    force_fill_slot = None
    missing_aspect = None
    target_slot_name = None
    if pending:
        target_slot_name = pending["target_slot"]
        gained_new_content = len(slots[target_slot_name]) > len(state["slots"][target_slot_name])

        if retry_count.get(target_slot_name, 0) == 0:
            if not gained_new_content:
                missing_aspect = "답변 없음"
                needs_detail_question = True
            else:
                slot_value = slots[target_slot_name][-1]
                suff_result = _timed_llm_call(
                    "SUFFICIENCY_CHECK",
                    prompts.SUFFICIENCY_CHECK_SYSTEM,
                    prompts.SUFFICIENCY_CHECK_TASK.format(
                        target_slot_key=target_slot_name,
                        slot_value=slot_value,
                        missing_aspect_categories=prompts.MISSING_ASPECT_CATEGORIES[target_slot_name],
                    ),
                )
                sufficient = bool(suff_result.get("sufficient", True))
                missing_aspect = suff_result.get("missing_aspect") or None
                debug_print(
                    f"[SLOT DEBUG] target_slot={target_slot_name} sufficient={sufficient} "
                    f"missing_aspect={missing_aspect}"
                )
                if not sufficient:
                    needs_detail_question = True
        elif not gained_new_content:
            # already used this slot's one re-ask and still got nothing --
            # stop asking and lock in "-"
            force_fill_slot = target_slot_name

    debug_print(
        f"[SLOT DEBUG] needs_detail_question={needs_detail_question} "
        f"force_fill_slot={force_fill_slot} retry_count={retry_count}"
    )

    if needs_detail_question:
        retry_count[target_slot_name] = retry_count.get(target_slot_name, 0) + 1
        target_slot = target_slot_name
        question_intent = SLOT_QUESTION_TEMPLATES[target_slot_name]
        aspect_guidance = prompts.DETAIL_GUIDANCE.get(
            missing_aspect, "구체적인 사실을 하나만 더 물으세요."
        )
        question = _generate_question(
            "detail",
            slot_goal=question_intent,
            user_last_answer=user_utterance,
            missing_aspect=missing_aspect or "구체성 부족",
            aspect_guidance=aspect_guidance,
        ) or question_intent
    else:
        if force_fill_slot and EXPLICIT_UNKNOWN_VALUE not in slots[force_fill_slot]:
            slots[force_fill_slot].append(EXPLICIT_UNKNOWN_VALUE)
        if force_fill_slot and next_slot == force_fill_slot:
            fresh_candidates = askable_slots(slots, state["asked_slots"])
            if fresh_candidates:
                target_slot = fresh_candidates[0]
            else:
                remaining = [s for s in SLOT_ORDER if not slots[s]]
                target_slot = remaining[0] if remaining else force_fill_slot
            question_intent = SLOT_QUESTION_TEMPLATES[target_slot]
        else:
            target_slot = next_slot
            question_intent = next_question_intent

        if (
            target_slot not in state["asked_slots"]
            and slots[target_slot]
            and retry_count.get(target_slot, 0) == 0
        ):
            # never asked about this slot directly, but it already has
            # content picked up incidentally while answering a different
            # slot's question -- treat that as Incomplete right away instead
            # of asking a plain generic question first
            retry_count[target_slot] = retry_count.get(target_slot, 0) + 1
            slot_value = slots[target_slot][-1]
            aspect_guidance = prompts.DETAIL_GUIDANCE["질문 전에 부수적으로 언급됨"]
            question = _generate_question(
                "detail",
                slot_goal=question_intent,
                user_last_answer=slot_value,
                missing_aspect="질문 전에 부수적으로 언급됨",
                aspect_guidance=aspect_guidance,
            ) or question_intent
        elif target_slot == next_slot and speculative_question is not None:
            question = speculative_question
        else:
            question = _generate_question(
                "next_slot",
                slot_goal=question_intent,
                already_asked=already_covered,
            ) or question_intent

    message = f"{reflection} {question}".strip()

    asked_slots = state["asked_slots"]
    if target_slot not in asked_slots:
        asked_slots = asked_slots + [target_slot]

    log = state["conversation_log"]
    if user_utterance:
        log = _log(state, "user", user_utterance)

    return {
        "slots": slots,
        "bot_message": message,
        "pending": {"target_slot": target_slot, "question_intent": question_intent},
        "asked_slots": asked_slots,
        "reflection_prefix": None,
        "turn_count": turn_count,
        "retry_count": retry_count,
        "conversation_log": log + [{"role": "bot", "content": message}],
    }

def gate_check_node(state: SessionState) -> dict:
    return {"gate": compute_gate(state)}


def _parse_value_selection(user_utterance: str) -> list[str] | None:
    seen: list[int] = []
    for token in re.findall(r"\d+", user_utterance):
        n = int(token)
        if 1 <= n <= len(VALUE_IDS) and n not in seen:
            seen.append(n)
    if len(seen) != 5:
        return None
    return [VALUE_IDS[n - 1] for n in seen]


def values_node(state: SessionState) -> dict:
    # gate_check can flip coverage_ok True and fall through to this node
    # within the same invoke() call that just processed the answer to the
    # *last loop slot* -- state["user_input"] at that point is still that
    # slot's answer, not a reply to the values question. Key off
    # values_prompted (only ever set True after this node has actually
    # paused and returned control to the user) instead of user_input's mere
    # presence, so that stale input is never mistaken for a values answer.
    if not state.get("values_prompted"):
        message = prompts.VALUES_INTRO.format(name=state["name"], value_list=value_list_text())
        return {
            "bot_message": message,
            "stage": "values",
            "values_prompted": True,
            "conversation_log": _log(state, "bot", message),
        }

    user_input = state.get("user_input") or ""
    log = _log(state, "user", user_input)
    selected = _parse_value_selection(user_input)

    if selected is None:
        message = prompts.VALUES_RETRY
        return {
            "bot_message": message,
            "stage": "values",
            "conversation_log": log + [{"role": "bot", "content": message}],
        }

    print("[선택된 가치]")
    for vid in selected:
        v = get_value(vid)
        print(f"  - {v['name_ko']} ({v['name_en']})")

    return {
        "selected_values": selected,
        "conversation_log": log,
    }


# placeholder-ish non-answers the consolidate model sometimes emits when a
# slot had no real content -- these must not be accepted as genuine content
_NULL_LIKE_VALUES = {"없음", "(없음)", "null", "none", "-"}


def consolidate_slots_node(state: SessionState) -> dict:
    slots = state["slots"]
    result = _timed_llm_call(
        "CONSOLIDATE",
        prompts.CONSOLIDATE_SYSTEM,
        prompts.CONSOLIDATE_TASK.format(
            **{slot: " ".join(slots[slot]) or "(없음)" for slot in SLOT_ORDER}
        ),
    )
    consolidated = {}
    for slot in SLOT_ORDER:
        text = str(result.get(slot) or "").strip()
        if text.lower() in _NULL_LIKE_VALUES:
            text = ""
        consolidated[slot] = [text] if text else list(slots[slot])
    debug_print(f"[CONSOLIDATE DEBUG] before: {slots}")
    debug_print(f"[CONSOLIDATE DEBUG] after: {consolidated}")

    print("[최종 슬롯 정리]")
    for slot in SLOT_ORDER:
        content = " / ".join(consolidated[slot]) if consolidated[slot] else "X"
        print(f"  - {SLOT_KOREAN_LABELS[slot]}({slot}): {content}")

    return {"slots": consolidated}


def _format_selected_values(value_ids: list[str]) -> str:
    lines = []
    for vid in value_ids:
        v = get_value(vid)
        lines.append(f"- {v['name_ko']} ({v['name_en']}): {v['definition']}")
    return "\n".join(lines)


def insight_report_node(state: SessionState) -> dict:
    if state.get("report"):
        user_input = state.get("user_input") or ""
        log = list(state["conversation_log"])
        if user_input:
            log.append({"role": "user", "content": user_input})
        log.append({"role": "bot", "content": prompts.SESSION_CLOSED_MESSAGE})
        return {"bot_message": prompts.SESSION_CLOSED_MESSAGE, "conversation_log": log}

    slots = state["slots"]

    result = _timed_llm_call(
        "INSIGHT_REPORT",
        prompts.INSIGHT_REPORT_SYSTEM,
        prompts.INSIGHT_REPORT_TASK.format(
            situation=" ".join(slots["situation"]) or "(없음)",
            thought=" ".join(slots["thought"]) or "(없음)",
            emotion=" ".join(slots["emotion"]) or "(없음)",
            cause=" ".join(slots["cause"]) or "(없음)",
            behavior=" ".join(slots["behavior"]) or "(없음)",
            impact=" ".join(slots["impact"]) or "(없음)",
            duration=" ".join(slots["duration"]) or "(없음)",
            relationship=" ".join(slots["relationship"]) or "(없음)",
            coping=" ".join(slots["coping"]) or "(없음)",
            goal=" ".join(slots["goal"]) or "(없음)",
            self_message=" ".join(slots["self_message"]) or "(없음)",
            values=_format_selected_values(state["selected_values"]),
        ),
        reasoning_effort="medium",
    )
    report = str(result.get("report", "")).strip()

    return {
        "report": report,
        "bot_message": report,
        "stage": "done",
        "conversation_log": _log(state, "bot", report),
    }
