import time
from concurrent.futures import ThreadPoolExecutor

from data import get_pattern, get_self_help, self_help_context
from debug import debug_print, is_debug
from gate import compute_gate
from pattern_matching import match_pattern
from state import (
    SessionState,
    SLOT_ORDER,
    PATTERN_MATCH_SLOTS,
    SLOT_QUESTION_TEMPLATES,
    SLOT_KOREAN_LABELS,
    EXPLICIT_UNKNOWN_VALUE,
    askable_slots,
)
from llm import call_openai_json
import prompts
SELF_HELP_TOP_K = 3


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

        if target_slot == next_slot and speculative_question is not None:
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


def pattern_mapping_node(state: SessionState) -> dict:
    slots = state["slots"]
    if not any(slots[slot] for slot in PATTERN_MATCH_SLOTS):
        return {
            "pattern_candidates": [],
            "pattern_final": [],
        }

    thought_text = " ".join(slots["thought"]) or "(없음)"
    emotion_text = " ".join(slots["emotion"]) or "(없음)"
    behavior_text = " ".join(slots["behavior"]) or "(없음)"
    debug_print(f"[PATTERN MATCH DEBUG] thought={thought_text!r} emotion={emotion_text!r} behavior={behavior_text!r}")

    match_start = time.perf_counter()
    match = match_pattern(thought_text, emotion_text, behavior_text, verbose=is_debug())
    debug_print(f"[TIMING DEBUG] pattern match (embedding): {time.perf_counter() - match_start:.3f}s")
    debug_print(
        f"[PATTERN MATCH DEBUG] best match: {match['pattern_id']} (score={match['score']:.4f}): "
        f"{get_pattern(match['pattern_id'])['description']}"
    )

    return {
        "pattern_candidates": match["scores"],
        "pattern_final": [{"pattern_id": match["pattern_id"], "score": match["score"]}],
    }


def _format_self_help_candidate_block(self_help_ids: list[str]) -> str:
    blocks = []
    for sid in self_help_ids:
        item = get_self_help(sid)
        blocks.append(
            f"[{sid}]\n"
            f"제목: {item['title']}\n"
            f"의도: {item['intent']}\n"
            f"적용 상황: {', '.join(item['use_when'])}"
        )
    return "\n\n".join(blocks)


def _rerank_self_help(self_help_ids: list[str], slots) -> list[str]:
    if len(self_help_ids) <= SELF_HELP_TOP_K:
        return self_help_ids

    candidate_ids = set(self_help_ids)
    result = _timed_llm_call(
        "SELF_HELP_RERANK",
        prompts.SELF_HELP_RERANK_SYSTEM,
        prompts.SELF_HELP_RERANK_TASK.format(
            thought=" ".join(slots["thought"]) or "(없음)",
            emotion=" ".join(slots["emotion"]) or "(없음)",
            behavior=" ".join(slots["behavior"]) or "(없음)",
            coping=" ".join(slots["coping"]) or "(없음)",
            goal=" ".join(slots["goal"]) or "(없음)",
            candidate_block=_format_self_help_candidate_block(self_help_ids),
        ),
    )
    ranked = [
        r["self_help_id"] for r in result.get("ranked", [])
        if isinstance(r, dict) and r.get("self_help_id") in candidate_ids
    ]
    if not ranked:
        return self_help_ids[:SELF_HELP_TOP_K]
    return ranked[:SELF_HELP_TOP_K]


def self_help_node(state: SessionState) -> dict:
    self_help_ids: list[str] = []
    for match in state["pattern_final"]:
        for sid in get_pattern(match["pattern_id"])["self_help_ids"]:
            if sid not in self_help_ids:
                self_help_ids.append(sid)
    debug_print(f"[RAG DEBUG] self_help_ids candidates: {self_help_ids}")

    self_help_ids = _rerank_self_help(self_help_ids, state["slots"])
    debug_print(f"[RAG DEBUG] self_help_ids after rerank (top-{SELF_HELP_TOP_K}): {self_help_ids}")
    debug_print(f"[RAG DEBUG] self_help_context:\n{self_help_context(self_help_ids)}")
    return {"self_help_ids": self_help_ids}


def summary_node(state: SessionState) -> dict:
    if state.get("summary"):
        user_input = state.get("user_input") or ""
        log = list(state["conversation_log"])
        if user_input:
            log.append({"role": "user", "content": user_input})
        log.append({"role": "bot", "content": prompts.SESSION_CLOSED_MESSAGE})
        return {"bot_message": prompts.SESSION_CLOSED_MESSAGE, "conversation_log": log}

    slots = state["slots"]
    pattern_descriptions = "\n".join(
        get_pattern(p["pattern_id"])["description"] for p in state["pattern_final"]
    )
    debug_print(f"[RAG DEBUG] pattern_descriptions fed into summary prompt:\n{pattern_descriptions or '(없음)'}")
    debug_print(f"[RAG DEBUG] self_help_context fed into summary prompt:\n{self_help_context(state['self_help_ids']) or '(없음)'}")

    result = _timed_llm_call(
        "SUMMARY",
        prompts.SUMMARY_SYSTEM,
        prompts.SUMMARY_TASK.format(
            situation=" ".join(slots["situation"]) or "(없음)",
            thought=" ".join(slots["thought"]) or "(없음)",
            emotion=" ".join(slots["emotion"]) or "(없음)",
            cause=" ".join(slots["cause"]) or "(없음)",
            behavior=" ".join(slots["behavior"]) or "(없음)",
            impact=" ".join(slots["impact"]) or "(없음)",
            duration=" ".join(slots["duration"]) or "(없음)",
            coping=" ".join(slots["coping"]) or "(없음)",
            goal=" ".join(slots["goal"]) or "(없음)",
            pattern_descriptions=pattern_descriptions or "(없음)",
            self_help_context=self_help_context(state["self_help_ids"]),
        ),
        reasoning_effort="low",
    )
    summary = str(result.get("summary", "")).strip()

    return {
        "summary": summary,
        "bot_message": summary,
        "stage": "done",
        "conversation_log": _log(state, "bot", summary),
    }
