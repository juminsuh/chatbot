import time
from concurrent.futures import ThreadPoolExecutor

from data import get_pattern, get_self_help, self_help_context
from gate import compute_gate
from pattern_matching import match_pattern
from state import (
    SessionState,
    SLOT_ORDER,
    PATTERN_MATCH_SLOTS,
    SLOT_QUESTION_TEMPLATES,
    SLOT_KOREAN_LABELS,
    RETRY_ELIGIBLE_SLOTS,
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
    print(f"[TIMING DEBUG] llm call '{label}': {elapsed:.3f}s")
    return result


def _log(state: SessionState, role: str, content: str, off_topic: bool | None = None) -> list[dict]:
    log = list(state["conversation_log"])
    entry = {"role": role, "content": content}
    if off_topic is not None:
        entry["off_topic"] = off_topic
    log.append(entry)
    return log


def _already_covered_label(slots) -> str:
    labels = [SLOT_KOREAN_LABELS[slot] for slot in SLOT_ORDER if slots[slot]]
    return ", ".join(labels) if labels else "(없음)"


def _situation_context(slots) -> str:
    parts = [text for slot in ("situation", "emotion", "behavior") for text in slots[slot]]
    return " ".join(parts) if parts else "(아직 파악된 내용 없음)"


def _detect_offtopic(situation_context: str, bot_question: str, user_utterance: str) -> tuple[bool, bool, bool]:
    result = _timed_llm_call(
        "OFFTOPIC_DETECT",
        prompts.OFFTOPIC_DETECT_SYSTEM,
        prompts.OFFTOPIC_DETECT_TASK.format(
            situation_context=situation_context,
            bot_question=bot_question,
            user_utterance=user_utterance,
        ),
    )
    return (
        bool(result.get("off_topic", False)),
        bool(result.get("resume_intent", False)),
        bool(result.get("explicit_unknown", False)),
    )


def _silent_extract(user_utterance: str, situation_context: str) -> dict:
    return _timed_llm_call(
        "SILENT_EXTRACT",
        prompts.SILENT_EXTRACT_SYSTEM,
        prompts.SILENT_EXTRACT_TASK.format(
            situation_context=situation_context,
            user_utterance=user_utterance,
        ),
    )


def _offtopic_reply(user_utterance: str) -> str:
    result = _timed_llm_call(
        "OFFTOPIC_REPLY",
        prompts.RESPONSE_SYSTEM,
        prompts.OFFTOPIC_REPLY_TASK.format(user_utterance=user_utterance),
    )
    return str(result.get("message", "")).strip()


def _reflect(bot_question: str, user_utterance: str) -> str:
    result = _timed_llm_call(
        "REFLECT",
        prompts.RESPONSE_SYSTEM,
        prompts.REFLECT_TASK.format(bot_question=bot_question, user_utterance=user_utterance),
    )
    return str(result.get("reflection", "")).strip()


def _generate_question(mode: str, **kwargs) -> str:
    if mode == "thin_retry":
        task = prompts.THIN_RETRY_QUESTION_TASK.format(**kwargs)
        label = "THIN_RETRY_QUESTION"
    else:
        task = prompts.NEXT_QUESTION_TASK.format(**kwargs)
        label = "NEXT_QUESTION"
    result = _timed_llm_call(label, prompts.QUESTION_PROMPT, task)
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

    if step == "who" or next_step == "who":
        prefix_text = ""
    else:
        prefix_text = _reflect(state.get("bot_message") or "", user_input)
    prefix = f"{prefix_text} " if prefix_text else ""
    log = _log(state, "user", user_input)

    if next_step is None:
        closing = f"{prefix}{prompts.RAPPORT_CLOSING}".strip()
        return {
            "stage": "loop",
            "offtopic_message": closing,
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
    retry_attempted = dict(state.get("retry_attempted") or {s: False for s in RETRY_ELIGIBLE_SLOTS})
    thin_retry_count = dict(state.get("thin_retry_count") or {s: 0 for s in SLOT_ORDER})
    pending = state.get("pending")

    speculative_question = None
    explicit_unknown = False
    sufficient = True
    missing_aspect = None
    user_utterance = ""

    if not state.get("already_extracted"):
        previous_question_intent = pending["question_intent"] if pending else "(없음)"
        target_slot_key = pending["target_slot"] if pending else "(없음)"

        if state["asked_slots"]:
            user_utterance = state.get("user_input") or ""
        else:
            user_utterance = ""

        result = _timed_llm_call(
            "EXTRACT_AND_QUESTION",
            prompts.QUESTION_PROMPT,
            prompts.EXTRACT_AND_QUESTION_TASK.format(
                previous_question_intent=previous_question_intent,
                target_slot_key=target_slot_key,
                user_utterance=user_utterance,
                slot_goal=next_question_intent,
                already_asked=already_covered,
            ),
        )
        explicit_unknown = bool(result.get("explicit_unknown", False))
        for slot in SLOT_ORDER:
            value = result.get(slot)
            if not value:
                continue
            text = str(value).strip()
            if text and text not in slots[slot]:
                slots[slot].append(text)
        speculative_question = str(result.get("question", "")).strip() or next_question_intent

    retry_slot = None
    thin_retry_needed = False
    target_slot_name = None
    if pending and not state.get("already_extracted"):
        target_slot_name = pending["target_slot"]
        gained_new_content = len(slots[target_slot_name]) > len(state["slots"][target_slot_name])
        if not gained_new_content:
            if target_slot_name in RETRY_ELIGIBLE_SLOTS:
                # emotion/behavior/thought matter too much to silently skip --
                # retry whether the miss was an explicit "don't know" or the
                # user's answer just drifted onto unrelated content instead.
                retry_slot = target_slot_name
            elif explicit_unknown or retry_attempted.get(target_slot_name, False):
                # explicit "don't know", or this slot already missed once before
                # (e.g. it cycled back around via the askable_slots fallback) --
                # stop asking and force-fill instead of looping on it forever.
                retry_slot = target_slot_name
            else:
                # first miss on a non-retry-eligible slot: let the conversation
                # move on naturally (no immediate re-ask), but remember the miss
                # so a second miss later gets force-filled instead of asked again.
                retry_attempted[target_slot_name] = True
        elif thin_retry_count.get(target_slot_name, 0) == 0:
            # content landed -- check whether it's specific enough before deciding
            # to move on. Separate LLM call, scoped to just this slot's category
            # table, so it only runs on the turn that actually needs it.
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
            print(
                f"[SLOT DEBUG] target_slot={target_slot_name} sufficient={sufficient} "
                f"missing_aspect={missing_aspect}"
            )
            if not sufficient:
                # one re-ask, separate cap from the explicit-unknown retry above
                # (thin_retry_count vs retry_attempted)
                thin_retry_needed = True

    print(
        f"[SLOT DEBUG] retry_slot={retry_slot} thin_retry_needed={thin_retry_needed} "
        f"thin_retry_count={thin_retry_count} retry_attempted={retry_attempted}"
    )

    if retry_slot and retry_slot in RETRY_ELIGIBLE_SLOTS and not retry_attempted.get(retry_slot, False):
        retry_attempted[retry_slot] = True
        target_slot = retry_slot
        question_intent = SLOT_QUESTION_TEMPLATES[retry_slot]
        question = prompts.RETRY_QUESTIONS[retry_slot]
    elif thin_retry_needed:
        thin_retry_count[target_slot_name] = thin_retry_count.get(target_slot_name, 0) + 1
        target_slot = target_slot_name
        question_intent = SLOT_QUESTION_TEMPLATES[target_slot_name]
        aspect_guidance = prompts.THIN_RETRY_ASPECT_GUIDANCE.get(
            missing_aspect, "구체적인 사실을 하나만 더 물으세요."
        )
        question = _generate_question(
            "thin_retry",
            slot_goal=question_intent,
            user_last_answer=user_utterance,
            missing_aspect=missing_aspect or "구체성 부족",
            aspect_guidance=aspect_guidance,
        ) or question_intent
    else:
        if retry_slot and EXPLICIT_UNKNOWN_VALUE not in slots[retry_slot]:
            slots[retry_slot].append(EXPLICIT_UNKNOWN_VALUE)

        # next_slot/next_question_intent were computed at the top of this
        # function from state["slots"], before this turn's fills (and any
        # force-fill above) landed. Normally that's still the right slot to
        # ask about next, but when askable_slots() was already exhausted
        # (every slot asked at least once) and the fallback happened to pick
        # the very slot we just force-filled, re-derive the true next gap so
        # we don't re-ask the same slot or, on a later turn, crash trying to
        # find a "first unfilled slot" that no longer exists.
        if retry_slot and next_slot == retry_slot:
            fresh_candidates = askable_slots(slots, state["asked_slots"])
            if fresh_candidates:
                target_slot = fresh_candidates[0]
            else:
                remaining = [s for s in SLOT_ORDER if not slots[s]]
                target_slot = remaining[0] if remaining else retry_slot
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

    prefix = state.get("offtopic_message") or ""
    message = f"{prefix} {question}".strip()

    asked_slots = state["asked_slots"]
    if target_slot not in asked_slots:
        asked_slots = asked_slots + [target_slot]

    return {
        "slots": slots,
        "bot_message": message,
        "pending": {"target_slot": target_slot, "question_intent": question_intent},
        "asked_slots": asked_slots,
        "offtopic_message": None,
        "already_extracted": False,
        "thin_retry_count": thin_retry_count,
        "retry_attempted": retry_attempted,
        "conversation_log": _log(state, "bot", message),
    }


def extract_and_detect_node(state: SessionState) -> dict:
    user_input = state.get("user_input") or ""
    bot_message = state.get("bot_message") or ""
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        offtopic_future = executor.submit(
            _detect_offtopic, _situation_context(state["slots"]), bot_message, user_input
        )
        reply_future = executor.submit(_offtopic_reply, user_input)
        reflect_future = executor.submit(_reflect, bot_message, user_input)
        off_topic, resume_intent, explicit_unknown = offtopic_future.result()
        speculative_reply = reply_future.result()
        speculative_reflect = reflect_future.result()

    print(f"[OFFTOPIC DEBUG] loop off_topic={off_topic} resume_intent={resume_intent} explicit_unknown={explicit_unknown}")
    original_off_topic = off_topic
    if resume_intent or explicit_unknown:
        off_topic = False

    slots = state["slots"]
    already_extracted = False
    if original_off_topic:
        slots = {slot: list(state["slots"][slot]) for slot in SLOT_ORDER}
        extracted = _silent_extract(user_input, _situation_context(state["slots"]))
        found_content = False
        for slot in SLOT_ORDER:
            value = extracted.get(slot)
            if not value:
                continue
            found_content = True
            text = str(value).strip()
            if text and text not in slots[slot]:
                slots[slot].append(text)

        if not off_topic:
            # resume_intent/explicit_unknown forced this turn back on-topic even
            # though the utterance itself carried no real content -- it's a meta
            # statement ("let's resume", "I don't know" about the off-topic reply),
            # not a genuine attempt at answering the pending slot. Mark it as already
            # handled so render_question_node doesn't score it as a failed answer and
            # burn the slot's one-shot retry (which would surface the canned
            # RETRY_QUESTIONS wording instead of a normal follow-up question).
            already_extracted = True
        elif found_content:
            print("[OFFTOPIC DEBUG] loop off_topic reclassified False (extraction found content)")
            off_topic = False
            already_extracted = True

    log = _log(state, "user", user_input, off_topic=off_topic)

    if off_topic:
        streak = state.get("offtopic_streak", 0) + 1

        reply = speculative_reply
        if streak % prompts.OFFTOPIC_STREAK_LIMIT == 0: # 3-turn마다 다시 상담으로 돌아오자는 멘트 출력
            message = f"{reply} {prompts.OFFTOPIC_CONFIRM_QUESTION}".strip()
        else:
            message = reply

        pending = state.get("pending")
        asked_slots = state["asked_slots"]
        if pending and pending["target_slot"] in asked_slots:
            asked_slots = [s for s in asked_slots if s != pending["target_slot"]]

        return {
            "slots": slots,
            "off_topic": True,
            "offtopic_streak": streak,
            "asked_slots": asked_slots,
            "bot_message": message,
            "turn_count": state["turn_count"] + 1,
            "conversation_log": log + [{"role": "bot", "content": message}],
        }

    if resume_intent:
        prefix = prompts.OFFTOPIC_RESUME_ACK
    else:
        prefix = speculative_reflect

    return {
        "slots": slots,
        "off_topic": False,
        "offtopic_streak": 0,
        "offtopic_message": prefix,
        "already_extracted": already_extracted,
        "turn_count": state["turn_count"] + 1,
        "conversation_log": log,
    }

def gate_check_node(state: SessionState) -> dict:
    return {"gate": compute_gate(state)}


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
        consolidated[slot] = [text] if text else list(slots[slot])
    print(f"[CONSOLIDATE DEBUG] before: {slots}")
    print(f"[CONSOLIDATE DEBUG] after: {consolidated}")
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
    print(f"[PATTERN MATCH DEBUG] thought={thought_text!r} emotion={emotion_text!r} behavior={behavior_text!r}")

    match_start = time.perf_counter()
    match = match_pattern(thought_text, emotion_text, behavior_text)
    print(f"[TIMING DEBUG] pattern match (embedding): {time.perf_counter() - match_start:.3f}s")
    print(
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
    print(f"[RAG DEBUG] self_help_ids candidates: {self_help_ids}")

    self_help_ids = _rerank_self_help(self_help_ids, state["slots"])
    print(f"[RAG DEBUG] self_help_ids after rerank (top-{SELF_HELP_TOP_K}): {self_help_ids}")
    print(f"[RAG DEBUG] self_help_context:\n{self_help_context(self_help_ids)}")
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
    print(f"[RAG DEBUG] pattern_descriptions fed into summary prompt:\n{pattern_descriptions or '(없음)'}")
    print(f"[RAG DEBUG] self_help_context fed into summary prompt:\n{self_help_context(state['self_help_ids']) or '(없음)'}")

    result = _timed_llm_call(
        "SUMMARY",
        prompts.SUMMARY_SYSTEM,
        prompts.SUMMARY_TASK.format(
            situation=" ".join(slots["situation"]) or "(없음)",
            thought=" ".join(slots["thought"]) or "(없음)",
            emotion=" ".join(slots["emotion"]) or "(없음)",
            behavior=" ".join(slots["behavior"]) or "(없음)",
            impact=" ".join(slots["impact"]) or "(없음)",
            duration=" ".join(slots["duration"]) or "(없음)",
            coping=" ".join(slots["coping"]) or "(없음)",
            goal=" ".join(slots["goal"]) or "(없음)",
            pattern_descriptions=pattern_descriptions or "(없음)",
            self_help_context=self_help_context(state["self_help_ids"]),
        ),
        temperature=0.5,
    )
    summary = str(result.get("summary", "")).strip()

    return {
        "summary": summary,
        "bot_message": summary,
        "stage": "done",
        "conversation_log": _log(state, "bot", summary),
    }
