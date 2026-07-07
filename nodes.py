"""LangGraph node functions, one per pipeline stage.

Bodies not yet covered by an implementation step are stubs -- this module is being
rebuilt incrementally, node by node. Each stub notes which step fills it in.
"""
from data import get_pattern, self_help_context
from gate import compute_gate
from retrieval import top_k_candidates
from state import SessionState, SLOT_ORDER, PATTERN_MATCH_SLOTS
from llm import call_openai_json, chat_qwen
import prompts


def _log(state: SessionState, role: str, content: str) -> list[dict]:
    log = list(state["conversation_log"])
    log.append({"role": role, "content": content})
    return log


def _already_covered_label(slots) -> str:
    labels = [prompts.SLOT_KOREAN_LABELS[slot] for slot in SLOT_ORDER if slots[slot]]
    return ", ".join(labels) if labels else "(없음)"


def rapport_node(state: SessionState) -> dict:
    """Fixed greeting (turn 0) + mood check reply hand-off (turn 1), no LLM call."""
    if not state.get("user_input"):
        message = f"{prompts.RAPPORT_GREETING} {prompts.RAPPORT_MOOD_CHECK}"
        return {
            "bot_message": message,
            "conversation_log": _log(state, "bot", message),
        }

    # user just answered the mood check -> hand off to the loop stage.
    # render_question_node runs next in this same invoke and asks the first slot
    # question, weaving this reply in as reflection context, so no bot_message here.
    return {
        "stage": "loop",
        "conversation_log": _log(state, "user", state["user_input"]),
    }


def render_question_node(state: SessionState) -> dict:
    """Qwen: reflect + turn a target_slot/question_intent into one Korean question.
    Also resumes a pending question after an off-topic detour, combining it with
    offtopic_message when present -- no fresh reflection in that case, since the
    last user utterance was off-topic, not a real answer."""
    pending = state.get("pending") or {
        "target_slot": "situation",
        "question_intent": prompts.SLOT_QUESTION_TEMPLATES["situation"],
    }
    already_covered = _already_covered_label(state["slots"])

    if state.get("offtopic_message"):
        result = chat_qwen(
            prompts.QWEN_BASE,
            prompts.SLOT_RESUME_TASK.format(
                slot_goal=pending["question_intent"],
                already_asked=already_covered,
            ),
            required_keys=["question"],
        )
        # local 4B model occasionally returns valid JSON missing the expected keys;
        # never let a rendering hiccup send the user an empty turn
        question = str(result.get("question", "")).strip() or pending["question_intent"]
        message = f"{state['offtopic_message']} {question}".strip()
        return {
            "bot_message": message,
            "pending": pending,
            "offtopic_message": None,
            "conversation_log": _log(state, "bot", message),
        }

    result = chat_qwen(
        prompts.QWEN_BASE,
        prompts.SLOT_TASK.format(
            user_utterance=state.get("user_input") or "",
            slot_goal=pending["question_intent"],
            already_asked=already_covered,
        ),
        required_keys=["question"],
    )
    reflection = str(result.get("reflection", "")).strip()
    question = str(result.get("question", "")).strip() or pending["question_intent"]
    message = f"{reflection} {question}".strip()

    return {
        "bot_message": message,
        "pending": pending,
        # every genuinely new question (not an off-topic resume, which re-renders
        # the same still-pending question rather than posing a new one) counts
        # toward that slot's ask total, so supervisor_node can enforce the repeat cap
        "asked_slots": state["asked_slots"] + [pending["target_slot"]],
        "conversation_log": _log(state, "bot", message),
    }


def extract_and_detect_node(state: SessionState) -> dict:
    """Step 3: GPT-4o-mini, single call -- slot extraction + off-topic detection.
    Scans all 8 categories every turn (not just the slot that was actively asked
    about), so one rich reply can enrich several slots at once. When off_topic is
    true, extracted values are discarded and slots are left untouched."""
    user_input = state.get("user_input") or ""
    pending_question = state.get("bot_message") or ""

    result = call_openai_json(
        prompts.EXTRACT_AND_DETECT_SYSTEM,
        prompts.EXTRACT_AND_DETECT_TASK.format(
            pending_question=pending_question,
            user_utterance=user_input,
        ),
    )
    off_topic = bool(result.get("off_topic", False))

    slots = {slot: list(state["slots"][slot]) for slot in SLOT_ORDER}
    if not off_topic:
        for slot in SLOT_ORDER:
            value = result.get(slot)
            if not value:
                continue
            text = str(value).strip()
            if text and text not in slots[slot]:
                slots[slot].append(text)

    return {
        "slots": slots,
        "off_topic": off_topic,
        "turn_count": state["turn_count"] + 1,
        "conversation_log": _log(state, "user", user_input),
    }


def offtopic_response_node(state: SessionState) -> dict:
    """Step 4 (off-topic branch): GPT-4o-mini free-text chat reply. Does not touch
    slots/asked_slots/pending -- state is otherwise unchanged. The message is held
    in the transient offtopic_message field; render_question_node combines it with
    the re-asked pending question and logs the merged bot turn."""
    result = call_openai_json(
        prompts.OFFTOPIC_RESPONSE_SYSTEM,
        prompts.OFFTOPIC_RESPONSE_TASK.format(user_utterance=state.get("user_input") or ""),
    )
    message = str(result.get("message", "")).strip()
    return {"offtopic_message": message}


def _format_candidate_block(candidates: list[dict]) -> str:
    blocks = []
    for c in candidates:
        pattern = get_pattern(c["pattern_id"])
        features = pattern["matching_features"]
        blocks.append(
            f"[{c['pattern_id']}]\n"
            f"설명: {pattern['description']}\n"
            f"생각 단서: {', '.join(features['thought'])}\n"
            f"감정 단서: {', '.join(features['emotion'])}\n"
            f"행동 단서: {', '.join(features['behavior'])}"
        )
    return "\n\n".join(blocks)


def _rerank_patterns(candidates: list[dict], thought: list[str], emotion: list[str], behavior: list[str]) -> list[dict]:
    """Step 4b: GPT-4o-mini reorders/scores the embedding-shortlisted candidates by
    textual fit. Falls back to the embedding order if the model returns nothing
    usable, so a malformed call degrades gracefully instead of crashing the turn."""
    candidate_ids = {c["pattern_id"] for c in candidates}

    result = call_openai_json(
        prompts.PATTERN_RERANK_SYSTEM,
        prompts.PATTERN_RERANK_TASK.format(
            thought=" ".join(thought) or "(없음)",
            emotion=" ".join(emotion) or "(없음)",
            behavior=" ".join(behavior) or "(없음)",
            candidate_block=_format_candidate_block(candidates),
        ),
    )
    ranked = [
        r for r in result.get("ranked", [])
        if isinstance(r, dict) and r.get("pattern_id") in candidate_ids
    ]
    if not ranked:
        return [
            {"pattern_id": c["pattern_id"], "score": c["embedding_score"]}
            for c in candidates
        ]
    return ranked


def _top1_and_margin(ranked: list[dict]) -> tuple[float, float]:
    """Step 4c: pure computation, no LLM call."""
    if not ranked:
        return 0.0, 0.0
    top1 = float(ranked[0]["score"])
    second = float(ranked[1]["score"]) if len(ranked) > 1 else 0.0
    return top1, top1 - second


def pattern_mapping_node(state: SessionState) -> dict:
    """Step 4: (a) embedding top-k -> pattern_candidates, (b) GPT-4o-mini rerank,
    (c) recompute top1_score/margin from the reranked order -> gate. Runs once, right
    before self_help_node, not every turn -- self_help_node and summary_node are the
    only consumers of pattern_final, and both only run once the loop is exiting
    (coverage_ok or force_end), so there's nothing mid-loop to feed anymore now that
    there's no follow-up branch."""
    slots = state["slots"]
    if not any(slots[slot] for slot in PATTERN_MATCH_SLOTS):
        # no thought/emotion/behavior content yet -- nothing to match against
        return {
            "pattern_candidates": [],
            "pattern_final": [],
            "gate": {**state["gate"], "top1_score": 0.0, "margin": 0.0},
        }

    query = " ".join(text for slot in PATTERN_MATCH_SLOTS for text in slots[slot])
    candidates = top_k_candidates(query)

    ranked = _rerank_patterns(candidates, slots["thought"], slots["emotion"], slots["behavior"])
    top1_score, margin = _top1_and_margin(ranked)

    pattern_final = [
        {"pattern_id": r["pattern_id"], "rerank_score": float(r["score"])}
        for r in ranked[:2]
    ]

    gate = dict(state["gate"])
    gate["top1_score"] = top1_score
    gate["margin"] = margin

    return {
        "pattern_candidates": candidates,
        "pattern_final": pattern_final,
        "gate": gate,
    }


def gate_check_node(state: SessionState) -> dict:
    """Step 5: pure coverage/gate computation, no LLM call."""
    return {"gate": compute_gate(state)}


# hard cap, enforced in code rather than left to the model's judgment (asked_slots
# was previously just informational prompt context, which wasn't enough to stop
# the conversation circling back to the same slot)
MAX_ASKS_PER_SLOT = 2


def _unfilled_slots(slots, asked_slots: list[str]) -> list[str]:
    return [s for s in SLOT_ORDER if not slots[s] and asked_slots.count(s) < MAX_ASKS_PER_SLOT]


def _allowed_actions_block(unfilled: list[str], early_summary_allowed: bool) -> str:
    lines = []
    if unfilled:
        options = "\n".join(f"  - {slot}: {prompts.SLOT_QUESTION_TEMPLATES[slot]}" for slot in unfilled)
        lines.append(f"ask_intake_question (target_slot 선택지, question_intent는 오른쪽 문구 그대로 사용):\n{options}")
    if early_summary_allowed:
        lines.append("recommend_self_help (target_slot/question_intent 없음)")
        lines.append("end (target_slot/question_intent 없음)")
    return "\n\n".join(lines)


def _validate_supervisor_output(result: dict, unfilled: list[str], early_summary_allowed: bool) -> dict:
    """Controlled agent: the model proposes, this clamps the choice to what's
    actually allowed so a malformed or hallucinated response can't break routing."""
    valid_actions = []
    if unfilled:
        valid_actions.append("ask_intake_question")
    if early_summary_allowed:
        valid_actions += ["recommend_self_help", "end"]

    next_action = result.get("next_action")
    if next_action not in valid_actions:
        next_action = valid_actions[0] if valid_actions else "end"

    if next_action == "ask_intake_question":
        target_slot = result.get("target_slot")
        if target_slot not in unfilled:
            target_slot = unfilled[0]
        return {
            "next_action": next_action,
            "target_slot": target_slot,
            "question_intent": prompts.SLOT_QUESTION_TEMPLATES[target_slot],
        }

    return {"next_action": next_action, "target_slot": None, "question_intent": None}


def supervisor_node(state: SessionState) -> dict:
    """Step 6: GPT-4o-mini controlled agent -- picks next_action from the
    dynamically constructed allowed_actions set. question_intent only ever comes
    from the fixed intake templates -- there is no follow-up branch, so the model
    never picks question content, only which unfilled slot to ask about next."""
    slots = state["slots"]
    unfilled = _unfilled_slots(slots, state["asked_slots"])
    early_summary_allowed = state["gate"]["early_summary_allowed"]

    slot_summary = "\n".join(
        f"- {prompts.SLOT_KOREAN_LABELS[s]}: {' '.join(slots[s]) or '(없음)'}" for s in SLOT_ORDER
    )

    result = call_openai_json(
        prompts.SUPERVISOR_SYSTEM,
        prompts.SUPERVISOR_TASK.format(
            slot_summary=slot_summary,
            asked_slots=", ".join(state["asked_slots"]) or "(없음)",
            turn_count=state["turn_count"],
            allowed_actions_block=_allowed_actions_block(unfilled, early_summary_allowed),
        ),
    )
    decision = _validate_supervisor_output(result, unfilled, early_summary_allowed)

    update: dict = {"supervisor_action": decision["next_action"]}
    if decision["next_action"] == "ask_intake_question":
        # the ask itself (and its asked_slots bookkeeping) happens in
        # render_question_node, which runs next in this same turn -- that's also
        # where the very-first default question (before supervisor ever runs) gets
        # counted, so this stays the single place ask counts are incremented
        update["pending"] = {
            "target_slot": decision["target_slot"],
            "question_intent": decision["question_intent"],
        }

    return update


def self_help_node(state: SessionState) -> dict:
    """Step 7: pure function -- union of self_help_ids across pattern_final,
    order-preserving, deduped. pattern_final may be empty on the force_end path
    (low turn/confidence) -- that's reflected in summary_node's tone, not here."""
    self_help_ids: list[str] = []
    for match in state["pattern_final"]:
        for sid in get_pattern(match["pattern_id"])["self_help_ids"]:
            if sid not in self_help_ids:
                self_help_ids.append(sid)
    return {"self_help_ids": self_help_ids}


def summary_node(state: SessionState) -> dict:
    """Step 8: GPT-4o-mini -- final summary. General path only; the force_end tone
    branch (partial-information wording) is a separate follow-up.

    Generates once. Once stage is "done", route_entry sends every later turn
    straight here (bypassing extract_and_detect_node), so without this check it
    would silently call GPT-4o-mini and hand back a freshly-regenerated summary on
    every stray post-session message -- this returns a fixed closing reply instead,
    while still logging the user's message (extract_and_detect_node isn't in this
    path anymore to do it)."""
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

    result = call_openai_json(
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
    )
    summary = str(result.get("summary", "")).strip()

    return {
        "summary": summary,
        "bot_message": summary,
        "stage": "done",
        "conversation_log": _log(state, "bot", summary),
    }
