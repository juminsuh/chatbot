from typing import Literal, TypedDict


class SlotState(TypedDict):
    situation: list[str]
    thought: list[str]
    emotion: list[str]
    behavior: list[str]
    impact: list[str]
    duration: list[str]
    coping: list[str]
    goal: list[str]


SLOT_ORDER = [
    "situation",
    "thought",
    "emotion",
    "behavior",
    "impact",
    "duration",
    "coping",
    "goal",
]

# slots the gate requires before early_summary_allowed can be true; impact/duration
# are context-only and never block coverage
REQUIRED_SLOTS = ["situation", "thought", "emotion", "behavior", "coping", "goal"]

# slots consumed by pattern_mapping_node; situation is summary-only context
PATTERN_MATCH_SLOTS = ["thought", "emotion", "behavior"]


class PendingQuestion(TypedDict):
    target_slot: str
    question_intent: str


class PatternCandidate(TypedDict):
    pattern_id: str
    embedding_score: float


class PatternFinal(TypedDict):
    pattern_id: str
    rerank_score: float


class GateState(TypedDict):
    coverage_ok: bool
    top1_score: float
    margin: float
    early_summary_allowed: bool
    force_end: bool


class SessionState(TypedDict):
    stage: Literal["rapport", "loop", "done"]
    turn_count: int

    user_input: str | None
    bot_message: str | None
    # transient: written by offtopic_response_node, consumed and cleared by
    # render_question_node when it re-renders the still-pending question
    offtopic_message: str | None

    pending: PendingQuestion | None
    asked_slots: list[str]
    # transient: written by supervisor_node, consumed by route_after_supervisor
    supervisor_action: Literal["ask_intake_question", "recommend_self_help", "end"] | None

    slots: SlotState
    off_topic: bool
    conversation_log: list[dict]  # [{"role": "user" | "bot", "content": str}]

    pattern_candidates: list[PatternCandidate]  # top-k, tentative (embedding-only)
    pattern_final: list[PatternFinal]  # top-2 after rerank, used by gate/self-help
    gate: GateState

    self_help_ids: list[str]
    summary: str | None


def new_session() -> SessionState:
    return SessionState(
        stage="rapport",
        turn_count=0,
        user_input=None,
        bot_message=None,
        offtopic_message=None,
        pending=None,
        asked_slots=[],
        supervisor_action=None,
        slots=SlotState(**{slot: [] for slot in SLOT_ORDER}),
        off_topic=False,
        conversation_log=[],
        pattern_candidates=[],
        pattern_final=[],
        gate=GateState(
            coverage_ok=False,
            top1_score=0.0,
            margin=0.0,
            early_summary_allowed=False,
            force_end=False,
        ),
        self_help_ids=[],
        summary=None,
    )
