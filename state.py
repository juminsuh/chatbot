from typing import Literal, TypedDict

# priority order the loop asks slots in -- render_question_node always targets
# the first slot here that askable_slots still allows
SLOT_ORDER = [
    "situation",
    "emotion",
    "behavior",
    "impact",
    "duration",
    "coping",
    "goal",
    "cause",
    "thought",
]

# slots consumed by pattern_mapping_node
PATTERN_MATCH_SLOTS = ["thought", "emotion", "behavior"]

SKIP_IF_FILLED_SLOTS = {"duration", "cause"}

# slots allowed one retry (via the static RETRY_QUESTIONS prompt) when the user
# explicitly says they don't know, before being force-filled and moved past
RETRY_ELIGIBLE_SLOTS = {"thought", "emotion", "behavior"}

EXPLICIT_UNKNOWN_VALUE = "-"

# intent descriptions, not literal questions -- render_question_node's LLM call
# phrases these into a natural Korean question
SLOT_QUESTION_TEMPLATES = {
    "situation": "사용자가 상담을 하는 이유와 배경 상황 (신경쓰이는 일)에 대해 질의",
    "emotion": "사용자가 최근 상황에 대해 느끼는 감정에 대해 질의",
    "behavior": "사용자의 최근 행동 패턴 또는 경향에 대해 질의. 이는 사용자의 행동 습관이나 문제 때문에 자주 하게 된 행동 패턴에 관한 것으로, 특정 활동 (e.g., 여행, 취미 활동)의 맥락이 아님.",
    "impact": "문제가 구체적으로 일상에 어떤 영향을 주는지 질의",
    "duration": "문제의 지속 기간과 빈도수에 대해 질의",
    "coping": "문제를 해결하기 위해 과거 시도해본 방법에 대해 질의",
    "goal": "사용자의 앞으로의 바램, 기대하는 나아진 모습, 목표에 대해 질의",
    "cause": "문제가 발생한 이유가 무엇이라고 생각하는지 질의",
    "thought": "문제에 대해 스스로 드는 생각이 무엇인지 질의",
}

SLOT_KOREAN_LABELS = {
    "situation": "상황",
    "emotion": "감정",
    "behavior": "행동",
    "impact": "영향",
    "duration": "기간·빈도",
    "coping": "과거 시도",
    "goal": "목표",
    "cause": "원인",
    "thought": "생각",
}


class SlotState(TypedDict):
    situation: list[str]
    emotion: list[str]
    behavior: list[str]
    impact: list[str]
    duration: list[str]
    coping: list[str]
    goal: list[str]
    cause: list[str]
    thought: list[str]


def askable_slots(slots: SlotState, asked_slots: list[str]) -> list[str]:
    result = []
    for s in SLOT_ORDER:
        if s in asked_slots:
            continue
        if s in SKIP_IF_FILLED_SLOTS and slots[s]:
            continue
        result.append(s)
    return result


class PendingQuestion(TypedDict):
    target_slot: str
    question_intent: str


class PatternCandidate(TypedDict):
    pattern_id: str
    thought_score: float
    emotion_score: float
    behavior_score: float
    pattern_score: float


class PatternFinal(TypedDict):
    pattern_id: str
    score: float


class GateState(TypedDict):
    coverage_ok: bool


class SessionState(TypedDict):
    stage: Literal["rapport", "loop", "done"]
    rapport_step: Literal["greeting", "mood", "how", "who"]
    name: str
    turn_count: int

    user_input: str | None
    bot_message: str | None
    
    offtopic_message: str | None
    offtopic_streak: int
    already_extracted: bool

    pending: PendingQuestion | None
    asked_slots: list[str]
    retry_attempted: dict[str, bool]
    # tracks thin-content re-asks (sufficient=false), independent of retry_attempted
    # (explicit-unknown re-asks) -- separate caps since the causes differ
    thin_retry_count: dict[str, int]

    slots: SlotState
    off_topic: bool
    conversation_log: list[dict] 

    pattern_candidates: list[PatternCandidate]  
    pattern_final: list[PatternFinal]  
    gate: GateState

    self_help_ids: list[str]
    summary: str | None


def new_session(name: str) -> SessionState:
    return SessionState(
        stage="rapport",
        rapport_step="greeting",
        name=name,
        turn_count=0,
        user_input=None,
        bot_message=None,
        offtopic_message=None,
        offtopic_streak=0, # offtopic으로 판정받는 수
        already_extracted=False,
        pending=None, # target_slot, question_intent for debugging
        asked_slots=[],
        retry_attempted={s: False for s in RETRY_ELIGIBLE_SLOTS},
        thin_retry_count={s: 0 for s in SLOT_ORDER},
        slots=SlotState(**{slot: [] for slot in SLOT_ORDER}),
        off_topic=False,
        conversation_log=[],
        pattern_candidates=[],
        pattern_final=[],
        gate=GateState(
            coverage_ok=False,
        ),
        self_help_ids=[],
        summary=None,
    )
