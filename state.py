from typing import Literal, TypedDict

SLOT_ORDER = [
    "situation",
    "emotion",
    "thought",
    "cause",
    "behavior",
    "duration",
    "impact",
    "coping",
    "goal",
]

# slots consumed by pattern_mapping_node
PATTERN_MATCH_SLOTS = ["thought", "emotion", "behavior"]

SKIP_IF_FILLED_SLOTS = {}

EXPLICIT_UNKNOWN_VALUE = "-"

# intent descriptions
SLOT_QUESTION_TEMPLATES = {
    "situation": "사용자가 상담을 하는 이유와 배경 상황 (신경쓰이는 일)에 대해 질의",
    "emotion": "사용자가 최근 상황에 대해 느끼는 감정에 대해 질의",
    "thought": "문제에 대해 스스로 드는 생각이 무엇인지 질의",
    "cause": "왜 그런 감정과 생각이 드는지 이유가 무엇이라고 생각하는지 질의",
    "behavior": "사용자의 최근 행동 패턴 또는 경향에 대해 질의. 이는 사용자의 행동 습관이나 문제 때문에 자주 하게 된 행동 패턴에 관한 것으로, 특정 활동 (e.g., 여행, 취미 활동)의 맥락이 아님.",
    "duration": "문제의 지속 기간과 빈도수에 대해 질의",
    "impact": "문제가 구체적으로 일상에 어떤 영향을 주는지 질의",
    "coping": "문제를 해결하기 위해 과거 시도해본 방법에 대해 질의",
    "goal": "사용자의 앞으로의 바램, 기대하는 나아진 모습, 목표에 대해 질의",
}

SLOT_KOREAN_LABELS = {
    "situation": "상황",
    "emotion": "감정",
    "thought": "생각",
    "cause": "원인",
    "behavior": "행동",
    "duration": "기간·빈도",
    "impact": "영향",
    "coping": "과거 시도",
    "goal": "목표",
}


class SlotState(TypedDict):
    situation: list[str]
    emotion: list[str]
    thought: list[str]
    cause: list[str]
    behavior: list[str]
    duration: list[str]
    impact: list[str]
    coping: list[str]
    goal: list[str]


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

    reflection_prefix: str | None

    pending: PendingQuestion | None
    asked_slots: list[str]
    
    retry_count: dict[str, int]

    slots: SlotState
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
        reflection_prefix=None,
        pending=None, # target_slot, question_intent for debugging
        asked_slots=[],
        retry_count={s: 0 for s in SLOT_ORDER},
        slots=SlotState(**{slot: [] for slot in SLOT_ORDER}),
        conversation_log=[],
        pattern_candidates=[],
        pattern_final=[],
        gate=GateState(
            coverage_ok=False,
        ),
        self_help_ids=[],
        summary=None,
    )
