"""All prompt templates and fixed user-facing text.

Qwen prompts (QWEN_BASE, SLOT_TASK, SLOT_RESUME_TASK) treat the model strictly as a
converter: a fixed slot template goes in, one natural Korean question comes out. It
is never given a persona or asked to judge/compose free content -- that's GPT-4o-mini's
job (OFFTOPIC_RESPONSE_*, EXTRACT_AND_DETECT_*, PATTERN_RERANK_*, SUPERVISOR_*, SUMMARY_*).
"""

# ─── rapport (fixed text, no LLM call) ──────────────────────────────────────

RAPPORT_GREETING = "안녕하세요, 오늘 여기서 편하게 이야기 나눠봐요."
RAPPORT_MOOD_CHECK = "오늘 컨디션이나 기분은 좀 어떠세요?"

# ─── Qwen: reflect + ask (single call, loop node) ───────────────────────────

QWEN_BASE = """
너는 한국어 문장 생성기다.

규칙:
- 입력된 목표 1개만 질문으로 바꾼다.
- 출력은 반드시 JSON 하나만 작성한다.
- question 값에는 질문 1문장만 넣는다.
- 질환명, 진단명, 점수, 의학 용어를 말하지 않는다.
- 조언, 해결책, 평가, 판단을 말하지 않는다.
- 40단어 이내, 존댓말, 한국어만 사용한다.
"""

SLOT_TASK = """
사용자 마지막 발화:
{user_utterance}

이번에 확인할 내용:
{slot_goal}

이미 확인한 내용:
{already_asked}

아래 JSON만 출력하세요.
{{
  "reflection": "사용자 말을 1문장으로 짧게 반영",
  "question": "이번에 확인할 내용만 묻는 질문 1문장"
}}

규칙:
- reflection은 질문이 아닌 사용자의 상황 또는 감정에 대한 공감과 재표현이어야 합니다. 
- question은 반드시 질문 1개만 포함합니다.
- already_asked와 같은 의미를 다시 묻지 마세요.
- 새로운 주제나 증상을 추가하지 마세요.
"""

# resuming a pending question after an off-topic detour: there is no fresh
# user_utterance to reflect (the last reply was off-topic and already handled by
# offtopic_response_node), so this only regenerates the question half
SLOT_RESUME_TASK = """
다시 물어야 할 내용:
{slot_goal}

이미 확인한 내용:
{already_asked}

아래 JSON만 출력하세요.
{{"question": "이번에 확인할 내용만 묻는 질문 1문장"}}

규칙:
- question은 반드시 질문 1개만 포함합니다.
- already_asked와 같은 의미를 다시 묻지 마세요.
- 새로운 주제나 증상을 추가하지 마세요.
"""

# ─── canonical slot templates (fixed; Qwen only rephrases these) ───────────

SLOT_QUESTION_TEMPLATES = {
    "situation": "오늘은 어떤 마음이나 상황 때문에 이야기해보고 싶으셨나요?",
    "thought": "그럴 때 스스로에게 어떤 생각이나 말이 가장 자주 떠오르나요?",
    "emotion": "요즘 스스로의 마음은 슬픔, 불안, 짜증, 무기력 중에 뭐에 가장 가까웠는지 이름을 붙여봐 주세요.",
    "behavior": "그 마음이 들 때 보통 어떻게 행동하게 되나요?",
    "impact": "그 패턴이 일, 사람과의 관계, 생활 리듬 등 일상 생활에 큰 영향을 미치고 있나요?",
    "duration": "이런 흐름이 언제부터 나타났나요? 하루 종일 그런 감정이 드나요, 아니면 드물게 그런 감정이 드나요?",
    "coping": "문제를 해결하기 위해 시도해본 방법이 있으신가요?",
    "goal": "지금 당장 문제가 다 해결되지 않더라도, 어떤 부분이 조금 가벼워지면 좋겠나요?",
}

SLOT_KOREAN_LABELS = {
    "situation": "상황",
    "thought": "생각",
    "emotion": "감정",
    "behavior": "행동",
    "impact": "영향",
    "duration": "기간·빈도",
    "coping": "과거 시도",
    "goal": "목표",
}

# ─── gpt-4o-mini: off-topic chat reply (off-topic branch of step 3/4) ───────
# unlike a slot question, this needs genuinely new content (replying to whatever
# the user actually said), which is why it's GPT-4o-mini and not Qwen -- Qwen only
# ever rephrases fixed content, it doesn't compose free responses

OFFTOPIC_RESPONSE_SYSTEM = (
    "당신은 한국어 상담 챗봇의 잡담 응답 담당입니다. 사용자의 잡담이나 시스템에 대한 "
    "질문에 짧게 답하고, 원래 상담 흐름으로 돌아오자는 뉘앙스를 남깁니다. "
    "지시된 JSON 형식으로만 응답하세요."
)

OFFTOPIC_RESPONSE_TASK = """
사용자 발화: {user_utterance}

이 발화는 상담 내용과 무관한 잡담이거나 시스템 자체에 대한 질문입니다.

아래 원칙을 지키며 응답하세요.
- 사용자의 발화에 자연스럽고 짧게 반응합니다 (1~2문장).
- 진단, 조언, 평가를 하지 않습니다.
- 질환명, 진단명, 의학 용어를 말하지 않습니다.
- 시스템에 대한 질문이면, "진단을 내리는 도구가 아니라 당신의 상황과 감정을 듣고 정리해드리는
  상담 시스템"이라는 취지로 짧게 설명합니다. 없는 기능이나 역할을 지어내지 않습니다.
- 다시 원래 이야기로 돌아오자는 뉘앙스로 마무리하되, 질문을 포함하지 않습니다
  (원래 질문은 이 응답 뒤에 별도로 다시 물어봅니다).
- 존댓말, 한국어만 사용, 60단어 이내.

출력 형식 (JSON, 다른 텍스트 없이): {{"message": "..."}}
"""

# ─── gpt-4o-mini: slot extraction + off-topic detection, single call (step 3) ──

EXTRACT_AND_DETECT_SYSTEM = (
    "당신은 상담 대화에서 사실만 추출하고 이탈 여부를 판단하는 분석기입니다. "
    "지시된 JSON 형식으로만 응답하세요."
)

EXTRACT_AND_DETECT_TASK = """
방금 사용자에게 한 질문: {pending_question}
사용자 마지막 발화: {user_utterance}

1) 아래 8개 항목 각각에 대해, 이번 발화에서 실제로 확인할 수 있는 내용이 있으면
그 내용을 1문장 이내로 요약해서 채우고, 확인할 수 없으면 null로 두세요.
이번 발화에 없는 내용을 추측해서 만들어내지 마세요.

항목:
- situation: 상담을 시작하게 된 마음이나 상황
- thought: 힘든 순간에 스스로에게 떠오르는 생각이나 말
- emotion: 이름 붙일 수 있는 감정 상태
- behavior: 그 마음이 들 때 하게 되는 행동
- impact: 일/관계/생활 리듬 등 일상에 미치는 영향
- duration: 시작된 시기, 지속 기간, 빈도
- coping: 스스로 시도해본 해결 방법
- goal: 가벼워지고 싶은 부분, 원하는 변화

2) 사용자의 발화가 위 질문에 대한 답변인지, 아니면 상담과 무관한 잡담(예: 날씨, 취미)이거나
시스템 자체에 대한 질문(예: 너는 누구야, 뭘 할 수 있어, 이거 어떻게 작동해)인지 판단하세요.
질문에 대한 답변으로 볼 수 있으면 off_topic: false, 무관한 발화이면 off_topic: true로 판단하세요.
명확하지 않으면 false로 판단하세요.
off_topic이 true이면 1)의 모든 항목은 null로 두세요.

출력 형식 (JSON, 다른 텍스트 없이, 아래 9개 key 모두 포함):
{{"situation": "..." 또는 null, "thought": "..." 또는 null, "emotion": "..." 또는 null,
"behavior": "..." 또는 null, "impact": "..." 또는 null, "duration": "..." 또는 null,
"coping": "..." 또는 null, "goal": "..." 또는 null, "off_topic": true 또는 false}}
"""

# ─── gpt-4o-mini: pattern rerank (step 4b) ──────────────────────────────────
# candidates come pre-filtered by embedding similarity (step 4a); this call only
# reorders/scores that shortlist by textual fit, it never expands the candidate set

PATTERN_RERANK_SYSTEM = (
    "당신은 상담 대화 내용과 어려움의 흐름 설명을 비교해 적합도를 판단하는 평가자입니다. "
    "지시된 JSON 형식으로만 응답하세요."
)

PATTERN_RERANK_TASK = """
사용자에게서 확인된 내용:
- 생각: {thought}
- 감정: {emotion}
- 행동: {behavior}

아래는 후보로 압축된 어려움의 흐름 설명들입니다.

후보:
{candidate_block}

각 후보에 대해 사용자의 생각-감정-행동 흐름과 얼마나 잘 맞는지 0.0~1.0 사이 점수로 판단하세요.
설명에 나온 단서 문구와 표면적으로 겹치는 단어가 있는지가 아니라, 사용자가 실제로 겪고 있는
흐름이 그 설명이 가리키는 것과 같은 종류인지를 기준으로 판단하세요.
전달받은 후보 전체에 대해 점수를 매기고, 점수가 높은 순으로 정렬해서 출력하세요.

출력 형식 (JSON, 다른 텍스트 없이):
{{"ranked": [{{"pattern_id": "...", "score": 0.0~1.0}}, ...]}}
"""

# ─── gpt-4o-mini: supervisor, controlled agent (step 6) ─────────────────────
# the model only ever selects a target_slot from the fixed intake catalog built by
# code -- it never composes new question content, that stays Qwen's job downstream

SUPERVISOR_SYSTEM = (
    "당신은 상담 대화의 다음 행동을 정하는 controlled agent입니다. "
    "반드시 제공된 선택지 중에서만 고르고, 지시된 JSON 형식으로만 응답하세요."
)

SUPERVISOR_TASK = """
지금까지 확인된 내용:
{slot_summary}

이미 물어본 slot: {asked_slots}
현재 turn 수: {turn_count}

선택 가능한 다음 행동:
{allowed_actions_block}

위 선택지 중에서만 다음 행동을 정하세요. ask_intake_question을 고를 경우 target_slot과
question_intent는 반드시 위에 제시된 선택지 중 하나를 그대로 사용하세요.
새로운 질문 내용을 만들어내지 마세요.

출력 형식 (JSON, 다른 텍스트 없이):
{{"next_action": "ask_intake_question 또는 recommend_self_help 또는 end",
"target_slot": "..." 또는 null, "question_intent": "..." 또는 null, "reason": "선택 이유 1문장"}}
"""

# ─── gpt-4o-mini: summary, general path (step 8) ────────────────────────────
# force_end tone branch is a separate follow-up prompt, not yet added here

SUMMARY_SYSTEM = "당신은 상담 요약을 작성하는 어시스턴트입니다. 지시된 JSON 형식으로만 응답하세요."

SUMMARY_TASK = """
아래는 상담 대화에서 확인된 내용입니다.

상황 맥락: {situation}
생각: {thought}
감정: {emotion}
행동: {behavior}
일상에 미치는 영향: {impact}
기간·빈도: {duration}
과거 시도: {coping}
목표: {goal}

참고할 어려움의 흐름 설명(그대로 인용하지 말고 서술에 녹여서 사용): {pattern_descriptions}
제안할 자가관리 방법: {self_help_context}

위 내용을 바탕으로 다음 순서로 상담 요약을 작성하세요.
1. 대화 내용을 공감적으로 요약 (2~3문장)
2. 어려움의 흐름을 서술형으로 정리 (패턴 이름, 진단 용어, 증상 라벨을 그대로 언급하지 않음)
3. 자가관리 제안 (구체적인 다음 행동 1~2가지)

규칙:
- "당신은 ~형입니다", "~증상이 있습니다" 같은 단정적 진단 문구를 사용하지 않습니다.
- "~한 흐름이 보이네요", "~해보면 도움이 될 수 있어요" 같은 잠정적 어조를 사용합니다.
- 질환명, 증상 라벨, 점수를 언급하지 않습니다.

출력 형식 (JSON, 다른 텍스트 없이): {{"summary": "..."}}
"""

# ─── done stage: fixed text, no LLM call ────────────────────────────────────
# shown for any message after the summary has already been generated once, instead
# of silently regenerating a new summary on every stray post-session message

SESSION_CLOSED_MESSAGE = "오늘 이야기 나눠주셔서 감사해요. 여기서 상담을 마칠게요."
