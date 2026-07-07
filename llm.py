"""LLM wrappers.

Qwen3-4B (Ollama) is used ONLY as a strict template-to-Korean-question converter
(render_question_node) -- it never judges or composes free content.

gpt-4o-mini handles everything that requires judgment or content generation:
slot extraction, off-topic detection, pattern rerank, supervisor routing,
off-topic chat replies, and the final summary.
"""
import json
import re
from pathlib import Path

import ollama
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

OLLAMA_MODEL = "pume-voice-qwen3-4b"
OPENAI_MODEL = "gpt-4o-mini"


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


# modern Korean text has no business containing CJK Han ideographs (Hanja);
# the 4B model occasionally leaks Chinese tokens despite the "no language mixing" rule
_HAN_RE = re.compile(r"[一-鿿]")

_openai_client = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()  # reads OPENAI_API_KEY from env
    return _openai_client


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}


def _chat_qwen_once(system: str, user: str) -> str:
    # the model tag's own template pre-fills an empty <think></think> block to
    # suppress reasoning (the classic Qwen3 "nothink" trick), but this Ollama
    # version applies its own runtime-level thinking handling for qwen3-architecture
    # models regardless -- without an explicit think=False, that runtime handling
    # wins, reasoning goes into a separate "thinking" field, and content can come
    # back empty (or generation can run unbounded before content ever appears)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.3},
        format="json",
        think=False,
    )
    return _strip_think(response["message"]["content"])


def chat_qwen(system: str, user: str, required_keys: list[str] | None = None) -> dict:
    """Returns the parsed JSON dict from a converter-role Qwen call. Regenerates
    once if the model leaks Chinese characters, or if any required_keys came back
    missing/blank -- the 4B model occasionally returns syntactically valid JSON
    that's just missing the key a caller actually needs."""
    text = _chat_qwen_once(system, user)
    parsed = _parse_json(text)

    missing_required = required_keys and any(not str(parsed.get(k, "")).strip() for k in required_keys)
    if _HAN_RE.search(text) or missing_required:
        text = _chat_qwen_once(system, user)
        parsed = _parse_json(text)

    return parsed


def call_openai_json(system: str, user: str) -> dict:
    """JSON-forced gpt-4o-mini call, for judgment (extraction, meta-detection) and
    for generation that needs to be reliably non-diagnostic (summary)."""
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return _parse_json(response.choices[0].message.content)
