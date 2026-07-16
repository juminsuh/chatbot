"""LLM wrappers.

gpt-4o-mini handles everything: slot extraction, off-topic detection, pattern
rerank, supervisor routing, off-topic chat replies, question rendering, and
the final summary.
"""
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

OPENAI_MODEL = "gpt-4o-mini"

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


def call_openai_json(system: str, user: str, temperature: float = 0.3) -> dict:
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    return _parse_json(response.choices[0].message.content)
