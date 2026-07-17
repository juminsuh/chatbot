"""LLM wrappers.

gpt-5.4-mini (via the Responses API) handles everything: slot extraction,
pattern rerank, question rendering, and the final summary.
"""
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

OPENAI_MODEL = "gpt-5.4-mini"

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


def call_openai_json(system: str, user: str, reasoning_effort: str = "none") -> dict:
    client = _get_openai_client()
    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=system,
        input=user,
        reasoning={"effort": reasoning_effort},
        text={"format": {"type": "json_object"}},
    )
    return _parse_json(response.output_text)
