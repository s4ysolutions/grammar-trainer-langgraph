import json
import logging
import os
import re
import socket
from functools import lru_cache
from typing import Literal
from urllib.parse import urlparse

import openai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

logger = logging.getLogger(__name__)

try:
    from google.api_core.exceptions import ResourceExhausted as _GoogleRateLimitError
    _RATE_LIMIT_EXCEPTIONS = (openai.RateLimitError, _GoogleRateLimitError)
except ImportError:
    _RATE_LIMIT_EXCEPTIONS = (openai.RateLimitError,)

from .state import TutorState
from .prompts import EXERCISE_PROMPT, GRADE_PROMPT


_PROVIDER_DEFAULTS = {
    "openai": "gpt-5-nano",
    "huggingface": "meta-llama/Llama-3.1-8B-Instruct",
    "gemini": "gemma-4-26b-a4b-it",
    "openrouter": "meta-llama/llama-3.1-8b-instruct",
}

_PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

_HUGGINGFACE_DEFAULT_BASE_URL = "https://router.huggingface.co/hf-inference/v1"
_OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _huggingface_base_url() -> str:
    return os.getenv("HUGGINGFACE_BASE_URL") or _HUGGINGFACE_DEFAULT_BASE_URL


def _openrouter_base_url() -> str:
    return os.getenv("OPENROUTER_BASE_URL") or _OPENROUTER_DEFAULT_BASE_URL


def _tcp_probe(url: str, label: str, url_env_var: str, default_url: str) -> None:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        raise ValueError(
            f"Cannot connect to {label}: URL is empty or invalid (resolved to '{url}'). "
            f"Unset {url_env_var} to use the default ({default_url}), "
            f"or set it to a valid URL."
        )
    try:
        socket.create_connection((host, port), timeout=5).close()
    except OSError as e:
        raise ValueError(
            f"Cannot connect to {label} at {url} (host={host}, port={port}): {e}. "
            f"Check your network or set {url_env_var} to override "
            f"(default: {default_url})."
        ) from e


@lru_cache(maxsize=1)
def validate_config() -> None:
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    valid = set(_PROVIDER_DEFAULTS)
    if provider not in valid:
        raise ValueError(
            f"LLM_PROVIDER='{provider}' is not supported. "
            f"Set LLM_PROVIDER to one of: {', '.join(sorted(valid))}"
        )
    key_var = _PROVIDER_KEY_ENV[provider]
    if not os.getenv(key_var):
        raise ValueError(
            f"LLM_PROVIDER='{provider}' requires {key_var} to be set. "
            f"Add it to your .env file or environment."
        )
    if provider == "huggingface":
        _tcp_probe(
            _huggingface_base_url(), "HuggingFace API",
            "HUGGINGFACE_BASE_URL", _HUGGINGFACE_DEFAULT_BASE_URL,
        )
    elif provider == "openrouter":
        _tcp_probe(
            _openrouter_base_url(), "OpenRouter API",
            "OPENROUTER_BASE_URL", _OPENROUTER_DEFAULT_BASE_URL,
        )

    for i in (1, 2):
        fb_provider = os.getenv(f"FALLBACK{i}_PROVIDER", "").lower()
        if not fb_provider:
            continue
        if fb_provider not in valid:
            logger.warning(
                "FALLBACK%d_PROVIDER='%s' not supported, ignoring. Valid: %s",
                i, fb_provider, ", ".join(sorted(valid)),
            )
            continue
        fb_key_var = _PROVIDER_KEY_ENV[fb_provider]
        if not os.getenv(fb_key_var):
            logger.warning(
                "FALLBACK%d_PROVIDER='%s' requires %s — fallback will be skipped",
                i, fb_provider, fb_key_var,
            )


def _make_llm(model: str, temperature: float, provider: str | None = None):
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        validate_config()
    if provider == "openai":
        return ChatOpenAI(model=model, temperature=temperature)
    if provider == "huggingface":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=os.getenv("HUGGINGFACE_API_KEY", ""),
            openai_api_base=_huggingface_base_url(),
        )
    if provider == "openrouter":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            openai_api_base=_openrouter_base_url(),
        )
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def _get_llms():
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    default = _PROVIDER_DEFAULTS.get(provider, "gemma-4-26b-a4b-it")
    generator = _make_llm(os.getenv("GENERATOR_MODEL") or default, 0.7)
    grader = _make_llm(os.getenv("GRADER_MODEL") or default, 0.0)

    fb_generators = []
    fb_graders = []
    for i in (1, 2):
        fb_provider = os.getenv(f"FALLBACK{i}_PROVIDER", "").lower()
        if not fb_provider or fb_provider not in _PROVIDER_DEFAULTS:
            continue
        if not os.getenv(_PROVIDER_KEY_ENV[fb_provider]):
            continue
        fb_default = _PROVIDER_DEFAULTS[fb_provider]
        fb_gen = os.getenv(f"FALLBACK{i}_GENERATOR_MODEL") or fb_default
        fb_grd = os.getenv(f"FALLBACK{i}_GRADER_MODEL") or fb_default
        fb_generators.append(_make_llm(fb_gen, 0.7, fb_provider))
        fb_graders.append(_make_llm(fb_grd, 0.0, fb_provider))

    if fb_generators:
        generator = generator.with_fallbacks(fb_generators, exceptions_to_handle=_RATE_LIMIT_EXCEPTIONS)
    if fb_graders:
        grader = grader.with_fallbacks(fb_graders, exceptions_to_handle=_RATE_LIMIT_EXCEPTIONS)

    return generator, grader


def _extract_text(response) -> str:
    content = response.content
    if isinstance(content, str):
        return content
    block = next(
        (b for b in content if isinstance(b, dict) and b.get("type") == "text"),
        None,
    )
    return block["text"] if block else str(content)


def collect_language(state: TutorState) -> dict:
    language = interrupt("Waiting for language")
    return {"language": language, "phase": "topic"}


def collect_topic(state: TutorState) -> dict:
    topic = interrupt("Waiting for topic")
    return {"topic": topic, "phase": "active"}


def generate_exercise(state: TutorState) -> dict:
    generator, _ = _get_llms()
    past = state.get("past_exercises", [])
    past_text = "\n".join(f"- {ex}" for ex in past) if past else "нет"
    prompt = EXERCISE_PROMPT.format(
        language=state["language"],
        topic=state["topic"],
        past=past_text,
    )
    exercise = _extract_text(generator.invoke(prompt)).strip()
    return {
        "last_exercise": exercise,
        "past_exercises": past + [exercise],
    }


def wait_for_answer(state: TutorState) -> dict:
    answer = interrupt("Waiting for answer")
    if answer.strip() == "/end":
        return {"last_answer": "", "phase": "done"}
    return {"last_answer": answer}


def check_answer(state: TutorState) -> dict:
    _, grader = _get_llms()
    prompt = GRADE_PROMPT.format(
        language=state["language"],
        topic=state["topic"],
        exercise=state["last_exercise"],
        answer=state["last_answer"],
    )
    raw = _extract_text(grader.invoke(prompt))
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    data = json.loads(match.group()) if match else {}
    verdict = data.get("verdict", "INCORRECT").upper()
    feedback = data.get("feedback", raw.strip())
    is_correct = verdict == "CORRECT"
    return {
        "feedback": feedback,
        "last_verdict": verdict,
        "correct_count": state.get("correct_count", 0) + (1 if is_correct else 0),
        "incorrect_count": state.get("incorrect_count", 0) + (0 if is_correct else 1),
    }


def on_correct(state: TutorState) -> dict:
    return {}


def on_incorrect(state: TutorState) -> dict:
    return {}


def update_state(state: TutorState) -> dict:
    return {
        "turn_count": state.get("turn_count", 0) + 1,
        "last_answer": "",
    }


def route_phase(state: TutorState) -> str:
    return state.get("phase", "language")


def route_after_wait(state: TutorState) -> str:
    if state.get("phase") == "done":
        return "end"
    return "check_answer"


def route_verdict(state: TutorState) -> Literal["on_correct", "on_incorrect"]:
    if state.get("last_verdict") == "CORRECT":
        return "on_correct"
    return "on_incorrect"
