# english-grammar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Python LangGraph grammar tutor with multi-language support, Russian UI, CLI + Telegram bot (polling and webhook).

**Architecture:** LangGraph StateGraph with phase-based routing (language → topic → exercise loop). Shared `agent/` library used by two products (`cli/` and `tgbot/`). Telegram bot split into separate `polling.py` and `webhook.py` entry points that share `handlers.py`.

**Tech Stack:** Python 3.11+, LangGraph ≥0.2, LangChain (OpenAI + Google GenAI), python-telegram-bot ≥21, python-dotenv, pytest, uv.

---

## File Map

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project deps, uv tooling |
| `.env.example` | Env var template |
| `messages.py` | All Russian UI strings |
| `agent/__init__.py` | Exports `build_graph`, `make_initial_state` |
| `agent/state.py` | `TutorState` TypedDict + `make_initial_state()` factory |
| `agent/prompts.py` | Russian LLM prompt templates |
| `agent/nodes.py` | All graph nodes + LLM factory + routing functions |
| `agent/graph.py` | LangGraph `StateGraph` assembly |
| `cli/__init__.py` | Empty |
| `cli/main.py` | Single-user terminal product |
| `tgbot/__init__.py` | Empty |
| `tgbot/handlers.py` | All Telegram handlers + `build_app()` factory |
| `tgbot/polling.py` | Entry point: `run_polling()` |
| `tgbot/webhook.py` | Entry point: `run_webhook()` |
| `tests/__init__.py` | Empty |
| `tests/test_state.py` | State creation tests |
| `tests/test_prompts.py` | Prompt formatting tests |
| `tests/test_nodes.py` | Node unit tests with mocked LLM |
| `tests/test_graph.py` | Graph integration test with mocked nodes |

All paths relative to `/Users/dsa/s4y/llm-agents/english-grammar/`.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `agent/__init__.py`, `cli/__init__.py`, `tgbot/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "english-grammar"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2",
    "langchain-openai>=0.1",
    "langchain-google-genai>=4.0",
    "python-telegram-bot>=21.0",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["agent", "cli", "tgbot"]
```

- [ ] **Step 2: Create .env.example**

```env
# Required for Telegram bot
TELEGRAM_BOT_TOKEN=

# LLM provider: gemini | openai | huggingface  (default: gemini)
LLM_PROVIDER=gemini
GOOGLE_API_KEY=
OPENAI_API_KEY=
HUGGINGFACE_API_KEY=

# Optional model overrides
GENERATOR_MODEL=
GRADER_MODEL=

# Webhook mode only
WEBHOOK_URL=
PORT=8443
```

- [ ] **Step 3: Create empty __init__ files**

Create these four files, each with a single empty line:
- `agent/__init__.py`
- `cli/__init__.py`
- `tgbot/__init__.py`
- `tests/__init__.py`

- [ ] **Step 4: Install deps and verify**

```bash
cd /Users/dsa/s4y/llm-agents/english-grammar
uv sync --extra dev
```

Expected: resolves and installs without errors.

- [ ] **Step 5: Commit**

```bash
git init
git add pyproject.toml .env.example agent/__init__.py cli/__init__.py tgbot/__init__.py tests/__init__.py
git commit -m "chore: scaffold english-grammar project"
```

---

## Task 2: messages.py — Russian UI strings

**Files:**
- Create: `messages.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_messages.py`:

```python
import messages


def test_all_keys_present():
    assert messages.WELCOME
    assert messages.CHOOSE_TOPIC
    assert messages.NEXT_PROMPT
    assert messages.STATS
    assert messages.CORRECT
    assert messages.INCORRECT
    assert messages.LANG_CHANGED
    assert messages.TOPIC_CHANGED
    assert messages.NO_SESSION


def test_stats_format():
    result = messages.STATS.format(correct=3, total=5, pct=60)
    assert "3" in result
    assert "5" in result
    assert "60" in result


def test_correct_format():
    result = messages.CORRECT.format(feedback="Правильно использован глагол.")
    assert "Правильно использован глагол." in result


def test_incorrect_format():
    result = messages.INCORRECT.format(feedback="Нужен past simple.")
    assert "Нужен past simple." in result


def test_lang_changed_format():
    result = messages.LANG_CHANGED.format(language="Spanish")
    assert "Spanish" in result


def test_topic_changed_format():
    result = messages.TOPIC_CHANGED.format(topic="past tense")
    assert "past tense" in result
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /Users/dsa/s4y/llm-agents/english-grammar
uv run pytest tests/test_messages.py -v
```

Expected: `ModuleNotFoundError: No module named 'messages'`

- [ ] **Step 3: Create messages.py**

```python
WELCOME = "Привет! Я помогу тебе практиковать грамматику. Какой язык изучаем?"

CHOOSE_TOPIC = "Отлично! Тема для практики?"

CORRECT = "✅ Правильно!\n💡 {feedback}"

INCORRECT = "❌ Неверно.\n💡 {feedback}"

NEXT_PROMPT = "Напиши ответ или /end чтобы завершить сессию."

STATS = (
    "📊 Сессия завершена\n"
    "Правильно: {correct}/{total} ({pct}%)\n"
    "Упражнений: {total}"
)

LANG_CHANGED = "Язык изменён на {language}. Выбери тему:"

TOPIC_CHANGED = "Тема изменена: {topic}. Начинаем новое упражнение."

NO_SESSION = "Используй /start чтобы начать сессию."
```

- [ ] **Step 4: Run to verify it passes**

```bash
uv run pytest tests/test_messages.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add messages.py tests/test_messages.py
git commit -m "feat: add Russian UI messages"
```

---

## Task 3: agent/state.py — TutorState

**Files:**
- Modify: `agent/state.py`
- Create: `tests/test_state.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_state.py`:

```python
from agent.state import TutorState, make_initial_state


def test_make_initial_state_defaults():
    s = make_initial_state()
    assert s["phase"] == "language"
    assert s["language"] == ""
    assert s["topic"] == ""
    assert s["past_exercises"] == []
    assert s["last_exercise"] == ""
    assert s["last_answer"] == ""
    assert s["feedback"] == ""
    assert s["last_verdict"] == ""
    assert s["turn_count"] == 0
    assert s["correct_count"] == 0
    assert s["incorrect_count"] == 0


def test_make_initial_state_with_language():
    s = make_initial_state(phase="topic", language="Spanish")
    assert s["phase"] == "topic"
    assert s["language"] == "Spanish"
    assert s["topic"] == ""


def test_make_initial_state_with_topic():
    s = make_initial_state(phase="active", language="French", topic="présent")
    assert s["phase"] == "active"
    assert s["language"] == "French"
    assert s["topic"] == "présent"


def test_tutor_state_is_dict():
    s = make_initial_state()
    assert isinstance(s, dict)
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_state.py -v
```

Expected: `ImportError: cannot import name 'TutorState' from 'agent.state'`

- [ ] **Step 3: Write agent/state.py**

```python
from typing import TypedDict, Literal


class TutorState(TypedDict):
    phase: Literal["language", "topic", "active", "done"]
    language: str
    topic: str
    past_exercises: list[str]
    last_exercise: str
    last_answer: str
    feedback: str
    last_verdict: str
    turn_count: int
    correct_count: int
    incorrect_count: int


def make_initial_state(
    phase: str = "language",
    language: str = "",
    topic: str = "",
) -> TutorState:
    return {
        "phase": phase,
        "language": language,
        "topic": topic,
        "past_exercises": [],
        "last_exercise": "",
        "last_answer": "",
        "feedback": "",
        "last_verdict": "",
        "turn_count": 0,
        "correct_count": 0,
        "incorrect_count": 0,
    }
```

- [ ] **Step 4: Run to verify it passes**

```bash
uv run pytest tests/test_state.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/state.py tests/test_state.py
git commit -m "feat: add TutorState with phase and language fields"
```

---

## Task 4: agent/prompts.py — Russian LLM prompt templates

**Files:**
- Create: `agent/prompts.py`
- Create: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prompts.py`:

```python
from agent.prompts import EXERCISE_PROMPT, GRADE_PROMPT


def test_exercise_prompt_format():
    result = EXERCISE_PROMPT.format(
        language="English",
        topic="past simple",
        past="- I went to school.",
    )
    assert "English" in result
    assert "past simple" in result
    assert "I went to school." in result


def test_exercise_prompt_no_past():
    result = EXERCISE_PROMPT.format(
        language="Spanish",
        topic="ser vs estar",
        past="нет",
    )
    assert "Spanish" in result
    assert "ser vs estar" in result
    assert "нет" in result


def test_grade_prompt_format():
    result = GRADE_PROMPT.format(
        language="French",
        topic="passé composé",
        exercise="Conjuguez le verbe aller: Je ___ au marché.",
        answer="suis allé",
    )
    assert "French" in result
    assert "passé composé" in result
    assert "Conjuguez le verbe aller" in result
    assert "suis allé" in result
    assert "CORRECT" in result
    assert "INCORRECT" in result


def test_exercise_prompt_is_russian():
    assert "Ты репетитор" in EXERCISE_PROMPT


def test_grade_prompt_is_russian():
    assert "Ты репетитор" in GRADE_PROMPT
    assert "русском" in GRADE_PROMPT
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_prompts.py -v
```

Expected: `ImportError: cannot import name 'EXERCISE_PROMPT' from 'agent.prompts'`

- [ ] **Step 3: Create agent/prompts.py**

```python
EXERCISE_PROMPT = (
    "Ты репетитор по грамматике {language}. "
    "Составь одно короткое упражнение на тему: '{topic}'. "
    "Попроси пользователя вставить пропущенное слово, исправить предложение "
    "или выбрать правильную форму. "
    "Упражнение должно быть на языке {language}. "
    "Выведи только текст упражнения, без вступления.\n"
    "Не повторяй эти прошлые упражнения:\n{past}"
)

GRADE_PROMPT = (
    "Ты репетитор по грамматике {language}.\n"
    "Тема: '{topic}'\n"
    "Упражнение: {exercise}\n"
    "Ответ ученика: {answer}\n"
    "Оцени ответ. Объяснение давай на русском языке. "
    "Ответь ТОЛЬКО JSON:\n"
    '{{"verdict": "CORRECT" or "INCORRECT", "feedback": "краткое объяснение на русском"}}'
)
```

- [ ] **Step 4: Run to verify it passes**

```bash
uv run pytest tests/test_prompts.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/prompts.py tests/test_prompts.py
git commit -m "feat: add Russian LLM prompt templates"
```

---

## Task 5: agent/nodes.py — Graph nodes + LLM factory

**Files:**
- Create: `agent/nodes.py`
- Create: `tests/test_nodes.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_nodes.py`:

```python
from unittest.mock import MagicMock, patch
from agent.state import make_initial_state


def _mock_llm(text: str):
    m = MagicMock()
    m.invoke.return_value = MagicMock(content=text)
    return m


@patch("agent.nodes._get_llms")
def test_generate_exercise(mock_get_llms):
    from agent.nodes import generate_exercise
    mock_get_llms.return_value = (_mock_llm("Fill in: I ___ go."), MagicMock())

    state = make_initial_state(phase="active", language="English", topic="past simple")
    result = generate_exercise(state)

    assert result["last_exercise"] == "Fill in: I ___ go."
    assert result["past_exercises"] == ["Fill in: I ___ go."]


@patch("agent.nodes._get_llms")
def test_generate_exercise_avoids_past(mock_get_llms):
    from agent.nodes import generate_exercise
    mock_gen = _mock_llm("New exercise.")
    mock_get_llms.return_value = (mock_gen, MagicMock())

    state = make_initial_state(phase="active", language="English", topic="past simple")
    state["past_exercises"] = ["Old exercise."]
    generate_exercise(state)

    call_args = mock_gen.invoke.call_args[0][0]
    assert "Old exercise." in call_args


@patch("agent.nodes._get_llms")
def test_check_answer_correct(mock_get_llms):
    from agent.nodes import check_answer
    mock_get_llms.return_value = (
        MagicMock(),
        _mock_llm('{"verdict": "CORRECT", "feedback": "Правильно!"}'),
    )

    state = make_initial_state(phase="active", language="English", topic="past simple")
    state["last_exercise"] = "Fill in: I ___ go."
    state["last_answer"] = "went"
    result = check_answer(state)

    assert result["last_verdict"] == "CORRECT"
    assert result["feedback"] == "Правильно!"
    assert result["correct_count"] == 1
    assert result["incorrect_count"] == 0


@patch("agent.nodes._get_llms")
def test_check_answer_incorrect(mock_get_llms):
    from agent.nodes import check_answer
    mock_get_llms.return_value = (
        MagicMock(),
        _mock_llm('{"verdict": "INCORRECT", "feedback": "Нужен past simple."}'),
    )

    state = make_initial_state(phase="active", language="English", topic="past simple")
    state["last_exercise"] = "Fill in: I ___ go."
    state["last_answer"] = "go"
    result = check_answer(state)

    assert result["last_verdict"] == "INCORRECT"
    assert result["feedback"] == "Нужен past simple."
    assert result["correct_count"] == 0
    assert result["incorrect_count"] == 1


@patch("agent.nodes._get_llms")
def test_check_answer_malformed_json(mock_get_llms):
    from agent.nodes import check_answer
    mock_get_llms.return_value = (MagicMock(), _mock_llm("Not JSON at all."))

    state = make_initial_state(phase="active", language="English", topic="past simple")
    state["last_exercise"] = "Ex."
    state["last_answer"] = "answer"
    result = check_answer(state)

    assert result["last_verdict"] == "INCORRECT"
    assert result["feedback"] == "Not JSON at all."


def test_route_phase_language():
    from agent.nodes import route_phase
    state = make_initial_state(phase="language")
    assert route_phase(state) == "language"


def test_route_phase_active():
    from agent.nodes import route_phase
    state = make_initial_state(phase="active")
    assert route_phase(state) == "active"


def test_route_after_wait_check():
    from agent.nodes import route_after_wait
    state = make_initial_state(phase="active")
    state["last_answer"] = "went"
    assert route_after_wait(state) == "check_answer"


def test_route_after_wait_end():
    from agent.nodes import route_after_wait
    state = make_initial_state(phase="done")
    assert route_after_wait(state) == "end"


def test_route_verdict_correct():
    from agent.nodes import route_verdict
    state = make_initial_state(phase="active")
    state["last_verdict"] = "CORRECT"
    assert route_verdict(state) == "on_correct"


def test_route_verdict_incorrect():
    from agent.nodes import route_verdict
    state = make_initial_state(phase="active")
    state["last_verdict"] = "INCORRECT"
    assert route_verdict(state) == "on_incorrect"


def test_update_state():
    from agent.nodes import update_state
    state = make_initial_state(phase="active")
    state["turn_count"] = 2
    state["last_answer"] = "some answer"
    state["last_verdict"] = "CORRECT"
    result = update_state(state)
    assert result["turn_count"] == 3
    assert result["last_answer"] == ""
    assert "last_verdict" not in result
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_nodes.py -v
```

Expected: `ImportError: cannot import name 'generate_exercise' from 'agent.nodes'`

- [ ] **Step 3: Create agent/nodes.py**

```python
import json
import os
import re
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from .state import TutorState
from .prompts import EXERCISE_PROMPT, GRADE_PROMPT


_PROVIDER_DEFAULTS = {
    "openai": "gpt-5-nano",
    "huggingface": "meta-llama/Llama-3.1-8B-Instruct",
    "gemini": "gemma-4-26b-a4b-it",
}


def _make_llm(model: str, temperature: float):
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "openai":
        return ChatOpenAI(model=model, temperature=temperature)
    if provider == "huggingface":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=os.getenv("HUGGINGFACE_API_KEY", ""),
            openai_api_base="https://api-inference.huggingface.co/v1",
        )
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def _get_llms():
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    default = _PROVIDER_DEFAULTS.get(provider, "gemma-4-26b-a4b-it")
    generator = _make_llm(os.getenv("GENERATOR_MODEL") or default, 0.7)
    grader = _make_llm(os.getenv("GRADER_MODEL") or default, 0.0)
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
```

- [ ] **Step 4: Run to verify it passes**

```bash
uv run pytest tests/test_nodes.py -v
```

Expected: all 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/nodes.py tests/test_nodes.py
git commit -m "feat: add graph nodes, LLM factory, routing functions"
```

---

## Task 6: agent/graph.py — LangGraph StateGraph

**Files:**
- Create: `agent/graph.py`
- Modify: `agent/__init__.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_graph.py`:

```python
from unittest.mock import patch, MagicMock
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from agent.state import make_initial_state


def _node_mock(return_value: dict):
    def node(state):
        return return_value
    return node


@patch("agent.nodes.generate_exercise")
@patch("agent.nodes.check_answer")
def test_full_session(mock_check, mock_gen):
    from agent.graph import build_graph

    mock_gen.side_effect = [
        {"last_exercise": "Ex 1", "past_exercises": ["Ex 1"]},
        {"last_exercise": "Ex 2", "past_exercises": ["Ex 1", "Ex 2"]},
    ]
    mock_check.return_value = {
        "feedback": "Правильно!",
        "last_verdict": "CORRECT",
        "correct_count": 1,
        "incorrect_count": 0,
    }

    graph = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "test-full"}}

    # /start — hits collect_language interrupt
    graph.invoke(make_initial_state(), config=config)

    # Provide language — hits collect_topic interrupt
    state = graph.invoke(Command(resume="English"), config=config)
    assert state["language"] == "English"
    assert state["phase"] == "topic"

    # Provide topic — runs generate_exercise, hits wait_for_answer interrupt
    state = graph.invoke(Command(resume="past simple"), config=config)
    assert state["topic"] == "past simple"
    assert state["last_exercise"] == "Ex 1"
    assert state["phase"] == "active"

    # Answer — check_answer, loops, hits wait_for_answer again with Ex 2
    state = graph.invoke(Command(resume="went"), config=config)
    assert state["last_verdict"] == "CORRECT"
    assert state["feedback"] == "Правильно!"
    assert state["last_exercise"] == "Ex 2"
    assert state["turn_count"] == 1
    assert state["correct_count"] == 1

    # /end — routes to END
    state = graph.invoke(Command(resume="/end"), config=config)
    assert state["phase"] == "done"


@patch("agent.nodes.generate_exercise")
def test_lang_skip(mock_gen):
    from agent.graph import build_graph

    mock_gen.return_value = {"last_exercise": "Ex", "past_exercises": ["Ex"]}

    graph = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "test-lang-skip"}}

    # Start with phase="topic" (language already set, e.g. from /lang command)
    state = graph.invoke(
        make_initial_state(phase="topic", language="Spanish"),
        config=config,
    )
    assert state["language"] == "Spanish"
    assert state["phase"] == "topic"

    # Provide topic — should skip collect_language and run exercise
    state = graph.invoke(Command(resume="ser vs estar"), config=config)
    assert state["topic"] == "ser vs estar"
    assert state["last_exercise"] == "Ex"


@patch("agent.nodes.generate_exercise")
def test_topic_skip(mock_gen):
    from agent.graph import build_graph

    mock_gen.return_value = {"last_exercise": "Ex", "past_exercises": ["Ex"]}

    graph = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "test-topic-skip"}}

    # Start with phase="active" (both language and topic already set)
    state = graph.invoke(
        make_initial_state(phase="active", language="French", topic="présent"),
        config=config,
    )
    assert state["last_exercise"] == "Ex"
    assert state["phase"] == "active"
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_graph.py -v
```

Expected: `ImportError: cannot import name 'build_graph' from 'agent.graph'`

- [ ] **Step 3: Create agent/graph.py**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import TutorState
from .nodes import (
    collect_language,
    collect_topic,
    generate_exercise,
    wait_for_answer,
    check_answer,
    on_correct,
    on_incorrect,
    update_state,
    route_phase,
    route_after_wait,
    route_verdict,
)


def build_graph(checkpointer=None):
    if checkpointer is None:
        checkpointer = MemorySaver()

    builder = StateGraph(TutorState)

    builder.add_node("collect_language", collect_language)
    builder.add_node("collect_topic", collect_topic)
    builder.add_node("generate_exercise", generate_exercise)
    builder.add_node("wait_for_answer", wait_for_answer)
    builder.add_node("check_answer", check_answer)
    builder.add_node("on_correct", on_correct)
    builder.add_node("on_incorrect", on_incorrect)
    builder.add_node("update_state", update_state)

    builder.add_conditional_edges(
        START,
        route_phase,
        {
            "language": "collect_language",
            "topic": "collect_topic",
            "active": "generate_exercise",
            "done": END,
        },
    )

    builder.add_edge("collect_language", "collect_topic")
    builder.add_edge("collect_topic", "generate_exercise")
    builder.add_edge("generate_exercise", "wait_for_answer")

    builder.add_conditional_edges(
        "wait_for_answer",
        route_after_wait,
        {
            "check_answer": "check_answer",
            "end": END,
        },
    )

    builder.add_conditional_edges(
        "check_answer",
        route_verdict,
        {
            "on_correct": "on_correct",
            "on_incorrect": "on_incorrect",
        },
    )

    builder.add_edge("on_correct", "update_state")
    builder.add_edge("on_incorrect", "update_state")
    builder.add_edge("update_state", "generate_exercise")

    return builder.compile(checkpointer=checkpointer)
```

- [ ] **Step 4: Update agent/__init__.py**

```python
from .graph import build_graph
from .state import make_initial_state

__all__ = ["build_graph", "make_initial_state"]
```

- [ ] **Step 5: Run all tests to verify**

```bash
uv run pytest tests/ -v
```

Expected: all tests PASS (test_graph.py tests now included).

- [ ] **Step 6: Commit**

```bash
git add agent/graph.py agent/__init__.py tests/test_graph.py
git commit -m "feat: assemble LangGraph StateGraph with phase routing"
```

---

## Task 7: cli/main.py — Terminal product

**Files:**
- Create: `cli/main.py`

No unit tests for CLI — it is the integration point. Verify with a manual smoke test described below.

- [ ] **Step 1: Create cli/main.py**

```python
#!/usr/bin/env python3
import sys
import os
import io

from dotenv import load_dotenv
load_dotenv()

sys.stdin  = io.TextIOWrapper(sys.stdin.buffer,  encoding="utf-8", errors="replace")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langgraph.types import Command
from agent import build_graph, make_initial_state
import messages

THREAD_ID = "cli-session"


def _show_stats(state: dict) -> None:
    total = state.get("turn_count", 0)
    correct = state.get("correct_count", 0)
    pct = round(correct / total * 100) if total else 0
    print(messages.STATS.format(correct=correct, total=total, pct=pct))


def main():
    graph = build_graph()
    config = {"configurable": {"thread_id": THREAD_ID}}

    graph.invoke(make_initial_state(), config=config)
    print(messages.WELCOME)

    try:
        language = input().strip()
    except (EOFError, KeyboardInterrupt):
        return

    graph.invoke(Command(resume=language), config=config)
    print(messages.CHOOSE_TOPIC)

    try:
        topic = input().strip()
    except (EOFError, KeyboardInterrupt):
        return

    state = graph.invoke(Command(resume=topic), config=config)

    print(f"\n{state['last_exercise']}")
    print(messages.NEXT_PROMPT)

    while True:
        try:
            answer = input().strip()
        except (EOFError, KeyboardInterrupt):
            _show_stats(state)
            break

        state = graph.invoke(Command(resume=answer), config=config)

        if state.get("phase") == "done":
            _show_stats(state)
            break

        verdict = state.get("last_verdict", "")
        feedback = state.get("feedback", "")
        if verdict == "CORRECT":
            print(messages.CORRECT.format(feedback=feedback))
        elif verdict:
            print(messages.INCORRECT.format(feedback=feedback))

        print(f"\n{state['last_exercise']}")
        print(messages.NEXT_PROMPT)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all tests to confirm nothing broke**

```bash
uv run pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 3: Manual smoke test (requires .env with API key)**

```bash
echo -e "English\npast simple\nI went\n/end" | uv run python -m cli.main
```

Expected output (with real LLM):
```
Привет! Я помогу тебе практиковать грамматику. Какой язык изучаем?
Отлично! Тема для практики?

<exercise about past simple>
Напиши ответ или /end чтобы завершить сессию.
✅ Правильно!  (or ❌ Неверно.)
💡 <Russian feedback>

📊 Сессия завершена
Правильно: 1/1 (100%)
Упражнений: 1
```

- [ ] **Step 4: Commit**

```bash
git add cli/main.py
git commit -m "feat: add CLI product"
```

---

## Task 8: tgbot/handlers.py — Shared Telegram handlers

**Files:**
- Create: `tgbot/handlers.py`

- [ ] **Step 1: Create tgbot/handlers.py**

```python
import logging
import os

from dotenv import load_dotenv
load_dotenv()

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from langgraph.types import Command

from agent import build_graph, make_initial_state
import messages

logger = logging.getLogger(__name__)

graph = build_graph()

_user_sessions: dict[int, int] = {}
_user_languages: dict[int, str] = {}


def _config(chat_id: int) -> dict:
    session = _user_sessions.get(chat_id, 0)
    return {"configurable": {"thread_id": f"{chat_id}_{session}"}}


def _new_session(chat_id: int) -> dict:
    _user_sessions[chat_id] = _user_sessions.get(chat_id, 0) + 1
    return _config(chat_id)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    config = _new_session(chat_id)
    graph.invoke(make_initial_state(), config=config)
    await update.message.reply_text(messages.WELCOME)


async def cmd_lang(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    lang = " ".join(context.args).strip() if context.args else ""
    if not lang:
        await update.message.reply_text(
            "Использование: /lang <язык>  Пример: /lang Spanish"
        )
        return
    _user_languages[chat_id] = lang
    config = _new_session(chat_id)
    graph.invoke(make_initial_state(phase="topic", language=lang), config=config)
    await update.message.reply_text(messages.LANG_CHANGED.format(language=lang))


async def cmd_topic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    topic = " ".join(context.args).strip() if context.args else ""
    if not topic:
        await update.message.reply_text(
            "Использование: /topic <тема>  Пример: /topic past tense"
        )
        return
    lang = _user_languages.get(chat_id, "")
    if not lang:
        config = _new_session(chat_id)
        graph.invoke(make_initial_state(), config=config)
        await update.message.reply_text(messages.WELCOME)
        return
    config = _new_session(chat_id)
    state = graph.invoke(
        make_initial_state(phase="active", language=lang, topic=topic),
        config=config,
    )
    await update.message.reply_text(
        f"{state['last_exercise']}\n\n{messages.NEXT_PROMPT}"
    )


async def cmd_end(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    config = _config(chat_id)
    try:
        state = graph.invoke(Command(resume="/end"), config=config)
    except Exception:
        await update.message.reply_text(messages.NO_SESSION)
        return
    total = state.get("turn_count", 0)
    correct = state.get("correct_count", 0)
    pct = round(correct / total * 100) if total else 0
    await update.message.reply_text(
        messages.STATS.format(correct=correct, total=total, pct=pct)
    )


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in _user_sessions:
        await update.message.reply_text(messages.NO_SESSION)
        return

    text = update.message.text.strip()
    config = _config(chat_id)

    try:
        state = graph.invoke(Command(resume=text), config=config)
    except Exception:
        await update.message.reply_text(messages.NO_SESSION)
        return

    if state.get("phase") == "done":
        total = state.get("turn_count", 0)
        correct = state.get("correct_count", 0)
        pct = round(correct / total * 100) if total else 0
        await update.message.reply_text(
            messages.STATS.format(correct=correct, total=total, pct=pct)
        )
        return

    if state.get("phase") == "topic":
        _user_languages[chat_id] = state.get("language", "")
        await update.message.reply_text(messages.CHOOSE_TOPIC)
        return

    reply_parts = []
    verdict = state.get("last_verdict", "")
    feedback = state.get("feedback", "")
    if verdict == "CORRECT":
        reply_parts.append(messages.CORRECT.format(feedback=feedback))
    elif verdict == "INCORRECT":
        reply_parts.append(messages.INCORRECT.format(feedback=feedback))

    exercise = state.get("last_exercise", "")
    if exercise:
        reply_parts.append(f"\n{exercise}\n\n{messages.NEXT_PROMPT}")

    if reply_parts:
        await update.message.reply_text("\n".join(reply_parts))


def build_app(token: str) -> Application:
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("lang", cmd_lang))
    app.add_handler(CommandHandler("topic", cmd_topic))
    app.add_handler(CommandHandler("end", cmd_end))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    return app
```

- [ ] **Step 2: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all PASS (handlers.py has no unit tests — it's integration-level).

- [ ] **Step 3: Commit**

```bash
git add tgbot/handlers.py
git commit -m "feat: add shared Telegram handlers with /start /lang /topic /end"
```

---

## Task 9: tgbot/polling.py + tgbot/webhook.py — Entry points

**Files:**
- Create: `tgbot/polling.py`
- Create: `tgbot/webhook.py`

- [ ] **Step 1: Create tgbot/polling.py**

```python
#!/usr/bin/env python3
"""
Telegram bot — polling mode.

Usage:
    uv run python -m tgbot.polling
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tgbot.handlers import build_app


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        sys.exit(1)
    app = build_app(token)
    logger.info("Starting polling...")
    app.run_polling()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create tgbot/webhook.py**

```python
#!/usr/bin/env python3
"""
Telegram bot — webhook mode.

Usage:
    WEBHOOK_URL=https://your.domain/hook uv run python -m tgbot.webhook

Requires a public URL. For local dev, use ngrok:
    ngrok http 8443
    WEBHOOK_URL=https://<ngrok-id>.ngrok.io/hook uv run python -m tgbot.webhook
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tgbot.handlers import build_app


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        sys.exit(1)
    webhook_url = os.environ.get("WEBHOOK_URL")
    if not webhook_url:
        logger.error("WEBHOOK_URL not set")
        sys.exit(1)
    port = int(os.environ.get("PORT", "8443"))

    app = build_app(token)
    logger.info("Starting webhook on port %d...", port)
    app.run_webhook(
        listen="0.0.0.0",
        port=port,
        webhook_url=webhook_url,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tgbot/polling.py tgbot/webhook.py
git commit -m "feat: add polling and webhook Telegram entry points"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Multi-language — `collect_language` node + `{language}` in all prompts
- [x] Russian UI — `messages.py` imported everywhere
- [x] Russian LLM prompts — `prompts.py` with Russian text
- [x] CLI product — `cli/main.py`
- [x] Telegram bot polling — `tgbot/polling.py`
- [x] Telegram bot webhook — `tgbot/webhook.py`
- [x] Shared handlers — `tgbot/handlers.py`
- [x] Stats display — `_show_stats()` in CLI, inline in bot's `/end` and `on_message`
- [x] `/lang` command — `cmd_lang` in handlers
- [x] `/topic` command — `cmd_topic` in handlers
- [x] Memory-only checkpointer — `MemorySaver()` in `build_graph()`
- [x] Phase routing from START — `route_phase` conditional edge

**No placeholders:** none found.

**Type consistency:**
- `make_initial_state()` defined in Task 3, used in Tasks 6, 7, 8 ✓
- `build_graph()` defined in Task 6, used in Tasks 7, 8 ✓
- `route_phase`, `route_after_wait`, `route_verdict` defined in Task 5, used in Task 6 ✓
- `messages.CORRECT`, `messages.INCORRECT`, `messages.STATS` etc. defined in Task 2, used in Tasks 7, 8 ✓
- `EXERCISE_PROMPT`, `GRADE_PROMPT` defined in Task 4, used in Task 5 ✓
