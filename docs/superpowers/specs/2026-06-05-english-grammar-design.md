# english-grammar: Design Spec

**Date:** 2026-06-05  
**Project:** `/Users/dsa/s4y/llm-agents/english-grammar`  
**Source projects:** `english-grammar-gc` (TypeScript/Golem, multi-language), `english-grammar-lg` (Python/LangGraph, English-only)

---

## Goal

Python-based grammar tutor that combines:
- Multi-language support from `english-grammar-gc`
- Python CLI + Telegram bot architecture from `english-grammar-lg`
- Russian UI system messages
- Session scoring/stats display

---

## Project Structure

```
english-grammar/
├── agent/
│   ├── __init__.py
│   ├── state.py        # TutorState TypedDict
│   ├── nodes.py        # All graph nodes + LLM factory
│   ├── graph.py        # LangGraph StateGraph definition
│   └── prompts.py      # Prompt templates with {language} injection
├── cli/
│   └── main.py         # Single-user terminal product
├── tgbot/
│   ├── handlers.py     # Shared: message handlers, command handlers
│   ├── polling.py      # Entry point: run_polling()
│   └── webhook.py      # Entry point: run_webhook(...)
├── messages.py         # Russian UI strings (all bot/CLI user-facing text)
├── pyproject.toml
├── .env.example
└── README.md
```

---

## State

```python
class TutorState(TypedDict):
    phase: Literal["language", "topic", "active", "done"]
    language: str
    topic: str
    past_exercises: list[str]
    last_exercise: str
    last_answer: str
    feedback: str
    last_verdict: str        # "CORRECT" | "INCORRECT" | ""
    turn_count: int
    correct_count: int
    incorrect_count: int
```

`phase` drives initial routing:
- `"language"` → `collect_language` node
- `"topic"` → `collect_topic` node
- `"active"` → exercise loop
- `"done"` → END (stats displayed by CLI/bot handlers)

---

## Graph Flow

```
START
  → collect_language   (interrupt: bot asks language, user replies)
  → collect_topic      (interrupt: bot asks topic, user replies)
  → generate_exercise  (LLM call, temp=0.7)
  → wait_for_answer    (interrupt: user types answer or /end)
  → check_answer       (LLM call, temp=0.0, returns JSON)
  → route_verdict      (conditional edge)
      CORRECT   → on_correct   → update_state → generate_exercise
      INCORRECT → on_incorrect → update_state → generate_exercise
      /end      → END
```

`/end` detected in `wait_for_answer` node: if input equals `/end`, sets `phase="done"` and returns without calling `check_answer`. Conditional edge after `wait_for_answer` routes to `END` when `phase=="done"`, else to `check_answer`. Stats are displayed by the CLI and bot handlers, not by a graph node.

---

## LLM Integration

### Provider factory (`agent/nodes.py`)

```python
LLM_PROVIDER=gemini|openai|huggingface   # env var
GENERATOR_MODEL=...                       # optional override
GRADER_MODEL=...                          # optional override
```

Defaults:
- `gemini`: `gemma-4-26b-a4b-it`
- `openai`: `gpt-5-nano`
- `huggingface`: `meta-llama/Llama-3.1-8B-Instruct`

Uses LangChain wrappers (`ChatGoogleGenerativeAI`, `ChatOpenAI`).

### Prompt templates (`agent/prompts.py`)

Prompts are in Russian so LLM feedback is in Russian. Exercise content is still in `{language}`.

```python
EXERCISE_PROMPT = (
    "Ты репетитор по грамматике {language}. "
    "Составь одно короткое упражнение на тему: '{topic}'. "
    "Попроси пользователя вставить пропущенное слово, исправить предложение или выбрать правильную форму. "
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
    '{"verdict": "CORRECT" or "INCORRECT", "feedback": "краткое объяснение на русском"}'
)
```

---

## Russian System Messages (`messages.py`)

```python
WELCOME      = "Привет! Я помогу тебе практиковать грамматику. Какой язык изучаем?"
CHOOSE_TOPIC = "Отлично! Тема для практики?"
CORRECT      = "✅ Правильно!\n💡 {feedback}"
INCORRECT    = "❌ Неверно.\n💡 {feedback}"
NEXT_PROMPT  = "Напиши ответ или /end чтобы завершить сессию."
STATS        = (
    "📊 Сессия завершена\n"
    "Правильно: {correct}/{total} ({pct}%)\n"
    "Упражнений: {total}"
)
```

All user-facing strings live only in `messages.py`. Nodes import from here.

---

## Checkpointer

Memory-only (`MemorySaver`). Sessions do not survive restart. Simple default — no external DB required.

---

## Telegram Bot

### `tgbot/handlers.py`
- `/start` → welcome message, start graph for this `chat_id`
- `/lang <language>` → reset graph for this user, set new language, resume from collect_topic
- `/topic <topic>` → reset to generate_exercise with new topic (language unchanged)
- `/end` → inject `/end` signal into graph, show stats
- Text messages → resume graph via `Command(resume=text)`

### `tgbot/polling.py`
```python
application.run_polling()
```
No public URL needed. Default for dev.

### `tgbot/webhook.py`
```python
application.run_webhook(
    listen="0.0.0.0",
    port=int(os.getenv("PORT", "8443")),
    webhook_url=os.getenv("WEBHOOK_URL"),
)
```
Requires `WEBHOOK_URL` env var. For prod (with ngrok or real domain).

---

## CLI Product (`cli/main.py`)

Single-user interactive loop:
- Builds graph once
- Fixed `thread_id = "cli-session"`
- Accepts `/end` to quit and show stats
- Prints all Russian messages to terminal

---

## Dependencies

```toml
[project]
name = "english-grammar"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2",
    "langchain-openai>=0.1",
    "langchain-google-genai>=4.0",
    "python-telegram-bot>=21.0",
    "python-dotenv>=1.0",
]
```

No `langgraph-checkpoint-sqlite` — memory-only checkpointer.

---

## Environment Variables

```env
# Required for bot
TELEGRAM_BOT_TOKEN=

# LLM provider (gemini | openai | huggingface)
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

---

## What's New vs Source Projects

| Feature | english-grammar-gc | english-grammar-lg | english-grammar |
|---|---|---|---|
| Language | multi | English-only | multi ✓ |
| CLI | no | yes | yes ✓ |
| Telegram | webhook only | polling only | both ✓ |
| UI language | English | English | Russian ✓ |
| Stats display | no | no | yes ✓ |
| Runtime | TypeScript/WASM | Python | Python ✓ |
| Persistence | in-memory | sqlite | in-memory |
