# english-grammar

Grammar tutor bot. Supports any language. UI language is configurable (default: Russian). Exercises in the target language.

## Setup

```bash
uv sync --extra dev
cp .env.example .env
# Edit .env — set API key for your LLM provider
```

## Configuration

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | No | `gemini` (default) \| `openai` \| `huggingface` \| `openrouter` |
| `GOOGLE_API_KEY` | If gemini | Google AI API key |
| `OPENAI_API_KEY` | If openai | OpenAI API key |
| `HUGGINGFACE_API_KEY` | If huggingface | HuggingFace token |
| `OPENROUTER_API_KEY` | If openrouter | OpenRouter API key |
| `HUGGINGFACE_BASE_URL` | No | Override HuggingFace endpoint (default: `https://router.huggingface.co/hf-inference/v1`) |
| `OPENROUTER_BASE_URL` | No | Override OpenRouter endpoint (default: `https://openrouter.ai/api/v1`) |
| `UI_LANG` | No | UI language code — loads `messages-{UI_LANG}.py` if present (default: Russian) |
| `GENERATOR_MODEL` | No | Override exercise generator model |
| `GRADER_MODEL` | No | Override answer grader model |
| `TELEGRAM_BOT_TOKEN` | For bot | Token from @BotFather |
| `WEBHOOK_URL` | Webhook mode | Public URL, e.g. `https://abc.ngrok.io/hook` |
| `PORT` | Webhook mode | Local port (default `8443`) |
| `TELEGRAM_WEBHOOK_SECRET` | Webhook mode | Secret token Telegram sends in `X-Telegram-Bot-Api-Secret-Token`; requests without it are rejected with 401. Generate with `uv run python -m cli.gen_secret` |

## Launch

### List available models

```bash
uv run python -m cli.list_models
```

### Probe (verify provider/model works)

```bash
uv run python -m cli.probe
```

### CLI

```bash
uv run python -m cli.main
```

### Telegram bot — polling (dev)

```bash
uv run python -m tgbot.polling
```

### Telegram bot — webhook (prod)

```bash
# Generate and save a webhook secret (one-time):
uv run python -m cli.gen_secret

# With ngrok:
ngrok http 8443
WEBHOOK_URL=https://<ngrok-id>.ngrok.io/hook uv run python -m tgbot.webhook
```

## Bot commands

| Command | Action |
|---|---|
| `/start` | Start new session, ask for language |
| `/lang <language>` | Change language, restart from topic selection |
| `/topic <topic>` | Change topic, restart exercises |
| `/end` | End session, show stats |

## Tests

```bash
uv run pytest
```
