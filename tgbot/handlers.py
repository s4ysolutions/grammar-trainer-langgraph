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
    await graph.ainvoke(make_initial_state(), config=config)
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
    await graph.ainvoke(make_initial_state(phase="topic", language=lang), config=config)
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
        await graph.ainvoke(make_initial_state(), config=config)
        await update.message.reply_text(messages.WELCOME)
        return
    config = _new_session(chat_id)
    state = await graph.ainvoke(
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
        state = await graph.ainvoke(Command(resume="/end"), config=config)
    except Exception:
        logger.exception("cmd_end: graph.ainvoke failed for chat_id=%s", chat_id)
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
        state = await graph.ainvoke(Command(resume=text), config=config)
    except Exception:
        logger.exception("on_message: graph.ainvoke failed for chat_id=%s", chat_id)
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
