import importlib.util
import os
from pathlib import Path

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

ERROR = "Произошла ошибка. Попробуй ещё раз или начни сессию заново с /start."

RATE_LIMIT = (
    "Провайдер перегружен (429). Подожди минуту и попробуй снова — "
    "используй /topic чтобы начать упражнение заново."
)

_ui_lang = os.getenv("UI_LANG", "").strip().lower()
if _ui_lang:
    _locale_file = Path(__file__).parent / f"messages-{_ui_lang}.py"
    if _locale_file.exists():
        _spec = importlib.util.spec_from_file_location(f"_messages_{_ui_lang}", _locale_file)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        for _k, _v in vars(_mod).items():
            if _k.isupper():
                globals()[_k] = _v
