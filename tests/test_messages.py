import messages


def test_all_keys_present():
    for attr in [
        "WELCOME", "CHOOSE_TOPIC", "NEXT_PROMPT", "STATS",
        "CORRECT", "INCORRECT", "LANG_CHANGED", "TOPIC_CHANGED", "NO_SESSION",
    ]:
        assert hasattr(messages, attr), f"messages.{attr} missing"
        assert isinstance(getattr(messages, attr), str), f"messages.{attr} not a str"
        assert getattr(messages, attr), f"messages.{attr} is empty"


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
