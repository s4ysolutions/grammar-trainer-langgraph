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
