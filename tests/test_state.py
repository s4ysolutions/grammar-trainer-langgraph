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
