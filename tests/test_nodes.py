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
