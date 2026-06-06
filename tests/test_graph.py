from unittest.mock import patch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from agent.state import make_initial_state


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
