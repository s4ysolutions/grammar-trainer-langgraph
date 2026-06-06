from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import TutorState
from . import nodes as _nodes


def build_graph(checkpointer=None):
    if checkpointer is None:
        checkpointer = MemorySaver()

    _nodes.validate_config()

    builder = StateGraph(TutorState)

    # Use lambdas so that patches applied to agent.nodes are respected at call time.
    builder.add_node("collect_language", lambda s: _nodes.collect_language(s))
    builder.add_node("collect_topic", lambda s: _nodes.collect_topic(s))
    builder.add_node("generate_exercise", lambda s: _nodes.generate_exercise(s))
    builder.add_node("wait_for_answer", lambda s: _nodes.wait_for_answer(s))
    builder.add_node("check_answer", lambda s: _nodes.check_answer(s))
    builder.add_node("on_correct", lambda s: _nodes.on_correct(s))
    builder.add_node("on_incorrect", lambda s: _nodes.on_incorrect(s))
    builder.add_node("update_state", lambda s: _nodes.update_state(s))

    builder.add_conditional_edges(
        START,
        lambda s: _nodes.route_phase(s),
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
        lambda s: _nodes.route_after_wait(s),
        {
            "check_answer": "check_answer",
            "end": END,
        },
    )

    builder.add_conditional_edges(
        "check_answer",
        lambda s: _nodes.route_verdict(s),
        {
            "on_correct": "on_correct",
            "on_incorrect": "on_incorrect",
        },
    )

    builder.add_edge("on_correct", "update_state")
    builder.add_edge("on_incorrect", "update_state")
    builder.add_edge("update_state", "generate_exercise")

    return builder.compile(checkpointer=checkpointer)
