#!/usr/bin/env python3
import sys
import os
import io

from dotenv import load_dotenv
load_dotenv()

sys.stdin  = io.TextIOWrapper(sys.stdin.buffer,  encoding="utf-8", errors="replace")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langgraph.types import Command
from agent import build_graph, make_initial_state
import messages

THREAD_ID = "cli-session"


def _show_stats(state: dict) -> None:
    total = state.get("turn_count", 0)
    correct = state.get("correct_count", 0)
    pct = round(correct / total * 100) if total else 0
    print(messages.STATS.format(correct=correct, total=total, pct=pct))


def main():
    graph = build_graph()
    config = {"configurable": {"thread_id": THREAD_ID}}

    graph.invoke(make_initial_state(), config=config)
    print(messages.WELCOME)

    try:
        language = input().strip()
    except (EOFError, KeyboardInterrupt):
        return

    graph.invoke(Command(resume=language), config=config)
    print(messages.CHOOSE_TOPIC)

    try:
        topic = input().strip()
    except (EOFError, KeyboardInterrupt):
        return

    state = graph.invoke(Command(resume=topic), config=config)

    print(f"\n{state['last_exercise']}")
    print(messages.NEXT_PROMPT)

    while True:
        try:
            answer = input().strip()
        except (EOFError, KeyboardInterrupt):
            _show_stats(state)
            break

        if answer == "/end":
            _show_stats(state)
            break

        state = graph.invoke(Command(resume=answer), config=config)

        if state.get("phase") == "done":
            _show_stats(state)
            break

        verdict = state.get("last_verdict", "")
        feedback = state.get("feedback", "")
        if verdict == "CORRECT":
            print(messages.CORRECT.format(feedback=feedback))
        elif verdict:
            print(messages.INCORRECT.format(feedback=feedback))

        print(f"\n{state['last_exercise']}")
        print(messages.NEXT_PROMPT)


if __name__ == "__main__":
    main()
