from typing import TypedDict, Literal


class TutorState(TypedDict):
    phase: Literal["language", "topic", "active", "done"]
    language: str
    topic: str
    past_exercises: list[str]
    last_exercise: str
    last_answer: str
    feedback: str
    last_verdict: str
    turn_count: int
    correct_count: int
    incorrect_count: int


def make_initial_state(
    phase: str = "language",
    language: str = "",
    topic: str = "",
) -> TutorState:
    return {
        "phase": phase,
        "language": language,
        "topic": topic,
        "past_exercises": [],
        "last_exercise": "",
        "last_answer": "",
        "feedback": "",
        "last_verdict": "",
        "turn_count": 0,
        "correct_count": 0,
        "incorrect_count": 0,
    }
