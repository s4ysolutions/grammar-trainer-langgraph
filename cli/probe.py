#!/usr/bin/env python3
"""Smoke-test the configured LLM provider/model. Exit 0 = OK, exit 1 = failed."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from agent.nodes import validate_config, _get_llms, _extract_text


def main():
    provider = os.getenv("LLM_PROVIDER", "gemini")
    print(f"Provider : {provider}")

    print("Config   : checking...")
    try:
        validate_config()
    except ValueError as e:
        print(f"FAIL     : {e}")
        sys.exit(1)
    print("Config   : OK")

    print("LLM call : sending test prompt...")
    try:
        generator, grader = _get_llms()
        t0 = time.monotonic()
        response = _extract_text(generator.invoke("Reply with exactly one word: OK"))
        elapsed = time.monotonic() - t0
    except Exception as e:
        print(f"FAIL     : {e}")
        sys.exit(1)

    print(f"LLM call : OK ({elapsed:.1f}s) — response: {response[:120]!r}")
    print("Result   : PASS")


if __name__ == "__main__":
    main()
