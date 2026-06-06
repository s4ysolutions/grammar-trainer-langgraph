#!/usr/bin/env python3
"""List available models for the configured LLM provider."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from agent.nodes import validate_config


def list_gemini():
    import google.genai as genai
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    for m in client.models.list():
        if "generateContent" in getattr(m, "supported_actions", []):
            print(m.name)


def list_openai():
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    for m in sorted(client.models.list(), key=lambda m: m.id):
        print(m.id)


def list_huggingface():
    from agent.nodes import _huggingface_base_url
    import openai
    base_url = _huggingface_base_url()
    client = openai.OpenAI(api_key=os.environ["HUGGINGFACE_API_KEY"], base_url=base_url)
    for m in sorted(client.models.list(), key=lambda m: m.id):
        print(m.id)


def list_openrouter():
    from agent.nodes import _openrouter_base_url
    import openai
    base_url = _openrouter_base_url()
    client = openai.OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=base_url)
    for m in sorted(client.models.list(), key=lambda m: m.id):
        print(m.id)


def main():
    provider = os.getenv("LLM_PROVIDER", "gemini")
    print(f"Provider: {provider}\n")
    try:
        validate_config()
    except ValueError as e:
        print(f"FAIL: {e}")
        sys.exit(1)

    try:
        if provider == "gemini":
            list_gemini()
        elif provider == "openai":
            list_openai()
        elif provider == "huggingface":
            list_huggingface()
        elif provider == "openrouter":
            list_openrouter()
    except Exception as e:
        print(f"FAIL: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
