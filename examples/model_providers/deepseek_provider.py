"""Example that uses DeepSeek's OpenAI-compatible API directly.

DeepSeek serves an OpenAI-compatible Chat Completions endpoint, so you can
wire it up with the standard ``AsyncOpenAI`` client and
``OpenAIChatCompletionsModel``. Set ``DEEPSEEK_API_KEY`` (or pass
``--api-key``) and run:

    uv run examples/model_providers/deepseek_provider.py

The default model is ``deepseek-v4-flash``. The ``deepseek-chat`` and
``deepseek-reasoner`` aliases were deprecated on 2026-07-24, so use the v4
family names directly (``deepseek-v4-flash`` or ``deepseek-v4-pro``).
"""

from __future__ import annotations

import argparse
import asyncio
import os

from openai import AsyncOpenAI

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    set_tracing_disabled,
)
from agents.decorators import tool

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

# DeepSeek does not support OpenAI's tracing endpoint, so disable tracing.
set_tracing_disabled(disabled=True)


@tool
def get_weather(city: str):
    """Get the current weather for a city."""

    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."


async def main(model: str, api_key: str):
    if api_key == "dummy":
        print("Skipping run because no valid DEEPSEEK_API_KEY was provided.")
        return

    client = AsyncOpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    chat_completions_model = OpenAIChatCompletionsModel(
        model=model,
        openai_client=client,
    )

    agent = Agent(
        name="DeepSeek Assistant",
        instructions="You only respond in haikus.",
        model=chat_completions_model,
        tools=[get_weather],
    )

    result = await Runner.run(agent, "What's the weather in Tokyo?")
    print(result.final_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("DEEPSEEK_API_KEY", "dummy"),
    )
    args = parser.parse_args()

    if not args.model:
        print(f"Using default model: {args.model}")
    if not args.api_key:
        print("Using DEEPSEEK_API_KEY from environment (or dummy placeholder).")

    asyncio.run(main(args.model, args.api_key))
