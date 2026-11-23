"""Run MoralAgent on a moral scenario from the command line.

Example:
    python -m moralagent.main "A maintenance worker..." --model llama3.1 --show-steps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from moralagent.config import load_settings
from moralagent.judge import MoralAgent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Moral judgment with a LLM-based agent (Ollama)")
    parser.add_argument(
        "scenario",
        nargs="?",
        help="The moral scenario text (the behavior to judge is described at the end)",
    )
    parser.add_argument("-f", "--file", type=Path, help="Read the scenario from a text file")
    parser.add_argument("--model", help="Ollama model name (env: OLLAMA_MODEL, default: llama3.1)")
    parser.add_argument(
        "--base-url", help="Ollama base URL (env: OLLAMA_BASE_URL, default: http://localhost:11434)"
    )
    parser.add_argument(
        "--temperature", type=float, help="Sampling temperature (env: OLLAMA_TEMPERATURE, default: 0)"
    )
    parser.add_argument("--show-steps", action="store_true", help="Print each analysis step")
    args = parser.parse_args(argv)

    if args.scenario:
        text = args.scenario
    elif args.file:
        text = args.file.read_text(encoding="utf-8")
    else:
        text = sys.stdin.read().strip()
    if not text:
        parser.error("a moral scenario is required (argument, --file, or stdin)")

    settings = load_settings(model=args.model, base_url=args.base_url, temperature=args.temperature)
    agent = MoralAgent(settings=settings)
    judgment = agent.judge(text, stream_steps=args.show_steps)
    if not args.show_steps:
        print(f"{judgment.judgment} - {judgment.reasons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
