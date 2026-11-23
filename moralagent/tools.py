"""Moral-principles knowledge graph exposed as a callable tool (T_k).

The paper condenses internationally recognized AI-ethics norms from
governments, institutions, and academic bodies into a knowledge graph of
principle nodes (fairness, justice, peace, freedom, cooperation, respect for
others, safety) plus guiding principles for special situations and vulnerable
groups. The LLM calls `lookup_moral_principles` during the multidimensional
analysis step to retrieve the relevant principles.
"""

from __future__ import annotations

import re

from langchain_core.tools import tool

from moralagent.principles_data import PRINCIPLES

_WORD = re.compile(r"[a-z]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _match_score(name: str, principle: dict, query_tokens: set[str]) -> int:
    """Score how relevant a principle node is to the query."""
    keywords = {k.lower() for k in principle["keywords"]}
    score = len(query_tokens & keywords) * 3
    score += len(query_tokens & (_tokenize(name))) * 2
    return score


def _neighbors(name: str) -> list[str]:
    return PRINCIPLES.get(name, {}).get("related", [])


@tool
def lookup_moral_principles(query: str) -> str:
    """Look up moral principles relevant to a moral scenario or situation.

    Use this tool before making a multidimensional moral analysis. Pass a
    short description of the scenario, the behavior to judge, or a specific
    concern (e.g. 'cutting in line affects other people waiting',
    'damaging someone else's property').

    Args:
        query: Short description of the scenario, behavior, or concern.

    Returns:
        The relevant moral principles with their descriptions, together with
        related guiding principles for special situations and specific groups.
    """
    query_tokens = _tokenize(query)
    scored = sorted(
        ((_match_score(name, p, query_tokens), name) for name, p in PRINCIPLES.items()),
        key=lambda item: item[0],
        reverse=True,
    )
    matches = [name for score, name in scored[:3] if score > 0]
    if not matches:
        matches = [name for name in PRINCIPLES if PRINCIPLES[name]["tier"] == "basic"]
    extra: list[str] = []
    for name in matches:
        for neighbor in _neighbors(name):
            if neighbor not in matches and neighbor in PRINCIPLES and neighbor not in extra:
                extra.append(neighbor)
    lines = []
    for name in matches + extra:
        p = PRINCIPLES[name]
        related = ", ".join(p["related"]) if p["related"] else "none"
        lines.append(
            f"- {name} ({p['tier']}): {p['description']} Related principles: {related}."
        )
    return "\n".join(lines)
