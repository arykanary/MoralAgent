"""Memory module (MM) of MoralAgent.

After each analysis step the updated template P_i' (instruction plus result)
is passed to the memory module, which stores it and uses the LLM to summarize
the important information, obtaining M_Pi. M_Pi then fills the variable fields
of the next dynamic prompt template. This mirrors Len(P') storage plus LLM
summarization as described in the paper (Eqs. 1-2, 5, 8, 11).
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

from moralagent.prompts import SUMMARIZE

_SUMMARY_PROMPT = PromptTemplate.from_template(SUMMARIZE)


class MemoryModule:
    """Summarizing memory coordinated with the dynamic prompt templates."""

    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._entries: list[tuple[str, str, str]] = []

    @property
    def entries(self) -> list[tuple[str, str, str]]:
        """Stored (step, updated prompt P_i', summary M_Pi) tuples."""
        return list(self._entries)

    def remember(self, step: str, updated_prompt: str) -> str:
        """Store P_i' and summarize it into M_Pi."""
        summary = _SUMMARY_PROMPT.invoke({"step": step, "updated_prompt": updated_prompt})
        m_pi = self._llm.invoke(summary.to_messages()).content
        if isinstance(m_pi, list):
            m_pi = "".join(str(part) for part in m_pi)
        self._entries.append((step, updated_prompt, str(m_pi).strip()))
        return str(m_pi).strip()

    def clear(self) -> None:
        self._entries.clear()
