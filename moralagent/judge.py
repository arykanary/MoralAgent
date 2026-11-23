"""High-level MoralAgent interface: judge(Text) -> MoralJudgment."""

from __future__ import annotations

from typing import Any, Iterable

from langchain_core.language_models import BaseChatModel

from moralagent.config import Settings, load_settings
from moralagent.graph import build_moralagent, make_llm
from moralagent.memory import MemoryModule
from moralagent.schemas import MoralJudgment


class MoralAgent:
    """Moral judgment agent backed by a local Ollama model.

    Runs the paper's four-step analysis plan (key information extraction ->
    preliminary analysis -> multidimensional analysis with the moral-principles
    tool -> comprehensive decision) as a LangGraph graph, coordinated with a
    summarizing memory module.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        llm: BaseChatModel | None = None,
    ):
        self.settings = settings or load_settings()
        self.llm = llm if llm is not None else make_llm(self.settings)
        self.memory = MemoryModule(self.llm)
        self.graph = build_moralagent(settings=self.settings, llm=self.llm, memory=self.memory)

    def judge(self, text: str, stream_steps: bool = False) -> MoralJudgment:
        """Run the full moral judgment process on a moral scenario.

        Args:
            text: The moral scenario text ending with the behavior to judge.
            stream_steps: If True, print each step's output as it completes.

        Returns:
            The final judgment T4 with reasons.
        """
        self.memory.clear()
        if not stream_steps:
            result = self.graph.invoke({"text": text}, config={"recursion_limit": 30})
            return result["t4"]

        t4: MoralJudgment | None = None
        for update in self.graph.stream(
            {"text": text}, stream_mode="updates", config={"recursion_limit": 30}
        ):
            for node, value in update.items():
                self._print_step(node, value)
                if node == "comprehensive_decision":
                    t4 = value["t4"]
        if t4 is None:
            raise RuntimeError("The graph finished without producing a judgment")
        return t4

    def _print_step(self, node: str, value: dict) -> None:
        if node == "extract_key_info" and "t1" in value:
            print("=== Step 1: key information ===")
            print(value["t1"].model_dump_json(indent=2))
        elif node == "preliminary_analysis" and "t2" in value:
            print("=== Step 2: preliminary analysis ===")
            print(value["t2"])
        elif node == "moral_principles_tool" and value.get("tk"):
            print("--- retrieved moral principles (T_k) ---")
            print(value["tk"])
        elif node == "remember_p3" and "t3" in value:
            print("=== Step 3: multidimensional analysis ===")
            print(value["t3"])
        elif node == "comprehensive_decision" and "t4" in value:
            print("=== Step 4: decision ===")
            print(f"{value['t4'].judgment} - {value['t4'].reasons}")

    @property
    def memory_entries(self) -> Iterable[tuple[str, str, str]]:
        """Stored (step, P_i', M_Pi) tuples from the last judgment."""
        return self.memory.entries
