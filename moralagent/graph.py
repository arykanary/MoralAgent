"""MoralAgent graph built with LangGraph.

Topology:

    START
      -> extract_key_info        (P1, formatted output = {Json} formatting tool)
      -> remember_p1             (MM summarizes P1'  -> M_P1)
      -> preliminary_analysis    (P2 = Regular_2 + {M_P1} + {T1})
      -> remember_p2             (MM summarizes P2'  -> M_P2)
      -> multidim_analysis       (P3 = Regular_3 + {M_P2} + {Text} + {T1} + {T2};
                                  the LLM may call the moral-principles tool {T_k})
          ^  | tool_calls -> moral_principles_tool
          +--+   no tool_calls
      -> remember_p3             (MM summarizes P3'  -> M_P3)
      -> comprehensive_decision  (P4 = Regular_4 + {M_P3} -> T4)
      -> END
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from moralagent.config import Settings, load_settings
from moralagent.memory import MemoryModule
from moralagent.prompts import P1_KEY_INFO, P2_PRELIMINARY, P3_MULTIDIM, P4_DECISION
from moralagent.schemas import KeyInformation, MoralJudgment
from moralagent.tools import lookup_moral_principles


class MoralAgentState(TypedDict, total=False):
    text: str
    t1: KeyInformation
    t2: str
    tk: str
    t3: str
    t4: MoralJudgment
    p1_prime: str
    p2_prime: str
    p3_prime: str
    mp1: str
    mp2: str
    mp3: str
    md_messages: Annotated[list[Any], lambda old, new: new]


def make_llm(settings: Settings) -> BaseChatModel:
    """Create the Ollama chat model used by the agent."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.model,
        base_url=settings.base_url,
        temperature=settings.temperature,
        num_ctx=settings.num_ctx,
    )


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(str(part) for part in content if isinstance(part, str))


def build_moralagent(
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    memory: MemoryModule | None = None,
):
    """Compile the MoralAgent graph.

    Args:
        settings: Ollama/model settings; defaults are loaded from the environment.
        llm: Explicit LLM override (useful for testing with a local fake model).
        memory: Explicit memory module override (must use the same LLM).
    """
    settings = settings or load_settings()
    llm = llm if llm is not None else make_llm(settings)
    memory = memory if memory is not None else MemoryModule(llm)
    tool_llm = llm.bind_tools([lookup_moral_principles])

    def extract_key_info(state: MoralAgentState) -> dict:
        rendered = P1_KEY_INFO.format(text=state["text"])
        t1 = llm.with_structured_output(KeyInformation).invoke(rendered)
        p1_prime = rendered + "\n\nKey information: " + t1.model_dump_json()
        return {"t1": t1, "p1_prime": p1_prime}

    def remember_p1(state: MoralAgentState) -> dict:
        return {"mp1": memory.remember("key information extraction", state["p1_prime"])}

    def preliminary_analysis(state: MoralAgentState) -> dict:
        rendered = P2_PRELIMINARY.format(mp1=state["mp1"], t1=state["t1"].model_dump_json())
        t2 = _as_text(llm.invoke(rendered).content).strip()
        return {"t2": t2, "p2_prime": rendered + "\n\nPreliminary analysis: " + t2}

    def remember_p2(state: MoralAgentState) -> dict:
        return {"mp2": memory.remember("preliminary analysis", state["p2_prime"])}

    def multidim_analysis(state: MoralAgentState) -> dict:
        messages = state.get("md_messages")
        if not messages:
            rendered = P3_MULTIDIM.format(
                mp2=state["mp2"],
                text=state["text"],
                t1=state["t1"].model_dump_json(),
                t2=state["t2"],
            )
            messages = [
                SystemMessage(_as_text(rendered)),  # type: ignore[list-item]
                HumanMessage(
                    "Analyze this moral scenario now. Call `lookup_moral_principles` first, "
                    "then give the multidimensional analysis."
                ),
            ]
        response = tool_llm.invoke(messages)
        return {"md_messages": [*messages, response]}

    def call_moral_principles_tool(state: MoralAgentState) -> dict:
        last = state["md_messages"][-1]
        messages = list(state["md_messages"])
        observations: list[str] = []
        for tool_call in last.tool_calls:
            observation = lookup_moral_principles.invoke(tool_call["args"])
            observations.append(str(observation))
            messages.append(
                ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )
        update: dict = {"md_messages": messages}
        if observations:
            update["tk"] = "\n\n".join(observations)
        return update

    def route_after_multidim(state: MoralAgentState) -> str:
        last = state["md_messages"][-1]
        if getattr(last, "tool_calls", None):
            return "moral_principles_tool"
        return "remember_p3"

    def remember_p3(state: MoralAgentState) -> dict:
        t3 = _as_text(state["md_messages"][-1].content).strip()
        p3_prime = "Task: multidimensional analysis of the moral scenario.\n\nAnalysis result: " + t3
        return {"t3": t3, "p3_prime": p3_prime, "mp3": memory.remember("multidimensional analysis", p3_prime)}

    def comprehensive_decision(state: MoralAgentState) -> dict:
        rendered = P4_DECISION.format(mp3=state["mp3"])
        t4 = llm.with_structured_output(MoralJudgment).invoke(rendered)
        return {"t4": t4}

    graph = StateGraph(MoralAgentState)
    graph.add_node("extract_key_info", extract_key_info)
    graph.add_node("remember_p1", remember_p1)
    graph.add_node("preliminary_analysis", preliminary_analysis)
    graph.add_node("remember_p2", remember_p2)
    graph.add_node("multidim_analysis", multidim_analysis)
    graph.add_node("moral_principles_tool", call_moral_principles_tool)
    graph.add_node("remember_p3", remember_p3)
    graph.add_node("comprehensive_decision", comprehensive_decision)

    graph.add_edge(START, "extract_key_info")
    graph.add_edge("extract_key_info", "remember_p1")
    graph.add_edge("remember_p1", "preliminary_analysis")
    graph.add_edge("preliminary_analysis", "remember_p2")
    graph.add_edge("remember_p2", "multidim_analysis")
    graph.add_conditional_edges(
        "multidim_analysis",
        route_after_multidim,
        {"moral_principles_tool": "moral_principles_tool", "remember_p3": "remember_p3"},
    )
    graph.add_edge("moral_principles_tool", "multidim_analysis")
    graph.add_edge("remember_p3", "comprehensive_decision")
    graph.add_edge("comprehensive_decision", END)

    return graph.compile()
