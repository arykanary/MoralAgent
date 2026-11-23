"""MoralAgent: moral judgment with an LLM-based agent (Xiong et al., Electronics 2025).

Implementation of the four-step analysis plan:
  P1 key information extraction -> T1   (memory -> M_P1)
  P2 intuitive preliminary analysis -> T2   (memory -> M_P2)
  P3 multidimensional analysis + moral-principles knowledge-graph tool -> T3
                                        (memory -> M_P3)
  P4 comprehensive decision making -> T4 (Yes/No)
built with LangGraph and a local Ollama model.
"""

from moralagent.config import Settings, load_settings
from moralagent.graph import build_moralagent
from moralagent.judge import MoralAgent

__all__ = ["MoralAgent", "Settings", "build_moralagent", "load_settings"]
__version__ = "0.1.0"
