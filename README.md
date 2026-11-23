# MoralAgent

Implementation of **MoralAgent** from *"Moral Judgment with a Large Language Model-Based Agent"*
(Xiong, Gu, Liang, Yin — Electronics 2025, 14, 2580, https://doi.org/10.3390/electronics14132580),
built with the latest **LangGraph** and a local **Ollama** model.

## Architecture

The agent condenses six human moral-judgment theories (Social Intuitionist Model, Causal-Intentional
Model, Dual Process Theory, Path Model of Blame, Dynamic Systems Model, SIP-MDM) into a four-step
analysis plan of dynamic prompt templates P1–P4, coordinated with a summarizing memory module
(MM) and a moral-principles knowledge graph tool:

```
START
  -> extract_key_info        P1 = Regular1 + {Text} + {T1} + {Json}   -> T1 (who/what/when/where/why/how)
  -> remember_p1             MM summarizes P1' -> M_P1
  -> preliminary_analysis    P2 = Regular2 + {M_P1} + {T1}            -> T2 (intuitive first pass)
  -> remember_p2             MM summarizes P2' -> M_P2
  -> multidim_analysis       P3 = Regular3 + {M_P2} + {Text} + {T1} + {T2}
       |  (LLM calls the moral-principles knowledge-graph tool -> T_k)
       v
  -> remember_p3             MM summarizes P3' (+T_k) -> M_P3
  -> comprehensive_decision  P4 = Regular4 + {M_P3}                   -> T4 (Yes/No + reasons)
END
```

- **Dynamic prompt templates** (`moralagent/prompts.py`): fixed fields give the step's task
  (Regular1..Regular4); variable fields are replaced as the task progresses.
- **Memory module** (`moralagent/memory.py`): after every step, the updated template P_i'
  (instruction + result) is stored and summarized by the LLM into M_Pi, which fills the variable
  fields of the next template.
- **Moral principles tool** (`moralagent/tools.py`): a knowledge graph of internationally
  recognized AI-ethics principles (basic principles such as fairness, justice, peace, freedom,
  cooperation, respect for others, safety; guiding principles for special situations and vulnerable
  groups). During multidimensional analysis the LLM retrieves the relevant principles via
  `lookup_moral_principles`.
- **Formatting/formatted output**: steps 1 and 4 return strictly formatted JSON equivalent to the
  paper's formatting tool, via LangChain structured output.

## Setup

1. Install Ollama and pull a tool-calling model:

```bash
ollama pull llama3.1     # or qwen2.5, mistral-nemo, etc. (tool calling required)
```

2. Install the package:

```bash
pip install -r requirements.txt
```

3. Run:

```bash
export OLLAMA_MODEL=llama3.1
python -m moralagent.main \
  "A restaurant was nearly empty when a maintenance worker jumped ahead in line so that he could get \
his food and get back to work before his break ended. Is the behavior appropriate?" \
  --show-steps
```

You can also pass a scenario file (`--file scenario.txt`) or read from stdin. Run without
`--show-steps` to print only the final judgment.

### Programmatic use

```python
from moralagent import MoralAgent, load_settings

agent = MoralAgent(settings=load_settings(model="llama3.1"))
judgment = agent.judge(text)          # -> MoralJudgment(judgment="Yes"|"No", reasons=...)
for step, p_prime, m_p in agent.memory_entries:   # stored P_i' and summaries M_Pi
    print(step, "->", m_p)
```

### Configuration (environment variables)

| Variable             | Default                  |
|----------------------|--------------------------|
| `OLLAMA_MODEL`       | `llama3.1`               |
| `OLLAMA_BASE_URL`    | `http://localhost:11434` |
| `OLLAMA_TEMPERATURE` | `0.0`                    |
| `OLLAMA_NUM_CTX`     | `8192`                   |

A tool-calling model is required for the multidimensional analysis step
(llama3.1+, qwen2.5, mistral-nemo, firefunction-v2, ...).

## File layout

```
moralagent/
  config.py          Settings loaded from environment variables
  schemas.py         T1 (KeyInformation) and T4 (MoralJudgment) schemas
  prompts.py         Dynamic prompt templates P1-P4 + memory summarization prompt
  principles_data.py Moral-principles knowledge graph nodes
  tools.py           lookup_moral_principles tool (T_k retrieval)
  memory.py          Memory module (MM): store P_i' and summarize into M_Pi
  graph.py           LangGraph state graph wiring all steps together
  judge.py           MoralAgent high-level interface
  main.py            CLI entry point
```
