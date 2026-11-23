"""The four dynamic prompt templates P1-P4 and the memory summarization prompt.

Each template has fixed fields (the task instruction Regular_1..Regular_4) and
variable fields ({text}, {t1}, {t2}, {t3}, {m_p?}, {t_k}, {json}) that are
dynamically replaced as the analysis progresses, guiding the LLM through the
four-step analysis plan.
"""

from __future__ import annotations

REGULAR_1 = """\
You are the key information extraction step of a moral judgment agent.
Read the moral scenario below and extract its key information: who (the actors \
and their roles), what (the behavior that must be judged), when, where, why \
(the stated or implied motivation), and how (how the behavior is carried out, \
plus relevant background). Complete every field; where the scenario does not \
state something explicitly, infer the most reasonable value and keep it faithful \
to the text. Do not judge whether the behavior is right or wrong."""

P1_KEY_INFO = REGULAR_1 + """

Moral scenario text:
{text}

Return the key information."""

REGULAR_2 = """\
You are the intuitive preliminary analysis step of a moral judgment agent.
Using the summarized information from earlier steps and the extracted key \
information, give a brief, intuitive first-pass analysis of the moral scenario: \
what behavior is at issue, whom it affects, what the motivation appears to be, \
and what the likely direction of a moral judgment is. Keep it short and \
surface-level; you are building a starting point, not the final judgment."""

P2_PRELIMINARY = REGULAR_2 + """

Summary of previous steps (M_P1):
{mp1}

Extracted key information:
{t1}

Return the preliminary analysis."""

REGULAR_3 = """\
You are the multidimensional analysis step of a moral judgment agent.
First call the `lookup_moral_principles` tool to retrieve the moral principles \
relevant to this scenario. Then analyze the scenario in depth from multiple \
dimensions: emotional impact (how the behavior makes those affected feel), \
causality (what consequences the behavior causes and for whom), purpose and \
intent (why the actor did it and whether the intent was reasonable), \
perspective-taking (the situation from the viewpoints of the actor, the \
affected parties, and third parties), and moral principles (which of the \
retrieved principles support or condemn the behavior, and whether any \
exception or special circumstance applies)."""

P3_MULTIDIM = REGULAR_3 + """

Summary of previous steps (M_P2):
{mp2}

Moral scenario text:
{text}

Extracted key information:
{t1}

Preliminary analysis:
{t2}

Return the multidimensional analysis."""

REGULAR_4 = """\
You are the comprehensive decision-making step of a moral judgment agent.
Based on the summary of all previous analyses, decide whether the behavior \
described in the moral scenario is morally appropriate. Weigh the preliminary \
analysis and the multidimensional analysis; if the behavior is a justified \
exception to a general moral rule, judge it as appropriate. Answer with \
'Yes' if the behavior is appropriate and 'No' if it is not, together with the \
reasons for your judgment."""

P4_DECISION = REGULAR_4 + """

Summary of previous steps (M_P3):
{mp3}

Return your judgment."""

SUMMARIZE = """\
Summarize the important information contained in the following instruction and \
result from analysis step '{step}' of a moral judgment. Keep only what later \
steps of the judgment need: the actors and behavior, relevant contextual \
details, the analysis conclusions, and any moral principles considered. Be \
faithful and concise.

Instruction and result:
{updated_prompt}

Return the summary."""
