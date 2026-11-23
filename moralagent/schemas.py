"""Structured schemas for the dynamic prompt templates.

{Json} variable fields of the paper: the "formatting tool" fixes the content
and format of what the LLM has to recognize and produce in each step.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class KeyInformation(BaseModel):
    """T1: key information of the moral scenario (who, what, when, where, why, how)."""

    who: str = Field(description="The agents/actors involved and their roles")
    what: str = Field(description="The behavior or action that must be judged")
    when: str = Field(description="When the event happens, if stated")
    where: str = Field(description="Where the event happens, if stated")
    why: str = Field(description="The stated or implied motivation for the behavior")
    how: str = Field(description="How the behavior is carried out and any relevant background")


class MoralJudgment(BaseModel):
    """T4: final judgment on whether the behavior is morally appropriate."""

    judgment: Literal["Yes", "No"] = Field(
        description="'Yes' if the behavior is appropriate, 'No' if it is not"
    )
    reasons: str = Field(description="The reasons supporting the judgment, based on the analysis")
