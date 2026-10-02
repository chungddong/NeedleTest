"""Disaster outage / equipment-damage report schema for the Korean fine-tune.

Keys and enum values stay in English: the Needle tokenizer has no Hangul pieces,
so every Korean character costs 3 byte tokens. Only free-text values (location)
are copied from the Korean input.
"""
import json
from typing import Annotated, Literal, Optional

import needle

IncidentType = Literal["outage", "pole_down", "line_down", "transformer_noise", "spark", "fire"]


@needle.tool
def report_incident(
    incident_type: IncidentType,
    location: str,
    households: Annotated[Optional[int], needle.Field(ge=1, le=100000)] = None,
    hazard: Optional[Literal["electrocution", "fire"]] = None,
):
    """Record a power outage or electrical equipment damage report.

    Args:
        incident_type: outage, pole_down, line_down, transformer_noise, spark, or fire.
        location: Where it happened, copied as the reporter said it.
        households: Number of affected households, only when stated.
        hazard: Immediate danger, only when the reporter describes one.
    """
    return {"ok": True}


TOOLS = [report_incident]
TOOLS_JSON = [t._needle_tool for t in TOOLS]


def reasoning(evidence, answers):
    """The one-line `reasoning` a training row carries.

    The engine always opens a <think> block before the tool call, and `needle finetune`
    trains that block only from this field; rows without it leave the tuned model
    reasoning like the base model at inference. The label part is derived from
    `answers` so the reasoning cannot contradict them, and it stays English because
    every Korean character costs 3 byte tokens.
    """
    if not answers:
        return f"{evidence} -> no report"
    parts = [f"{a['arguments'].get('incident_type')}; "
             f"households {a['arguments'].get('households', 'none')}; "
             f"hazard {a['arguments'].get('hazard', 'none')}" for a in answers]
    return f"{evidence} -> " + " | ".join(parts)

if __name__ == "__main__":
    print(json.dumps(TOOLS_JSON, ensure_ascii=False, indent=1))
