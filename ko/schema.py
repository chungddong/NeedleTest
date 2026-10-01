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

if __name__ == "__main__":
    print(json.dumps(TOOLS_JSON, ensure_ascii=False, indent=1))
