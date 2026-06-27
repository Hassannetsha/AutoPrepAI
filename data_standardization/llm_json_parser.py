"""

==================
Utility for parsing JSON out of LLM responses.

LLM responses are often wrapped in markdown fences or contain
surrounding explanation text. This module handles all of that
in one place so individual strategies don't duplicate the logic.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

T = TypeVar("T", list, dict)


def parse_llm_json(raw: str, expected_type: type[T]) -> T:
    """
    Extract and parse a JSON value from a raw LLM response string.

    Handles:
      - Markdown code fences (```json ... ``` or ``` ... ```)
      - Leading/trailing whitespace
      - JSON embedded within surrounding explanation text

    Parameters
    ----------
    raw           : Raw string from the LLM.
    expected_type : `list` or `dict` — the expected top-level type.

    Returns
    -------
    Parsed JSON value of the expected type.

    Raises
    ------
    ValueError
        If no valid JSON of the expected type can be extracted.
    """
    cleaned = raw.strip()

    # Strip markdown fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    # Attempt direct parse first
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, expected_type):
            return parsed
        raise ValueError(
            f"Expected {expected_type.__name__}, "
            f"got {type(parsed).__name__}"
        )
    except json.JSONDecodeError:
        pass

    # Fall back: extract first bracket block matching expected type
    pattern = r"\[.*?\]" if expected_type is list else r"\{.*?\}"
    match = re.search(pattern, raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, expected_type):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not extract a valid {expected_type.__name__} "
        f"from LLM response: {raw[:200]!r}"
    )