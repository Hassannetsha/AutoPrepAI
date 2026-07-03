from __future__ import annotations

import json
import re
from typing import TypeVar

T = TypeVar("T", list, dict)


def parse_llm_json(raw: str, expected_type: type[T]) -> T:
    # Remove surrounding whitespace from the LLM response.
    cleaned = raw.strip()

    # Remove Markdown code fences (```json ... ```), if present.
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
    
    # No valid JSON could be extracted
    raise ValueError(
        f"Could not extract a valid {expected_type.__name__} "
        f"from LLM response: {raw[:200]!r}"
    )