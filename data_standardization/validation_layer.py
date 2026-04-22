import re


class ValidationLayer:
    """
    Rule-based validation that runs BEFORE any LLM suggestion is accepted.

    Rules are registered per column. If no rules are registered for a column,
    the layer acts as a pass-through (accepts everything).

    Supported rule types
    --------------------
    allowed_values  : set/list  – exact allowed strings (case-insensitive match)
    regex           : str       – the normalized value must fully match this pattern
    max_length      : int       – character length ceiling
    min_length      : int       – character length floor
    disallow_digits : bool      – reject values that are purely numeric after normalization
    custom          : callable  – f(original, normalized) -> (bool, reason_str)
    """

    def __init__(self):
        self._rules: dict[str, dict] = {}

    def register(self, column: str, **rules):
        """Register rules for a column. Call multiple times to accumulate rules."""
        self._rules.setdefault(column, {}).update(rules)

    def get_rules(self, column: str) -> dict:
        """Return a shallow copy of the registered rules for a column."""
        return dict(self._rules.get(column, {}))

    def validate(self, column: str, original: str, normalized: str) -> tuple[bool, str]:
        """
        Returns (is_valid, reason).
        is_valid=True  → accept normalized value
        is_valid=False → reject; caller should fall back to original
        """
        rules = self._rules.get(column, {})
        if not rules:
            return True, "no rules defined — accepted"

        # allowed_values
        if "allowed_values" in rules:
            allowed = {str(v).lower() for v in rules["allowed_values"]}
            if normalized.lower() not in allowed:
                return False, (
                    f"'{normalized}' not in allowed values "
                    f"({list(rules['allowed_values'])[:5]}{'...' if len(rules['allowed_values']) > 5 else ''})"
                )

        # regex
        if "regex" in rules:
            if not re.fullmatch(rules["regex"], normalized):
                return False, f"'{normalized}' does not match pattern '{rules['regex']}'"

        # max_length
        if "max_length" in rules and len(normalized) > rules["max_length"]:
            return False, f"length {len(normalized)} exceeds max {rules['max_length']}"

        # min_length
        if "min_length" in rules and len(normalized) < rules["min_length"]:
            return False, f"length {len(normalized)} below min {rules['min_length']}"

        # disallow_digits
        if rules.get("disallow_digits") and normalized.isdigit():
            return False, f"'{normalized}' is purely numeric — unexpected for this column"

        # custom callable
        if "custom" in rules:
            ok, reason = rules["custom"](original, normalized)
            if not ok:
                return False, reason

        if "min_value" in rules or "max_value" in rules:
            try:
                value = float(normalized)
            except (TypeError, ValueError):
                return False, f"'{normalized}' is not numeric"

            if "min_value" in rules and value < rules["min_value"]:
                return False, f"{value} below min {rules['min_value']}"
            if "max_value" in rules and value > rules["max_value"]:
                return False, f"{value} above max {rules['max_value']}"

        return True, "passed all rules"


