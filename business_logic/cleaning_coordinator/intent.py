"""
Intent: Represents a detected user intent from NLP processing.
"""
from typing import Any, List, Dict


class Intent:
    """
    Represents a user intent detected by NLP processing.
    """
    
    def __init__(self, name: str, columns: List[str] = None, parameters: Dict[str, Any] = None):
        self.name = name
        self.columns = columns or []
        self.parameters = parameters or {}

    def has_column(self, column: str) -> bool:
        """Check if a specific column is referenced in this intent."""
        return column in self.columns

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get a parameter value with a default fallback."""
        return self.parameters.get(key, default)

    def __repr__(self):
        return f"Intent(name={self.name}, columns={self.columns}, parameters={self.parameters})"