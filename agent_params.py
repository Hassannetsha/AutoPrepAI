"""
AgentParams: Parameter container for pipeline agents.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentParams:
    """
    Parameters passed to agents during execution.
    """
    columns: List[str] = field(default_factory=list)
    strategy: str = ""
    options: Dict[str, Any] = field(default_factory=dict)

    def get_option(self, key: str, default: Any = None) -> Any:
        """Get an option value with a default fallback."""
        return self.options.get(key, default)

    def has_columns(self) -> bool:
        """Check if columns are specified."""
        return len(self.columns) > 0