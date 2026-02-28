"""
DataContext: Holds the state and data throughout the pipeline execution.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List
import pandas as pd


@dataclass
class DataContext:
    """
    Context object that flows through the pipeline, carrying data and metadata.
    """
    data: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    user_id: str = ""
    conversation_id: str = ""

    def log(self, message: str) -> None:
        """Add a log message to the context."""
        self.logs.append(message)

    def get_metadata(self, key: str) -> Any:
        """Retrieve metadata value by key."""
        return self.metadata.get(key)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        self.metadata[key] = value

    def get_data(self) -> pd.DataFrame:
        """Get the current dataframe."""
        return self.data

    def set_data(self, data: pd.DataFrame) -> None:
        """Set the dataframe."""
        self.data = data

