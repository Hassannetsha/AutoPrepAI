"""
ParameterResolver: Resolves parameters for agent execution from context.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from data_context import DataContext
from agent_params import AgentParams
from intent import Intent


class ParameterResolver(ABC):
    """
    Interface for parameter resolution strategies.
    """
    
    @abstractmethod
    def resolve(self, context: DataContext) -> AgentParams:
        """
        Resolve parameters from the context.
        
        Args:
            context: The current data context
            
        Returns:
            AgentParams containing resolved parameters
        """
        pass


class IntentColumnResolver(ParameterResolver):
    """
    Resolves parameters by extracting columns and options from intents.
    """
    
    def __init__(self, intent_names: List[str], default_strategy: str = ""):
        """
        Args:
            intent_names: List of intent names to extract parameters from
            default_strategy: Default strategy if not specified in intents
        """
        self.intent_names = intent_names
        self.default_strategy = default_strategy

    def resolve(self, context: DataContext) -> AgentParams:
        """Resolve parameters from matching intents."""
        intents = context.get_metadata("intents") or []
        
        columns = self._extract_columns_from_intents(intents)
        parameters = self._merge_parameters(intents)
        
        strategy = parameters.get("strategy", self.default_strategy)
        
        return AgentParams(
            columns=columns,
            strategy=strategy,
            options=parameters
        )

    def _extract_columns_from_intents(self, intents: List) -> List[str]:
        """Extract column names from intents that match our intent names."""
        columns = []
        
        for item in intents:
            intent_name = None
            intent_columns = []
            
            # Handle Intent objects
            if isinstance(item, Intent):
                intent_name = item.name
                intent_columns = item.columns
            # Handle tuples/lists like ["intent_name", "col1", "col2"]
            elif isinstance(item, (list, tuple)) and len(item) > 0:
                intent_name = item[0]
                if len(item) > 1:
                    # Check if second element is a list of columns
                    if isinstance(item[1], (list, tuple)):
                        intent_columns = list(item[1])
                    else:
                        intent_columns = list(item[1:])
            
            # If this intent matches one we're looking for, add its columns
            if intent_name in self.intent_names:
                columns.extend(intent_columns)
        
        return columns

    def _merge_parameters(self, intents: List) -> Dict[str, Any]:
        """Merge parameters from all matching intents."""
        merged = {}
        
        for item in intents:
            intent_name = None
            intent_params = {}
            
            if isinstance(item, Intent):
                intent_name = item.name
                intent_params = item.parameters
            elif isinstance(item, (list, tuple)) and len(item) > 0:
                intent_name = item[0]
            
            if intent_name in self.intent_names and intent_params:
                merged.update(intent_params)
        
        return merged