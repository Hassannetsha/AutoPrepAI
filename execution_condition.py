"""
ExecutionCondition: Determines whether a pipeline node should execute.
"""
from abc import ABC, abstractmethod
from typing import List
from data_context import DataContext
from intent import Intent


class ExecutionCondition(ABC):
    """
    Interface for execution conditions.
    """
    
    @abstractmethod
    def evaluate(self, context: DataContext) -> bool:
        """
        Evaluate whether the condition is met.
        
        Args:
            context: The current data context
            
        Returns:
            True if the condition is satisfied
        """
        pass


class AlwaysCondition(ExecutionCondition):
    """
    Condition that always evaluates to True.
    """

    def evaluate(self, context: DataContext) -> bool:
        return True


class IntentBasedCondition(ExecutionCondition):
    """
    Condition that checks for specific intents in the context.
    """
    
    def __init__(self, required_intents: List[str], operator: str = "any"):
        """
        Args:
            required_intents: List of intent names to check for
            operator: "any" or "all" - determines if any or all intents must be present
        """
        self.required_intents = required_intents
        self.operator = operator.lower()
        
        if self.operator not in ["any", "all"]:
            raise ValueError("Operator must be 'any' or 'all'")

    def evaluate(self, context: DataContext) -> bool:
        """Evaluate if the required intents are present."""
        intents = context.get_metadata("intents") or []
        
        # Handle both Intent objects and simple tuples/lists
        intent_names = []
        for item in intents:
            if isinstance(item, Intent):
                intent_names.append(item.name)
            elif isinstance(item, (list, tuple)) and len(item) > 0:
                intent_names.append(item[0])
            elif isinstance(item, str):
                intent_names.append(item)
        
        if self.operator == "all":
            return self._check_all_intents(intent_names)
        else:
            return self._check_any_intent(intent_names)

    def _check_all_intents(self, intent_names: List[str]) -> bool:
        """Check if all required intents are present."""
        return all(intent in intent_names for intent in self.required_intents)

    def _check_any_intent(self, intent_names: List[str]) -> bool:
        """Check if any of the required intents are present."""
        return any(intent in intent_names for intent in self.required_intents)
