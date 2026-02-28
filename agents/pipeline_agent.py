"""
PipelineAgent: Base interface for all pipeline agents.
"""
from abc import ABC, abstractmethod
from data_context import DataContext
from agent_params import AgentParams


class PipelineAgent(ABC):
    """
    Interface for all preprocessing agents in the pipeline.
    """
    
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        """
        Execute the agent's logic on the given context.
        
        Args:
            context: The data context to process
            params: Parameters for this agent
            
        Returns:
            Updated DataContext
        """
        pass