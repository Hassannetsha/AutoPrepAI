"""
PipelineNode: Wraps an agent with execution conditions and parameter resolution.
"""
from ml_layer.agents.pipeline_agent import PipelineAgent
from business_logic.cleaning_coordinator.execution_condition import ExecutionCondition
from business_logic.cleaning_coordinator.parameter_resolver import ParameterResolver
from business_logic.cleaning_coordinator.data_context import DataContext
from business_logic.cleaning_coordinator.agent_params import AgentParams


class PipelineNode:
    """
    A node in the pipeline that wraps an agent with conditional execution logic.
    """
    
    def __init__(
        self, 
        agent: PipelineAgent, 
        condition: ExecutionCondition, 
        resolver: ParameterResolver
    ):
        """
        Args:
            agent: The agent to execute
            condition: Condition determining if this node should run
            resolver: Resolver for extracting agent parameters
        """
        self.agent = agent
        self.condition = condition
        self.param_resolver = resolver

    def should_run(self, context: DataContext) -> bool:
        """Check if this node should execute based on the condition."""
        return self.condition.evaluate(context)

    def resolve_params(self, context: DataContext) -> AgentParams:
        """Resolve parameters for the agent from the context."""
        return self.param_resolver.resolve(context)

    def execute(self, context: DataContext) -> DataContext:
        """Execute the agent with resolved parameters."""
        params = self.resolve_params(context)
        return self.agent.execute(context, params)

    def get_agent_name(self) -> str:
        """Get the name of the wrapped agent."""
        return self.agent.name