from ml_layer.agents.pipeline_agent import PipelineAgent
from business_logic.cleaning_coordinator.data_context import DataContext
from business_logic.cleaning_coordinator.agent_params import AgentParams
from ml_layer.outliers.outliers_service import OutliersService


class OutliersAgent(PipelineAgent):
    def __init__(self):
        super().__init__("Outlier Remover")

    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        context.log("Handling outliers")
        context.metadata["outliers_handled"] = True
        c = OutliersService(dataframe=context.data)
        context.data = c.process()
        context.metadata["outlier_strategy"] = c.get_strategy_name()
        context.log(f"Outlier strategy used: {c.get_strategy_name()}")
        return context
