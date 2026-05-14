from agents.pipeline_agent import PipelineAgent
from data_context import DataContext
from agent_params import AgentParams
from services.outliers_service import OutliersService


class OutliersAgent(PipelineAgent):
    def __init__(self):
        super().__init__("Outlier Remover")

    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        context.log("Handling outliers")
        context.metadata["outliers_handled"] = True
        c = OutliersService(dataframe=context.data)
        cleaned = c.process()
        context.data = cleaned
        return context
