from agents.pipeline_agent import PipelineAgent
from data_context import DataContext
from agent_params import AgentParams
from agents.semantic_duplicate_remover import SemanticDuplicateRemover
from agents.exact_duplicates_agent import ExactDuplicateRemover

class DuplicateRemoverAgent(PipelineAgent):
    def __init__(self):
        super().__init__("Duplicate Remover")

    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        context.data = context.data.reset_index(drop=True)
        context.log("Removing duplicate rows")
        context.metadata["duplicates_removed"] = True
        exact_remover = ExactDuplicateRemover()
        context = exact_remover.execute(context, params)
        semantic_remover = SemanticDuplicateRemover()
        context = semantic_remover.execute(context, params)
        return context
