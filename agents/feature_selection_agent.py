from agents.pipeline_agent import PipelineAgent
from data_context import DataContext
from agent_params import AgentParams
from services.feature_selection_service import FeatureSelectionService

class FeatureSelectionAgent(PipelineAgent):
    def __init__(self):
        super().__init__("Feature Selection")

    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        columns = params.columns or []
        context.log("Selecting features")
        try:
            service = FeatureSelectionService(random_state=context.metadata.get("random_state", 42))
            threshold = params.get_option("threshold")
            n_features = params.get_option("n_features")
            selected, pruned_df = service.run(
                context.data,
                columns=columns,
                threshold=threshold,
                n_features=n_features,
                metadata=context.metadata
            )
        except ValueError as e:
            context.log(f"Feature selection skipped: {e}")
            return context
        except Exception as e:
            context.log(f"Feature selection error: {e}")
            return context
        context.data = pruned_df
        context.metadata["features_selected"] = True
        context.metadata["selected_features"] = selected
        context.log(f"Selected features: {selected}")
        return context
