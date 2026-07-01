from ml_layer.agents.pipeline_agent import PipelineAgent
from business_logic.cleaning_coordinator.data_context import DataContext
from business_logic.cleaning_coordinator.agent_params import AgentParams
from ml_layer.features.selection_service import FeatureSelectionService

class FeatureSelectionAgent(PipelineAgent):
    def __init__(self):
        super().__init__("Feature Selection")

    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        context.data = context.data.reset_index(drop=True)
        columns = params.columns or []
        context.log("Selecting features")
        try:
            service = FeatureSelectionService(random_state=context.metadata.get("random_state", 42))
            n_features = params.get_option("n_features")
            selected, dropped, excluded, pruned_df = service.run(
                context.data,
                columns=columns,
                n_features=n_features,
                metadata=context.metadata
            )
        except ValueError as e:
            reason = str(e).replace("NaN", "missing values")
            context.log(f"Feature selection skipped: {reason}")
            print(f"Feature selection skipped: {reason}")
            return context
        except Exception as e:
            context.log(f"Feature selection failed.")
            print(f"Feature selection error: {e}")
            return context
        context.data = pruned_df
        context.metadata["features_selected"] = True
        context.metadata["selected_features"] = selected
        context.log(f"Selected features: {selected}")
        if dropped:
            context.log(f"Dropped features: {dropped}")
        if excluded:
            context.log(f"Excluded by importance: {excluded}")
        return context
