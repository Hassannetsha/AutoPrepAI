from ml_layer.agents.pipeline_agent import PipelineAgent
from business_logic.cleaning_coordinator.data_context import DataContext
from business_logic.cleaning_coordinator.agent_params import AgentParams
from ml_layer.duplicates.exact_duplicate_remover_service import ExactDuplicateRemoverService


class ExactDuplicateRemover(PipelineAgent):
    def __init__(self):
        super().__init__("Exact Duplicate Remover")

    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        context.data = context.data.reset_index(drop=True)
        columns = params.columns or []
        context.log("Removing exact duplicate rows")
        
        try:
            target_col = context.metadata.get("target_col")
            if target_col and target_col not in columns:
                columns = [*columns, target_col]
            subset = columns if columns else None

            # Initialize exact duplicate remover
            remover = ExactDuplicateRemoverService(
                subset=subset,
                keep='first',
                auto_exclude_ids=True
            )
            
            # Remove duplicates
            df_dedup, df_duplicates = remover.remove_duplicates(
                context.data,
                verbose=True
            )
            
            num_duplicates = len(df_duplicates)
            context.data = df_dedup
            context.metadata["exact_duplicates_removed"] = True
            context.metadata["exact_duplicates_count"] = num_duplicates
            
            if num_duplicates > 0:
                context.log(f"Removed {num_duplicates} exact duplicate rows")
            else:
                context.log("No exact duplicates found")
                
        except Exception as e:
            context.log(f"Exact duplicate removal failed.")
            print(f"Exact duplicate removal failed: {e}")
            import traceback
            print(traceback.format_exc())
        
        return context
