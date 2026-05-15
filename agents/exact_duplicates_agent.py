from agents.pipeline_agent import PipelineAgent
from data_context import DataContext
from agent_params import AgentParams
from duplicates.exact_duplicate_remover_service import ExactDuplicateRemoverService


class ExactDuplicateRemover(PipelineAgent):
    def __init__(self):
        super().__init__("Exact Duplicate Remover")

    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        context.data = context.data.reset_index(drop=True)
        columns = params.columns or []
        context.log("Removing exact duplicate rows")
        
        try:
            target_col = context.metadata.get("target_col")
            subset = columns if columns else None
            if subset and target_col in context.data.columns and target_col not in subset:
                # A feature duplicate with a conflicting label is not safe to drop.
                # Including the target keeps only fully identical supervised rows.
                subset = [*subset, target_col]

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
            context.log(f"Exact duplicate removal error: {e}")
            import traceback
            context.log(traceback.format_exc())
        
        return context
