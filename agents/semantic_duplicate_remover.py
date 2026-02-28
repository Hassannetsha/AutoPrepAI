from agents.pipeline_agent import PipelineAgent
from data_context import DataContext
from agent_params import AgentParams
from duplicates.semantic_duplicate_remover_service import SemanticDuplicateRemoverService


class SemanticDuplicateRemover(PipelineAgent):
    def __init__(self):
        super().__init__("Semantic Duplicate Remover")

    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        columns = params.columns or []
        context.log("Removing semantic duplicate rows")
        
        try:
            # Check if data is suitable for semantic duplicate detection
            text_columns = context.data.select_dtypes(include=['object']).columns.tolist()
            
            if not text_columns:
                context.log("No text columns found for semantic duplicate detection")
                return context
            
            # Determine which text column to use
            # Priority: specified column > column with longest average text > first text column
            target_column = None
            
            if columns and len(columns) > 0:
                # Use specified column if it's a text column
                for col in columns:
                    if col in text_columns:
                        target_column = col
                        break
            
            if not target_column:
                # Find the text column with longest average text length
                avg_lengths = {}
                for col in text_columns:
                    avg_lengths[col] = context.data[col].astype(str).str.len().mean()
                
                # Only use semantic detection if average text length is substantial (> 20 chars)
                max_col = max(avg_lengths, key=avg_lengths.get)
                if avg_lengths[max_col] > 20:
                    target_column = max_col
                else:
                    context.log(f"Text columns have short content (avg < 20 chars). Skipping semantic duplicate detection.")
                    return context
            
            if not target_column:
                context.log("No suitable text column found for semantic duplicate detection")
                return context
            
            context.log(f"Using column '{target_column}' for semantic duplicate detection")
            
            # Initialize semantic duplicate remover
            remover = SemanticDuplicateRemoverService(
                model_name="paraphrase-MiniLM-L6-v2",
                threshold=0.85,
                k_neighbors=10,
                batch_size=512
            )
            
            # Remove semantic duplicates
            df_dedup, df_duplicates = remover.remove_duplicates(
                context.data,
                text_column=target_column
            )
            
            num_duplicates = len(df_duplicates) if not df_duplicates.empty else 0
            context.data = df_dedup
            context.metadata["semantic_duplicates_removed"] = True
            context.metadata["semantic_duplicates_count"] = num_duplicates
            context.metadata["semantic_column_used"] = target_column
            
            if num_duplicates > 0:
                context.log(f"Removed {num_duplicates} semantic duplicate rows from column '{target_column}'")
            else:
                context.log(f"No semantic duplicates found in column '{target_column}'")
                
        except Exception as e:
            context.log(f"Semantic duplicate removal error: {e}")
            import traceback
            context.log(traceback.format_exc())
        
        return context
