from agents.pipeline_agent import PipelineAgent
from data_context import DataContext
from agent_params import AgentParams
from services.spelling_corrector_service import SpellingCorrectorService

class SpellingCorrectorAgent(PipelineAgent):
    def __init__(self):
        super().__init__("Spelling Corrector")

    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        context.data = context.data.reset_index(drop=True)
        columns = params.columns or []
        context.log("Correcting spelling errors in categorical columns")
        
        try:
            # Identify categorical columns
            categorical_cols = context.data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not categorical_cols:
                context.log("No categorical columns found for spelling correction")
                return context
            
            # If specific columns are provided, use them; otherwise process all categorical columns
            target_cols = columns if columns else categorical_cols
            target_cols = [col for col in target_cols if col in categorical_cols]
            
            if not target_cols:
                context.log("No valid categorical columns to process")
                return context
            
            corrector = SpellingCorrectorService(max_edit_distance=2, prefix_length=7)
            corrected_columns = []
            
            for col in target_cols:
                try:
                    # Build dictionary from the column itself
                    corrector.build_dictionary_from_dataframe(
                        context.data, 
                        col, 
                        show_progress=False
                    )
                    
                    # Correct the column
                    context.data[col] = corrector.correct_dataframe_column(
                        context.data,
                        col,
                        show_progress=False,
                        inplace=False
                    )
                    
                    corrected_columns.append(col)
                    context.log(f"Corrected spelling in column '{col}'")
                    
                except Exception as col_error:
                    context.log(f"Error correcting column '{col}': {col_error}")
            
            context.metadata["spelling_corrected"] = True
            context.metadata["spelling_corrected_columns"] = corrected_columns
            context.log(f"Spelling correction completed for {len(corrected_columns)} columns")
            
        except Exception as e:
            context.log(f"Spelling correction error: {e}")
            import traceback
            context.log(traceback.format_exc())
        
        return context
