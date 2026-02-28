from agents.pipeline_agent import PipelineAgent
from data_context import DataContext
from agent_params import AgentParams
from data_standardization.data_standardizing_service import DataStandardizingService
from api_key_manager import get_key_manager 

from groq import Groq
import pandas as pd


class DataStandardizerAgent(PipelineAgent):
    def __init__(self):
        super().__init__("Data Standardizer")

    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        columns = params.columns or []
        context.log("Standardizing data values using LLM")
        
        try:
            # Initialize Groq client
            key_manager = get_key_manager()
            api_key = key_manager.get_current_key()
            client = Groq(api_key=api_key)
            model = "llama-3.3-70b-versatile"
            
            standardizer = DataStandardizingService(context.data, client, model)
            
            # Run detection on all columns or specified columns
            if columns:
                # Process only specified columns
                for col in columns:
                    if col in context.data.columns:
                        if pd.api.types.is_numeric_dtype(context.data[col]):
                            standardizer.detect_numeric_issues(col)
                        else:
                            standardizer.detect_categorical_issues(col)
            else:
                # Process all columns
                standardizer.run_detection()
            
            # Apply categorical fixes
            standardizer.apply_categorical_fixes(columns=columns if columns else None)
            
            # Update context with standardized data
            context.data = standardizer.df
            context.metadata["data_standardized"] = True
            context.metadata["standardization_results"] = standardizer.results
            context.log("Data standardization completed")
            
        except Exception as e:
            context.log(f"Data standardization error: {e}")
            import traceback
            context.log(traceback.format_exc())
        
        return context
