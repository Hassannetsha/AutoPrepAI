from agents.pipeline_agent import PipelineAgent
from data_context import DataContext
from agent_params import AgentParams
from Class_missingValues import MissingValuesDemo

import numpy as np

class MissingValueAgent(PipelineAgent):
    def __init__(self):
        super().__init__("Missing Values")
    
    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        # Get columns from resolver params
        columns = params.columns or []
        
        # Get strategy from params (default to "mean")
        strategy = params.strategy or params.get_option("strategy", "mean")
        
        # If user didn't specify strategy, default to mean
        if not strategy or not isinstance(strategy, str):
            strategy = "mean"
        
        # If columns is empty, use all numeric columns
        if not columns:
            columns = context.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not columns:
            context.log("No columns to handle missing values")
            return context
        
        context.log(f"Handling missing values for columns: {columns}")
        context.log(f"Using strategy: {strategy}")
        
        demo = MissingValuesDemo()
        context.data = demo.run(
            context.data, 
            strategy=strategy, 
            selected_cols=columns
        )
        
        return context
    
