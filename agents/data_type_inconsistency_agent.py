from agents.pipeline_agent import PipelineAgent
from data_context import DataContext
from agent_params import AgentParams
from data_type_inconsistency_detector import DataTypeInconsistencyDetector
from data_type_inconsistency_resolver import DataResolvingService

class DataTypeInconsistencyAgent(PipelineAgent):
    def __init__(self):
        super().__init__("Data Type Inconsistency Handler")

    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        context.data = context.data.reset_index(drop=True)
        columns = params.columns or []
        context.log("Detecting and resolving data type inconsistencies")
        
        try:
            # Step 1: Detect inconsistencies
            detector = DataTypeInconsistencyDetector()
            detection_results = detector.analyze_dataframe(context.data)
            
            # Store detection results in metadata
            context.metadata["datatype_detection_results"] = detection_results
            
            # Step 2: Identify columns with inconsistencies
            inconsistent_columns = []
            for col_name, result in detection_results.items():
                if len(result.get('detected_types', {})) > 1:
                    inconsistent_columns.append(col_name)
                    context.log(f"Column '{col_name}': {result['detected_types']}")
            
            if not inconsistent_columns:
                context.log("No data type inconsistencies detected")
                context.metadata["datatype_inconsistencies_fixed"] = True
                return context
            
            # Step 3: Resolve inconsistencies
            resolver = DataResolvingService(context.data, detection_results)
            
            for col_name in inconsistent_columns:
                result = detection_results[col_name]
                recommended_type = result.get('recommended_type')
                
                # Apply conversion strategy based on recommended type
                if recommended_type and recommended_type != 'empty_column':
                    context.log(f"Converting '{col_name}' to {recommended_type}")
                    resolver.resolve(
                        strategy_name="convert_to_type",
                        column_name=col_name,
                        target_type=recommended_type
                    )
            
            # Update context with resolved data
            context.data = resolver.df_resolved
            context.metadata["datatype_inconsistencies_fixed"] = True
            context.metadata["resolution_log"] = resolver.resolution_log
            context.log(f"Fixed data type inconsistencies in {len(inconsistent_columns)} columns")
            
        except Exception as e:
            context.log(f"Data type inconsistency handling error: {e}")
            import traceback
            context.log(traceback.format_exc())
        
        return context
