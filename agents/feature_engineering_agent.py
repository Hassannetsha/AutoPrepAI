
from agents.pipeline_agent import PipelineAgent
from data_context import DataContext
from agent_params import AgentParams
from api_key_manager import get_key_manager
from services.feature_engineering_service import FeatureEngineeringService, SuggestFeatures

import dspy 
import json

class FeatureEngineeringAgent(PipelineAgent):
    def __init__(self):
        super().__init__("Feature Engineering")
        # Get API key from multi-key manager (supports automatic rotation)
        key_manager = get_key_manager()
        api_key = key_manager.get_current_key()
        
        # Configure DSPy with a language model if not already configured
        if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
            lm = dspy.LM(model="groq/llama-3.3-70b-versatile", api_key=api_key, max_tokens=1000)
            dspy.settings.configure(lm=lm)
    
    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        context.data = context.data.reset_index(drop=True)
        context.log("Starting feature engineering")
        
        try:
            suggest_predictor = dspy.ChainOfThought(SuggestFeatures)
            
            target_col = context.metadata.get("target_col")
            feature_columns = [col for col in context.data.columns.tolist() if col != target_col]
            dataset_columns = json.dumps(feature_columns)
            sample_rows = context.data.head(5).to_json(orient='records')
            top_n = params.get_option('top_n', '5')
            
            result = suggest_predictor(
                dataset_columns=dataset_columns,
                sample_rows=sample_rows,
                top_n=top_n
            )
            
            suggested_str = result.suggested_features
            
            if not suggested_str or not suggested_str.strip():
                context.log("No feature suggestions generated; skipping")
                return context
            
            context.log(f"Generated suggestions:\n{suggested_str}")
            
            fe = FeatureEngineeringService()
            new_df, features_added, feature_log = fe.engineer(
                context.data,
                suggested_str,
                target_col=target_col,
            )
            
            context.data = new_df
            context.metadata["features_engineered"] = True
            context.metadata["features_added_count"] = features_added
            context.metadata["feature_engineering_suggestions"] = suggested_str
            context.metadata["feature_engineering_log"] = feature_log
            context.metadata["engineered_columns"] = [
                item.get("output_column")
                for item in feature_log
                if item.get("applied")
            ]
            context.log(f"Successfully added {features_added} new features")
            
        except Exception as e:
            context.log(f"Feature engineering error: {e}")
            import traceback
            context.log(traceback.format_exc())
        
        return context
