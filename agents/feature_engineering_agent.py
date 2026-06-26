
import os
import json

from agents.pipeline_agent import PipelineAgent
from data_context import DataContext
from agent_params import AgentParams
from api_key_manager import get_key_manager
from services.feature_engineering_service import FeatureEngineeringService, SuggestFeatures
from utils.retry_handler import GroqRetryHandler

import dspy

_key_manager = get_key_manager()

class FeatureEngineeringAgent(PipelineAgent):
    def __init__(self):
        super().__init__("Feature Engineering")
        self._ensure_lm()

    @staticmethod
    def _ensure_lm():
        if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
            api_key = _key_manager.get_current_key()
            lm = dspy.LM(model="groq/llama-3.3-70b-versatile", api_key=api_key, max_tokens=1000)
            dspy.settings.configure(lm=lm)

    @staticmethod
    def _rotate_and_reconfigure():
        api_key = _key_manager.rotate_key()
        os.environ["GROQ_API_KEY"] = api_key
        lm = dspy.LM(model="groq/llama-3.3-70b-versatile", api_key=api_key, max_tokens=1000)
        dspy.settings.configure(lm=lm)
        return api_key
    
    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        context.data = context.data.reset_index(drop=True)
        context.log("Starting feature engineering")

        self._ensure_lm()
        
        try:
            suggest_predictor = dspy.ChainOfThought(SuggestFeatures)
            
            target_col = context.metadata.get("target_col")
            feature_columns = [col for col in context.data.columns.tolist() if col != target_col]
            dataset_columns = json.dumps(feature_columns)
            sample_rows = context.data.head(5).to_json(orient='records')
            top_n = params.get_option('top_n', '5')

            handler = GroqRetryHandler(_key_manager, log_fn=context.log)
            suggested_str = None
            try:
                result = handler.execute(
                    task=lambda: suggest_predictor(
                        dataset_columns=dataset_columns,
                        sample_rows=sample_rows,
                        top_n=top_n
                    ),
                    after_rotate=lambda new_key: self._rotate_and_reconfigure(),
                    task_name="feature_engineering"
                )
                suggested_str = result.suggested_features
            except RuntimeError as e:
                context.log(str(e))
                return context
            if not suggested_str or not suggested_str.strip():
                context.log("No feature suggestions generated; skipping")
                return context
            
            context.log(f"Generated suggestions:\n{suggested_str}")
            
            fe = FeatureEngineeringService()
            new_df, features_added = fe.engineer(
                context.data,
                suggested_str,
            )
            
            context.data = new_df
            context.metadata["features_engineered"] = True
            context.metadata["features_added_count"] = features_added
            context.metadata["feature_engineering_suggestions"] = suggested_str
            context.log(f"Successfully added {features_added} new features")
            
        except Exception as e:
            context.log(f"Feature engineering Failed.")
            print(f"Feature engineering error: {e}")
            import traceback
            print(traceback.format_exc())
        
        return context
