
import os
import re
import time

from agents.pipeline_agent import PipelineAgent
from data_context import DataContext
from agent_params import AgentParams
from api_key_manager import get_key_manager
from services.feature_engineering_service import FeatureEngineeringService, SuggestFeatures

import dspy 
import json

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

            max_retries = _key_manager.get_total_keys_count()
            retry_count = 0
            suggested_str = None

            while retry_count < max_retries:
                try:
                    result = suggest_predictor(
                        dataset_columns=dataset_columns,
                        sample_rows=sample_rows,
                        top_n=top_n
                    )
                    suggested_str = result.suggested_features
                    break
                except Exception as e:
                    error_msg = str(e).lower()
                    if "rate" in error_msg or "quota" in error_msg or "limit" in error_msg or "429" in error_msg:
                        available = _key_manager.get_available_keys_count()
                        context.log(f"Rate limit hit. Rotating key... ({available} keys available)")
                        _key_manager.mark_key_failed()

                        if available <= 1:
                            retry_seconds = "unknown"
                            if "please try again in" in error_msg:
                                match = re.search(r'please try again in ([^s]+s)', error_msg)
                                if match:
                                    retry_seconds = match.group(1)
                            context.log(f"All API keys exhausted (TPD limit reached). Retry in {retry_seconds}.")
                            return context

                        try:
                            self._rotate_and_reconfigure()
                            retry_count += 1
                            time.sleep(1)
                            continue
                        except RuntimeError:
                            context.log("All API keys exhausted.")
                            return context
                    else:
                        raise
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
