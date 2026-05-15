from agents.pipeline_agent import PipelineAgent
from data_context import DataContext
from agent_params import AgentParams
from data_standardization.data_standardizing_service import DataStandardizingService
from data_standardization.validation_layer import ValidationLayer
from api_key_manager import get_key_manager 

from groq import Groq
import pandas as pd


class DataStandardizerAgent(PipelineAgent):
    def __init__(self):
        super().__init__("Data Standardizer")

    def _build_validation_layer(self, context: DataContext, params: AgentParams) -> ValidationLayer:
        validation = ValidationLayer()
        validation_rules = (
            params.get_option("validation_rules")
            or context.metadata.get("standardization_validation_rules")
            or {}
        )

        for column, rules in validation_rules.items():
            if column in context.data.columns and isinstance(rules, dict):
                validation.register(column, **rules)

        allowed_values = (
            params.get_option("allowed_values")
            or context.metadata.get("standardization_allowed_values")
            or {}
        )

        for column, values in allowed_values.items():
            if column in context.data.columns:
                validation.register(column, allowed_values=set(values))

        numeric_ranges = (
            params.get_option("numeric_ranges")
            or context.metadata.get("standardization_numeric_ranges")
            or {}
        )

        for column, bounds in numeric_ranges.items():
            if column in context.data.columns and isinstance(bounds, dict):
                range_rules = {
                    key: bounds[key]
                    for key in ("min_value", "max_value")
                    if key in bounds
                }
                if range_rules:
                    validation.register(column, **range_rules)

        return validation

    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        context.data = context.data.reset_index(drop=True)
        columns = params.columns or []
        target_columns = [col for col in columns if col in context.data.columns]
        selected_columns = target_columns if columns else list(context.data.columns)
        categorical_columns = [
            col for col in selected_columns
            if not pd.api.types.is_numeric_dtype(context.data[col])
        ]
        numeric_ranges = (
            params.get_option("numeric_ranges")
            or context.metadata.get("standardization_numeric_ranges")
            or {}
        )
        numeric_columns = [
            col for col in selected_columns
            if pd.api.types.is_numeric_dtype(context.data[col]) and col in numeric_ranges
        ]
        context.log("Standardizing data values using LLM")
        
        try:
            if columns and not target_columns:
                context.log("Data standardization skipped: none of the requested columns exist")
                context.metadata["data_standardized"] = False
                return context

            if not categorical_columns and not numeric_columns:
                context.log("Data standardization skipped: no categorical columns selected")
                context.metadata["data_standardized"] = False
                return context

            # Initialize Groq client
            key_manager = get_key_manager()
            api_key = key_manager.get_current_key()
            client = Groq(api_key=api_key)
            model = params.get_option("model", "llama-3.3-70b-versatile")
            confidence_threshold = params.get_option(
                "confidence_threshold",
                DataStandardizingService.DEFAULT_CONFIDENCE_THRESHOLD,
            )
            validation_layer = self._build_validation_layer(context, params)
            
            standardizer = DataStandardizingService(
                df=context.data,
                client=client,
                model=model,
                confidence_threshold=confidence_threshold,
                similarity_threshold=params.get_option(
                    "similarity_threshold",
                    DataStandardizingService.DEFAULT_SIMILARITY_THRESHOLD,
                ),
                validation_layer=validation_layer,
                requests_per_minute=params.get_option("requests_per_minute", 20),
                tokens_per_minute=params.get_option("tokens_per_minute", 30000),
                max_retries=params.get_option("max_retries", 5),
                max_unique_values=params.get_option("max_unique_values", 500),
            )
            
            skipped_numeric = [
                col for col in selected_columns
                if pd.api.types.is_numeric_dtype(context.data[col]) and col not in numeric_columns
            ]
            if skipped_numeric:
                context.log(
                    "Skipping numeric columns in data standardization "
                    f"(no numeric_ranges rules): {skipped_numeric}"
                )

            standardizer.standardize(
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
            )
            
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
