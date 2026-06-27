from agents.pipeline_agent import PipelineAgent
from data_context import DataContext
from agent_params import AgentParams
from data_standardization.data_standardizing_service import DataStandardizingService
from data_standardization.validation_layer import ValidationLayer
from api_key_manager import get_key_manager 
from data_standardization.rate_limiter import RateLimiter
from data_standardization.groq_llm_client import GroqLLMClient

from groq import Groq
import pandas as pd


class DataStandardizerAgent(PipelineAgent):
    def __init__(self):
        super().__init__("Data Standardizer")

    def _build_validation_layer(self, context: DataContext, params: AgentParams) -> ValidationLayer:
        """
        Builds the validation layer for CATEGORICAL standardization.
        """
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

        return validation

    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        context.data = context.data.reset_index(drop=True)
        columns = params.columns or []
        target_columns = [col for col in columns if col in context.data.columns]
        selected_columns = target_columns if columns else list(context.data.columns)
        
        # Only extract categorical columns — numeric is ignored entirely
        categorical_columns = [
            col for col in selected_columns
            if not pd.api.types.is_numeric_dtype(context.data[col])
        ]

        # ── Filter out High-Cardinality Columns (e.g., IDs, Names) ──
        columns_to_process = []
        ID_KEYWORDS = {
            "id",
            "_id",
            "identifier",
            "uuid",
            "code",
            "serial"
        }

        for col in categorical_columns:
            series = context.data[col].dropna()

            # Skip identifier columns
            if any(k in col.lower() for k in ID_KEYWORDS):
                context.log(f"Skipped identifier column '{col}'")
                print(f"Skipped identifier column '{col}'")
                continue

            unique_count = series.nunique()

            # Skip very high-cardinality columns
            if unique_count > params.get_option("max_unique_values", 500):
                context.log(
                    f"Skipped '{col}' ({unique_count} unique values)"
                )
                print(f"Skipped '{col}' ({unique_count} unique values)")
                continue

            columns_to_process.append(col)
        # ─────────────────────────────────────────────────────────

        context.log("Standardizing categorical data values using LLM")

        try:
            if columns and not target_columns:
                context.log("Data standardization skipped: none of the requested columns exist")
                context.metadata["data_standardized"] = False
                return context

            if not columns_to_process:
                context.log("Data standardization skipped: no valid categorical columns to process")
                context.metadata["data_standardized"] = False
                return context

            # ── Build LLM client ────────────────────────────────
            key_manager = get_key_manager()
            api_key = key_manager.get_current_key()

            rate_limiter = RateLimiter(
                requests_per_minute=params.get_option("requests_per_minute", 20),
                tokens_per_minute=params.get_option("tokens_per_minute", 30_000),
            )
            llm_client = GroqLLMClient(
                groq_client=Groq(api_key=api_key),
                model=params.get_option("model", "llama-3.3-70b-versatile"),
                rate_limiter=rate_limiter,
                max_retries=params.get_option("max_retries", 5),
            )
            # ───────────────────────────────────────────────────

            validation_layer = self._build_validation_layer(context, params)

            standardizer = DataStandardizingService(
                df=context.data,
                llm_client=llm_client,
                validation=validation_layer,
                confidence_threshold=params.get_option("confidence_threshold", 0.7),
                similarity_threshold=params.get_option("similarity_threshold", 0.35),
                max_unique_values=params.get_option("max_unique_values", 500),
            )

            # Pass the FILTERED list
            standardizer.standardize(
                categorical_columns=columns_to_process,
            )
            
            context.data = standardizer.df
            context.metadata["data_standardized"] = True
            context.metadata["standardization_results"] = standardizer.results
            context.log("Data standardization completed")

        except Exception as e:
            context.log("Data standardization failed.")
            context.print(f"Data standardization error: {e}")
            import traceback
            print(traceback.format_exc())

        return context