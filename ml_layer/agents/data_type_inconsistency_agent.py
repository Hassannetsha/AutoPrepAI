from ml_layer.agents.pipeline_agent import PipelineAgent
from business_logic.cleaning_coordinator.data_context import DataContext
from business_logic.cleaning_coordinator.agent_params import AgentParams
from ml_layer.data_type_inconsistency.detector import DataTypeInconsistencyDetector
from ml_layer.data_type_inconsistency.resolver import DataResolvingService

class DataTypeInconsistencyAgent(PipelineAgent):
    def __init__(self):
        super().__init__("Data Type Inconsistency Handler")

    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        context.data = context.data.reset_index(drop=True)
        context.log("Detecting and resolving data type inconsistencies")

        try:
            # Step 1: Detect inconsistencies
            detector = DataTypeInconsistencyDetector()
            detection_results = detector.analyze_dataframe(context.data)

            context.metadata["datatype_detection_results"] = detection_results

            # Step 2: Identify inconsistent columns
            inconsistent_columns = []

            for col_name, result in detection_results.items():
                if len(result.get("detected_types", {})) > 1:
                    inconsistent_columns.append(col_name)

            if not inconsistent_columns:
                context.log("No data type inconsistencies detected")
                context.metadata["datatype_inconsistencies_fixed"] = True
                return context

            # Step 3: Resolve inconsistencies
            resolver = DataResolvingService(context.data, detection_results)

            total_failed = 0
            lossy_columns = 0

            for col_name in inconsistent_columns:
                result = detection_results[col_name]
                recommended_type = result.get("recommended_type")

                if not recommended_type or recommended_type == "empty_column":
                    continue

                confidence_level = result.get("recommendation_confidence", "Low")

                confidence_scores = {
                    "High": 0.9,
                    "Medium": 0.6,
                    "Low": 0.3
                }

                confidence = confidence_scores.get(confidence_level, 0)
                
                context.log(f"Column '{col_name}'")
                context.log(f"Detected types: {result['detected_types']}")
                context.log(
                    f"Recommended type: {recommended_type} "
                    f"({confidence_level} confidence, "
                    f"{result['conversion_failure_rate']:.1%} conversion failures)"
                )

                if (
                    context.metadata.get("mode") == "full_auto"
                    and confidence < 0.7
                ):
                    context.log(
                        f"Skipped '{col_name}' conversion due to low confidence."
                    )
                    continue

                _, message = resolver.resolve(
                    strategy_name="convert_to_type",
                    column_name=col_name,
                    target_type=recommended_type,
                )

                context.log(message)

                conversion = result.get("conversion_issues", {})
                failed_count = conversion.get("failed_count", 0)
                failed_values = conversion.get("failed_values", [])

                if failed_count > 0:
                    total_failed += failed_count
                    lossy_columns += 1

                    context.log(
                        f"Warning: {failed_count} values ({result['conversion_failure_rate']:.1%}) "
                        f"could not be converted and will become missing values."
                    )

                    if failed_values:
                        examples = ", ".join(map(str, failed_values[:3]))
                        context.log(f"Examples: {examples}")

            # Update context
            context.data = resolver.df_resolved
            context.metadata["datatype_inconsistencies_fixed"] = True
            context.metadata["resolution_log"] = resolver.resolution_log

            context.log(f"Resolved {len(inconsistent_columns)} inconsistent columns.")

            if total_failed > 0:
                context.log(
                    f"{lossy_columns} column(s) required lossy conversion. "
                    f"{total_failed} values became missing."
                )

        except Exception as e:
            context.log("Data type inconsistency handling failed.")
            context.log(f"Data type inconsistency handling error: {e}")
            
            import traceback
            print(traceback.format_exc())

        return context