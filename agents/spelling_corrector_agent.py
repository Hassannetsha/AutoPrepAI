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
            # Find categorical columns
            categorical_cols = context.data.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            if not categorical_cols:
                context.log("No categorical columns found")
                return context

            # Use specified columns if provided
            target_cols = columns if columns else categorical_cols
            target_cols = [c for c in target_cols if c in categorical_cols]

            if not target_cols:
                context.log("No valid categorical columns selected")
                return context

            # Load SymSpell packaged dictionary
            corrector = SpellingCorrectorService(
                max_edit_distance=2,
                prefix_length=7
            )

            corrected_columns = []

            for col in target_cols:
                try:
                    # Skip columns that are mostly unique (IDs, names, emails, etc.)
                    unique_ratio = (
                        context.data[col].nunique(dropna=True)
                        / max(len(context.data), 1)
                    )

                    if unique_ratio > 0.9:
                        context.log(f"Skipping '{col}' (mostly unique values)")
                        continue

                    original = context.data[col].copy()

                    corrected = corrector.correct_dataframe_column(
                        context.data,
                        column_name=col,
                        show_progress=False,
                        inplace=False
                    )

                    context.data[col] = corrected

                    changed = (original != corrected).sum()

                    corrected_columns.append(col)
                    context.log(
                        f"'{col}': corrected {changed} value(s)"
                    )

                except Exception as e:
                    context.log(f"Failed to correct '{col}': {e}")

            context.metadata["spelling_corrected"] = True
            context.metadata["spelling_corrected_columns"] = corrected_columns

            context.log(
                f"Spelling correction completed for {len(corrected_columns)} columns."
            )

        except Exception as e:
            context.log("Spelling correction failed.")
            print(e)
            import traceback
            print(traceback.format_exc())

        return context