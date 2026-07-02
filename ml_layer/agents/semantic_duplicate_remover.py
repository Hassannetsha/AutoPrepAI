from ml_layer.agents.pipeline_agent import PipelineAgent
from business_logic.cleaning_coordinator.data_context import DataContext
from business_logic.cleaning_coordinator.agent_params import AgentParams
from ml_layer.duplicates.semantic_duplicate_remover_service import SemanticDuplicateRemoverService

import pandas as pd

class SemanticDuplicateRemover(PipelineAgent):
    def __init__(self):
        super().__init__("Semantic Duplicate Remover")

    def _semantic_text_candidates(
        self,
        context: DataContext,
        columns: list[str]
    ) -> tuple[list[str], list[dict]]:

        text_columns = context.data.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()

        requested = (
            [col for col in columns if col in text_columns]
            if columns else text_columns
        )

        candidates = []
        analysis = []

        for col in requested:

            values = context.data[col].dropna().astype(str)

            if values.empty:
                continue

            avg_length = values.str.len().mean()
            avg_words = values.str.split().str.len().mean()
            unique_ratio = values.nunique() / max(1, len(values))

            url_ratio = values.str.contains(
                r"https?://|www\.|\.jpg|\.jpeg|\.png|\.gif",
                case=False,
                regex=True,
            ).mean()

            identifier_ratio = values.str.fullmatch(
                r"[A-Za-z0-9_-]{12,}"
            ).mean()

            numeric_ratio = pd.to_numeric(
                values.str.replace(",", "", regex=False),
                errors="coerce"
            ).notna().mean()

            selected = True
            reason = "Passed all heuristics"

            if numeric_ratio >= 0.8:
                selected = False
                reason = "numeric values"

            elif avg_length < 20:
                selected = False
                reason = "short text"

            elif avg_words < 2:
                selected = False
                reason = "categorical values"

            elif unique_ratio < 0.05:
                selected = False
                reason = "Low uniqueness"

            elif url_ratio >= 0.5:
                selected = False
                reason = "mostly URLs"

            elif identifier_ratio >= 0.8:
                selected = False
                reason = "identifier-like values"

            analysis.append({
                "column": col,
                "selected": selected,
                "reason": reason,
            })

            if selected:
                candidates.append(col)

        return candidates, analysis

    def execute(self, context: DataContext, params: AgentParams) -> DataContext:
        context.data = context.data.reset_index(drop=True)
        columns = params.columns or []
        context.log("Removing semantic duplicate rows")
        
        try:
            # Check if data is suitable for semantic duplicate detection
            text_columns = context.data.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
            
            if not text_columns:
                context.log("No text columns found for semantic duplicate detection")
                return context
            
            # Determine which text column to use
            candidate_columns, analysis = self._semantic_text_candidates(context, columns)

            if not candidate_columns:
                context.log("No suitable text columns found for semantic duplicate detection.")
                return context

            context.log(f"Analyzing {len(text_columns)} text columns...")

            context.log(f"Selected {len(candidate_columns)} semantic text columns:")
            for col in candidate_columns:
                context.log(f"• {col}")

            rejected = [c for c in analysis if not c["selected"]]

            if rejected:
                context.log(f"Ignored {len(rejected)} columns:")
                for info in rejected:
                    context.log(f"• {info['column']} ({info['reason']})")

            # Initialize semantic duplicate remover
            remover = SemanticDuplicateRemoverService(
                model_name="paraphrase-MiniLM-L6-v2",
                threshold=0.85,
                k_neighbors=10,
                batch_size=512
            )
            
            # Remove semantic duplicates
            if len(candidate_columns) == 1:
                df_dedup, df_duplicates = remover.remove_duplicates(
                    context.data,
                    text_column=candidate_columns[0]
                )
            else:
                df_dedup, df_duplicates = remover.remove_duplicates_multicolumn(
                    context.data,
                    text_columns=candidate_columns
                )
            
            num_duplicates = len(df_duplicates) if not df_duplicates.empty else 0

            MAX_EXAMPLES = 3
            MAX_CHARS = 150

            examples = []

            for _, pair in df_duplicates.head(MAX_EXAMPLES).iterrows():

                example = {
                    "similarity": round(pair["similarity"] * 100, 1),
                    "kept": {},
                    "removed": {},
                }

                for col in candidate_columns:

                    kept = str(context.data.loc[pair["query_index_1"], col])
                    removed = str(context.data.loc[pair["query_index_2"], col])

                    if len(kept) > MAX_CHARS:
                        kept = kept[:MAX_CHARS] + "..."

                    if len(removed) > MAX_CHARS:
                        removed = removed[:MAX_CHARS] + "..."

                    example["kept"][col] = kept
                    example["removed"][col] = removed

                examples.append(example)

            context.data = df_dedup
            context.metadata["semantic_duplicates_removed"] = True
            context.metadata["semantic_duplicates_count"] = num_duplicates
            context.metadata["semantic_columns_used"] = candidate_columns
            context.metadata["semantic_column_analysis"] = analysis
            context.metadata["semantic_duplicate_examples"] = examples
            
            if num_duplicates > 0:

                context.log(f"Removed {num_duplicates} semantic duplicate rows.")

                if examples:

                    context.log("Examples of removed semantic duplicates:")

                    for i, example in enumerate(examples, 1):

                        context.log(f"Example {i}")

                        context.log("Kept row:")
                        for col, value in example["kept"].items():
                            context.log(f"  {col}: {value}")

                        context.log("Removed row:")
                        for col, value in example["removed"].items():
                            context.log(f"  {col}: {value}")

                        context.log(f"Similarity: {example['similarity']}%")
                        context.log("----------------------------------------")
            else:
                context.log("No semantic duplicate rows found.")
                
        except Exception as e:
            context.log(f"Semantic duplicate removal failed.")
            print(f"Semantic duplicate removal failed: {e}")
            import traceback
            print(traceback.format_exc())
        
        return context
