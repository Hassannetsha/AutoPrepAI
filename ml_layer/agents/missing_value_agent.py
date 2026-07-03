from ml_layer.agents.pipeline_agent import PipelineAgent
from business_logic.cleaning_coordinator.data_context import DataContext
from business_logic.cleaning_coordinator.agent_params import AgentParams
from ml_layer.missing_values.Class_missingValues import MissingValuesDemo

import numpy as np
import pandas as pd

from sklearn.impute import (
    KNNImputer,
    SimpleImputer
)

from sklearn.experimental import (
    enable_iterative_imputer
)

from sklearn.impute import (
    IterativeImputer
)

from sklearn.metrics import (
    mean_squared_error
)


class MissingValueAgent(PipelineAgent):

    def __init__(self):
        super().__init__("Missing Values")

    # AUTO EVALUATION MODEL
    def choose_best_strategy(
        self,
        data: pd.DataFrame,
        numeric_cols: list
    ):

        has_integer = any(
            pd.api.types.is_integer_dtype(data[c])
            for c in numeric_cols
        )

        strategies = [
            "median",
            "knn",
            "mice"
        ]
        if not has_integer:
            strategies.insert(0, "mean")

        scores = {}

        temp_data = (
            data[numeric_cols]
            .copy()
        )

        original_data = (
            temp_data.copy()
        )


        np.random.seed(42)

        mask = (
            temp_data.notnull()
            & (
                np.random.rand(
                    *temp_data.shape
                ) < 0.1
            )
        )

        temp_data[mask] = np.nan

        # -------------------------------------------------
        # Try all strategies
        # -------------------------------------------------
        for strategy in strategies:

            try:

                test_df = (
                    temp_data.copy()
                )

                if strategy == "mean":

                    imputer = SimpleImputer(
                        strategy="mean"
                    )

                    test_df[:] = (
                        imputer.fit_transform(
                            test_df
                        )
                    )

                elif strategy == "median":

                    imputer = SimpleImputer(
                        strategy="median"
                    )

                    test_df[:] = (
                        imputer.fit_transform(
                            test_df
                        )
                    )


                elif strategy == "knn":

                    imputer = KNNImputer(
                        n_neighbors=5
                    )

                    test_df[:] = (
                        imputer.fit_transform(
                            test_df
                        )
                    )

                elif strategy == "mice":

                    imputer = IterativeImputer(
                        random_state=42,
                        max_iter=10
                    )

                    test_df[:] = (
                        imputer.fit_transform(
                            test_df
                        )
                    )
                    
                predicted = (
                    test_df[mask]
                )

                actual = (
                    original_data[mask]
                )

                rmse = np.sqrt(
                    mean_squared_error(
                        actual.values.flatten(),
                        predicted.values.flatten()
                    )
                )

                scores[strategy] = rmse

            except Exception:

                scores[strategy] = float("inf")

        best_strategy = min(
            scores,
            key=scores.get
        )

        return best_strategy, scores


    @staticmethod
    def _replace_placeholder_markers(data: pd.DataFrame) -> pd.DataFrame:
        """Replace single-character special-symbol placeholders (?, $, #, etc.)
        in object columns with NaN, so they are treated as missing values."""
        import string
        special_chars = set(string.punctuation)  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        data = data.copy()
        for col in data.select_dtypes(include="object").columns:
            mask = data[col].apply(
                lambda x: isinstance(x, str)
                and len(x) == 1
                and x in special_chars
            )
            if mask.any():
                data.loc[mask, col] = np.nan
                count = mask.sum()
                print(f"[MissingValueAgent] Replaced {count} placeholder marker(s) in '{col}' with NaN")
        return data

    def execute(
        self,
        context: DataContext,
        params: AgentParams
    ) -> DataContext:
        context.data = self._replace_placeholder_markers(context.data)
        context.data = context.data.reset_index(drop=True)

        columns = params.columns or []
        strategy = params.strategy

        for intent in (
            context.metadata.get(
                "intents",
                []
            )
        ):

            if (
                isinstance(
                    intent,
                    (list, tuple)
                )
                and len(intent) > 0
            ):

                if (
                    intent[0]
                    == "handle_missing_values"
                ):

                    if (
                        len(intent) > 2
                        and isinstance(
                            intent[2],
                            str
                        )
                    ):

                        strategy = intent[2]

                    break

        print(
            f"MissingValueAgent: "
            f"strategy={strategy}, "
            f"columns={columns}"
        )

        if not columns:

            numeric_cols = (
                context.data
                .select_dtypes(
                    include=[np.number]
                )
                .columns
                .tolist()
            )

            categorical_cols = (
                context.data
                .select_dtypes(
                    exclude=[np.number]
                )
                .columns
                .tolist()
            )

        else:

            numeric_cols = [
                col for col in columns
                if pd.api.types.is_numeric_dtype(
                    context.data[col]
                )
            ]

            categorical_cols = [
                col for col in columns
                if not pd.api.types.is_numeric_dtype(
                    context.data[col]
                )
            ]

        # Filter out numeric columns where ALL values are NaN (SimpleImputer cannot handle them)
        numeric_cols = [
            col for col in numeric_cols
            if not context.data[col].isna().all()
        ]

        if (
            not numeric_cols
            and not categorical_cols
        ):

            context.log(
                "No valid columns "
                "to handle missing values"
            )

            return context

        if (
            not strategy
            or not isinstance(
                strategy,
                str
            )
        ):

            if numeric_cols:

                context.log(
                    "Evaluating best "
                    "imputation strategy..."
                )

                strategy, scores = (
                    self.choose_best_strategy(
                        context.data,
                        numeric_cols
                    )
                )

                context.log(
                    f"Best strategy selected: "
                    f"{strategy}"
                )

                context.log(
                    f"Strategy scores: "
                    f"{scores}"
                )

            else:

                strategy = "mode"

        if numeric_cols:

            context.log(
                f"Handling numeric columns: "
                f"{numeric_cols}"
            )

            context.log(
                f"Using strategy: "
                f"{strategy}"
            )

            if strategy == "knn":

                try:

                    imputer = KNNImputer(
                        n_neighbors=5
                    )

                    context.data[
                        numeric_cols
                    ] = (
                        imputer.fit_transform(
                            context.data[
                                numeric_cols
                            ]
                        )
                    )

                    context.log(
                        "KNN imputation completed"
                    )

                except Exception as e:

                    context.log(
                        f"KNN failed: {str(e)}"
                    )


            elif strategy == "mice":

                try:

                    imputer = (
                        IterativeImputer(
                            random_state=42,
                            max_iter=10
                        )
                    )

                    context.data[
                        numeric_cols
                    ] = (
                        imputer.fit_transform(
                            context.data[
                                numeric_cols
                            ]
                        )
                    )

                    context.log(
                        "MICE imputation completed"
                    )

                except Exception as e:

                    context.log(
                        f"MICE failed: {str(e)}"
                    )

            elif strategy == "mean":

                imputer = SimpleImputer(
                    strategy="mean"
                )

                context.data[
                    numeric_cols
                ] = (
                    imputer.fit_transform(
                        context.data[
                            numeric_cols
                        ]
                    )
                )

                context.log(
                    "Mean imputation completed"
                )

            elif strategy == "median":

                imputer = SimpleImputer(
                    strategy="median"
                )

                context.data[
                    numeric_cols
                ] = (
                    imputer.fit_transform(
                        context.data[
                            numeric_cols
                        ]
                    )
                )

                context.log(
                    "Median imputation completed"
                )


            else:

                demo = MissingValuesDemo()

                context.data = demo.run(
                    context.data,
                    strategy="mean",
                    selected_cols=numeric_cols
                )


        if categorical_cols:

            context.log(
                f"Handling categorical columns "
                f"with mode: "
                f"{categorical_cols}"
            )

            for col in categorical_cols:

                if (
                    context.data[col]
                    .isnull()
                    .sum() > 0
                ):

                    mode_value = (
                        context.data[col]
                        .mode(dropna=True)
                    )

                    if not mode_value.empty:

                        context.data[col] = (
                            context.data[col]
                            .fillna(
                                mode_value[0]
                            )
                        )
                    else:
                        context.data[col] = (
                            context.data[col]
                            .fillna("unknown")
                        )
                        context.log(
                            f"Column '{col}' has all NaN — "
                            f"filled with 'unknown'"
                        )

        remaining_missing = (
            context.data.isnull().sum().sum()
        )

        # Fallback: fill any remaining NaN with 0 for numeric, 'unknown' for non-numeric
        if remaining_missing > 0:
            for col in context.data.columns:
                if context.data[col].isna().any():
                    if pd.api.types.is_numeric_dtype(context.data[col]):
                        context.data[col] = context.data[col].fillna(0)
                    else:
                        context.data[col] = context.data[col].fillna("unknown")
            context.log(
                f"Fallback fill completed "
                f"for remaining NaN values"
            )

        remaining_missing = (
            context.data.isnull().sum().sum()
        )

        context.log(
            f"Remaining missing values: "
            f"{remaining_missing}"
        )

        return context
