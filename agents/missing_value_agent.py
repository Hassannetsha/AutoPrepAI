from agents.pipeline_agent import PipelineAgent
from data_context import DataContext
from agent_params import AgentParams
from Class_missingValues import MissingValuesDemo

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

        strategies = [
            "mean",
            "median",
            "knn",
            "mice"
        ]

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


    def execute(
        self,
        context: DataContext,
        params: AgentParams
    ) -> DataContext:
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

        remaining_missing = (
            context.data
            .isnull()
            .sum()
            .sum()
        )

        context.log(
            f"Remaining missing values: "
            f"{remaining_missing}"
        )

        return context
