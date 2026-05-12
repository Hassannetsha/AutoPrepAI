import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from agents.missing_value_agent import MissingValueAgent
from data_context import DataContext
from agent_params import AgentParams


url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
# Remove unnecessary columns
df = df.drop(columns=[
    'PassengerId',
    'Name',
    'Ticket',
    'Cabin'
])

# Remove existing missing values
df = df.dropna().reset_index(drop=True)

target = "Survived"

original_df = df.copy()



def create_mcar(df, columns, missing_rate=0.2):

    df_missing = df.copy()

    for col in columns:

        mask = np.random.rand(len(df_missing)) < missing_rate

        df_missing.loc[mask, col] = np.nan

    return df_missing


def create_mar(df):

    df_missing = df.copy()

    mask = (
        (df_missing["Pclass"] == 3)
        &
        (np.random.rand(len(df_missing)) < 0.4)
    )

    # Make Fare missing more often for passengers in Pclass 3
    if "Fare" in df_missing.columns:
        df_missing.loc[mask, "Fare"] = np.nan

    return df_missing


def create_mnar(df):

    df_missing = df.copy()

    # MNAR: make higher Fare values more likely to be missing
    threshold = df_missing["Fare"].median()

    mask = (
        (df_missing["Fare"] > threshold)
        &
        (np.random.rand(len(df_missing)) < 0.5)
    )

    df_missing.loc[mask, "Fare"] = np.nan

    return df_missing


# GET MISSING MASK

def get_missing_mask(original_df, missing_df):

    return missing_df.isna() & ~original_df.isna()


# SIMPLE IMPUTATION

def simple_imputation(df, strategy="mean"):

    df_imputed = df.copy()

    numeric_cols = df_imputed.select_dtypes(
        include=np.number
    ).columns

    categorical_cols = df_imputed.select_dtypes(
        exclude=np.number
    ).columns

    # Numerical
    num_imputer = SimpleImputer(strategy=strategy)

    df_imputed[numeric_cols] = num_imputer.fit_transform(
        df_imputed[numeric_cols]
    )

    # Categorical
    cat_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    df_imputed[categorical_cols] = cat_imputer.fit_transform(
        df_imputed[categorical_cols]
    )

    return df_imputed


# KNN IMPUTATION

def knn_imputation(df):

    df_imputed = df.copy()

    numeric_cols = df_imputed.select_dtypes(
        include=np.number
    ).columns

    imputer = KNNImputer(n_neighbors=5)

    df_imputed[numeric_cols] = imputer.fit_transform(
        df_imputed[numeric_cols]
    )

    return df_imputed


# YOUR AGENT IMPUTATION

def agent_imputation(df, strategy=None):

    context = DataContext(data=df.copy())
    params = AgentParams(
        strategy=(strategy or ""),
        columns=[]
    )

    agent = MissingValueAgent()

    try:
        result_context = agent.execute(context, params)
    except Exception as e:
        raise RuntimeError(f"MissingValueAgent failed: {e}") from e

    if hasattr(result_context, "data"):
        return result_context.data
    if isinstance(result_context, pd.DataFrame):
        return result_context
    raise RuntimeError("Unexpected return type from MissingValueAgent.execute()")


# EVALUATE NUMERICAL COLUMNS

def evaluate_numeric(
        original_df,
        imputed_df,
        missing_mask,
        column
):

    true_values = original_df.loc[
        missing_mask[column],
        column
    ]

    # If there were no missing values inserted for this column, return NaNs
    # so that downstream aggregations or displays can handle the absence.
    if true_values.empty:
        return np.nan, np.nan

    pred_values = imputed_df.loc[
        missing_mask[column],
        column
    ]

    rmse = np.sqrt(
        mean_squared_error(
            true_values,
            pred_values
        )
    )

    mae = mean_absolute_error(
        true_values,
        pred_values
    )

    return rmse, mae


# DOWNSTREAM MODEL EVALUATION

def evaluate_downstream_model(
        df,
        target_column=target
):

    df_model = df.copy()

    # Encode categorical columns
    for col in df_model.select_dtypes(
            include="object"
    ).columns:

        le = LabelEncoder()

        df_model[col] = le.fit_transform(
            df_model[col].astype(str)
        )

    X = df_model.drop(columns=[target_column])

    y = df_model[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    acc = accuracy_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    return acc, f1


# RUN ALL EXPERIMENTS

results = []

columns_to_corrupt = ["Fare"]

missing_scenarios = [

    (
        "MCAR",
        create_mcar(
            original_df,
            columns_to_corrupt
        )
    ),

    (
        "MAR",
        create_mar(original_df)
    ),

    (
        "MNAR",
        create_mnar(original_df)
    )
]

for missing_type, missing_df in missing_scenarios:

    print(f"\nRunning {missing_type} Experiment...\n")

    missing_mask = get_missing_mask(
        original_df,
        missing_df
    )

    # ======== MEAN =============
    mean_df = simple_imputation(
        missing_df,
        strategy="mean"
    )

    rmse, mae = evaluate_numeric(
        original_df,
        mean_df,
        missing_mask,
        "Fare"
    )

    acc, f1 = evaluate_downstream_model(
        mean_df
    )

    results.append([
        "Mean/Mode",
        missing_type,
        rmse,
        mae,
        acc,
        f1
    ])

    # ======= MEDIAN =============

    median_df = simple_imputation(
        missing_df,
        strategy="median"
    )

    rmse, mae = evaluate_numeric(
        original_df,
        median_df,
        missing_mask,
        "Fare"
    )

    acc, f1 = evaluate_downstream_model(
        median_df
    )

    results.append([
        "Median/Mode",
        missing_type,
        rmse,
        mae,
        acc,
        f1
    ])

    # ======= KNN =============
    knn_df = knn_imputation(
        missing_df
    )

    rmse, mae = evaluate_numeric(
        original_df,
        knn_df,
        missing_mask,
        "Fare"
    )

    acc, f1 = evaluate_downstream_model(
        knn_df
    )

    results.append([
        "KNN",
        missing_type,
        rmse,
        mae,
        acc,
        f1
    ])


    # Do not pass the literal string "None" — pass no strategy so the agent
    # can auto-select based on data statistics.
    agent_df = agent_imputation(missing_df)

    rmse, mae = evaluate_numeric(
        original_df,
        agent_df,
        missing_mask,
        "Fare"
    )

    acc, f1 = evaluate_downstream_model(
        agent_df
    )

    results.append([
        "AutoPrepAI Agent",
        missing_type,
        rmse,
        mae,
        acc,
        f1
    ])


results_df = pd.DataFrame(
    results,
    columns=[
        "Method",
        "Missing Type",
        "RMSE",
        "MAE",
        "Accuracy",
        "F1"
    ]
)

print("\n================ FINAL RESULTS ================\n")

print(results_df)
