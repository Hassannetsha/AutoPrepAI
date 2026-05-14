import pandas as pd

from outliers.strategies.zscore_strategy import ZScoreStrategy
from outliers.strategies.iqr_strategy import IQRStrategy
from outliers.strategies.isolation_forest_strategy import IsolationForestStrategy

# Import the corrected evaluator (replace your old import path if needed)
from outliers.evaluation.benchmark_evaluator import OutlierBenchmarkEvaluator

from services.outliers_service import OutliersService



datasets = [
     {
        "name": "Titanic",
        "path": r"Input\Titanic-Dataset.csv",
        "target": "Survived"
    },

    {
        "name": "Adult Income",
        "path": r"Input\UCI_Adult_Income_Dataset.csv",
        "target": "income"
    },

    {
        "name": "Wine Quality",
        "path": r"Input\WineQT.csv",
        "target": "quality"
    },

   
    {
        "name": "Credit Card Fraud",
        "path": r"Input\creditcard.csv",
        "target": "Class",

    },
]


all_results = []

for dataset_info in datasets:


    # ─── Load ─────────────────────────────────────────────────────────
    df = pd.read_csv(dataset_info["path"])
    target_column = dataset_info["target"]

    # ─── Strategies ───────────────────────────────────────────────────
    strategies = {
        "Z-Score":          ZScoreStrategy(),
        "IQR":              IQRStrategy(),
        "Isolation Forest": IsolationForestStrategy(),
    }

    # ─── Evaluate ─────────────────────────────────────────────────────
    evaluator = OutlierBenchmarkEvaluator(
        dataframe=df,
        target_column=target_column,
        strategies=strategies,
        random_state=42,
        cv_folds=5,          # 5-fold CV — stable estimate
    )

    results_df = evaluator.evaluate()
    # ─── Selector decision ────────────────────────────────────────────
    service = OutliersService(df)
    service.process()
    selected_strategy = service.get_strategy_name()

    print(f"\nSelector Decision: {selected_strategy}")

    results_df["Dataset"]           = dataset_info["name"]
    results_df["Selected Strategy"] = selected_strategy
    all_results.append(results_df)

    evaluator.print_summary()
final_results = pd.concat(all_results, ignore_index=True)


columns_order = [
    "Dataset",
    "Strategy",
    "Selected Strategy",
    "Rows Before",
    "Rows After",
    "Removed Rows",
    "Removal Rate",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
    "Execution Time (s)",
]

final_results = final_results[columns_order]

final_results.to_csv("all_outlier_benchmark_results.csv", index=False)

print("\n" + "=" * 80)
print("FINAL BENCHMARK RESULTS")
print("=" * 80)
print(final_results.to_string(index=False))

print("\nResults exported to: all_outlier_benchmark_results.csv")

selector_summary = (
    final_results
    .drop_duplicates(subset=["Dataset"])
    [["Dataset", "Selected Strategy"]]
)

print("\n" + "=" * 80)
print("SELECTOR ACCURACY SUMMARY")
print("=" * 80)
print(selector_summary.to_string(index=False))
total_count   = len(selector_summary)
print("\nDone.")