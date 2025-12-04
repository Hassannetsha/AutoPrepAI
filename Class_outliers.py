# ...existing code...
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import glob
from typing import Optional, List

class class_outliers:
    """
    Simple wrapper to run IsolationForest outlier detection on a CSV.
    """

    def __init__(self, csv_glob: str = "./Input/placement.csv"):
        self.csv_glob = csv_glob
        self.dataframe: Optional[pd.DataFrame] = None
        self.numerical_cols: List[str] = []
        self.iso: Optional[IsolationForest] = None

    def load(self):
        csv_files = glob.glob(self.csv_glob, recursive=True)
        if not csv_files:
            raise FileNotFoundError(f"Please provide a dataset matching: {self.csv_glob}")
        DATA_PATH = csv_files[0]
        print("Using dataset:", DATA_PATH)
        self.dataframe = pd.read_csv(DATA_PATH)
        print("\nOriginal Dataset Shape:", self.dataframe.shape)
        print(self.dataframe)
        self.numerical_cols = self.dataframe.select_dtypes(include=[np.number]).columns.tolist()
        if not self.numerical_cols:
            raise Exception("No numerical columns found for outlier detection!")
        print("\nNumerical columns considered for outliers:", self.numerical_cols)
        print(self.dataframe[self.numerical_cols].describe())

    def run_isolation_forest(self,
                             contamination="auto",
                             random_state=42,
                             n_estimators=200,
                             max_samples="auto",
                             max_features=1.0):
        if self.dataframe is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        print("\nRunning Isolation Forest...")
        X = self.dataframe[self.numerical_cols].fillna(0)
        self.iso = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features
        )
        self.iso.fit(X)
        self.dataframe['IF_score'] = self.iso.decision_function(X)
        self.dataframe['is_outlier_IF'] = self.iso.predict(X)
        print("\nIsolationForest predictions (+1=inlier, -1=outlier):")
        print(self.dataframe['is_outlier_IF'].value_counts())

    def get_cleaned(self) -> pd.DataFrame:
        if self.dataframe is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        df_cleaned = self.dataframe[self.dataframe['is_outlier_IF'] == 1].drop(columns=['is_outlier_IF'])
        print(f"\nAfter IsolationForest filtering: {df_cleaned.shape[0]} rows remain (from {self.dataframe.shape[0]} original)")
        return df_cleaned

    def plot_scatter(self):
        if self.dataframe is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        if len(self.numerical_cols) < 2:
            print("Need at least two numerical columns to scatter plot.")
            return
        plt.figure(figsize=(6,4))
        plt.scatter(self.dataframe[self.numerical_cols[0]],
                    self.dataframe[self.numerical_cols[1]],
                    c=self.dataframe['is_outlier_IF'], cmap='coolwarm', edgecolor='k')
        plt.xlabel(self.numerical_cols[0])
        plt.ylabel(self.numerical_cols[1])
        plt.title("IsolationForest Outlier Detection")
        plt.show()

    def plot_score_hist(self):
        if self.dataframe is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        plt.figure(figsize=(6,4))
        plt.hist(self.dataframe["IF_score"], bins=30, edgecolor="k")
        plt.title("Distribution of Isolation Forest Scores")
        plt.xlabel("IF_score")
        plt.ylabel("Frequency")
        plt.show()

    def summary(self):
        if self.dataframe is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        df_cleaned = self.get_cleaned()
        print("\nSummary of Outlier Handling:")
        print(f"- Original dataset rows: {self.dataframe.shape[0]}")
        print(f"- After IsolationForest: {df_cleaned.shape[0]} rows remain")
        print(f"- Outliers detected: {(self.dataframe.shape[0] - df_cleaned.shape[0])}")

if __name__ == "__main__":
    c = class_outliers()
    c.load()
    c.run_isolation_forest()
    cleaned = c.get_cleaned()
    c.plot_score_hist()
    try:
        c.plot_scatter()
    except Exception:
        pass
    c.summary()
    print(cleaned)
# ...existing code...