import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from typing import Optional, List, Tuple


class FeatureSelectionAgent:
    """Reusable feature selection engine that supports threshold or top-N selection.

    Methods are headless and safe to use from the preprocessing pipeline (no Streamlit side-effects).
    """
    def __init__(self, estimator: Optional[RandomForestClassifier] = None, random_state: int = 42):
        self.random_state = random_state
        self.estimator = estimator or RandomForestClassifier(n_estimators=100, random_state=random_state)

    def select_features(self, df: pd.DataFrame, target_col: str, threshold: str = "median", n_features: Optional[int] = None) -> Tuple[List[str], pd.DataFrame]:
        """Return (selected_feature_names, pruned_dataframe_with_target).

        - df: full DataFrame containing target_col
        - target_col: the name of the target/label column
        - threshold: threshold to pass to SelectFromModel (e.g., 'median', 'mean')
        - n_features: if provided, select top-n by feature importance instead of threshold
        """
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not found in DataFrame columns")

        # Prepare features and target
        X = df.drop(columns=[target_col])
        X_encoded = pd.get_dummies(X)
        y = df[target_col]

        if n_features is not None:
            # Fit a RandomForest and take the top-n features by importance
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            rf.fit(X_encoded, y)
            importances = pd.Series(rf.feature_importances_, index=X_encoded.columns).sort_values(ascending=False)
            selected = list(importances.index[:n_features])
        else:
            selector = SelectFromModel(self.estimator, threshold=threshold)
            selector.fit(X_encoded, y)
            selected = list(X_encoded.columns[selector.get_support()])

        # Build the pruned DataFrame (include selected features and the target column)
        result_df = pd.concat([X_encoded[selected], df[[target_col]].reset_index(drop=True)], axis=1)
        return selected, result_df

    def run(self, df: pd.DataFrame, columns: Optional[List] = None, threshold: Optional[str] = None, n_features: Optional[int] = None, metadata: Optional[dict] = None) -> Tuple[List[str], pd.DataFrame]:
        """Higher-level runner that accepts pipeline-style `columns`/`metadata` configuration.

        Supported column formats:
          - ['target=Label', 'top=10']
          - ['Label', 'top=10']
          - ['median'] or ['mean'] to set the threshold
          - an integer value in the list treated as `n_features`

        Falls back to metadata.get('target_col') when target is not in `columns`.
        """
        columns = columns or []
        target = None
        if threshold is None:
            threshold = "median"

        for item in columns:
            if isinstance(item, str) and item.startswith("target="):
                target = item.split("=", 1)[1]
            elif isinstance(item, str) and item.startswith("top="):
                try:
                    n_features = int(item.split("=", 1)[1])
                except Exception:
                    pass
            elif isinstance(item, str) and item in ("median", "mean"):
                threshold = item
            elif isinstance(item, str) and "=" not in item and target is None:
                # treat a lone string as target column
                target = item
            elif isinstance(item, int) and n_features is None:
                n_features = item

        # metadata fallback
        if target is None and isinstance(metadata, dict):
            target = metadata.get("target_col")
            # allow n_features from metadata
            if n_features is None:
                n_features = metadata.get("n_features")
            if threshold is None:
                threshold = metadata.get("threshold", threshold)

        if target is None:
            raise ValueError("FeatureSelectionAgent.run requires a target column via columns or metadata['target_col']")

        # delegate to select_features
        return self.select_features(df, target_col=target, threshold=threshold, n_features=n_features)


if __name__ == "__main__":
    df = pd.read_csv("Input/emp.csv")
    target = "LeftCompany"
    engine = FeatureSelectionAgent()
    res = engine.run(df, columns=[f"target={target}"])
    print(res)
    