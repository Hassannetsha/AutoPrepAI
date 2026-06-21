import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from typing import Optional, List, Tuple


class FeatureSelectionService:
    """Reusable feature selection engine that supports threshold or top-N selection.

    Methods are headless and safe to use from the preprocessing pipeline.
    """
    def __init__(self, estimator: Optional[RandomForestClassifier] = None, random_state: int = 42):
        self.random_state = random_state
        self.estimator = estimator or RandomForestClassifier(n_estimators=100, random_state=random_state)

    def _filter_useless_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove constant and high-cardinality ID columns before feature selection."""
        before = set(X.columns)
        nunique = X.nunique()
        constant = nunique[nunique <= 1].index.tolist()
        X = X.drop(columns=constant, errors='ignore')
        for col in X.columns:
            if X[col].dtype in ("int64", "float64") and X[col].nunique() > 0.9 * len(X) and X[col].nunique() >= 100:
                X = X.drop(columns=[col])
        dropped = before - set(X.columns)
        if dropped:
            print(f"[FeatureSelection] Dropped useless columns: {sorted(dropped)}")
        return X

    def select_features(self, df: pd.DataFrame, target_col: str, threshold: str = "mean", n_features: Optional[int] = None, min_importance: float = 0.01) -> Tuple[List[str], pd.DataFrame]:
        """Return (selected_feature_names, pruned_dataframe_with_target).

        - df: full DataFrame containing target_col
        - target_col: the name of the target/label column (required)
        - threshold: threshold to pass to SelectFromModel (e.g., 'mean', 'median')
        - n_features: if provided, select top-n by feature importance instead of threshold
        - min_importance: minimum absolute importance a feature must have to be kept (default 0.01)
        """
        print(f"[FeatureSelection] Starting feature selection with target='{target_col}', threshold='{threshold}', n_features={n_features}, min_importance={min_importance}")
        df = df.reset_index(drop=True)

        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not found in DataFrame columns")

        X = df.drop(columns=[target_col])
        X = self._filter_useless_columns(X)
        X_encoded = pd.get_dummies(X)
        y = df[target_col]

        if n_features is not None:
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            rf.fit(X_encoded, y)
            importances = pd.Series(rf.feature_importances_, index=X_encoded.columns).sort_values(ascending=False)
            selected = list(importances.index[:n_features])
        else:
            selector = SelectFromModel(self.estimator, threshold=threshold)
            selector.fit(X_encoded, y)
            selected = list(X_encoded.columns[selector.get_support()])

        if min_importance > 0.0:
            importances = pd.Series(selector.estimator_.feature_importances_, index=X_encoded.columns)
            selected = [c for c in selected if importances.get(c, 0) >= min_importance]

        selected = [c for c in selected if c in X_encoded.columns]
        print(f"[FeatureSelection] Selected {len(selected)} features: {selected}")
        result_df = pd.concat(
            [
                X_encoded[selected].reset_index(drop=True),
                df[[target_col]].reset_index(drop=True),
            ],
            axis=1,
        )
        return selected, result_df

    def run(self, df: pd.DataFrame, columns: Optional[List] = None, threshold: Optional[str] = None, n_features: Optional[int] = None, metadata: Optional[dict] = None) -> Tuple[List[str], pd.DataFrame]:
        """Run feature selection with parameters resolved from the pipeline.

        Target column is required — it is extracted from (in priority order):
          1. ``columns`` list: ``'target=Name'`` or a bare string column name
          2. ``metadata['target_col']``

        Additional parameters from ``columns``:
          - ``'top=N'`` → select top-N features
          - ``'mean'`` / ``'median'`` → override threshold
          - an ``int`` → treated as ``n_features``
        """
        columns = columns or []
        target = None
        min_importance = 0.01

        # Parse columns list (highest priority)
        for item in columns:
            if isinstance(item, str):
                if item.startswith("target="):
                    target = item.split("=", 1)[1]
                elif item.startswith("top="):
                    try:
                        n_features = int(item.split("=", 1)[1])
                    except Exception:
                        pass
                elif item in ("median", "mean"):
                    threshold = item
                elif "=" not in item and target is None:
                    target = item
            elif isinstance(item, int):
                n_features = item

        # Metadata fallback
        if isinstance(metadata, dict):
            if target is None:
                target = metadata.get("target_col")
            if n_features is None:
                n_features = metadata.get("n_features")
            if threshold is None:
                threshold = metadata.get("threshold")
            min_importance = metadata.get("min_importance", min_importance)

        if threshold is None:
            threshold = "mean"

        if target is None:
            raise ValueError("FeatureSelection requires a target column. Provide it via columns (e.g. 'target=Label' or bare name) or metadata['target_col'].")

        return self.select_features(df, target_col=target, threshold=threshold, n_features=n_features, min_importance=min_importance)
    
if __name__ == "__main__":
    df = pd.read_csv("Input/test_pipeline_data.csv")
    target = "target"
    engine = FeatureSelectionService()
    selected,res = engine.run(df, columns=[f"target={target}"])
    print(selected)
    print(res)