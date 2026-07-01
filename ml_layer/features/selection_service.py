import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Optional, List, Tuple


class FeatureSelectionService:
    """Reusable feature selection engine that supports threshold or top-N selection.

    Methods are headless and safe to use from the preprocessing pipeline.
    """
    def __init__(self, estimator=None, random_state: int = 42):
        self.random_state = random_state
        self.estimator = estimator

    def _filter_useless_columns(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove constant and high-cardinality columns before feature selection.
        
        Returns (cleaned_X, dropped_column_names_with_reasons).
        """
        before = set(X.columns)
        nunique = X.nunique()
        n = len(X)
        dropped = []

        constant = nunique[nunique <= 1].index.tolist()
        for col in constant:
            dropped.append(f"{col} (constant)")
        X = X.drop(columns=constant, errors='ignore')

        for col in X.columns:
            if X[col].dtype in ("int64", "float64") and X[col].nunique() > 0.9 * n and X[col].nunique() >= 100:
                X = X.drop(columns=[col])
                dropped.append(f"{col} (high-cardinality numeric)")
                continue
            if X[col].dtype == "object" and X[col].nunique() > 0.5 * n:
                X = X.drop(columns=[col])
                dropped.append(f"{col} (high-cardinality object)")

        if dropped:
            print(f"[FeatureSelection] Dropped columns:")
            for d in dropped:
                print(f"  - {d}")
        return X, dropped

    def _is_classification_target(self, y: pd.Series) -> bool:
        """Detect if the target is classification (discrete) or regression (continuous)."""
        if y.dtype in ("int64", "int32", "object", "category", "bool"):
            return True
        # Float target: classification if few unique values, regression if many
        nunique = y.nunique()
        return nunique <= 0.1 * len(y) or nunique <= 20

    def _get_estimator(self, y: pd.Series):
        """Return a classifier or regressor depending on the target type."""
        if self.estimator is not None:
            return self.estimator
        if self._is_classification_target(y):
            return RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        return RandomForestRegressor(n_estimators=100, random_state=self.random_state)

    def _auto_n_features(self, n_total: int) -> int:
        """Automatically pick a reasonable number of features to keep."""
        return max(3, min(10, int(n_total * 0.5)))

    def select_features(self, df: pd.DataFrame, target_col: str, n_features: Optional[int] = None, min_importance: float = 0.01) -> Tuple[List[str], List[str], List[str], pd.DataFrame]:
        """Return (selected, dropped_with_reasons, excluded_by_importance, pruned_dataframe_with_target).

        - df: full DataFrame containing target_col
        - target_col: the name of the target/label column (required)
        - n_features: if provided, select top-n by feature importance; otherwise auto-determined
        - min_importance: minimum absolute importance a feature must have to be kept (default 0.01)
        """
        print(f"[FeatureSelection] Starting feature selection with target='{target_col}', n_features={n_features}, min_importance={min_importance}")
        df = df.reset_index(drop=True)

        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not found in DataFrame columns")

        X = df.drop(columns=[target_col])
        X, dropped = self._filter_useless_columns(X)

        # Detect columns that were mostly null before imputation (mode dominates)
        for col in X.select_dtypes(include="object").columns:
            top_freq = X[col].value_counts(normalize=True).iloc[0] if not X[col].value_counts().empty else 0
            if top_freq > 0.70:
                X = X.drop(columns=[col])
                dropped.append(f"{col} (mode-dominated, {top_freq:.0%} same value — likely too many missing values)")

        X_encoded = pd.get_dummies(X, drop_first=False)
        y = df[target_col]

        k = n_features if n_features is not None else self._auto_n_features(len(X_encoded.columns))
        print(f"[FeatureSelection] Selecting top-{k} of {len(X_encoded.columns)} encoded features")

        estimator = self._get_estimator(y)
        print(f"[FeatureSelection] Using {type(estimator).__name__} (target nunique={y.nunique()}, dtype={y.dtype})")
        estimator.fit(X_encoded, y)
        importances = pd.Series(estimator.feature_importances_, index=X_encoded.columns).sort_values(ascending=False)
        selected = list(importances.index[:k])

        if min_importance > 0.0 and selected:
            selected = [c for c in selected if importances.get(c, 0) >= min_importance]

        selected = [c for c in selected if c in X_encoded.columns]

        # Map encoded column names back to original DataFrame column names
        orig_cols = set(X.columns)
        def _to_original(encoded_name: str) -> str:
            for o in orig_cols:
                if encoded_name == o:
                    return o
                if encoded_name.startswith(o + '_'):
                    return o
            return encoded_name

        original_selected = sorted(set(_to_original(c) for c in selected))

        # Track encoded columns excluded by importance
        all_encoded = set(X_encoded.columns)
        selected_set = set(selected)
        excluded_set = all_encoded - selected_set
        excluded = sorted(
            f"{_to_original(c)} ({c}, importance: {importances.get(c, 0):.4f})"
            for c in excluded_set
        )

        print(f"[FeatureSelection] Selected {len(original_selected)} features: {original_selected}")
        if dropped:
            print(f"[FeatureSelection] Dropped features:")
            for d in dropped:
                print(f"  - {d}")
        if excluded:
            print(f"[FeatureSelection] Excluded by importance:")
            for e in excluded:
                print(f"  - {e}")
        result_df = pd.concat(
            [
                df[original_selected + [target_col]].reset_index(drop=True),
            ],
            axis=1,
        )
        return original_selected, dropped, excluded, result_df

    def run(self, df: pd.DataFrame, columns: Optional[List] = None, n_features: Optional[int] = None, metadata: Optional[dict] = None) -> Tuple[List[str], List[str], List[str], pd.DataFrame]:
        """Run feature selection with parameters resolved from the pipeline.

        Target column is required — it is extracted from (in priority order):
          1. ``columns`` list: ``'target=Name'`` or a bare string column name
          2. ``metadata['target_col']``

        Additional parameters from ``columns``:
          - ``'top=N'`` → select top-N features
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
            min_importance = metadata.get("min_importance", min_importance)

        if target is None:
            raise ValueError("FeatureSelection requires a target column. Provide it via columns (e.g. 'target=Label' or bare name) or metadata['target_col'].")

        selected, dropped, excluded, result_df = self.select_features(df, target_col=target, n_features=n_features, min_importance=min_importance)
        return selected, dropped, excluded, result_df
    
if __name__ == "__main__":
    df = pd.read_csv("Input/test_pipeline_data.csv")
    target = "target"
    engine = FeatureSelectionService()
    selected, dropped, excluded, res = engine.run(df, columns=[f"target={target}"])
    print(f"Selected: {selected}")
    print(f"Dropped: {dropped}")
    print(f"Excluded: {excluded}")
    print(res)