import numpy as np
import pandas as pd
from typing import Optional, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# ===============================
# Feature Selection Service (YOURS)
# ===============================
class FeatureSelectionService:
    def __init__(self, estimator: Optional[RandomForestClassifier] = None, random_state: int = 42):
        self.random_state = random_state
        self.estimator = estimator or RandomForestClassifier(
            n_estimators=100, random_state=random_state
        )

    def select_features(self, df: pd.DataFrame, target_col: str,
                        threshold: str = "median",
                        n_features: Optional[int] = None) -> Tuple[List[str], pd.DataFrame]:

        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not found")

        X = df.drop(columns=[target_col])
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

        result_df = pd.concat(
            [X_encoded[selected], df[[target_col]].reset_index(drop=True)], axis=1
        )

        return selected, result_df

    def run(self, df: pd.DataFrame, columns: Optional[List] = None,
            threshold: Optional[str] = None,
            n_features: Optional[int] = None,
            metadata: Optional[dict] = None):

        columns = columns or []
        target = None

        if threshold is None:
            threshold = "median"

        for item in columns:
            if isinstance(item, str) and item.startswith("target="):
                target = item.split("=", 1)[1]

        if target is None:
            raise ValueError("Target column required")

        return self.select_features(df, target, threshold, n_features)


# ===============================
# Evaluation Function
# ===============================
def evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')

    return acc, f1


# ===============================
# L1 Selection
# ===============================
def l1_selection(X, y):
    selector = SelectFromModel(
        LogisticRegression(C=1.0, l1_ratio=1, solver='liblinear', max_iter=1000)
    )
    selector.fit(X, y)
    selected = X.columns[selector.get_support()]
    return list(selected)

# ===============================
# Mutual Information Selection
# ===============================
def mi_selection(X, y, top_n=10):
    mi = mutual_info_classif(X, y)
    scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    return list(scores.head(top_n).index)


import shap

# ===============================
# SHAP Selection
# ===============================
def shap_selection(X, y, top_n=5):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle different shap output shapes
    if isinstance(shap_values, list):
        # Old format: list of arrays per class
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        # New format: 3D array (samples, features, classes)
        mean_shap = np.abs(shap_values).mean(axis=(0, 2))
    else:
        # Binary: 2D array (samples, features)
        mean_shap = np.abs(shap_values).mean(axis=0)

    scores = pd.Series(mean_shap, index=X.columns).sort_values(ascending=False)
    selected = list(scores.head(top_n).index)
    print(f"\nSHAP top {top_n} features: {selected}")
    return selected

# Load dataset
import pandas as pd

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    
	# Load and clean
	url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
	df = pd.read_csv(url)
	df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
	df = df.dropna()  # clean FIRST

	# THEN define X and y
	target = "Survived"
	X_full = pd.get_dummies(df.drop(columns=[target]))
	y = df[target].reset_index(drop=True)  # reset index after dropna
	X_full = X_full.reset_index(drop=True)

	print("Clean shape:", df.shape)
	print("X shape:", X_full.shape)
	print("y shape:", y.shape)

    # ===============================
    # 1. Baseline
    # ===============================
	acc_base, f1_base = evaluate_model(X_full, y)

    # ===============================
    # 2. RF (Your Method)
    # ===============================
    
	engine = FeatureSelectionService()
	df_clean = X_full.copy()
	df_clean[target] = y.values

	selected_rf, df_rf = engine.run(df_clean, columns=[f"target={target}"])

	X_rf = df_rf.drop(columns=[target]).reset_index(drop=True)
	acc_rf, f1_rf = evaluate_model(X_rf, y)
    # ===============================
    # 3. L1 Selection
    # ===============================
	selected_l1 = l1_selection(X_full, y)
	X_l1 = X_full[selected_l1]

	acc_l1, f1_l1 = evaluate_model(X_l1, y)

    # ===============================
    # 4. Mutual Information
    # ===============================
	selected_mi = mi_selection(X_full, y, top_n=10)
	X_mi = X_full[selected_mi]

	acc_mi, f1_mi = evaluate_model(X_mi, y)


	# ===============================
	# 5. SHAP Selection
	# ===============================
	selected_shap = shap_selection(X_full, y, top_n=5)
	X_shap = X_full[selected_shap]
	acc_shap, f1_shap = evaluate_model(X_shap, y)

    # ===============================
    # Results
    # ===============================
	print("\n=== Feature Selection Comparison ===")
	print(f"Baseline       -> Acc: {acc_base:.3f}, F1: {f1_base:.3f}, Features: {X_full.shape[1]}")
	print(f"RF (Yours)     -> Acc: {acc_rf:.3f}, F1: {f1_rf:.3f}, Features: {X_rf.shape[1]}")
	print(f"L1 Selection   -> Acc: {acc_l1:.3f}, F1: {f1_l1:.3f}, Features: {X_l1.shape[1]}")
	print(f"Mutual Info    -> Acc: {acc_mi:.3f}, F1: {f1_mi:.3f}, Features: {X_mi.shape[1]}")
	print(f"SHAP           -> Acc: {acc_shap:.3f}, F1: {f1_shap:.3f}, Features: {X_shap.shape[1]}")
