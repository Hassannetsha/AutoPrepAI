import dspy
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import re

class SuggestFeatures(dspy.Signature):
    """
    Suggest meaningful new features for a dataset.
    
    CRITICAL — focus on user_requested_columns if provided:
    Generate features USING those columns (transformations, combinations, ratios, etc.).
    Never ignore user_requested_columns — every feature must involve at least one of them.

    IMPORTANT:
    The target column is provided and should be considered when generating features.
    Create features that are useful and related to the target column.

    The target column can be used in feature generation.
    Analyze relationships between columns and the target to create meaningful features.

    CRITICAL: 
    Only reference column names that appear EXACTLY in dataset_columns (case-sensitive).
    NEVER invent or assume columns like 'income', 'salary', 'debt', 'balance' unless they are explicitly listed in dataset_columns.
    If a column doesn't exist, do NOT use it — find another approach with available columns.
    Example: if 'Credit amount' exists but 'income' does not, do NOT use df['income'].
    
    IMPORTANT: 
    Never suggest scalar/dataset-level statistics (like correlations or global means).
    
    Only suggest row-level features that produce a unique value per row.
    
    Outputs one feature per line: name: description | code: pandas expression
    
    IMPORTANT: 
    Never use pd.get_dummies() — it returns multiple columns which cannot be assigned as a single feature.
    Use df['column'].map() or label encoding instead for categorical columns.
    """

    dataset_columns = dspy.InputField(desc="Available column names (comma-separated)", default="")
    column_dtypes = dspy.InputField(
        desc="Column names and their data types as JSON dict, e.g. {\"Age\": \"int64\", \"Sex\": \"object\"}",
        default=""
    )
    user_requested_columns = dspy.InputField(
        desc="Columns the user specifically asked to engineer features from. Empty if none specified.",
        default=""
    )
    target_column = dspy.InputField(
        desc="Target column that should guide feature generation",
        default=""
    )
    sample_rows = dspy.InputField(desc="Sample rows as JSON (first N rows)", default="")
    top_n = dspy.InputField(desc="Number of suggestions to return", default="5")
    suggested_features = dspy.OutputField(desc="Suggested features, one per line (name: description | code: ...)")

def _fix_nan_in_feature(df: pd.DataFrame, col_name: str, code: str):
    """Fix NaN in a newly created feature column.
    For pd.cut-based features, extend bins to cover the data range.
    Falls back to filling with the most frequent value.
    """
    if col_name not in df.columns or not df[col_name].isna().any():
        return
    if 'pd.cut' in code:
        m = re.search(r"bins=\[([^\]]+)\]\s*,\s*labels=\[([^\]]+)\]", code)
        if m:
            try:
                raw_bins = [float(x.strip()) for x in m.group(1).split(',')]
                raw_labels = [x.strip().strip("'\"") for x in m.group(2).split(',')]
            except ValueError:
                raw_bins = raw_labels = None
            if raw_bins and len(raw_bins) >= 2 and len(raw_labels) == len(raw_bins) - 1:
                cut_m = re.search(r"pd\.cut\(\s*([^,]+)\s*,", code)
                if cut_m:
                    series_code = cut_m.group(1).strip()
                    try:
                        series = eval(series_code, {"df": df, "pd": pd, "np": np})
                        new_bins = list(raw_bins)
                        new_bins[0] = min(float(series.min()), raw_bins[0])
                        new_bins[-1] = max(float(series.max()) + 1, raw_bins[-1])  # Add 1 to include max
                        df[col_name] = pd.cut(series, bins=new_bins, labels=raw_labels, include_lowest=True)
                        return
                    except Exception:
                        pass
    # Fallback: fill NaN with most frequent value
    mode_val = df[col_name].mode()
    if not mode_val.empty:
        df[col_name] = df[col_name].fillna(mode_val.iloc[0])

class FeatureEngineeringService:
    """Feature engineering utilities encapsulated as a class to use from the pipeline.

    Example:
        fe = FeatureEngineer()
        df_new = fe.engineer(df, suggested_features_str)
    """

    @staticmethod
    def fix_column_references(code: str, df_columns: list) -> str:
        """Fix bare column names in code to use df['column'] syntax.
        Also corrects case mismatches for column names already inside df['...'].
        """
        def is_inside_quotes(text: str, pos: int) -> bool:
            """Check if position is inside a quoted string."""
            single_quote_count = 0
            double_quote_count = 0
            
            for i in range(pos):
                if text[i] == "'" and (i == 0 or text[i-1] != '\\'):
                    single_quote_count += 1
                elif text[i] == '"' and (i == 0 or text[i-1] != '\\'):
                    double_quote_count += 1
            
            # If odd number of quotes before position, we're inside quotes
            return (single_quote_count % 2 == 1) or (double_quote_count % 2 == 1)
        
        fixed_code = code
        sorted_columns = sorted(df_columns, key=len, reverse=True)
        
        # Step 1: Fix bare column names to df['col'] syntax
        for col in sorted_columns:
            pattern = r'\b' + re.escape(col) + r'\b'
            matches = list(re.finditer(pattern, fixed_code))
            
            for match in reversed(matches):
                start_pos = match.start()
                
                if is_inside_quotes(fixed_code, start_pos):
                    continue
                
                if start_pos >= 4:
                    before = fixed_code[start_pos-4:start_pos]
                    if before == "df['" or before == 'df["':
                        continue
                
                fixed_code = (fixed_code[:start_pos] + 
                            f"df['{col}']" + 
                            fixed_code[match.end():])
        
        # Step 2: Fix case of column names already inside df['...'] or df["..."]
        df_refs = list(re.finditer(r"df\s*\[\s*['\"]([^'\"]+)['\"]\s*\]", fixed_code))
        for ref_match in reversed(df_refs):
            ref_col = ref_match.group(1)
            for actual_col in sorted_columns:
                if ref_col.lower() == actual_col.lower() and ref_col != actual_col:
                    corrected = f"df['{actual_col}']"
                    fixed_code = (fixed_code[:ref_match.start()] + 
                                corrected + 
                                fixed_code[ref_match.end():])
                    break
        
        return fixed_code
    def apply(self, DataFrame: pd.DataFrame, suggested_features: str) -> tuple[pd.DataFrame, int]:
        """Apply feature engineering described in `suggested_features` to the DataFrame.

        `suggested_features` is expected to be one feature per line in the
        format: `name: description | code: pandas_expression`.

        Returns a tuple (new_dataframe, features_added_count).
        """
        df = DataFrame.copy()
        df_columns = list(df.columns)  # Keep as mutable list
        features_added = 0

        lines = suggested_features.strip().split("\n")
        for line in lines:
            if not line.strip():
                continue
            try:
                if "| code:" not in line:
                    print(f"Skipping invalid format: {line}")
                    continue
                name_desc, code_part = line.split("| code:", 1)
                if ":" not in name_desc:
                    print(f"Skipping line without name: {line}")
                    continue
                name, _ = name_desc.split(":", 1)
                name = name.strip()
                name = re.sub(r'^\d+\.\s*', '', name)
                name = re.sub(r'([A-Z])', r'_\1', name).lstrip('_').lower()  # CamelCase → snake_case
                name = re.sub(r'\s+', '_', name)  # spaces → underscores
                name = re.sub(r'_+', '_', name)
                code = code_part.strip()
                # Remove trailing parenthetical notes and sentence-ending periods
                code = re.sub(r'\s*\((?:Note|Again|Assuming|This)[^)]*\)\s*\.?\s*$', '', code)
                code = code.rstrip('.')

                print(f"\n{'='*50}")
                print(f"Processing feature: {name}")
                print(f"Original code: {code}")

                fixed_code = self.fix_column_references(code, df_columns)
                
                if re.search(r"groupby\(.+\)\[.+\]\.(mean|sum|std|min|max|median)\(\)", fixed_code):
                    fixed_code = re.sub(
                        r"(groupby\(.+?\)\[.+?\])\.(mean|sum|std|min|max|median)\(\)",
                        r"\1.transform('\2')",
                        fixed_code
                    )
                
                print(f"Fixed code: {fixed_code}")

                # Validate that all column references exist in the dataframe
                missing_cols = []
                for ref_match in re.finditer(r"df\s*\[\s*['\"]([^'\"]+)['\"]\s*\]", fixed_code):
                    ref_col = ref_match.group(1)
                    # Skip if this is the target column in an assignment (LHS)
                    after = fixed_code[ref_match.end():].lstrip()
                    if after.startswith('='):
                        continue
                    if ref_col not in df_columns:
                        # Check if there's a case-insensitive match already handled
                        if not any(ref_col.lower() == c.lower() for c in df_columns):
                            missing_cols.append(ref_col)
                if missing_cols:
                    print(f"[SKIP] Feature '{name}' references non-existent columns: {missing_cols}")
                    continue

                eval_context = {"df": df, "pd": pd, "np": np, "numpy": np,
                                "datetime": __import__('datetime'), "math": __import__('math'),
                                "re": __import__('re')}
                
                # Ensure the code is wrapped as an assignment if it isn't already
                assign_match = re.match(r"^\s*df\[(['\"])(.+?)\1\]\s*=", fixed_code)
                is_assignment = bool(assign_match)
                if is_assignment:
                    execution_code = fixed_code
                    name = assign_match.group(2)
                else:
                    execution_code = f"df['{name}'] = {fixed_code}"
                
                try:
                    # Use exec() to handle assignment statements
                    exec(execution_code, eval_context)
                    # Check if a variable with the feature name was created
                    if name in eval_context and name not in ["df", "pd", "np", "numpy"]:
                        result = eval_context[name]
                        df[name] = result

                    # Handle NaN in newly created feature
                    if df[name].isna().any():
                        _fix_nan_in_feature(df, name, fixed_code)
                    # UPDATE: Add new column to tracking list for future features
                    if name not in df_columns:
                        df_columns.append(name)
                    features_added += 1
                    print(f"[OK] Successfully added feature: {name}")
                except Exception as exec_error:
                    # If exec fails, try original fixed code as expression and assign
                    try:
                        result = eval(fixed_code, eval_context)
                        df[name] = result
                        # Handle NaN in newly created feature
                        if df[name].isna().any():
                            _fix_nan_in_feature(df, name, fixed_code)
                        # UPDATE: Add new column to tracking list for future features
                        if name not in df_columns:
                            df_columns.append(name)
                        features_added += 1
                        print(f"[OK] Successfully added feature: {name}")
                    except:
                        raise exec_error
            except Exception as e:
                print(f"[FAIL] Error processing line '{line}'")
                print(f"  Error: {e}")
                import traceback
                print(f"  Traceback: {traceback.format_exc()}")
                continue

        print(f"\n{'='*50}")
        print(f"Total features added: {features_added}")
        return df, features_added

    def engineer(self, df: Any, suggested_features: str) -> tuple[pd.DataFrame, int]:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("First argument must be a pandas DataFrame")
        if not suggested_features or not suggested_features.strip():
            print("No features to apply")
            return df, 0
        print(f"{'='*50}")
        print(f"Starting feature engineering")
        print(f"Dataframe shape: {df.shape}")
        print(f"Dataframe columns: {list(df.columns)}")
        print(f"{'='*50}")
        print(f"Features to apply:")
        print(suggested_features)
        print(f"{'='*50}")
        return self.apply(df, suggested_features)


# Backwards-compatible wrappers
def apply_feature_engineering_agent(DataFrame, suggested_features: str) -> tuple[pd.DataFrame, int]:
    fe = FeatureEngineeringService()
    return fe.apply(DataFrame, suggested_features)


def engineer_features(df: Any, suggested_features: str) -> tuple[pd.DataFrame, int]:
    fe = FeatureEngineeringService()
    return fe.engineer(df, suggested_features)

def review_features(suggested_features: str) -> str:
    """
    Interactively ask the user to accept or reject each suggested feature.
    Returns a filtered string containing only accepted features.
    """
    lines = [l.strip() for l in suggested_features.strip().split("\n") if l.strip()]
    accepted = []

    print("\n" + "="*50)
    print("FEATURE REVIEW — accept or reject each feature")
    print("="*50)

    for i, line in enumerate(lines, 1):
        if "| code:" not in line:
            continue

        name_desc, code_part = line.split("| code:", 1)
        name = name_desc.split(":")[0].strip()
        code = code_part.strip()

        print(f"\n[{i}/{len(lines)}] {name}")
        print(f"  code: {code}")

        while True:
            choice = input("  Accept? [y/n/s to skip all remaining]: ").strip().lower()
            if choice == 'y':
                accepted.append(line)
                print("  ✓ Accepted")
                break
            elif choice == 'n':
                print("  ✗ Rejected")
                break
            elif choice == 's':
                print("  Skipping remaining features...")
                print(f"\n{'='*50}")
                print(f"Accepted {len(accepted)}/{len(lines)} features")
                return "\n".join(accepted)
            else:
                print("  Please enter y, n, or s")

    print(f"\n{'='*50}")
    print(f"Accepted {len(accepted)}/{len(lines)} features")
    return "\n".join(accepted)