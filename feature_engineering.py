import dspy
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import re

class SuggestFeatures(dspy.Signature):
    """Suggest meaningful new features for a dataset.
    Outputs a set of suggested new column definitions, one per line in the format:
    name: description | code: pandas expression
    """
    dataset_columns = dspy.InputField(desc="Available column names (comma-separated)", default="")
    sample_rows = dspy.InputField(desc="Sample rows as JSON (first N rows)", default="")
    top_n = dspy.InputField(desc="Number of suggestions to return", default="5")
    suggested_features = dspy.OutputField(desc="Suggested features, one per line (name: description | code: ...)")

def fix_column_references(code: str, df_columns: list) -> str:
    """Fix bare column names in code to use df['column'] syntax.
    
    Args:
        code: The code expression (e.g., "Weight / Height")
        df_columns: List of actual dataframe column names
        
    Returns:
        Fixed code with proper df['column'] syntax
    """
    fixed_code = code
    
    # Sort columns by length (descending) to replace longer names first
    # This prevents replacing "Age" when we have both "Age" and "AgeGroup"
    sorted_columns = sorted(df_columns, key=len, reverse=True)
    
    for col in sorted_columns:
        # Use word boundaries to match whole column names only
        # Match column name not preceded by df[' or followed by ']
        pattern = r'(?<!df\[\')(?<!df\[")(?<!\w)' + re.escape(col) + r'(?!\w)(?!\'\])(?!\"\])'
        replacement = f"df['{col}']"
        fixed_code = re.sub(pattern, replacement, fixed_code)
    
    return fixed_code

def apply_feature_engineering_agent(DataFrame, suggested_features: str):
    """Apply feature engineering to a dataframe based on suggested features string.
    
    Args:
        DataFrame: pandas DataFrame to add features to
        suggested_features: String with features in format "name: description | code: expression"
    
    Returns:
        DataFrame with new features added
    """
    df = DataFrame.copy()
    df_columns = df.columns.tolist()
    features_added = 0
    
    # Split by newlines and process each feature
    lines = suggested_features.strip().split("\n")
    
    for line in lines:
        if not line.strip():
            continue
            
        try:
            # Check if line has the proper format
            if "| code:" not in line:
                print(f"Skipping invalid format: {line}")
                continue
                
            # Split into name/description and code
            name_desc, code_part = line.split("| code:", 1)
            
            if ":" not in name_desc:
                print(f"Skipping line without name: {line}")
                continue
                
            name, _ = name_desc.split(":", 1)
            
            # Clean the name (remove numbers and extra characters)
            name = name.strip()
            # Remove leading numbers like "1. ", "2. ", etc.
            name = re.sub(r'^\d+\.\s*', '', name)
            
            code = code_part.strip()
            
            print(f"\n{'='*50}")
            print(f"Processing feature: {name}")
            print(f"Original code: {code}")
            
            # Fix column references in the code
            fixed_code = fix_column_references(code, df_columns)
            print(f"Fixed code: {fixed_code}")
            
            # Evaluate the code in the context of the dataframe
            # Include common libraries that might be used
            eval_context = {
                "df": df, 
                "pd": pd, 
                "np": np,
                "numpy": np
            }
            
            df[name] = eval(fixed_code, eval_context)
            features_added += 1
            print(f"✓ Successfully added feature: {name}")
            
        except Exception as e:
            print(f"✗ Error processing line '{line}'")
            print(f"  Error: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            continue
    
    print(f"\n{'='*50}")
    print(f"Total features added: {features_added}")
    return df

def engineer_features(df: Any, suggested_features: str) -> Any:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("First argument must be a pandas DataFrame")
    
    if not suggested_features or not suggested_features.strip():
        print("No features to apply")
        return df
    
    print(f"{'='*50}")
    print(f"Starting feature engineering")
    print(f"Dataframe shape: {df.shape}")
    print(f"Dataframe columns: {list(df.columns)}")
    print(f"{'='*50}")
    print(f"Features to apply:")
    print(suggested_features)
    print(f"{'='*50}")
    
    return apply_feature_engineering_agent(df, suggested_features)