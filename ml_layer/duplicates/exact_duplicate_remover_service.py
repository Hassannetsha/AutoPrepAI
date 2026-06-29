import pandas as pd
from typing import Optional, List

class ExactDuplicateRemoverService:
    def __init__(
        self,
        subset: Optional[List[str]] = None,
        keep: str = 'first',
        auto_exclude_ids: bool = True
    ):
        self.subset = subset
        self.keep = keep
        self.auto_exclude_ids = auto_exclude_ids
    
    @staticmethod
    def detect_id_columns(
        df: pd.DataFrame,
        uniqueness_threshold: float = 0.95,
        verbose: bool = False
    ) -> List[str]:
        potential_ids = []
        n_rows = len(df)
        if n_rows == 0: return []

        for col in df.columns:
            series = df[col]
            if series.isnull().all(): continue

            col_lower = str(col).lower()
            id_keywords = {'id', 'uuid', 'guid', 'pk', 'key'}
            name_parts = set(col_lower.replace('_', ' ').replace('.', ' ').split())
            has_id_name = not id_keywords.isdisjoint(name_parts) or col_lower.endswith('_id')

            nunique = series.nunique()
            uniqueness_ratio = nunique / n_rows
            is_integer = pd.api.types.is_integer_dtype(series)
            is_object = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)

            if is_object and uniqueness_ratio > uniqueness_threshold:
                potential_ids.append(col)
                continue

            if is_integer:
                if uniqueness_ratio > uniqueness_threshold and has_id_name:
                    potential_ids.append(col)
                    continue
                if nunique == n_rows and (series.is_monotonic_increasing or series.is_monotonic_decreasing):
                    potential_ids.append(col)
                    continue
            
            if pd.api.types.is_float_dtype(series):
                if has_id_name and uniqueness_ratio > uniqueness_threshold:
                    potential_ids.append(col)
                continue

        return potential_ids
    
    def remove_duplicates(
        self, 
        df: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None,
        verbose: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Removes duplicates and provides a detailed report of the findings.
        """
        df = df.reset_index(drop=True)

        if df.empty:
            if verbose: print("DataFrame is empty. No duplicates to remove.")
            return df.copy(), pd.DataFrame()

        # 1. ID Identification & Integrity Check
        detected_ids = self.detect_id_columns(df, verbose=False) if self.auto_exclude_ids else []
        id_cols_in_df = [c for c in detected_ids if c in df.columns]
        
        if verbose and id_cols_in_df:
            primary_id = id_cols_in_df[0]
            dupe_ids_count = df.duplicated(subset=[primary_id]).sum()
            if dupe_ids_count > 0:
                print(f"ID Integrity Check: Found {dupe_ids_count} rows with shared ID '{primary_id}'.")
            else:
                print(f"ID Integrity Check: Column '{primary_id}' contains 100% unique keys.")

        # 2. Determine Columns to Check
        columns_to_exclude = set(exclude_columns or [])
        columns_to_exclude.update(detected_ids)
        
        if self.subset is not None:
            cols_to_check = [col for col in self.subset if col not in columns_to_exclude]
        else:
            cols_to_check = [col for col in df.columns if col not in columns_to_exclude]
        
        if not cols_to_check:
            if verbose: 
                print("Note: No columns left to check after ID exclusion. Treating all rows as unique.")
            return df.copy(), pd.DataFrame()

        # 3. Duplicate Detection
        duplicate_mask = df.duplicated(subset=cols_to_check, keep=self.keep)
        df_dedup = df[~duplicate_mask].reset_index(drop=True)
        df_duplicates = df[duplicate_mask].reset_index(drop=True)
        num_duplicates = len(df_duplicates)

        # 4. Final Reporting
        if verbose:
            print(f"Checking for duplicates based on: {cols_to_check}")
            
            if num_duplicates > 0:
                print(f"Result: Found {num_duplicates} duplicate rows.")
                print(f"Data Reduced: {len(df)} rows \u2192 {len(df_dedup)} rows.")
                print(f"Action: Removed duplicates, keeping the '{self.keep}' occurrence.")
                print("Preview of removed data:")
                print(df_duplicates.head(5))
            else:
                print(f"Result: No duplicate rows found.")
                print(f"Success: The dataset is clean. All {len(df)} rows are unique based on the selected criteria.")
        
        return df_dedup, df_duplicates

    def get_duplicate_groups(self, df: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        # Re-uses the same logic for consistency
        detected_ids = self.detect_id_columns(df) if self.auto_exclude_ids else []
        columns_to_exclude = set(exclude_columns or []) | set(detected_ids)
        
        cols_to_check = [col for col in (self.subset or df.columns) if col not in columns_to_exclude]
        if not cols_to_check: return pd.DataFrame()

        df = df.reset_index(drop=True)
        df_with_groups = df.copy()
        df_with_groups['duplicate_group'] = df_with_groups.groupby(cols_to_check, dropna=False).ngroup()
        
        counts = df_with_groups['duplicate_group'].value_counts()
        dupe_groups = counts[counts > 1].index
        return df_with_groups[df_with_groups['duplicate_group'].isin(dupe_groups)].sort_values('duplicate_group')
