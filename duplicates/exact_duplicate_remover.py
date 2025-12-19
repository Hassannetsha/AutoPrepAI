import pandas as pd
from typing import Optional, List


class ExactDuplicateRemover:
    def __init__(
        self,
        subset: Optional[List[str]] = None,
        keep: str = 'first',
        auto_exclude_ids: bool = True
    ):
        """
        Initialize the exact duplicate remover.
        
        Args:
            subset: List of column names to consider for duplicate detection.
                   If None, all columns are used.
            keep: Which duplicate to keep ('first', 'last', or False to drop all).
            auto_exclude_ids: Whether to automatically detect and exclude ID columns.
        """
        self.subset = subset
        self.keep = keep
        self.auto_exclude_ids = auto_exclude_ids
    
    @staticmethod
    def detect_id_columns(df: pd.DataFrame) -> List[str]:
        """
        Detect columns that are likely IDs or unique identifiers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of column names that appear to be ID columns
        """
        potential_ids = []
        
        for col in df.columns:
            # Check if column name suggests it's an ID
            col_lower = str(col).lower()
            id_keywords = ['id', 'key', 'index', 'uuid', 'guid', '_id', 'pk']
            if any(keyword in col_lower for keyword in id_keywords):
                potential_ids.append(col)
                continue
            
            # Check if column has all unique values (and sufficient data)
            if len(df) > 1 and df[col].nunique() == len(df):
                potential_ids.append(col)
                continue
            
            # Check if it's a sequential integer index
            if df[col].dtype in ['int64', 'int32', 'Int64', 'Int32']:
                try:
                    if (df[col].is_monotonic_increasing or 
                        df[col].is_monotonic_decreasing):
                        potential_ids.append(col)
                except (TypeError, AttributeError):
                    # Handle nullable integer types that might not support is_monotonic
                    pass
        
        return potential_ids
    
    def remove_duplicates(
        self, 
        df: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None,
        verbose: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove exact duplicates from a DataFrame.
        
        Args:
            df: Input DataFrame
            exclude_columns: Additional columns to exclude from duplicate detection
            verbose: Whether to print detailed information
            
        Returns:
            Tuple of (deduplicated DataFrame, DataFrame of removed duplicates)
        """
        if df.empty:
            if verbose:
                print("DataFrame is empty. No duplicates to remove.")
            return df.copy(), pd.DataFrame()
        
        # Determine which columns to exclude
        columns_to_exclude = set(exclude_columns or [])
        
        if self.auto_exclude_ids:
            detected_ids = self.detect_id_columns(df)
            columns_to_exclude.update(detected_ids)
            if verbose and detected_ids:
                print(f"Auto-detected ID columns: {detected_ids}")
        
        # Determine subset of columns to check
        if self.subset is not None:
            # User provided specific columns
            cols_to_check = [col for col in self.subset if col not in columns_to_exclude]
        elif columns_to_exclude:
            # Exclude detected/specified ID columns
            cols_to_check = [col for col in df.columns if col not in columns_to_exclude]
        else:
            # Use all columns
            cols_to_check = None
        
        if verbose:
            if columns_to_exclude:
                print(f"Excluding columns from duplicate check: {sorted(columns_to_exclude)}")
            if cols_to_check:
                print(f"Checking for duplicates based on columns: {cols_to_check}")
        
        # Find duplicates
        if cols_to_check is not None and len(cols_to_check) == 0:
            if verbose:
                print("Warning: No columns left to check after exclusions.")
            return df.copy(), pd.DataFrame()
        
        duplicate_mask = df.duplicated(subset=cols_to_check, keep=self.keep)
        
        # Separate duplicates from unique rows
        df_dedup = df[~duplicate_mask].reset_index(drop=True)
        df_duplicates = df[duplicate_mask].reset_index(drop=True)
        
        if verbose:
            num_duplicates = duplicate_mask.sum()
            print(f"\nTotal exact duplicates found: {num_duplicates}")
            print(f"Original length: {len(df)} → Remaining: {len(df_dedup)}")
            if num_duplicates > 0:
                print(f"Kept: '{self.keep}' occurrence(s)")
        
        return df_dedup, df_duplicates
    
    def get_duplicate_groups(
        self,
        df: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get all duplicate groups with a group ID for analysis.
        
        Args:
            df: Input DataFrame
            exclude_columns: Columns to exclude from duplicate detection
            
        Returns:
            DataFrame with an additional 'duplicate_group' column
        """
        columns_to_exclude = set(exclude_columns or [])
        
        if self.auto_exclude_ids:
            detected_ids = self.detect_id_columns(df)
            columns_to_exclude.update(detected_ids)
        
        if self.subset is not None:
            cols_to_check = [col for col in self.subset if col not in columns_to_exclude]
        elif columns_to_exclude:
            cols_to_check = [col for col in df.columns if col not in columns_to_exclude]
        else:
            cols_to_check = None
        
        # Create a copy and add group IDs
        df_with_groups = df.copy()
        
        if cols_to_check and len(cols_to_check) > 0:
            df_with_groups['duplicate_group'] = df_with_groups.groupby(
                cols_to_check, 
                dropna=False
            ).ngroup()
            
            # Filter to only rows that have duplicates
            group_counts = df_with_groups['duplicate_group'].value_counts()
            duplicate_groups = group_counts[group_counts > 1].index
            df_with_groups = df_with_groups[
                df_with_groups['duplicate_group'].isin(duplicate_groups)
            ].sort_values('duplicate_group').reset_index(drop=True)
        else:
            df_with_groups['duplicate_group'] = -1
            df_with_groups = df_with_groups.iloc[:0]  # Return empty with structure
        
        return df_with_groups