import io
from pathlib import Path

import datetime as _datetime

import pandas as pd


def save_dataframe_to_bytes(df: pd.DataFrame, filename: str = "data") -> tuple[bytes, str, str]:
    """
    Save a DataFrame to CSV or Excel bytes based on the filename extension.
    
    Returns:
        (bytes, content_type, output_filename)
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename


def read_excel_clean(file_source: str | Path | bytes | io.BytesIO, engine: str | None = None) -> pd.DataFrame:
    """Read an Excel file, auto-detect and remove unnecessary top rows and empty
    left columns, and return a clean DataFrame.

    Detection logic:
      1. Read raw (no header) to see all rows/columns.
      2. Drop columns where every cell is NaN.
      3. Drop rows where every cell is NaN.
      4. Find the first remaining row with >= 2 non-NaN values — treat it as the
         column header.  Rows before it (titles, blanks) are discarded.
      5. Assign header names, drop the header row from data, and strip
         whitespace from string columns / header names.

    Parameters
    ----------
    file_source : str | Path | bytes | io.BytesIO
        Path to an ``.xlsx`` / ``.xls`` file, or the raw bytes, or a
        BytesIO stream.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for further processing or CSV export.
    """
    # If raw bytes are passed, wrap them in a BytesIO stream for pandas to read
    if isinstance(file_source, bytes):
        file_source = io.BytesIO(file_source)

    # Read the Excel file with no header so we see every row as data
    read_kwargs = {"header": None}
    if engine:
        read_kwargs["engine"] = engine
    raw: pd.DataFrame = pd.read_excel(file_source, **read_kwargs)

    # Drop columns that are completely empty, then rows that are completely empty, and reset index
    raw = raw.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)

    # If nothing remains, return an empty DataFrame
    if raw.empty:
        return pd.DataFrame()

    # Determine the minimum number of non-null values needed to qualify as a header row
    ncols = raw.shape[1]
    min_header_cols = min(2, ncols)

    # Scan rows to find the first one with at least `min_header_cols` non-null values
    header_row_idx: int | None = None
    for idx in range(len(raw)):
        non_null_count = raw.iloc[idx].notna().sum()
        if non_null_count >= min_header_cols:
            header_row_idx = idx
            break

    # If no suitable header row was found, return the raw DataFrame as it is
    if header_row_idx is None:
        return raw

    # Extract the header row values and convert them to cleaned strings
    headers = raw.iloc[header_row_idx].tolist()
    headers = [
        str(h).strip() if pd.notna(h) else f"Unnamed: {i}"
        for i, h in enumerate(headers)
    ]

    # rename duplicate header names
    seen: set[str] = set()
    unique_headers: list[str] = []
    for h in headers:
        candidate = h
        suffix = 1
        while candidate in seen:
            candidate = f"{h}_{suffix}"
            suffix += 1
        seen.add(candidate)
        unique_headers.append(candidate)
    headers = unique_headers

    # Slice the data below the header row and assign the cleaned column names
    data = raw.iloc[header_row_idx + 1 :].reset_index(drop=True)
    data.columns = headers
    data.columns = data.columns.astype(str).str.strip()

    # Strip whitespace from string-type columns
    for col in data.columns:
        dtype = data[col].dtype
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            str_mask = data[col].apply(lambda x: isinstance(x, str))
            if str_mask.any():
                data.loc[str_mask, col] = data.loc[str_mask, col].str.strip()

    # Attempt type coercion: try numeric first, then datetime, for object/string columns
    for col in data.columns:
        dt = data[col].dtype
        # Skip columns that are already non-string types
        if not (pd.api.types.is_object_dtype(dt) or pd.api.types.is_string_dtype(dt)):
            continue

        # Attempt to convert the entire column to numeric (coerce failures to NaN)
        converted = pd.to_numeric(data[col], errors="coerce")
        if converted.notna().all():
            # All values were successfully converted — keep the numeric column
            data[col] = converted
            continue

        # Check the first 100 non-null values for any datetime objects
        has_datetime = any(
            isinstance(x, _datetime.datetime)
            for x in data[col].iloc[:100]
            if pd.notna(x)
        )
        if has_datetime:
            data[col] = pd.to_datetime(data[col])

    # Detect Excel serial date numbers in numeric columns with date-related names
    for col in data.columns:
        # Skip columns already parsed as datetime
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            continue
        # Only process numeric columns
        if not pd.api.types.is_numeric_dtype(data[col]):
            continue

        # Check if the column name suggests it contains dates
        col_lower = col.lower()
        is_date_col = any(kw in col_lower for kw in ["date", "start", "end", "time"])
        if not is_date_col:
            continue

        # Excel serial date numbers for reasonable dates fall in ~40000-60000 range
        if data[col].between(40000, 60000).all():
            try:
                # Convert Excel serial numbers to datetime (epoch = 1899-12-30)
                data[col] = pd.to_datetime(data[col], origin="1899-12-30", unit="D")
            except Exception:
                pass

    return data