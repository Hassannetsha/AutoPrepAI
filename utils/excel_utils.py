import io
from pathlib import Path

import pandas as pd


def read_excel_clean(file_source: str | Path | bytes | io.BytesIO) -> pd.DataFrame:
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
    if isinstance(file_source, bytes):
        file_source = io.BytesIO(file_source)

    raw: pd.DataFrame = pd.read_excel(file_source, header=None)
    raw = raw.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)

    if raw.empty:
        return pd.DataFrame()

    ncols = raw.shape[1]
    min_header_cols = min(2, ncols)

    header_row_idx: int | None = None
    for idx in range(len(raw)):
        non_null_count = raw.iloc[idx].notna().sum()
        if non_null_count >= min_header_cols:
            header_row_idx = idx
            break

    if header_row_idx is None:
        return raw

    headers = raw.iloc[header_row_idx].tolist()
    headers = [
        str(h).strip() if pd.notna(h) else f"Unnamed: {i}"
        for i, h in enumerate(headers)
    ]
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

    data = raw.iloc[header_row_idx + 1 :].reset_index(drop=True)
    data.columns = headers
    data.columns = data.columns.astype(str).str.strip()

    for col in data.columns:
        dtype = data[col].dtype
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            str_mask = data[col].apply(lambda x: isinstance(x, str))
            if str_mask.any():
                data.loc[str_mask, col] = data.loc[str_mask, col].str.strip()

    return data
