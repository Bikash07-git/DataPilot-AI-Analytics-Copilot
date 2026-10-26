import math
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from backend.config import MAX_PREVIEW_ROWS, MAX_CATEGORICAL_UNIQUE
from backend.models.schemas import ColumnProfile, DatasetOverview, ProfileResponse
from backend.services.file_service import FileValidationError, detect_encoding, detect_delimiter

def format_bytes(size: int) -> str:
    """Format bytes into human-readable string (KB, MB, GB)."""
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.2f} MB"
    else:
        return f"{size / (1024 * 1024 * 1024):.2f} GB"

def sanitize_value(val: Any) -> Any:
    """Convert pandas/numpy types to JSON-serializable Python native types, mapping NaN/Inf to None."""
    if val is None:
        return None
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (pd.Timestamp, datetime.datetime, datetime.date)):
        return val.isoformat()
    if pd.isna(val):
        return None
    return str(val)

def infer_column_type(series: pd.Series, dtype_str: str) -> str:
    """Infer column type (numeric, categorical, datetime, boolean, text)."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
        
    # Check if object column can be parsed as datetime
    non_null = series.dropna()
    if len(non_null) > 0 and len(non_null) <= 500:
        try:
            pd.to_datetime(non_null.iloc[:50], format="mixed")
            # If at least sample succeeded without crashing, test a larger slice
            return "datetime"
        except Exception:
            pass

    # Distinct ratio check
    n_unique = series.nunique()
    if n_unique <= 2 and len(non_null) > 0:
        unique_vals = {str(x).strip().lower() for x in non_null.unique()}
        if unique_vals.issubset({"true", "false", "0", "1", "yes", "no", "y", "n"}):
            return "boolean"

    if n_unique <= MAX_CATEGORICAL_UNIQUE or (len(series) > 0 and (n_unique / len(series)) < 0.15):
        return "categorical"
        
    return "text"

def profile_dataset(file_id: str, file_path: Path, original_filename: str) -> ProfileResponse:
    """
    Reads a CSV dataset from file_path, validates contents,
    computes accurate profiling metrics, and returns a ProfileResponse.
    """
    if not file_path.exists():
        raise FileValidationError("File not found on server.", "FILE_NOT_FOUND")
        
    encoding = detect_encoding(file_path)
    delimiter = detect_delimiter(file_path, encoding)
    
    try:
        df = pd.read_csv(
            file_path,
            encoding=encoding,
            sep=delimiter,
            low_memory=False,
            on_bad_lines="error"  # Strict: fail on malformed row lengths
        )
    except pd.errors.EmptyDataError:
        raise FileValidationError("The CSV file has no columns or data.", "EMPTY_CSV")
    except pd.errors.ParserError as pe:
        raise FileValidationError(f"CSV parsing error: {str(pe)}", "MALFORMED_CSV")
    except Exception as e:
        raise FileValidationError(f"Failed to read CSV: {str(e)}", "READ_ERROR")
        
    total_rows, total_columns = df.shape
    
    if total_rows == 0:
        raise FileValidationError("The CSV file contains a header but no data rows.", "NO_ROWS")
    if total_columns == 0:
        raise FileValidationError("The CSV file contains no columns.", "NO_COLUMNS")
        
    file_size_bytes = file_path.stat().st_size
    memory_usage_bytes = int(df.memory_usage(deep=True).sum())
    
    # Missing cells calculation
    null_matrix = df.isna()
    total_cells = total_rows * total_columns
    total_missing_cells = int(null_matrix.sum().sum())
    missing_cells_percentage = round((total_missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0
    
    # Duplicate rows calculation
    try:
        duplicate_rows_count = int(df.duplicated().sum())
    except Exception:
        # Fallback if any unhashable column types
        duplicate_rows_count = 0
    duplicate_rows_percentage = round((duplicate_rows_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
    
    columns_by_type: Dict[str, int] = {
        "numeric": 0,
        "categorical": 0,
        "datetime": 0,
        "boolean": 0,
        "text": 0
    }
    
    column_profiles: List[ColumnProfile] = []
    
    for col_name in df.columns:
        col_series = df[col_name]
        col_dtype = str(col_series.dtype)
        inferred = infer_column_type(col_series, col_dtype)
        columns_by_type[inferred] = columns_by_type.get(inferred, 0) + 1
        
        col_null_count = int(col_series.isna().sum())
        col_non_null_count = total_rows - col_null_count
        col_null_pct = round((col_null_count / total_rows) * 100, 2)
        
        unique_count = int(col_series.nunique(dropna=True))
        unique_pct = round((unique_count / total_rows) * 100, 2)
        
        # Sample non-null unique values
        raw_samples = col_series.dropna().unique()[:4]
        sample_values = [sanitize_value(v) for v in raw_samples]
        
        is_const = (unique_count <= 1 and col_non_null_count > 0)
        is_uniq = (unique_count == total_rows)
        
        # Numeric stats if applicable
        min_v, max_v, mean_v, med_v, std_v = None, None, None, None, None
        if pd.api.types.is_numeric_dtype(col_series) and col_non_null_count > 0:
            try:
                min_v = sanitize_value(col_series.min())
                max_v = sanitize_value(col_series.max())
                mean_v = sanitize_value(col_series.mean())
                med_v = sanitize_value(col_series.median())
                std_v = sanitize_value(col_series.std())
            except Exception:
                pass
                
        profile = ColumnProfile(
            name=str(col_name),
            dtype=col_dtype,
            inferred_type=inferred,
            total_count=total_rows,
            non_null_count=col_non_null_count,
            null_count=col_null_count,
            null_percentage=col_null_pct,
            unique_count=unique_count,
            unique_percentage=unique_pct,
            sample_values=sample_values,
            is_constant=is_const,
            is_unique=is_uniq,
            min_val=min_v,
            max_val=max_v,
            mean_val=mean_v,
            median_val=med_v,
            std_val=std_v
        )
        column_profiles.append(profile)
        
    # Preview rows
    preview_df = df.head(MAX_PREVIEW_ROWS)
    preview_rows: List[Dict[str, Any]] = []
    for _, row in preview_df.iterrows():
        row_dict = {str(k): sanitize_value(v) for k, v in row.to_dict().items()}
        preview_rows.append(row_dict)
        
    overview = DatasetOverview(
        file_id=file_id,
        file_name=original_filename,
        file_size_bytes=file_size_bytes,
        file_size_formatted=format_bytes(file_size_bytes),
        total_rows=total_rows,
        total_columns=total_columns,
        total_cells=total_cells,
        total_missing_cells=total_missing_cells,
        missing_cells_percentage=missing_cells_percentage,
        duplicate_rows_count=duplicate_rows_count,
        duplicate_rows_percentage=duplicate_rows_percentage,
        columns_by_type=columns_by_type,
        memory_usage_bytes=memory_usage_bytes,
        memory_usage_formatted=format_bytes(memory_usage_bytes),
        uploaded_at=datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    
    return ProfileResponse(
        success=True,
        message=f"Successfully analyzed dataset '{original_filename}' ({total_rows} rows, {total_columns} columns)",
        overview=overview,
        columns=column_profiles,
        preview_rows=preview_rows
    )

