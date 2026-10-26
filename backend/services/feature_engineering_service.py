import os
import json
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from backend.models.schemas import (
    TransformationStep, TransformationPreviewResponse,
    SessionHistoryItem, ExecutiveReportResponse
)
from backend.services.file_service import FileValidationError, detect_encoding, detect_delimiter
from backend.services.profiler_service import sanitize_value

# In-memory session transformation tracker: file_id -> list of TransformationStep
SESSION_TRANSFORMATIONS: Dict[str, List[TransformationStep]] = {}
SESSION_HISTORY: Dict[str, List[SessionHistoryItem]] = {}

def preview_transformation(
    file_id: str,
    file_path: Path,
    step: TransformationStep
) -> TransformationPreviewResponse:
    """Preview a single transformation step without mutating the base dataset."""
    if not file_path.exists():
        raise FileValidationError("File not found on server.", "FILE_NOT_FOUND")

    encoding = detect_encoding(file_path)
    delimiter = detect_delimiter(file_path, encoding)
    df = pd.read_csv(file_path, encoding=encoding, sep=delimiter, low_memory=False)

    col = step.column
    if col not in df.columns:
        raise FileValidationError(f"Column '{col}' does not exist.", "COLUMN_NOT_FOUND")

    before_stats = {
        "null_count": int(df[col].isna().sum()),
        "unique_count": int(df[col].nunique()),
        "dtype": str(df[col].dtype),
        "mean": float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) and not df[col].dropna().empty else None
    }

    # Create working copy
    series = df[col].copy()
    python_code = ""

    if step.transformation_type == "impute":
        method = step.parameters.get("method", "median")
        if method == "mean":
            val = series.mean()
            series = series.fillna(val)
            python_code = f"df['{col}'] = df['{col}'].fillna(df['{col}'].mean())"
        elif method == "median":
            val = series.median()
            series = series.fillna(val)
            python_code = f"df['{col}'] = df['{col}'].fillna(df['{col}'].median())"
        elif method == "mode":
            mode_series = series.mode()
            val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            series = series.fillna(val)
            python_code = f"df['{col}'] = df['{col}'].fillna(df['{col}'].mode()[0])"
        elif method == "constant":
            val = step.parameters.get("value", 0)
            series = series.fillna(val)
            python_code = f"df['{col}'] = df['{col}'].fillna({repr(val)})"
        elif method == "ffill":
            series = series.ffill()
            python_code = f"df['{col}'] = df['{col}'].ffill()"

    elif step.transformation_type == "scale":
        method = step.parameters.get("method", "standard")
        if method == "standard":
            mean_v = series.mean()
            std_v = series.std() if series.std() > 0 else 1.0
            series = (series - mean_v) / std_v
            python_code = f"df['{col}'] = (df['{col}'] - df['{col}'].mean()) / df['{col}'].std()"
        elif method == "minmax":
            min_v = series.min()
            max_v = series.max()
            rng = max_v - min_v if (max_v - min_v) > 0 else 1.0
            series = (series - min_v) / rng
            python_code = f"df['{col}'] = (df['{col}'] - df['{col}'].min()) / (df['{col}'].max() - df['{col}'].min())"

    elif step.transformation_type == "log":
        series = np.log1p(np.maximum(0, series.fillna(0)))
        python_code = f"import numpy as np\ndf['{col}'] = np.log1p(np.maximum(0, df['{col}']))"

    elif step.transformation_type == "winsorize":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        series = series.clip(lower=lower, upper=upper)
        python_code = f"q1 = df['{col}'].quantile(0.25)\nq3 = df['{col}'].quantile(0.75)\niqr = q3 - q1\ndf['{col}'] = df['{col}'].clip(lower=q1 - 1.5*iqr, upper=q3 + 1.5*iqr)"

    elif step.transformation_type == "encode":
        # Label Encoding
        labels, uniques = pd.factorize(series)
        series = pd.Series(labels, index=df.index)
        python_code = f"df['{col}'] = pd.factorize(df['{col}'])[0]"

    elif step.transformation_type == "datetime_extract":
        dt_series = pd.to_datetime(series, errors="coerce")
        feature = step.parameters.get("feature", "month")
        if feature == "year":
            series = dt_series.dt.year
            python_code = f"df['{col}_year'] = pd.to_datetime(df['{col}']).dt.year"
        elif feature == "month":
            series = dt_series.dt.month
            python_code = f"df['{col}_month'] = pd.to_datetime(df['{col}']).dt.month"
        elif feature == "day_of_week":
            series = dt_series.dt.dayofweek
            python_code = f"df['{col}_dayofweek'] = pd.to_datetime(df['{col}']).dt.dayofweek"
        elif feature == "is_weekend":
            series = dt_series.dt.dayofweek >= 5
            python_code = f"df['{col}_is_weekend'] = pd.to_datetime(df['{col}']).dt.dayofweek >= 5"

    after_stats = {
        "null_count": int(series.isna().sum()),
        "unique_count": int(series.nunique()),
        "dtype": str(series.dtype),
        "mean": float(series.mean()) if pd.api.types.is_numeric_dtype(series) and not series.dropna().empty else None
    }

    # Generate preview rows comparing before and after
    preview_rows = []
    sample_slice = df.head(10)
    for idx, row in sample_slice.iterrows():
        b_val = sanitize_value(row[col])
        a_val = sanitize_value(series.iloc[idx])
        preview_rows.append({
            "row_index": idx + 1,
            "original_value": b_val,
            "transformed_value": a_val
        })

    return TransformationPreviewResponse(
        success=True,
        transformation_type=step.transformation_type,
        column=col,
        before_stats=before_stats,
        after_stats=after_stats,
        preview_rows=preview_rows,
        python_snippet=python_code
    )

def apply_transformations(
    file_id: str,
    file_path: Path,
    steps: List[TransformationStep]
) -> pd.DataFrame:
    """Applies a list of transformations and returns the mutated DataFrame."""
    encoding = detect_encoding(file_path)
    delimiter = detect_delimiter(file_path, encoding)
    df = pd.read_csv(file_path, encoding=encoding, sep=delimiter, low_memory=False)

    for step in steps:
        col = step.column
        if col not in df.columns:
            continue

        if step.transformation_type == "impute":
            method = step.parameters.get("method", "median")
            if method == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif method == "median":
                df[col] = df[col].fillna(df[col].median())
            elif method == "mode":
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
            elif method == "constant":
                df[col] = df[col].fillna(step.parameters.get("value", 0))
            elif method == "ffill":
                df[col] = df[col].ffill()

        elif step.transformation_type == "scale":
            method = step.parameters.get("method", "standard")
            if method == "standard":
                std = df[col].std() if df[col].std() > 0 else 1.0
                df[col] = (df[col] - df[col].mean()) / std
            elif method == "minmax":
                rng = df[col].max() - df[col].min()
                rng = rng if rng > 0 else 1.0
                df[col] = (df[col] - df[col].min()) / rng

        elif step.transformation_type == "log":
            df[col] = np.log1p(np.maximum(0, df[col].fillna(0)))

        elif step.transformation_type == "winsorize":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df[col] = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

        elif step.transformation_type == "encode":
            df[col] = pd.factorize(df[col])[0]

        elif step.transformation_type == "datetime_extract":
            dt_series = pd.to_datetime(df[col], errors="coerce")
            feat = step.parameters.get("feature", "month")
            if feat == "year":
                df[f"{col}_year"] = dt_series.dt.year
            elif feat == "month":
                df[f"{col}_month"] = dt_series.dt.month
            elif feat == "day_of_week":
                df[f"{col}_dayofweek"] = dt_series.dt.dayofweek
            elif feat == "is_weekend":
                df[f"{col}_is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(int)

    # Save to session cache
    SESSION_TRANSFORMATIONS[file_id] = steps
    return df

def generate_reproducible_code(file_name: str, steps: List[TransformationStep]) -> str:
    """Generates clean standalone Python script to reproduce all data transformations."""
    code_lines = [
        '"""',
        f'Automated Data Preparation Pipeline for: {file_name}',
        f'Generated by AI Data Studio on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '"""',
        'import pandas as pd',
        'import numpy as np',
        '',
        f'# 1. Load dataset',
        f'df = pd.read_csv("{file_name}")',
        'print("Original Shape:", df.shape)',
        '',
        '# 2. Sequential Feature Engineering Transformations'
    ]

    for i, s in enumerate(steps, 1):
        code_lines.append(f"# Step {i}: {s.transformation_type.upper()} on column '{s.column}'")
        col = s.column
        if s.transformation_type == "impute":
            m = s.parameters.get("method", "median")
            if m == "mean":
                code_lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].mean())")
            elif m == "median":
                code_lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].median())")
            elif m == "mode":
                code_lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].mode()[0])")
            elif m == "constant":
                v = s.parameters.get("value", 0)
                code_lines.append(f"df['{col}'] = df['{col}'].fillna({repr(v)})")
            elif m == "ffill":
                code_lines.append(f"df['{col}'] = df['{col}'].ffill()")
        elif s.transformation_type == "scale":
            m = s.parameters.get("method", "standard")
            if m == "standard":
                code_lines.append(f"df['{col}'] = (df['{col}'] - df['{col}'].mean()) / df['{col}'].std()")
            elif m == "minmax":
                code_lines.append(f"df['{col}'] = (df['{col}'] - df['{col}'].min()) / (df['{col}'].max() - df['{col}'].min())")
        elif s.transformation_type == "log":
            code_lines.append(f"df['{col}'] = np.log1p(np.maximum(0, df['{col}']))")
        elif s.transformation_type == "winsorize":
            code_lines.append(f"q1 = df['{col}'].quantile(0.25)")
            code_lines.append(f"q3 = df['{col}'].quantile(0.75)")
            code_lines.append(f"iqr = q3 - q1")
            code_lines.append(f"df['{col}'] = df['{col}'].clip(lower=q1 - 1.5*iqr, upper=q3 + 1.5*iqr)")
        elif s.transformation_type == "encode":
            code_lines.append(f"df['{col}'] = pd.factorize(df['{col}'])[0]")
        elif s.transformation_type == "datetime_extract":
            feat = s.parameters.get("feature", "month")
            code_lines.append(f"dt = pd.to_datetime(df['{col}'], errors='coerce')")
            code_lines.append(f"df['{col}_{feat}'] = getattr(dt.dt, '{feat}')")
        code_lines.append("")

    code_lines.append("# 3. Export Cleaned Dataset")
    code_lines.append("output_filename = 'cleaned_dataset.csv'")
    code_lines.append("df.to_csv(output_filename, index=False)")
    code_lines.append("print('Pipeline executed successfully! Output saved to:', output_filename)")
    return "\n".join(code_lines)

def generate_executive_report(
    file_id: str,
    file_path: Path,
    dataset_name: str
) -> ExecutiveReportResponse:
    """Generates an executive summary and health assessment report."""
    encoding = detect_encoding(file_path)
    delimiter = detect_delimiter(file_path, encoding)
    df = pd.read_csv(file_path, encoding=encoding, sep=delimiter, low_memory=False)

    rows, cols = df.shape
    total_cells = rows * cols
    missing_cells = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    missing_pct = round((missing_cells / total_cells) * 100, 1) if total_cells > 0 else 0
    dup_pct = round((duplicates / rows) * 100, 1) if rows > 0 else 0

    # Calculate overall Data Health Score (0-100)
    health_penalty = (missing_pct * 1.5) + (dup_pct * 2.0)
    health_score = max(10.0, round(100.0 - health_penalty, 1))

    findings = [
        f"Dataset contains {rows:,} records across {cols} features.",
        f"Overall Data Completeness is {round(100.0 - missing_pct, 1)}% with {missing_cells:,} missing cells.",
        f"Identified {duplicates} exact duplicate records ({dup_pct}% duplication rate)."
    ]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        top_num = numeric_cols[0]
        findings.append(f"Primary numeric metric '{top_num}' ranges from {df[top_num].min()} to {df[top_num].max()} (mean: {round(df[top_num].mean(), 2)}).")

    caveats = []
    if missing_pct > 10.0:
        caveats.append(f"Elevated missing rate ({missing_pct}%): imputation or row filtering is recommended before model training.")
    if duplicates > 0:
        caveats.append(f"{duplicates} duplicate rows detected: recommend deduplication to prevent model sample bias.")
    if rows < 100:
        caveats.append("Small sample size: statistical conclusions should be interpreted with caution.")

    if not caveats:
        caveats.append("Dataset is clean with zero missing values and zero duplicate records.")

    data_dict = []
    for c in df.columns:
        data_dict.append({
            "column": c,
            "type": str(df[c].dtype),
            "missing": int(df[c].isna().sum()),
            "unique": int(df[c].nunique()),
            "sample": sanitize_value(df[c].dropna().iloc[0]) if not df[c].dropna().empty else None
        })

    summary = (
        f"Executive Data Audit for '{dataset_name}': The dataset exhibits a Data Health Score of {health_score}/100. "
        f"It consists of {rows} rows and {cols} attributes with an average completeness of {round(100.0 - missing_pct, 1)}%. "
        f"{'High data readiness for statistical modeling.' if health_score >= 80 else 'Requires feature engineering and imputation before downstream use.'}"
    )

    return ExecutiveReportResponse(
        file_id=file_id,
        dataset_name=dataset_name,
        generated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        executive_summary=summary,
        health_score=health_score,
        key_findings=findings,
        caveats_and_limitations=caveats,
        data_dictionary=data_dict
    )

