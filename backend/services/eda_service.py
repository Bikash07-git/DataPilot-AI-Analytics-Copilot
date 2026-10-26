import math
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from backend.models.schemas import (
    EDAResponse, NumericSummary, CategoricalSummary,
    HistogramData, HistogramBin, BoxPlotData,
    CorrelationMatrix, CorrelationPair, OutlierReport,
    SegmentComparison
)
from backend.services.file_service import FileValidationError, detect_encoding, detect_delimiter
from backend.services.profiler_service import sanitize_value

def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert any numeric value to float, guarding against NaN and Inf."""
    if val is None or pd.isna(val):
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, 4)
    except Exception:
        return default

def compute_eda(file_id: str, file_path: Path, dataset_name: str) -> EDAResponse:
    """
    Computes comprehensive Exploratory Data Analysis (EDA) on the dataset:
    - Numeric and categorical distributions
    - Histograms & box plots with IQR outlier detection
    - Correlation matrix & strong relationship detection
    - Outlier diagnostics (IQR & Z-score)
    - Segment comparisons
    """
    if not file_path.exists():
        raise FileValidationError("File not found on server.", "FILE_NOT_FOUND")

    encoding = detect_encoding(file_path)
    delimiter = detect_delimiter(file_path, encoding)

    try:
        df = pd.read_csv(file_path, encoding=encoding, sep=delimiter, low_memory=False)
    except Exception as e:
        raise FileValidationError(f"Failed to read dataset for EDA: {str(e)}", "READ_ERROR")

    total_rows, total_cols = df.shape
    if total_rows == 0:
        raise FileValidationError("Dataset has no data rows.", "EMPTY_DATASET")

    sampled = False
    sampling_note = None
    if total_rows > 100000:
        df = df.sample(n=100000, random_state=42)
        sampled = True
        sampling_note = "Sampled 100,000 rows for high-performance interactive EDA calculation."

    # Identify numeric and categorical columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    numeric_summaries: List[NumericSummary] = []
    histograms: List[HistogramData] = []
    box_plots: List[BoxPlotData] = []
    outlier_reports: List[OutlierReport] = []

    for col in numeric_cols:
        series = df[col].dropna()
        n_count = len(series)
        missing_count = int(df[col].isna().sum())

        if n_count > 0:
            mean_v = safe_float(series.mean())
            std_v = safe_float(series.std())
            min_v = safe_float(series.min())
            q25_v = safe_float(series.quantile(0.25))
            med_v = safe_float(series.median())
            q75_v = safe_float(series.quantile(0.75))
            max_v = safe_float(series.max())
            iqr_v = safe_float(q75_v - q25_v)
            skew_v = safe_float(series.skew()) if n_count > 2 else 0.0
            kurt_v = safe_float(series.kurt()) if n_count > 3 else 0.0

            numeric_summaries.append(NumericSummary(
                column=str(col),
                count=n_count,
                missing=missing_count,
                mean=mean_v,
                std=std_v,
                min=min_v,
                q25=q25_v,
                median=med_v,
                q75=q75_v,
                max=max_v,
                iqr=iqr_v,
                skewness=skew_v,
                kurtosis=kurt_v
            ))

            # Histogram
            try:
                num_bins = min(12, max(5, series.nunique()))
                counts, bin_edges = np.histogram(series, bins=num_bins)
                bins_list: List[HistogramBin] = []
                for i in range(len(counts)):
                    b_start = safe_float(bin_edges[i])
                    b_end = safe_float(bin_edges[i+1])
                    bins_list.append(HistogramBin(
                        bin_start=b_start,
                        bin_end=b_end,
                        count=int(counts[i]),
                        label=f"{b_start:.1f} - {b_end:.1f}"
                    ))
                histograms.append(HistogramData(column=str(col), bins=bins_list))
            except Exception:
                pass

            # Box Plot & Outliers (IQR Method)
            lower_fence = safe_float(q25_v - 1.5 * iqr_v)
            upper_fence = safe_float(q75_v + 1.5 * iqr_v)
            iqr_outliers = series[(series < lower_fence) | (series > upper_fence)]
            iqr_outlier_count = len(iqr_outliers)

            # Z-Score Outliers (|z| > 3)
            if std_v > 0:
                z_scores = np.abs((series - mean_v) / std_v)
                zscore_outlier_count = int((z_scores > 3.0).sum())
            else:
                zscore_outlier_count = 0

            box_plots.append(BoxPlotData(
                column=str(col),
                min=min_v,
                q1=q25_v,
                median=med_v,
                q3=q75_v,
                max=max_v,
                lower_fence=lower_fence,
                upper_fence=upper_fence,
                outlier_count=iqr_outlier_count,
                outliers=[safe_float(x) for x in iqr_outliers.head(20).tolist()]
            ))

            outlier_reports.append(OutlierReport(
                column=str(col),
                iqr_outlier_count=iqr_outlier_count,
                iqr_outlier_percentage=round((iqr_outlier_count / n_count) * 100, 2),
                zscore_outlier_count=zscore_outlier_count,
                lower_fence=lower_fence,
                upper_fence=upper_fence,
                sample_outliers=[safe_float(x) for x in iqr_outliers.head(5).tolist()]
            ))

    # Categorical Summaries
    categorical_summaries: List[CategoricalSummary] = []
    for col in categorical_cols:
        series = df[col].dropna()
        n_count = len(series)
        missing_count = int(df[col].isna().sum())
        u_count = int(series.nunique())

        # Top 6 most frequent categories
        top_cats = []
        if n_count > 0:
            val_counts = series.value_counts().head(6)
            for val, cnt in val_counts.items():
                pct = round((cnt / n_count) * 100, 1)
                top_cats.append({
                    "value": str(val),
                    "count": int(cnt),
                    "percentage": pct
                })

        categorical_summaries.append(CategoricalSummary(
            column=str(col),
            count=n_count,
            missing=missing_count,
            unique_count=u_count,
            top_categories=top_cats
        ))

    # Correlation Matrix (if >= 2 numeric columns)
    corr_matrix_data: Optional[CorrelationMatrix] = None
    if len(numeric_cols) >= 2:
        try:
            num_df = df[numeric_cols].dropna()
            if len(num_df) > 1:
                corr = num_df.corr(method="pearson")
                columns_list = [str(c) for c in corr.columns]
                matrix_rows = []
                top_pairs: List[CorrelationPair] = []

                for i, r_col in enumerate(columns_list):
                    row_vals = []
                    for j, c_col in enumerate(columns_list):
                        val = corr.iloc[i, j]
                        s_val = None if pd.isna(val) else round(float(val), 3)
                        row_vals.append(s_val)

                        # Check non-diagonal top correlations
                        if i < j and s_val is not None:
                            strength = "Moderate"
                            if abs(s_val) >= 0.7:
                                strength = "Strong Positive" if s_val > 0 else "Strong Negative"
                            elif abs(s_val) < 0.3:
                                strength = "Weak"
                            top_pairs.append(CorrelationPair(
                                col1=r_col,
                                col2=c_col,
                                pearson=s_val,
                                strength=strength
                            ))
                    matrix_rows.append(row_vals)

                top_pairs.sort(key=lambda x: abs(x.pearson), reverse=True)
                corr_matrix_data = CorrelationMatrix(
                    columns=columns_list,
                    matrix=matrix_rows,
                    top_correlations=top_pairs[:8]
                )
        except Exception:
            corr_matrix_data = None

    # Segment Comparisons: detect categorical column with 2-10 unique values and a numeric column
    segment_comparisons: List[SegmentComparison] = []
    candidate_cat = [c for c in categorical_cols if 2 <= df[c].nunique() <= 12]
    if candidate_cat and numeric_cols:
        cat_col = candidate_cat[0]
        for num_col in numeric_cols[:2]:
            try:
                grouped = df.groupby(cat_col)[num_col].agg(["mean", "sum", "count"]).reset_index()
                segments = []
                for _, row in grouped.iterrows():
                    segments.append({
                        "segment": str(row[cat_col]),
                        "mean": safe_float(row["mean"]),
                        "sum": safe_float(row["sum"]),
                        "count": int(row["count"])
                    })
                segments.sort(key=lambda x: x["mean"], reverse=True)
                segment_comparisons.append(SegmentComparison(
                    group_by_col=str(cat_col),
                    metric_col=str(num_col),
                    segments=segments
                ))
            except Exception:
                pass

    total_cells = total_rows * total_cols
    total_missing = int(df.isna().sum().sum())
    completeness = round(((total_cells - total_missing) / total_cells) * 100, 2) if total_cells > 0 else 100.0

    return EDAResponse(
        success=True,
        file_id=file_id,
        dataset_name=dataset_name,
        total_rows=total_rows,
        sampled=sampled,
        sampling_note=sampling_note,
        numeric_summaries=numeric_summaries,
        categorical_summaries=categorical_summaries,
        histograms=histograms,
        box_plots=box_plots,
        correlations=corr_matrix_data,
        outliers=outlier_reports,
        segment_comparisons=segment_comparisons,
        completeness_score=completeness
    )

