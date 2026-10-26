from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

# ----------------- Milestone 01: Profiling Schemas -----------------
class ColumnProfile(BaseModel):
    name: str
    dtype: str
    inferred_type: str  # numeric, categorical, datetime, boolean, text
    total_count: int
    non_null_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    sample_values: List[Any] = Field(default_factory=list)
    is_constant: bool = False
    is_unique: bool = False
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None
    mean_val: Optional[float] = None
    median_val: Optional[float] = None
    std_val: Optional[float] = None

class DatasetOverview(BaseModel):
    file_id: str
    file_name: str
    file_size_bytes: int
    file_size_formatted: str
    total_rows: int
    total_columns: int
    total_cells: int
    total_missing_cells: int
    missing_cells_percentage: float
    duplicate_rows_count: int
    duplicate_rows_percentage: float
    columns_by_type: Dict[str, int]
    memory_usage_bytes: int
    memory_usage_formatted: str
    uploaded_at: str

class ProfileResponse(BaseModel):
    success: bool = True
    message: str = "Dataset successfully profiled"
    overview: DatasetOverview
    columns: List[ColumnProfile]
    preview_rows: List[Dict[str, Any]] = Field(default_factory=list)

class ErrorResponse(BaseModel):
    success: bool = False
    error_type: str
    detail: str

# ----------------- Milestone 02: EDA Schemas -----------------
class NumericSummary(BaseModel):
    column: str
    count: int
    missing: int
    mean: float
    std: float
    min: float
    q25: float
    median: float
    q75: float
    max: float
    iqr: float
    skewness: float
    kurtosis: float

class CategoricalSummary(BaseModel):
    column: str
    count: int
    missing: int
    unique_count: int
    top_categories: List[Dict[str, Any]]  # [{"value": "Eng", "count": 10, "percentage": 45.0}]

class HistogramBin(BaseModel):
    bin_start: float
    bin_end: float
    count: int
    label: str

class HistogramData(BaseModel):
    column: str
    bins: List[HistogramBin]

class BoxPlotData(BaseModel):
    column: str
    min: float
    q1: float
    median: float
    q3: float
    max: float
    lower_fence: float
    upper_fence: float
    outlier_count: int
    outliers: List[float]

class CorrelationPair(BaseModel):
    col1: str
    col2: str
    pearson: float
    strength: str  # Strong Positive, Moderate, Weak, Negative

class CorrelationMatrix(BaseModel):
    columns: List[str]
    matrix: List[List[Optional[float]]]
    top_correlations: List[CorrelationPair]

class OutlierReport(BaseModel):
    column: str
    iqr_outlier_count: int
    iqr_outlier_percentage: float
    zscore_outlier_count: int
    lower_fence: float
    upper_fence: float
    sample_outliers: List[Any]

class SegmentComparison(BaseModel):
    group_by_col: str
    metric_col: str
    segments: List[Dict[str, Any]]  # [{"segment": "Engineering", "mean": 105000, "count": 6, "sum": 630000}]

class EDAResponse(BaseModel):
    success: bool = True
    file_id: str
    dataset_name: str
    total_rows: int
    sampled: bool = False
    sampling_note: Optional[str] = None
    numeric_summaries: List[NumericSummary]
    categorical_summaries: List[CategoricalSummary]
    histograms: List[HistogramData]
    box_plots: List[BoxPlotData]
    correlations: Optional[CorrelationMatrix] = None
    outliers: List[OutlierReport]
    segment_comparisons: List[SegmentComparison]
    completeness_score: float

# ----------------- Milestone 03: Column Selection Schemas -----------------
class ColumnRecommendation(BaseModel):
    column_name: str
    inferred_type: str
    status: str  # "RECOMMENDED", "OPTIONAL", "EXCLUDED"
    relevance_score: float  # 0.0 to 1.0
    inclusion_reason: Optional[str] = None
    exclusion_reason: Optional[str] = None
    data_quality_flags: List[str] = Field(default_factory=list)

class ColumnSelectionRequest(BaseModel):
    file_id: str
    objective_type: str  # "customer_churn", "sales_forecasting", "employee_attrition", "risk_detection", "custom"
    business_question: str
    strict_quality_filter: bool = True

class ColumnSelectionResponse(BaseModel):
    success: bool = True
    objective: str
    business_question: str
    recommended_columns: List[str]
    column_details: List[ColumnRecommendation]
    precision_estimate: float
    recall_estimate: float
    methodology: str

# ----------------- Milestone 04: AI Agent Schemas -----------------
class AgentQueryRequest(BaseModel):
    file_id: str
    question: str
    api_key: Optional[str] = None

class PlanStep(BaseModel):
    step_number: int
    action: str
    tool_name: str
    arguments: Dict[str, Any]
    reasoning: str

class ToolResult(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    status: str  # "SUCCESS", "FAILED"
    output: Any
    evidence: str

class AgentResponse(BaseModel):
    success: bool = True
    question: str
    plan: List[PlanStep]
    executed_tools: List[ToolResult]
    numerical_findings: List[str]
    answer: str
    ambiguity_notes: Optional[str] = None
    model_used: str

# ----------------- Milestone 05: Feature Engineering Schemas -----------------
class TransformationStep(BaseModel):
    transformation_type: str  # "impute", "scale", "log", "encode", "winsorize", "datetime_extract"
    column: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

class TransformationPreviewRequest(BaseModel):
    file_id: str
    transformation: TransformationStep

class TransformationPreviewResponse(BaseModel):
    success: bool = True
    transformation_type: str
    column: str
    before_stats: Dict[str, Any]
    after_stats: Dict[str, Any]
    preview_rows: List[Dict[str, Any]]
    python_snippet: str

class ApplyTransformationsRequest(BaseModel):
    file_id: str
    transformations: List[TransformationStep]
    export_filename: Optional[str] = None

class SessionHistoryItem(BaseModel):
    step_id: int
    timestamp: str
    action_type: str
    description: str
    columns_affected: List[str]

class ExecutiveReportResponse(BaseModel):
    file_id: str
    dataset_name: str
    generated_at: str
    executive_summary: str
    health_score: float
    key_findings: List[str]
    caveats_and_limitations: List[str]
    data_dictionary: List[Dict[str, Any]]
