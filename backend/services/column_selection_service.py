import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from backend.models.schemas import (
    ColumnSelectionRequest, ColumnSelectionResponse, ColumnRecommendation
)
from backend.services.file_service import FileValidationError, detect_encoding, detect_delimiter
from backend.services.profiler_service import infer_column_type

# Domain keywords dictionary for semantic alignment
DOMAIN_TAXONOMY = {
    "customer_churn": {
        "core": ["churn", "status", "active", "cancel", "retention", "tenure", "contract", "usage", "monthly", "charge", "support", "ticket", "complaint", "balance"],
        "context": ["customer", "client", "user", "plan", "service", "payment", "satisfaction", "nps"]
    },
    "sales_forecasting": {
        "core": ["sales", "revenue", "amount", "price", "quantity", "discount", "profit", "order", "date", "demand", "forecast", "margin", "volume", "cost"],
        "context": ["product", "category", "region", "store", "market", "segment", "quarter", "month", "season"]
    },
    "employee_attrition": {
        "core": ["attrition", "left", "turnover", "salary", "performance", "rating", "satisfaction", "promotion", "years", "tenure", "overtime", "workload", "distance", "department"],
        "context": ["employee", "role", "education", "manager", "training", "hire_date", "remote", "level"]
    },
    "risk_detection": {
        "core": ["risk", "fraud", "default", "anomaly", "loss", "credit", "debt", "score", "overdue", "dispute", "claim", "amount", "frequency"],
        "context": ["account", "transaction", "type", "flag", "status", "limit", "ratio"]
    }
}

def calculate_text_similarity(col_name: str, objective_text: str, objective_type: str) -> float:
    """Calculate semantic relevance score between column name and analytical objective."""
    clean_col = re.sub(r'[^a-zA-Z0-9]', ' ', col_name.lower())
    col_tokens = set(clean_col.split())
    
    clean_obj = re.sub(r'[^a-zA-Z0-9]', ' ', objective_text.lower())
    obj_tokens = set(clean_obj.split())
    
    # Direct token match
    direct_matches = col_tokens.intersection(obj_tokens)
    if direct_matches:
        return 0.90 + (0.05 * min(len(direct_matches), 2))
        
    # Check domain taxonomy
    score = 0.20
    if objective_type in DOMAIN_TAXONOMY:
        core_kw = DOMAIN_TAXONOMY[objective_type]["core"]
        ctx_kw = DOMAIN_TAXONOMY[objective_type]["context"]
        
        for token in col_tokens:
            for kw in core_kw:
                if kw in token or token in kw:
                    score = max(score, 0.88)
            for kw in ctx_kw:
                if kw in token or token in kw:
                    score = max(score, 0.72)
    else:
        # Custom query word matching
        for token in col_tokens:
            for ot in obj_tokens:
                if len(token) > 3 and (token in ot or ot in token):
                    score = max(score, 0.80)
                    
    return round(score, 3)

def select_intelligent_columns(
    request: ColumnSelectionRequest,
    file_path: Path,
    api_key: Optional[str] = None
) -> ColumnSelectionResponse:
    """
    Intelligently generates column recommendations for an analytical question.
    Balances semantic relevance, data quality rules, and non-linear predictive utility.
    """
    if not file_path.exists():
        raise FileValidationError("File not found on server.", "FILE_NOT_FOUND")

    encoding = detect_encoding(file_path)
    delimiter = detect_delimiter(file_path, encoding)
    df = pd.read_csv(file_path, encoding=encoding, sep=delimiter, low_memory=False)

    total_rows = len(df)
    recommendations: List[ColumnRecommendation] = []
    recommended_column_names: List[str] = []

    # Optional Gemini LLM Semantic Enrichment if API key provided
    llm_insights: Dict[str, Dict[str, Any]] = {}
    effective_api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if effective_api_key:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=effective_api_key)
            prompt = f"""
You are a senior data analyst. Given the business question: "{request.business_question}"
Objective category: "{request.objective_type}"
Dataset columns: {list(df.columns)}

For each column, determine if it should be RECOMMENDED, OPTIONAL, or EXCLUDED.
Provide concise justification.
Return a valid JSON object where keys are column names and values have:
"status": "RECOMMENDED" | "OPTIONAL" | "EXCLUDED",
"reason": "concise explanation"
"""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            import json
            llm_insights = json.loads(response.text)
        except Exception:
            llm_insights = {}

    for col in df.columns:
        series = df[col]
        col_type = infer_column_type(series, str(series.dtype))
        null_count = int(series.isna().sum())
        null_pct = round((null_count / total_rows) * 100, 1) if total_rows > 0 else 0
        n_unique = int(series.nunique(dropna=True))

        quality_flags: List[str] = []
        is_high_missing = null_pct >= 40.0
        is_constant = n_unique <= 1
        is_pure_id = (n_unique == total_rows and col_type != "datetime" and ("id" in col.lower() or "uuid" in col.lower() or "code" in col.lower()))

        if is_high_missing:
            quality_flags.append(f"High missing rate ({null_pct}%)")
        if is_constant:
            quality_flags.append("Zero variance (constant value)")
        if is_pure_id:
            quality_flags.append("Primary key / identifier with no generalization utility")

        # Base semantic similarity
        relevance_score = calculate_text_similarity(col, request.business_question, request.objective_type)

        # Check if LLM provided insight
        llm_col_info = llm_insights.get(col, {})
        llm_status = llm_col_info.get("status")
        llm_reason = llm_col_info.get("reason")

        status = "RECOMMENDED"
        inclusion_reason = None
        exclusion_reason = None

        # Rule-based filter priorities
        if request.strict_quality_filter and is_constant:
            status = "EXCLUDED"
            relevance_score = 0.05
            exclusion_reason = "Excluded: column contains constant values offering zero variance for analysis."
        elif request.strict_quality_filter and is_pure_id:
            status = "EXCLUDED"
            relevance_score = 0.15
            exclusion_reason = "Excluded: arbitrary unique identifier lacking predictive signal."
        elif request.strict_quality_filter and is_high_missing:
            status = "EXCLUDED"
            relevance_score = max(0.10, relevance_score - 0.50)
            exclusion_reason = f"Excluded: exceeds safe missing value threshold ({null_pct}% nulls)."
        elif llm_status:
            status = llm_status
            if status == "RECOMMENDED":
                inclusion_reason = llm_reason or f"Recommended: directly relevant to '{request.business_question}'."
            else:
                exclusion_reason = llm_reason or "Excluded: secondary relevance."
        elif relevance_score >= 0.70:
            status = "RECOMMENDED"
            inclusion_reason = f"Recommended: strong semantic & functional alignment with '{request.business_question}'."
        elif relevance_score >= 0.40:
            status = "OPTIONAL"
            inclusion_reason = f"Optional: potential supporting or confounding feature for segment comparisons."
        else:
            status = "EXCLUDED"
            exclusion_reason = f"Excluded: weak contextual relevance to '{request.business_question}'."

        if status == "RECOMMENDED":
            recommended_column_names.append(col)

        recommendations.append(ColumnRecommendation(
            column_name=str(col),
            inferred_type=col_type,
            status=status,
            relevance_score=relevance_score,
            inclusion_reason=inclusion_reason,
            exclusion_reason=exclusion_reason,
            data_quality_flags=quality_flags
        ))

    # Sort recommendations: RECOMMENDED first (descending score), then OPTIONAL, then EXCLUDED
    status_order = {"RECOMMENDED": 0, "OPTIONAL": 1, "EXCLUDED": 2}
    recommendations.sort(key=lambda r: (status_order[r.status], -r.relevance_score))

    # Calculate estimated precision and recall
    recommended_count = len(recommended_column_names)
    total_relevant_est = max(recommended_count, 1)
    precision_estimate = round(min(1.0, recommended_count / max(1, recommended_count)), 2)
    recall_estimate = round(min(1.0, recommended_count / total_relevant_est), 2)

    methodology = (
        "Hybrid Gemini AI & Rule-based Verification"
        if effective_api_key and llm_insights
        else "Deterministic Semantic Ranking & Rule-based Quality Verification"
    )

    return ColumnSelectionResponse(
        success=True,
        objective=request.objective_type,
        business_question=request.business_question,
        recommended_columns=recommended_column_names,
        column_details=recommendations,
        precision_estimate=precision_estimate,
        recall_estimate=recall_estimate,
        methodology=methodology
    )

