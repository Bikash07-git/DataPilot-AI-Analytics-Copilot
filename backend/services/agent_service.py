import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from backend.models.schemas import (
    AgentQueryRequest, AgentResponse, PlanStep, ToolResult
)
from backend.services.file_service import FileValidationError, detect_encoding, detect_delimiter
from backend.services.profiler_service import sanitize_value

class ApprovedToolExecutor:
    """Safely executes authorized analysis tools strictly validated against DataFrame schema."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.columns = list(df.columns)
        self.columns_lower = {c.lower(): c for c in df.columns}

    def validate_column(self, col: str) -> str:
        if col in self.columns:
            return col
        if col.lower() in self.columns_lower:
            return self.columns_lower[col.lower()]
        raise ValueError(f"Column '{col}' does not exist in dataset. Available columns: {self.columns}")

    def compute_aggregates(self, column: str, group_by: Optional[str] = None, agg_func: str = "mean") -> Dict[str, Any]:
        target_col = self.validate_column(column)
        agg_func = agg_func.lower()
        if agg_func not in ["mean", "sum", "count", "min", "max", "median"]:
            agg_func = "mean"

        if group_by:
            group_col = self.validate_column(group_by)
            grouped = self.df.groupby(group_col)[target_col].agg(agg_func).reset_index()
            results = []
            for _, r in grouped.iterrows():
                results.append({
                    "group": str(r[group_col]),
                    "value": round(float(r[target_col]), 2) if pd.notna(r[target_col]) else None
                })
            results.sort(key=lambda x: x["value"] if x["value"] is not None else -1e9, reverse=True)
            return {
                "aggregation": agg_func,
                "target_column": target_col,
                "group_by": group_col,
                "results": results
            }
        else:
            val = getattr(self.df[target_col], agg_func)()
            return {
                "aggregation": agg_func,
                "target_column": target_col,
                "value": round(float(val), 2) if pd.notna(val) else None
            }

    def compute_correlation(self, col1: str, col2: str) -> Dict[str, Any]:
        c1 = self.validate_column(col1)
        c2 = self.validate_column(col2)
        valid = self.df[[c1, c2]].dropna()
        if len(valid) < 2:
            return {"error": "Not enough valid numeric pairs to compute correlation."}
        r = float(valid[c1].corr(valid[c2]))
        strength = "Moderate"
        if abs(r) >= 0.7:
            strength = "Strong Positive" if r > 0 else "Strong Negative"
        elif abs(r) < 0.3:
            strength = "Weak"
        return {
            "col1": c1,
            "col2": c2,
            "pearson": round(r, 4),
            "strength": strength,
            "sample_size": len(valid)
        }

    def detect_outliers(self, column: str) -> Dict[str, Any]:
        c = self.validate_column(column)
        series = self.df[c].dropna()
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)].tolist()
        return {
            "column": c,
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "lower_fence": round(lower, 2),
            "upper_fence": round(upper, 2),
            "outlier_count": len(outliers),
            "sample_outliers": [round(float(x), 2) for x in outliers[:5]]
        }

    def filter_and_slice(self, column: str, operator: str, value: Any) -> Dict[str, Any]:
        c = self.validate_column(column)
        if operator == ">":
            subset = self.df[self.df[c] > float(value)]
        elif operator == "<":
            subset = self.df[self.df[c] < float(value)]
        elif operator == "==":
            subset = self.df[self.df[c].astype(str).str.lower() == str(value).lower()]
        else:
            subset = self.df
        return {
            "filtered_column": c,
            "matching_rows": len(subset),
            "total_rows": len(self.df),
            "percentage": round((len(subset) / len(self.df)) * 100, 1) if len(self.df) > 0 else 0
        }

def run_agent_query(
    request: AgentQueryRequest,
    file_path: Path
) -> AgentResponse:
    """
    Executes a structured, evidence-backed analytical query.
    1. Formulates analysis plan.
    2. Validates schema & executes approved tools.
    3. Produces grounded answer directly referencing computed numbers.
    """
    if not file_path.exists():
        raise FileValidationError("File not found on server.", "FILE_NOT_FOUND")

    encoding = detect_encoding(file_path)
    delimiter = detect_delimiter(file_path, encoding)
    df = pd.read_csv(file_path, encoding=encoding, sep=delimiter, low_memory=False)

    executor = ApprovedToolExecutor(df)
    question = request.question.strip()
    q_lower = question.lower()

    effective_api_key = request.api_key or os.environ.get("GEMINI_API_KEY")
    plan: List[PlanStep] = []
    executed_tools: List[ToolResult] = []
    numerical_findings: List[str] = []
    model_used = "Deterministic Evidence Engine"

    # Plan generation & tool routing
    # Match user intent to target columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

    # 1. Check for correlation query
    if "correlation" in q_lower or "relationship" in q_lower or "relate" in q_lower or "vs" in q_lower:
        matched_num = [c for c in numeric_cols if c.lower() in q_lower]
        if len(matched_num) >= 2:
            c1, c2 = matched_num[0], matched_num[1]
        elif len(numeric_cols) >= 2:
            c1, c2 = numeric_cols[0], numeric_cols[1]
        else:
            c1, c2 = None, None

        if c1 and c2:
            plan.append(PlanStep(
                step_number=1,
                action=f"Compute Pearson correlation between '{c1}' and '{c2}'",
                tool_name="compute_correlation",
                arguments={"col1": c1, "col2": c2},
                reasoning="Quantify statistical association and check linearity between the two metrics."
            ))

    # 2. Check for group aggregation (e.g., average salary by department)
    matched_cat = [c for c in categorical_cols if c.lower() in q_lower]
    matched_num = [c for c in numeric_cols if c.lower() in q_lower]

    agg_func = "mean"
    if "sum" in q_lower or "total" in q_lower:
        agg_func = "sum"
    elif "highest" in q_lower or "max" in q_lower or "top" in q_lower:
        agg_func = "max"
    elif "lowest" in q_lower or "min" in q_lower:
        agg_func = "min"
    elif "count" in q_lower or "how many" in q_lower:
        agg_func = "count"

    if matched_cat and matched_num:
        plan.append(PlanStep(
            step_number=len(plan) + 1,
            action=f"Aggregate '{matched_num[0]}' grouped by '{matched_cat[0]}' with {agg_func}",
            tool_name="compute_aggregates",
            arguments={"column": matched_num[0], "group_by": matched_cat[0], "agg_func": agg_func},
            reasoning=f"Break down {matched_num[0]} across segments to identify leaders and variance."
        ))
    elif matched_num:
        plan.append(PlanStep(
            step_number=len(plan) + 1,
            action=f"Calculate {agg_func} of '{matched_num[0]}'",
            tool_name="compute_aggregates",
            arguments={"column": matched_num[0], "group_by": None, "agg_func": agg_func},
            reasoning=f"Compute summary statistic for {matched_num[0]}."
        ))

    # 3. Check for outlier question
    if "outlier" in q_lower or "anomaly" in q_lower or "extreme" in q_lower:
        target_outlier_col = matched_num[0] if matched_num else (numeric_cols[0] if numeric_cols else None)
        if target_outlier_col:
            plan.append(PlanStep(
                step_number=len(plan) + 1,
                action=f"Detect outliers in '{target_outlier_col}' using IQR fences",
                tool_name="detect_outliers",
                arguments={"column": target_outlier_col},
                reasoning="Identify extreme values that lie beyond 1.5x the interquartile range."
            ))

    # Default fallback plan if question is broad
    if not plan:
        target_num = numeric_cols[0] if numeric_cols else df.columns[0]
        target_cat = categorical_cols[0] if categorical_cols else None
        plan.append(PlanStep(
            step_number=1,
            action=f"Compute summary statistics for '{target_num}'",
            tool_name="compute_aggregates",
            arguments={"column": target_num, "group_by": target_cat, "agg_func": "mean"},
            reasoning="Provide comprehensive baseline aggregation for the primary dataset metric."
        ))

    # Execute all planned tools
    for step in plan:
        try:
            if step.tool_name == "compute_aggregates":
                res = executor.compute_aggregates(**step.arguments)
                if "results" in res:
                    top_item = res["results"][0] if res["results"] else {"group": "N/A", "value": 0}
                    ev = f"Grouped by '{res['group_by']}': highest '{res['target_column']}' is '{top_item['group']}' with {top_item['value']}."
                else:
                    ev = f"The {res['aggregation']} of '{res['target_column']}' is {res['value']}."
            elif step.tool_name == "compute_correlation":
                res = executor.compute_correlation(**step.arguments)
                ev = f"Pearson correlation between '{res['col1']}' and '{res['col2']}' is {res['pearson']} ({res['strength']})."
            elif step.tool_name == "detect_outliers":
                res = executor.detect_outliers(**step.arguments)
                ev = f"Found {res['outlier_count']} outliers in '{res['column']}' outside fences [{res['lower_fence']}, {res['upper_fence']}]."
            else:
                res = {}
                ev = "Operation completed."

            executed_tools.append(ToolResult(
                tool_name=step.tool_name,
                arguments=step.arguments,
                status="SUCCESS",
                output=res,
                evidence=ev
            ))
            numerical_findings.append(ev)
        except Exception as err:
            executed_tools.append(ToolResult(
                tool_name=step.tool_name,
                arguments=step.arguments,
                status="FAILED",
                output={"error": str(err)},
                evidence=f"Failed to execute {step.tool_name}: {str(err)}"
            ))

    # LLM Synthesis if Gemini API Key is available
    answer_text = ""
    if effective_api_key:
        try:
            from google import genai
            client = genai.Client(api_key=effective_api_key)
            synth_prompt = f"""
You are a factual, evidence-backed AI Data Analyst.
User Question: "{question}"

The following tools were executed on the dataset with exact calculated evidence:
{json.dumps([t.model_dump() for t in executed_tools], indent=2)}

Instructions:
1. Answer the user's question directly.
2. Every number and statistic you cite MUST match the tool outputs above.
3. Do not invent or assume numbers.
4. If the question was ambiguous, state your interpretation clearly.
"""
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=synth_prompt
            )
            answer_text = resp.text
            model_used = "Gemini 2.5 Flash (Grounded Tool Execution)"
        except Exception:
            answer_text = ""

    if not answer_text:
        # High quality grounded synthesis
        evidence_summary = " ".join(numerical_findings)
        answer_text = f"Based on computed dataset analysis: {evidence_summary}"

    ambiguity_notes = None
    if len(matched_cat) == 0 and len(matched_num) == 0:
        ambiguity_notes = "Question did not explicitly name target columns; defaulted to primary numeric and categorical features."

    return AgentResponse(
        success=True,
        question=question,
        plan=plan,
        executed_tools=executed_tools,
        numerical_findings=numerical_findings,
        answer=answer_text,
        ambiguity_notes=ambiguity_notes,
        model_used=model_used
    )

