import os
import json
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Query, Body, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from backend.config import UPLOAD_DIR, BASE_DIR
from backend.models.schemas import (
    ProfileResponse, ErrorResponse, EDAResponse,
    ColumnSelectionRequest, ColumnSelectionResponse,
    AgentQueryRequest, AgentResponse,
    TransformationPreviewRequest, TransformationPreviewResponse,
    ApplyTransformationsRequest, ExecutiveReportResponse
)
from backend.services.file_service import save_and_validate_upload, FileValidationError
from backend.services.profiler_service import profile_dataset
from backend.services.eda_service import compute_eda
from backend.services.column_selection_service import select_intelligent_columns
from backend.services.agent_service import run_agent_query
from backend.services.feature_engineering_service import (
    preview_transformation, apply_transformations,
    generate_reproducible_code, generate_executive_report,
    SESSION_TRANSFORMATIONS
)

app = FastAPI(
    title="AI Data Studio & Autonomous EDA Engine",
    description="End-to-end full-stack data profiling, exploratory data analysis, intelligent column selection, AI agent querying, and feature engineering platform.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory caches for fast retrieval
PROFILES_CACHE: Dict[str, ProfileResponse] = {}
EDA_CACHE: Dict[str, EDAResponse] = {}
GLOBAL_SETTINGS = {
    "gemini_api_key": os.environ.get("GEMINI_API_KEY", "")
}

def resolve_file_path(file_id: str) -> Tuple[Path, str]:
    """Helper to locate file on disk from file_id."""
    if file_id.startswith("sample_"):
        sample_name = file_id.replace("sample_", "")
        target = BASE_DIR / "sample_data" / f"{sample_name}.csv"
        if not target.exists():
            target = BASE_DIR / "sample_data" / sample_name
        if target.exists():
            return target, target.name

    matches = list(UPLOAD_DIR.glob(f"{file_id}_*"))
    if not matches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with file_id '{file_id}' not found on server."
        )
    file_path = matches[0]
    orig_name = file_path.name.replace(f"{file_id}_", "")
    return file_path, orig_name

@app.exception_handler(FileValidationError)
async def file_validation_exception_handler(request, exc: FileValidationError):
    status_code = status.HTTP_400_BAD_REQUEST
    if exc.error_type == "FILE_TOO_LARGE":
        status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    elif exc.error_type == "FILE_NOT_FOUND":
        status_code = status.HTTP_404_NOT_FOUND
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            success=False,
            error_type=exc.error_type,
            detail=exc.message
        ).model_dump()
    )

@app.get("/api/health")
async def health_check():
    has_gemini = bool(GLOBAL_SETTINGS["gemini_api_key"] or os.environ.get("GEMINI_API_KEY"))
    return {
        "status": "healthy",
        "service": "AI Data Studio Autonomous EDA Engine",
        "version": "2.0.0",
        "milestones_active": ["01", "02", "03", "04", "05", "06"],
        "gemini_ai_configured": has_gemini
    }

# ----------------- Milestone 01: Profiling Endpoints -----------------
@app.post("/api/upload", response_model=ProfileResponse)
async def upload_csv(file: UploadFile = File(...)):
    file_id, file_path, orig_name = await save_and_validate_upload(file)
    try:
        profile = profile_dataset(file_id, file_path, orig_name)
        PROFILES_CACHE[file_id] = profile
        return profile
    except FileValidationError:
        if file_path.exists():
            file_path.unlink()
        raise

@app.get("/api/profile/{file_id}", response_model=ProfileResponse)
async def get_profile(file_id: str):
    if file_id in PROFILES_CACHE:
        return PROFILES_CACHE[file_id]
    file_path, orig_name = resolve_file_path(file_id)
    profile = profile_dataset(file_id, file_path, orig_name)
    PROFILES_CACHE[file_id] = profile
    return profile

@app.get("/api/sample-datasets")
async def list_sample_datasets():
    sample_dir = BASE_DIR / "sample_data"
    files = []
    if sample_dir.exists():
        for p in sample_dir.glob("*.csv"):
            files.append({
                "id": p.stem,
                "filename": p.name,
                "size_bytes": p.stat().st_size
            })
    return {"samples": files}

@app.post("/api/load-sample/{sample_name}", response_model=ProfileResponse)
async def load_sample_dataset(sample_name: str):
    file_id = f"sample_{sample_name}"
    file_path, orig_name = resolve_file_path(file_id)
    profile = profile_dataset(file_id, file_path, orig_name)
    PROFILES_CACHE[file_id] = profile
    return profile

# ----------------- Milestone 02: EDA Endpoints -----------------
@app.get("/api/eda/{file_id}", response_model=EDAResponse)
async def get_eda(file_id: str):
    if file_id in EDA_CACHE:
        return EDA_CACHE[file_id]
    file_path, orig_name = resolve_file_path(file_id)
    eda = compute_eda(file_id, file_path, orig_name)
    EDA_CACHE[file_id] = eda
    return eda

# ----------------- Milestone 03: Column Selection Endpoints -----------------
@app.post("/api/column-selection", response_model=ColumnSelectionResponse)
async def recommend_columns(request: ColumnSelectionRequest):
    file_path, _ = resolve_file_path(request.file_id)
    api_key = GLOBAL_SETTINGS["gemini_api_key"]
    return select_intelligent_columns(request, file_path, api_key=api_key)

# ----------------- Milestone 04: AI Agent Query Endpoints -----------------
@app.post("/api/agent/query", response_model=AgentResponse)
async def query_agent(request: AgentQueryRequest):
    file_path, _ = resolve_file_path(request.file_id)
    if not request.api_key:
        request.api_key = GLOBAL_SETTINGS["gemini_api_key"]
    return run_agent_query(request, file_path)

# ----------------- Milestone 05: Feature Engineering & Reports -----------------
@app.post("/api/transform/preview", response_model=TransformationPreviewResponse)
async def preview_transform_step(request: TransformationPreviewRequest):
    file_path, _ = resolve_file_path(request.file_id)
    return preview_transformation(request.file_id, file_path, request.transformation)

@app.post("/api/transform/apply")
async def apply_dataset_transformations(request: ApplyTransformationsRequest):
    file_path, orig_name = resolve_file_path(request.file_id)
    df_transformed = apply_transformations(request.file_id, file_path, request.transformations)
    
    # Save transformed version
    transformed_file_id = f"transformed_{request.file_id}"
    out_name = f"{transformed_file_id}_{orig_name}"
    out_path = UPLOAD_DIR / out_name
    df_transformed.to_csv(out_path, index=False)
    
    # Generate new profile for the transformed dataset
    new_profile = profile_dataset(transformed_file_id, out_path, f"transformed_{orig_name}")
    PROFILES_CACHE[transformed_file_id] = new_profile
    return {
        "success": True,
        "transformed_file_id": transformed_file_id,
        "new_profile": new_profile
    }

@app.get("/api/export-data/{file_id}")
async def export_data(file_id: str):
    file_path, orig_name = resolve_file_path(file_id)
    return FileResponse(
        path=str(file_path),
        filename=f"clean_{orig_name}",
        media_type="text/csv"
    )

@app.get("/api/export-code/{file_id}")
async def export_reproducible_code(file_id: str):
    file_path, orig_name = resolve_file_path(file_id)
    steps = SESSION_TRANSFORMATIONS.get(file_id, [])
    code = generate_reproducible_code(orig_name, steps)
    return PlainTextResponse(
        content=code,
        headers={"Content-Disposition": f'attachment; filename="reproduce_pipeline.py"'}
    )

@app.get("/api/report/{file_id}", response_model=ExecutiveReportResponse)
async def get_executive_report(file_id: str):
    file_path, orig_name = resolve_file_path(file_id)
    return generate_executive_report(file_id, file_path, orig_name)

# ----------------- Milestone 06: Settings & Deployment -----------------
@app.post("/api/settings/api-key")
async def set_gemini_api_key(data: Dict[str, str] = Body(...)):
    key = data.get("api_key", "").strip()
    GLOBAL_SETTINGS["gemini_api_key"] = key
    os.environ["GEMINI_API_KEY"] = key
    return {"success": True, "message": "Gemini API key updated successfully."}

@app.get("/api/settings")
async def get_settings():
    return {
        "has_gemini_key": bool(GLOBAL_SETTINGS["gemini_api_key"] or os.environ.get("GEMINI_API_KEY")),
        "max_upload_size_mb": 50,
        "supported_extensions": [".csv"],
        "server_time": datetime.datetime.now().isoformat()
    }

# Mount Frontend static files
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
