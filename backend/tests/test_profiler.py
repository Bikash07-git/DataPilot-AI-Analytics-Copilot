import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.profiler_service import profile_dataset
from backend.services.file_service import FileValidationError

client = TestClient(app)
SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "sample_data"

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["milestone"] == "01"

def test_valid_employees_profiling():
    file_path = SAMPLE_DIR / "valid_employees.csv"
    assert file_path.exists(), "Sample file valid_employees.csv must exist"
    
    profile = profile_dataset("test_emp", file_path, "valid_employees.csv")
    
    assert profile.success is True
    overview = profile.overview
    assert overview.total_rows == 15
    assert overview.total_columns == 7
    assert overview.total_cells == 15 * 7
    assert overview.total_missing_cells == 3  # Fiona salary, Hannah perf, Nina salary
    assert overview.duplicate_rows_count == 1  # Alice Johnson repeated
    assert len(profile.columns) == 7
    assert len(profile.preview_rows) == 15
    
    # Verify column metrics
    cols = {c.name: c for c in profile.columns}
    assert "salary" in cols
    assert cols["salary"].null_count == 2
    assert cols["salary"].inferred_type == "numeric"
    assert cols["salary"].min_val == 68000.0
    assert cols["salary"].max_val == 115000.0
    
    assert "department" in cols
    assert cols["department"].null_count == 0
    assert cols["department"].inferred_type == "categorical"
    assert cols["department"].unique_count == 4  # Engineering, Marketing, Design, Sales

def test_valid_sales_profiling():
    file_path = SAMPLE_DIR / "valid_sales.csv"
    profile = profile_dataset("test_sales", file_path, "valid_sales.csv")
    
    assert profile.overview.total_rows == 10
    assert profile.overview.total_columns == 9
    assert profile.overview.total_missing_cells == 0
    assert profile.overview.duplicate_rows_count == 0

def test_upload_valid_csv():
    file_path = SAMPLE_DIR / "valid_employees.csv"
    with open(file_path, "rb") as f:
        response = client.post(
            "/api/upload",
            files={"file": ("valid_employees.csv", f, "text/csv")}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["overview"]["total_rows"] == 15
    assert data["overview"]["total_columns"] == 7
    file_id = data["overview"]["file_id"]
    
    # Test retrieving by file_id
    get_res = client.get(f"/api/profile/{file_id}")
    assert get_res.status_code == 200
    assert get_res.json()["overview"]["file_id"] == file_id

def test_upload_unsupported_file():
    file_path = SAMPLE_DIR / "unsupported.txt"
    with open(file_path, "rb") as f:
        response = client.post(
            "/api/upload",
            files={"file": ("unsupported.txt", f, "text/plain")}
        )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error_type"] == "UNSUPPORTED_FILE_TYPE"
    assert "Only CSV files (.csv) are currently supported" in data["detail"]

def test_upload_malformed_csv():
    file_path = SAMPLE_DIR / "malformed.csv"
    with open(file_path, "rb") as f:
        response = client.post(
            "/api/upload",
            files={"file": ("malformed.csv", f, "text/csv")}
        )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error_type"] in ["MALFORMED_CSV", "READ_ERROR"]
    assert "parsing error" in data["detail"].lower() or "error" in data["detail"].lower()

def test_upload_header_only_csv():
    file_path = SAMPLE_DIR / "header_only.csv"
    with open(file_path, "rb") as f:
        response = client.post(
            "/api/upload",
            files={"file": ("header_only.csv", f, "text/csv")}
        )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error_type"] == "NO_ROWS"

