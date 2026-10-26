import asyncio
import sys
from pathlib import Path
import httpx

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.main import app
from backend.services.profiler_service import profile_dataset
from backend.services.file_service import FileValidationError

SAMPLE_DIR = BASE_DIR / "sample_data"

def test_unit_profiler():
    print("--- [1/6] Running Unit Test: valid_employees.csv ---")
    file_path = SAMPLE_DIR / "valid_employees.csv"
    p = profile_dataset("emp_1", file_path, "valid_employees.csv")
    assert p.success is True
    assert p.overview.total_rows == 15, f"Expected 15 rows, got {p.overview.total_rows}"
    assert p.overview.total_columns == 7, f"Expected 7 columns, got {p.overview.total_columns}"
    assert p.overview.total_missing_cells == 3, f"Expected 3 missing cells, got {p.overview.total_missing_cells}"
    assert p.overview.duplicate_rows_count == 1, f"Expected 1 duplicate row, got {p.overview.duplicate_rows_count}"
    
    # Check column inference & stats
    cols = {c.name: c for c in p.columns}
    assert cols["salary"].inferred_type == "numeric"
    assert cols["salary"].min_val == 68000.0
    assert cols["salary"].max_val == 115000.0
    assert cols["department"].inferred_type == "categorical"
    assert cols["remote"].inferred_type == "boolean"
    print("  -> PASSED: Accurate counts, types, missing values, duplicates, and ranges.")

    print("--- [2/6] Running Unit Test: valid_sales.csv ---")
    p2 = profile_dataset("sales_1", SAMPLE_DIR / "valid_sales.csv", "valid_sales.csv")
    assert p2.overview.total_rows == 10
    assert p2.overview.total_columns == 9
    assert p2.overview.total_missing_cells == 0
    assert p2.overview.duplicate_rows_count == 0
    print("  -> PASSED: Valid sales dataset correctly profiled.")

    print("--- [3/6] Running Unit Test: malformed.csv error handling ---")
    try:
        profile_dataset("malformed_1", SAMPLE_DIR / "malformed.csv", "malformed.csv")
        assert False, "Should have raised FileValidationError"
    except FileValidationError as e:
        assert e.error_type in ["MALFORMED_CSV", "READ_ERROR"]
        print(f"  -> PASSED: Expected error caught correctly: {e.error_type} - {e.message}")

    print("--- [4/6] Running Unit Test: header_only.csv error handling ---")
    try:
        profile_dataset("header_1", SAMPLE_DIR / "header_only.csv", "header_only.csv")
        assert False, "Should have raised FileValidationError"
    except FileValidationError as e:
        assert e.error_type == "NO_ROWS"
        print(f"  -> PASSED: Expected error caught correctly: {e.error_type} - {e.message}")

async def test_api_endpoints():
    print("--- [5/6] Testing FastAPI Endpoints via ASGI Client ---")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Health check
        res = await client.get("/api/health")
        assert res.status_code == 200
        assert res.json()["status"] == "healthy"
        print("  -> /api/health returned 200 OK")

        # Upload valid file
        with open(SAMPLE_DIR / "valid_employees.csv", "rb") as f:
            upload_res = await client.post(
                "/api/upload",
                files={"file": ("valid_employees.csv", f, "text/csv")}
            )
        assert upload_res.status_code == 200
        data = upload_res.json()
        assert data["success"] is True
        assert data["overview"]["total_rows"] == 15
        assert data["overview"]["total_columns"] == 7
        file_id = data["overview"]["file_id"]
        print(f"  -> /api/upload returned 200 OK with file_id: {file_id}")

        # Fetch profile
        prof_res = await client.get(f"/api/profile/{file_id}")
        assert prof_res.status_code == 200
        assert prof_res.json()["overview"]["file_id"] == file_id
        print("  -> /api/profile/{file_id} successfully fetched cached profile")

        # Upload unsupported file
        print("--- [6/6] Testing Upload Error Handling (Unsupported & Malformed) ---")
        with open(SAMPLE_DIR / "unsupported.txt", "rb") as f:
            unsup_res = await client.post(
                "/api/upload",
                files={"file": ("unsupported.txt", f, "text/plain")}
            )
        assert unsup_res.status_code == 400
        err_data = unsup_res.json()
        assert err_data["success"] is False
        assert err_data["error_type"] == "UNSUPPORTED_FILE_TYPE"
        print(f"  -> Unsupported file rejected with 400: {err_data['detail']}")

        # Upload malformed file
        with open(SAMPLE_DIR / "malformed.csv", "rb") as f:
            mal_res = await client.post(
                "/api/upload",
                files={"file": ("malformed.csv", f, "text/csv")}
            )
        assert mal_res.status_code == 400
        mal_data = mal_res.json()
        assert mal_data["success"] is False
        print(f"  -> Malformed file rejected with 400: {mal_data['detail']}")

def main():
    test_unit_profiler()
    asyncio.run(test_api_endpoints())
    print("\n=======================================================")
    print("ALL MILESTONE 01 TESTS PASSED WITH 100% ACCURACY!")
    print("=======================================================")

if __name__ == "__main__":
    main()

