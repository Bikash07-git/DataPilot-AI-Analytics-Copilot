import time
import urllib.request
import json
import threading
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import uvicorn
from backend.main import app

def run_server():
    config = uvicorn.Config(app, host="127.0.0.1", port=8002, log_level="warning")
    server = uvicorn.Server(config)
    server.run()

def http_post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as res:
        return json.loads(res.read().decode())

def http_get(url: str) -> dict:
    with urllib.request.urlopen(url) as res:
        return json.loads(res.read().decode())

def main():
    print("Starting background server thread on 127.0.0.1:8002...")
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(2)

    base = "http://127.0.0.1:8002"

    # 1. Health check
    print("1. Testing GET /api/health...")
    h = http_get(f"{base}/api/health")
    assert h["status"] == "healthy"
    print("   -> OK:", h["milestones_active"])

    # 2. Frontend HTML
    print("2. Testing GET / (React SPA Frontend)...")
    with urllib.request.urlopen(base) as res:
        html = res.read().decode()
        assert "AI Data Studio" in html
        assert 'id="root"' in html
    print("   -> OK: React root and scripts delivered")

    # 3. Load Sample Dataset
    print("3. Testing POST /api/load-sample/valid_employees...")
    req = urllib.request.Request(f"{base}/api/load-sample/valid_employees", method="POST")
    with urllib.request.urlopen(req) as res:
        p = json.loads(res.read().decode())
    assert p["overview"]["total_rows"] == 15
    print("   -> OK: Sample dataset profiled (15 rows, 7 cols)")

    # 4. EDA Endpoint
    print("4. Testing GET /api/eda/sample_valid_employees...")
    eda = http_get(f"{base}/api/eda/sample_valid_employees")
    assert len(eda["numeric_summaries"]) > 0
    assert len(eda["histograms"]) > 0
    print(f"   -> OK: EDA computed ({len(eda['numeric_summaries'])} numeric summaries, {len(eda['histograms'])} histograms)")

    # 5. Column Selection Endpoint
    print("5. Testing POST /api/column-selection...")
    col_sel = http_post(f"{base}/api/column-selection", {
        "file_id": "sample_valid_employees",
        "objective_type": "employee_attrition",
        "business_question": "Which features correlate with compensation and performance?",
        "strict_quality_filter": True
    })
    assert len(col_sel["recommended_columns"]) > 0
    print("   -> OK: Recommended columns:", col_sel["recommended_columns"])

    # 6. AI Agent Endpoint
    print("6. Testing POST /api/agent/query...")
    agent_res = http_post(f"{base}/api/agent/query", {
        "file_id": "sample_valid_employees",
        "question": "What is the average salary by department?"
    })
    assert agent_res["success"] is True
    assert len(agent_res["executed_tools"]) > 0
    print("   -> OK: Agent answered:", agent_res["answer"][:60], "...")

    # 7. Transformation Preview Endpoint
    print("7. Testing POST /api/transform/preview...")
    prev = http_post(f"{base}/api/transform/preview", {
        "file_id": "sample_valid_employees",
        "transformation": {
            "transformation_type": "impute",
            "column": "salary",
            "parameters": {"method": "median"}
        }
    })
    assert prev["after_stats"]["null_count"] == 0
    print("   -> OK: Imputation preview generated, nulls reduced to 0")

    # 8. Executive Report Endpoint
    print("8. Testing GET /api/report/sample_valid_employees...")
    rep = http_get(f"{base}/api/report/sample_valid_employees")
    assert rep["health_score"] > 0
    print(f"   -> OK: Executive Report generated (Health Score: {rep['health_score']}/100)")

    print("\n=======================================================")
    print("[SUCCESS] ALL 6 MILESTONES VERIFIED OVER LIVE HTTP ENDPOINTS!")
    print("=======================================================")

if __name__ == "__main__":
    main()
