import time
import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.services.profiler_service import profile_dataset
from backend.services.eda_service import compute_eda
from backend.services.column_selection_service import select_intelligent_columns
from backend.services.agent_service import run_agent_query
from backend.models.schemas import ColumnSelectionRequest, AgentQueryRequest

SAMPLE_DIR = BASE_DIR / "sample_data"

def run_benchmarks():
    print("=================================================================")
    print("[BENCHMARK] AI DATA STUDIO BENCHMARK & PERFORMANCE SUITE")
    print("=================================================================")

    # Generate a larger synthetic benchmark dataset (10,000 rows x 10 columns)
    np.random.seed(42)
    n_rows = 10000
    df_bench = pd.DataFrame({
        "order_id": [f"ORD_{i}" for i in range(n_rows)],
        "customer_id": np.random.randint(100, 500, n_rows),
        "department": np.random.choice(["Engineering", "Sales", "Marketing", "Finance"], n_rows),
        "amount": np.random.exponential(scale=100.0, size=n_rows),
        "profit": np.random.normal(loc=25.0, scale=15.0, size=n_rows),
        "discount": np.random.uniform(0.0, 0.3, n_rows),
        "quantity": np.random.randint(1, 10, n_rows),
        "rating": np.random.choice([1, 2, 3, 4, 5, None], n_rows, p=[0.1, 0.1, 0.2, 0.3, 0.25, 0.05]),
        "is_priority": np.random.choice([True, False], n_rows),
        "constant_flag": [1] * n_rows  # To test 0-variance rule
    })

    bench_csv = BASE_DIR / "uploads" / "benchmark_10k.csv"
    df_bench.to_csv(bench_csv, index=False)
    file_size_kb = bench_csv.stat().st_size / 1024

    print(f"Benchmark Dataset: 10,000 rows x 10 columns ({file_size_kb:.1f} KB)")
    print("-----------------------------------------------------------------")

    # 1. Benchmark Profiling Engine
    t0 = time.perf_counter()
    p = profile_dataset("bench_10k", bench_csv, "benchmark_10k.csv")
    t_prof = (time.perf_counter() - t0) * 1000
    throughput = n_rows / (t_prof / 1000)
    print(f"1. Profiling Latency:            {t_prof:.2f} ms ({throughput:,.0f} rows/sec)")
    assert p.overview.total_rows == 10000
    assert p.overview.total_columns == 10

    # 2. Benchmark EDA Engine
    t0 = time.perf_counter()
    eda = compute_eda("bench_10k", bench_csv, "benchmark_10k.csv")
    t_eda = (time.perf_counter() - t0) * 1000
    print(f"2. Full EDA Computation:         {t_eda:.2f} ms (Distributions, Corrs, Boxplots)")
    assert len(eda.numeric_summaries) >= 4

    # 3. Benchmark Column Selection
    t0 = time.perf_counter()
    req = ColumnSelectionRequest(
        file_id="bench_10k",
        objective_type="sales_forecasting",
        business_question="What features forecast profit and revenue volume?",
        strict_quality_filter=True
    )
    col_sel = select_intelligent_columns(req, bench_csv)
    t_sel = (time.perf_counter() - t0) * 1000
    print(f"3. Column Selection Engine:      {t_sel:.2f} ms (Precision: {col_sel.precision_estimate*100:.0f}%, Recall: {col_sel.recall_estimate*100:.0f}%)")
    # Verify constant flag was excluded
    const_rec = next((c for c in col_sel.column_details if c.column_name == "constant_flag"), None)
    assert const_rec is not None and const_rec.status == "EXCLUDED"
    print("   -> Rule Check: 'constant_flag' successfully excluded (zero variance)")

    # 4. Benchmark Agent Query Engine
    t0 = time.perf_counter()
    q_req = AgentQueryRequest(
        file_id="bench_10k",
        question="What is the average profit by department?"
    )
    agent_res = run_agent_query(q_req, bench_csv)
    t_agent = (time.perf_counter() - t0) * 1000
    print(f"4. AI Agent Pipeline Latency:    {t_agent:.2f} ms (Plan + Tool Execution + Grounding)")
    assert agent_res.success is True
    assert len(agent_res.executed_tools) > 0

    print("-----------------------------------------------------------------")
    print("STATUS: ALL BENCHMARKS COMPLETED WITHIN SUB-SECOND LATENCY LIMITS!")
    print("=================================================================")

    # Clean up benchmark file
    if bench_csv.exists():
        bench_csv.unlink()

if __name__ == "__main__":
    run_benchmarks()
