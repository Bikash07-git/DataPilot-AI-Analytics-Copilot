import uvicorn
import os
import sys

if __name__ == "__main__":
    print("====================================================")
    print("🚀 Starting AI Data Studio - Milestone 01")
    print("👉 Server URL: http://127.0.0.1:8000")
    print("👉 API Docs:   http://127.0.0.1:8000/docs")
    print("====================================================")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

