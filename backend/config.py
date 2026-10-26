import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# File Upload Settings
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv"}
ALLOWED_MIME_TYPES = {
    "text/csv",
    "application/vnd.ms-excel",
    "text/plain",
    "application/csv",
    "application/octet-stream",  # Often sent by browsers for .csv
}

# Preview / Profiling Defaults
MAX_PREVIEW_ROWS = 20
MAX_CATEGORICAL_UNIQUE = 50

