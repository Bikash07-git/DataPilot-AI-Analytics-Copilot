import os
import uuid
import re
from pathlib import Path
from typing import Tuple, Optional
try:
    from charset_normalizer import detect as detect_charset
except ImportError:
    try:
        from chardet import detect as detect_charset
    except ImportError:
        detect_charset = None
from fastapi import UploadFile, HTTPException
from backend.config import UPLOAD_DIR, MAX_FILE_SIZE_BYTES, ALLOWED_EXTENSIONS

class FileValidationError(Exception):
    def __init__(self, message: str, error_type: str = "VALIDATION_ERROR"):
        self.message = message
        self.error_type = error_type
        super().__init__(self.message)

def sanitize_filename(filename: str) -> str:
    """Sanitize original filename to prevent path traversal or unsafe characters."""
    # Strip directory components
    clean_name = os.path.basename(filename)
    # Replace unsafe characters with underscore
    clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', clean_name)
    if not clean_name:
        clean_name = "dataset.csv"
    return clean_name

def detect_encoding(file_path: Path) -> str:
    """Detect file encoding from sample bytes, defaulting to utf-8."""
    with open(file_path, "rb") as f:
        raw_sample = f.read(100000)
    
    if not raw_sample:
        return "utf-8"
        
    if detect_charset:
        try:
            result = detect_charset(raw_sample)
            encoding = result.get("encoding")
            if encoding and result.get("confidence", 0) >= 0.6:
                return encoding
        except Exception:
            pass

    # Fallback cascade
    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            raw_sample.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"

def detect_delimiter(file_path: Path, encoding: str) -> str:
    """Detect CSV delimiter (comma, semicolon, tab, pipe)."""
    try:
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            lines = [f.readline() for _ in range(5)]
            sample = "".join(lines)
            
        if not sample.strip():
            return ","
            
        delimiters = [",", ";", "\t", "|"]
        counts = {d: sample.count(d) for d in delimiters}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","
    except Exception:
        return ","

async def save_and_validate_upload(file: UploadFile) -> Tuple[str, Path, str]:
    """
    Validates file extension, size, and saves to safe temporary storage.
    Returns (file_id, file_path, original_filename).
    """
    if not file.filename:
        raise FileValidationError("No filename provided in upload.", "NO_FILENAME")
        
    orig_name = sanitize_filename(file.filename)
    suffix = Path(orig_name).suffix.lower()
    
    if suffix not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"Unsupported file type '{suffix}'. Only CSV files (.csv) are currently supported.",
            "UNSUPPORTED_FILE_TYPE"
        )
        
    # Generate unique file id and target storage path
    file_id = str(uuid.uuid4())
    stored_filename = f"{file_id}_{orig_name}"
    target_path = UPLOAD_DIR / stored_filename
    
    # Stream file to disk and monitor file size
    bytes_read = 0
    try:
        with open(target_path, "wb") as f_out:
            while True:
                chunk = await file.read(1024 * 64)  # 64KB chunks
                if not chunk:
                    break
                bytes_read += len(chunk)
                if bytes_read > MAX_FILE_SIZE_BYTES:
                    # Clean up partial file
                    f_out.close()
                    if target_path.exists():
                        target_path.unlink()
                    raise FileValidationError(
                        f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB.",
                        "FILE_TOO_LARGE"
                    )
                f_out.write(chunk)
    except FileValidationError:
        raise
    except Exception as e:
        if target_path.exists():
            target_path.unlink()
        raise FileValidationError(f"Failed to save uploaded file: {str(e)}", "SAVE_ERROR")
        
    if bytes_read == 0:
        if target_path.exists():
            target_path.unlink()
        raise FileValidationError("The uploaded file is empty (0 bytes).", "EMPTY_FILE")
        
    return file_id, target_path, orig_name
