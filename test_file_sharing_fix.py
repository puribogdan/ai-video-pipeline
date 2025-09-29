#!/usr/bin/env python3
"""
Test script to verify that the file sharing fix works correctly.
This test simulates the upload and worker processing workflow.
"""
import os
import tempfile
import time
import shutil
from pathlib import Path

def test_file_sharing():
    """Test that files are properly shared between upload and worker processes"""

    # Use the same uploads directory as the application
    uploads_dir = Path(os.getenv("UPLOADS_DIR", "/tmp/uploads"))

    # Create a test job directory
    job_id = f"test_{int(time.time())}"
    job_dir = uploads_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Create a test audio file
    test_audio = job_dir / "test_input.mp3"
    test_content = b"test audio content " + b"x" * 1000  # 1KB test file

    with open(test_audio, "wb") as f:
        f.write(test_content)
        f.flush()
        os.fsync(f.fileno())

    print(f"[OK] Created test audio file: {test_audio}")
    print(f"[OK] File size: {test_audio.stat().st_size} bytes")
    print(f"[OK] Expected size: {len(test_content)} bytes")

    # Verify file exists and has correct size
    assert test_audio.exists(), "Test audio file was not created"
    assert test_audio.stat().st_size == len(test_content), "File size mismatch"

    # Simulate what the worker does - look for the file
    from app.worker_tasks import _find_any_audio

    found_audio = _find_any_audio(job_dir)
    if found_audio:
        print(f"[OK] Worker successfully found audio file: {found_audio}")
        print(f"[OK] Found file size: {found_audio.stat().st_size} bytes")
        assert found_audio.stat().st_size == len(test_content), "Worker found file with wrong size"
    else:
        # List what the worker actually sees
        available_files = [p.name for p in job_dir.iterdir()] if job_dir.exists() else []
        print(f"[ERROR] Worker could not find audio file")
        print(f"[ERROR] Job directory: {job_dir}")
        print(f"[ERROR] Available files: {available_files}")
        raise AssertionError(f"Audio file not found by worker. Available files: {available_files}")

    # Clean up
    shutil.rmtree(job_dir)
    print(f"[OK] Cleaned up test directory: {job_dir}")

    print("\n[SUCCESS] All tests passed! File sharing fix is working correctly.")

if __name__ == "__main__":
    test_file_sharing()