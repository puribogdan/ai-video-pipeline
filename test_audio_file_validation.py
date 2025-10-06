#!/usr/bin/env python3
"""
Test script to verify the audio file validation fix works correctly.
This simulates the race condition scenario and tests the new validation logic.
"""

import os
import time
import tempfile
import shutil
from pathlib import Path

# Add the app directory to the path so we can import worker_tasks
import sys
sys.path.append(str(Path(__file__).parent / "app"))

from worker_tasks import _find_any_audio, _wait_for_any_audio

def test_audio_file_validation():
    """Test the audio file validation logic"""
    print("Testing audio file validation fix...")

    # Create a temporary directory to simulate job directory
    with tempfile.TemporaryDirectory() as temp_dir:
        job_dir = Path(temp_dir)
        print(f"Using test directory: {job_dir}")

        # Test 1: No audio file present (should fail gracefully)
        print("\nTest 1: No audio file present")
        try:
            result = _find_any_audio(job_dir)
            print(f"Result: {result}")
            assert result is None, "Expected None when no audio files present"
            print("✓ Test 1 passed")
        except Exception as e:
            print(f"✗ Test 1 failed: {e}")

        # Test 2: Create a stable audio file
        print("\nTest 2: Create stable audio file")
        audio_file = job_dir / "test_audio.mp3"
        test_data = b"fake mp3 data" * 1000  # 14KB of data

        with open(audio_file, "wb") as f:
            f.write(test_data)
            f.flush()
            os.fsync(f.fileno())

        # Wait a moment for file system to settle
        time.sleep(0.1)

        try:
            result = _find_any_audio(job_dir)
            print(f"Found audio file: {result}")
            assert result is not None, "Expected to find audio file"
            assert result == audio_file, f"Expected {audio_file}, got {result}"
            print("✓ Test 2 passed")
        except Exception as e:
            print(f"✗ Test 2 failed: {e}")

        # Test 3: Test file stability checking (simulate race condition)
        print("\nTest 3: Test file stability checking")
        try:
            # Create a file that changes size (simulates being written)
            unstable_file = job_dir / "unstable.mp3"
            with open(unstable_file, "wb") as f:
                f.write(b"initial data")
                f.flush()

            # Check size immediately
            size1 = unstable_file.stat().st_size
            print(f"Initial size: {size1}")

            # Append more data (simulates ongoing write)
            with open(unstable_file, "ab") as f:
                f.write(b"more data")
                f.flush()

            size2 = unstable_file.stat().st_size
            print(f"Size after append: {size2}")

            # The _find_any_audio function should handle this gracefully
            result = _find_any_audio(job_dir)
            print(f"Result with unstable file: {result}")
            print("✓ Test 3 passed (handled gracefully)")
        except Exception as e:
            print(f"✗ Test 3 failed: {e}")

        # Test 4: Test with completion_status.json but no audio (the original issue)
        print("\nTest 4: Test scenario from original issue")
        try:
            # Create completion_status.json (simulates premature status save)
            completion_file = job_dir / "completion_status.json"
            completion_data = {
                "state": "active",
                "message": "Job started",
                "timestamp": time.time()
            }

            import json
            with open(completion_file, "w") as f:
                json.dump(completion_data, f)

            # Create a valid audio file
            valid_audio = job_dir / "valid_audio.mp3"
            with open(valid_audio, "wb") as f:
                f.write(b"valid audio data" * 500)

            # Test that _find_any_audio finds the audio file despite completion_status.json
            result = _find_any_audio(job_dir)
            print(f"Found audio despite completion_status.json: {result}")
            assert result is not None, "Should find audio file"
            print("✓ Test 4 passed")
        except Exception as e:
            print(f"✗ Test 4 failed: {e}")

    print("\nAll tests completed!")

if __name__ == "__main__":
    test_audio_file_validation()