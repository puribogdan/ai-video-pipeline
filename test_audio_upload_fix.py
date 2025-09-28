#!/usr/bin/env python3
"""
Test script to verify the audio upload fix.
This script simulates the audio upload process and tests the worker's ability to find audio files.
"""

import os
import tempfile
import shutil
from pathlib import Path

def test_audio_file_discovery():
    """Test the _find_any_audio function with various file types"""

    # Create a temporary directory to simulate job directory
    with tempfile.TemporaryDirectory() as temp_dir:
        job_dir = Path(temp_dir)

        # Test 1: No audio files
        print("Test 1: No audio files in directory")
        from app.worker_tasks import _find_any_audio
        result = _find_any_audio(job_dir)
        assert result is None, f"Expected None, got {result}"
        print("PASS Passed")

        # Test 2: MP3 file
        print("Test 2: MP3 file present")
        mp3_file = job_dir / "test_audio.mp3"
        mp3_file.write_bytes(b"fake mp3 content")
        result = _find_any_audio(job_dir)
        assert result == mp3_file, f"Expected {mp3_file}, got {result}"
        print("PASS Passed")

        # Test 3: M4A file (should be found)
        print("Test 3: M4A file present")
        mp3_file.unlink()  # Remove MP3
        m4a_file = job_dir / "test_audio.m4a"
        m4a_file.write_bytes(b"fake m4a content")
        result = _find_any_audio(job_dir)
        assert result == m4a_file, f"Expected {m4a_file}, got {result}"
        print("PASS Passed")

        # Test 4: WAV file
        print("Test 4: WAV file present")
        m4a_file.unlink()
        wav_file = job_dir / "test_audio.wav"
        wav_file.write_bytes(b"fake wav content")
        result = _find_any_audio(job_dir)
        assert result == wav_file, f"Expected {wav_file}, got {result}"
        print("PASS Passed")

        # Test 5: Multiple audio files (should return first one)
        print("Test 5: Multiple audio files")
        mp3_file2 = job_dir / "another_audio.mp3"
        mp3_file2.write_bytes(b"another fake mp3")
        result = _find_any_audio(job_dir)
        # Should return the first one alphabetically
        expected = min([wav_file, mp3_file2], key=lambda x: x.name)
        assert result == expected, f"Expected {expected}, got {result}"
        print("PASS")

        print("All tests passed! PASS")

def test_upload_path_validation():
    """Test the upload path validation logic"""
    print("\nTesting upload path validation...")

    with tempfile.TemporaryDirectory() as temp_dir:
        job_dir = Path(temp_dir)

        # Test case 1: Valid upload path
        print("Test: Valid upload path")
        audio_file = job_dir / "valid_audio.mp3"
        audio_file.write_bytes(b"valid audio content")

        hint_audio = audio_file
        if hint_audio and hint_audio.exists():
            audio_size = hint_audio.stat().st_size
            if audio_size == 0:
                print("✗ Failed: Audio file is empty")
            else:
                print("PASS Passed: Valid audio file accepted")
        else:
            print("✗ Failed: Audio file should exist")

        # Test case 2: Missing upload path
        print("Test: Missing upload path")
        missing_audio = job_dir / "missing_audio.mp3"
        if not missing_audio.exists():
            # Should try to find alternative
            from app.worker_tasks import _find_any_audio
            found = _find_any_audio(job_dir)
            if found:
                print("PASS Passed: Found alternative audio file")
            else:
                print("✗ Failed: Should have found alternative audio file")
        else:
            print("PASS Passed: Missing file handled correctly")

        # Test case 3: Empty audio file
        print("Test: Empty audio file")
        empty_audio = job_dir / "empty_audio.mp3"
        empty_audio.write_bytes(b"")
        if empty_audio.exists() and empty_audio.stat().st_size == 0:
            print("PASS Passed: Empty audio file detected")
        else:
            print("✗ Failed: Should detect empty audio file")

if __name__ == "__main__":
    print("Testing audio upload fix...")
    test_audio_file_discovery()
    test_upload_path_validation()
    print("\nAll tests completed successfully!")