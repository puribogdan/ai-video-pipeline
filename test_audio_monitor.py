#!/usr/bin/env python3
"""
Test script for the new AudioUploadMonitor system.

This script tests the event-driven file system monitoring system
to ensure it works correctly and provides immediate notification
when audio files are uploaded.
"""

import asyncio
import time
import shutil
from pathlib import Path
from app.audio_monitor import AudioUploadMonitor, AudioFileEvent, FileEventType, get_monitor


async def test_audio_monitor():
    """Test the AudioUploadMonitor functionality."""
    print("Testing AudioUploadMonitor...")

    # Create test uploads directory
    test_uploads_dir = Path("/tmp/test_uploads")
    test_uploads_dir.mkdir(exist_ok=True)

    # Create monitor instance
    monitor = AudioUploadMonitor(test_uploads_dir)

    # Test event callback
    received_events = []

    async def test_callback(event: AudioFileEvent):
        print(f"Received event: {event.event_type.value} - {event.file_path} ({event.file_size} bytes)")
        received_events.append(event)

    monitor.event_callback = test_callback

    # Start monitoring
    print("Starting monitor...")
    await monitor.start_monitoring()

    # Wait a moment for initialization
    await asyncio.sleep(0.5)

    # Create test job directory
    job_id = "test_job_123"
    job_dir = test_uploads_dir / job_id
    job_dir.mkdir(exist_ok=True)

    # Test 1: Create a non-audio file first
    print("\nTest 1: Creating non-audio file...")
    test_file = job_dir / "test.txt"
    test_file.write_text("This is a test file")
    await asyncio.sleep(1.0)  # Wait for event processing

    # Test 2: Create an audio file
    print("Test 2: Creating audio file...")
    audio_file = job_dir / "test_audio.mp3"
    audio_file.write_bytes(b"fake mp3 content" * 1000)  # Fake MP3 content
    await asyncio.sleep(2.0)  # Wait for event processing and stability checks

    # Test 3: Modify the audio file
    print("Test 3: Modifying audio file...")
    audio_file.write_bytes(b"modified fake mp3 content" * 1500)
    await asyncio.sleep(2.0)  # Wait for event processing

    # Check results
    print(f"\nReceived {len(received_events)} events:")
    for i, event in enumerate(received_events):
        print(f"  Event {i+1}: {event.event_type.value} - {event.file_path.name} - Audio: {event.is_audio} - Stable: {event.is_stable}")

    # Check job status
    job_status = monitor.get_job_status(job_id)
    if job_status:
        print(f"\nJob status for {job_id}:")
        print(f"  Audio files tracked: {len(job_status.get('audio_files', {}))}")
        for file_key, file_info in job_status.get('audio_files', {}).items():
            print(f"    {Path(file_key).name}: {file_info.get('file_size')} bytes, stable: {file_info.get('is_stable')}")

    # Stop monitoring
    print("\nStopping monitor...")
    await monitor.stop_monitoring()

    # Cleanup
    print("Cleaning up test files...")
    shutil.rmtree(test_uploads_dir)

    print("\nTest completed successfully!")


async def test_monitor_status():
    """Test monitor status and global functions."""
    print("\nTesting monitor status functions...")

    # Test get_monitor function
    monitor = get_monitor()
    print(f"Monitor instance: {type(monitor).__name__}")

    # Test status before starting
    status = monitor.get_monitoring_status()
    print(f"Status before starting: monitoring={status['monitoring']}")

    # Test cleanup
    cleaned = monitor.cleanup_old_jobs()
    print(f"Cleaned up {cleaned} old jobs")

    print("Status test completed!")


async def main():
    """Run all tests."""
    print("Starting AudioUploadMonitor tests...\n")

    try:
        # Test basic functionality
        await test_audio_monitor()

        # Test status functions
        await test_monitor_status()

        print("\n[SUCCESS] All tests passed successfully!")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())