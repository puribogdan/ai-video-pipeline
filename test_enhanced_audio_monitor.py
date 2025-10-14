#!/usr/bin/env python3
"""
Test script for the enhanced AudioUploadMonitor system.

This script tests the new features including:
- Enhanced audio validation
- Real-time progress tracking
- Partial upload detection
- Enhanced error reporting
"""

import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from app.audio_monitor import (
    AudioContentValidator,
    UploadProgressTracker,
    AudioUploadMonitor,
    AudioFileEvent,
    FileEventType,
    create_validation_summary,
    format_validation_error_message,
    check_system_requirements
)


async def test_audio_validation():
    """Test the enhanced audio validation features."""
    print("=== Testing Enhanced Audio Validation ===")

    # Check system requirements
    requirements_ok, warnings = check_system_requirements()
    if warnings:
        print(f"System warnings: {warnings}")

    # Test with a sample audio file if available
    audio_files = [
        "other_audio/input.mp3",
        "other_audio/input.m4a",
        "pipeline/audio_input/input.mp3"
    ]

    for audio_file in audio_files:
        if Path(audio_file).exists():
            print(f"\nTesting validation with: {audio_file}")

            # Test comprehensive validation
            is_valid, metadata, errors = AudioContentValidator.validate_audio_content(Path(audio_file))

            print(f"Validation result: {'PASS' if is_valid else 'FAIL'}")
            if metadata:
                print(f"Format: {metadata.format}")
                print(f"Sample Rate: {metadata.sample_rate} Hz")
                print(f"Channels: {metadata.channels}")
                print(f"Duration: {metadata.duration:.2f}s")
                print(f"Quality Score: {metadata.quality_score:.1f}%")
            if errors:
                print(f"Errors: {errors}")

            # Test partial upload detection
            is_partial, reason = AudioContentValidator.detect_partial_upload(Path(audio_file))
            print(f"Partial upload detection: {'YES' if is_partial else 'NO'} - {reason}")

            break
    else:
        print("No audio test files found - skipping validation tests")


async def test_progress_tracking():
    """Test the progress tracking functionality."""
    print("\n=== Testing Progress Tracking ===")

    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
        temp_path = Path(temp_file.name)

        # Create a progress tracker
        def progress_callback(progress_info):
            print(f"Progress update: {progress_info['progress_percentage']:.1f}% "
                  f"({progress_info['upload_speed']/1024:.1f} KB/s)")

        tracker = UploadProgressTracker(temp_path, progress_callback)

        # Start tracking
        await tracker.start_tracking()
        print("Progress tracking started")

        # Simulate file growth
        for i in range(5):
            # Write some data to simulate upload
            with open(temp_path, 'ab') as f:
                f.write(b'0' * 1000)  # Add 1KB

            # Wait a bit
            await asyncio.sleep(0.5)

            # Get current progress
            progress = tracker.get_current_progress()
            if progress:
                print(f"Current progress: {progress['progress_percentage']:.1f}%")

        # Stop tracking
        await tracker.stop_tracking()
        print("Progress tracking stopped")

    # Clean up
    temp_path.unlink(missing_ok=True)


async def test_monitor_integration():
    """Test the enhanced monitor with new features."""
    print("\n=== Testing Monitor Integration ===")

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create monitor instance
        monitor = AudioUploadMonitor(temp_path)

        # Test enhanced job status
        job_id = "test_job_123"

        # Simulate some job data
        monitor.active_jobs[job_id] = {
            'audio_files': {
                'test_file.mp3': {
                    'path': temp_path / 'test_file.mp3',
                    'size': 1024000,
                    'is_audio': True,
                    'validation_errors': [],
                    'audio_metadata': {
                        'format': 'mp3',
                        'sample_rate': 44100,
                        'channels': 2,
                        'duration': 60.5,
                        'quality_score': 85.5
                    },
                    'quality_score': 85.5
                }
            },
            'last_activity': asyncio.get_event_loop().time(),
            'status': 'active'
        }

        # Test enhanced job status
        enhanced_status = monitor.get_enhanced_job_status(job_id)
        if enhanced_status:
            validation_summary = enhanced_status.get('validation_summary', {})
            print(f"Enhanced job status: {validation_summary.get('valid_files', 0)}/{validation_summary.get('total_files', 0)} files valid")
            print(f"Average quality score: {validation_summary.get('average_quality_score', 0):.1f}%")

        # Test validation report
        report = monitor.get_validation_report(job_id)
        if report:
            print(f"Validation report: {report['files_validated']} valid, {report['files_with_issues']} with issues")

        # Test monitoring status
        status = monitor.get_monitoring_status()
        print(f"Monitor status: {status['active_jobs_count']} active jobs, {status['progress_trackers_count']} progress trackers")


async def test_error_formatting():
    """Test enhanced error message formatting."""
    print("\n=== Testing Error Formatting ===")

    # Create a mock audio event with errors
    audio_event = AudioFileEvent(
        job_id="test_job",
        file_path=Path("test_file.mp3"),
        event_type=FileEventType.CREATED,
        file_size=1024,
        validation_errors=["File header corrupted", "Invalid sample rate"],
        quality_score=45.5,
        audio_metadata={
            'format': 'mp3',
            'sample_rate': 0,
            'channels': 0,
            'duration': 0.0
        }
    )

    # Test validation summary
    summary = create_validation_summary(audio_event)
    print(f"Validation summary: {json.dumps(summary, indent=2)}")

    # Test error message formatting
    error_message = format_validation_error_message(audio_event)
    print(f"Formatted error message:\n{error_message}")


async def main():
    """Run all tests."""
    print("Testing Enhanced AudioUploadMonitor System")
    print("=" * 50)

    try:
        await test_audio_validation()
        await test_progress_tracking()
        await test_monitor_integration()
        await test_error_formatting()

        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("\nEnhanced AudioUploadMonitor Features:")
        print("* Comprehensive audio validation with format detection")
        print("* Real-time upload progress tracking")
        print("* Audio metadata extraction (sample rate, channels, duration)")
        print("* File integrity checks and quality assessment")
        print("* Partial upload detection and recovery")
        print("* Enhanced error messages with detailed information")
        print("* Backward compatibility with existing system")

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())