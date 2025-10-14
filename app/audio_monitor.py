"""
Event-driven file system monitoring system for audio uploads.

This module replaces the polling-based audio detection system with an event-driven approach
using file system events (inotify/fsevents) for immediate notification when audio files are ready.
"""

import asyncio
import json
import logging
import mimetypes
import os
import struct
import subprocess
import sys
import time
import threading
import wave
from asyncio import Queue, Task
from pathlib import Path
from typing import Dict, Optional, Set, Callable, Any, Tuple
from queue import Queue as ThreadQueue
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent, FileDeletedEvent
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileEventType(Enum):
    """Types of file system events we're interested in."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class AudioFileEvent:
    """Represents an audio file event with metadata."""
    job_id: str
    file_path: Path
    event_type: FileEventType
    file_size: int
    mime_type: Optional[str] = None
    is_audio: bool = False
    timestamp: float = field(default_factory=time.time)
    is_stable: bool = False
    stability_checks: int = 0
    # Enhanced validation fields
    audio_metadata: Optional[Dict[str, Any]] = None
    validation_errors: list[str] = field(default_factory=list)
    quality_score: Optional[float] = None
    upload_progress: Optional[Dict[str, Any]] = None


class AudioFileValidator:
    """Validates audio files and checks stability."""

    AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma'}
    AUDIO_MIME_TYPES = {'audio/mpeg', 'audio/wav', 'audio/mp4', 'audio/aac', 'audio/ogg', 'audio/flac', 'audio/wma'}

    @classmethod
    def is_audio_file(cls, file_path: Path) -> tuple[bool, Optional[str]]:
        """
        Check if a file is an audio file based on extension and mime type.

        Returns:
            Tuple of (is_audio: bool, mime_type: Optional[str])
        """
        try:
            # Check by extension first
            if file_path.suffix.lower() in cls.AUDIO_EXTENSIONS:
                mime_type, _ = mimetypes.guess_type(str(file_path))
                return True, mime_type

            # Check by mime type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type and mime_type in cls.AUDIO_MIME_TYPES:
                return True, mime_type

            return False, mime_type
        except Exception as e:
            logger.warning(f"Error validating audio file {file_path}: {e}")
            return False, None

    @classmethod
    def check_file_stability(cls, file_path: Path, min_checks: int = 2) -> tuple[bool, int]:
        """
        Check if a file size is stable (not being written to).

        Returns:
            Tuple of (is_stable: bool, current_size: int)
        """
        try:
            if not file_path.exists():
                return False, 0

            current_size = file_path.stat().st_size
            if current_size == 0:
                return False, 0

            # Perform multiple size checks with small delays
            for check in range(min_checks):
                time.sleep(0.5)
                new_size = file_path.stat().st_size
                if new_size != current_size:
                    return False, new_size
                current_size = new_size

            return True, current_size
        except Exception as e:
            logger.warning(f"Error checking file stability for {file_path}: {e}")
            return False, 0


@dataclass
class AudioMetadata:
    """Audio file metadata extracted from file headers."""
    format: str
    sample_rate: int
    channels: int
    duration: float
    bit_rate: Optional[int] = None
    bit_depth: Optional[int] = None
    codec: Optional[str] = None
    file_size: int = 0
    quality_score: Optional[float] = None


class AudioContentValidator:
    """Comprehensive audio file validator with content validation and quality assessment."""

    # Audio format signatures (magic bytes)
    AUDIO_SIGNATURES = {
        b'RIFF': 'wav',
        b'ID3': 'mp3',
        b'ftypM4A': 'm4a',
        b'M4A': 'm4a',
        b'FORM': 'aiff',
        b'OggS': 'ogg',
        b'fLaC': 'flac',
    }

    # Audio format specific validation
    FORMAT_VALIDATORS = {
        'wav': '_validate_wav_file',
        'mp3': '_validate_mp3_file',
        'm4a': '_validate_m4a_file',
        'ogg': '_validate_ogg_file',
        'flac': '_validate_flac_file',
    }

    @classmethod
    def validate_audio_content(cls, file_path: Path) -> Tuple[bool, Optional[AudioMetadata], list[str]]:
        """
        Comprehensive audio file validation.

        Returns:
            Tuple of (is_valid: bool, metadata: Optional[AudioMetadata], errors: list[str])
        """
        errors = []

        try:
            # Check if file exists and has content
            if not file_path.exists():
                errors.append("File does not exist")
                return False, None, errors

            if file_path.stat().st_size == 0:
                errors.append("File is empty")
                return False, None, errors

            # Detect audio format from magic bytes
            detected_format = cls._detect_audio_format(file_path)
            if not detected_format:
                errors.append("Could not detect audio format from file signature")
                return False, None, errors

            # Validate format-specific structure
            if detected_format in cls.FORMAT_VALIDATORS:
                format_valid, format_errors = getattr(cls, cls.FORMAT_VALIDATORS[detected_format])(file_path)
                if not format_valid:
                    errors.extend(format_errors)
                    return False, None, errors

            # Extract metadata
            metadata = cls._extract_audio_metadata(file_path, detected_format)
            if not metadata:
                errors.append("Could not extract audio metadata")
                return False, None, errors

            # Perform quality assessment
            quality_score = cls._assess_audio_quality(metadata, file_path)

            # Update metadata with quality score
            metadata.quality_score = quality_score

            return True, metadata, errors

        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            logger.warning(f"Error validating audio content {file_path}: {e}")
            return False, None, errors

    @classmethod
    def _detect_audio_format(cls, file_path: Path) -> Optional[str]:
        """Detect audio format from file signature (magic bytes)."""
        try:
            with open(file_path, 'rb') as f:
                # Read first 12 bytes for signature detection
                header = f.read(12)

                # Check for each known signature
                for signature, format_type in cls.AUDIO_SIGNATURES.items():
                    if header.startswith(signature):
                        return format_type

                # Special case for MP3 (ID3 tag might be at beginning)
                if len(header) >= 3 and header[0] == 0x49 and header[1] == 0x44 and header[2] == 0x33:
                    return 'mp3'

                # Check for MPEG audio without ID3
                if len(header) >= 2:
                    # MPEG audio frames start with 0xFFF or 0xFFE
                    if header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
                        return 'mp3'

        except Exception as e:
            logger.warning(f"Error detecting format for {file_path}: {e}")

        return None

    @classmethod
    def _validate_wav_file(cls, file_path: Path) -> Tuple[bool, list[str]]:
        """Validate WAV file structure."""
        errors = []
        try:
            with wave.open(str(file_path), 'rb') as wav_file:
                # Check basic WAV structure
                if wav_file.getnchannels() < 1:
                    errors.append("Invalid number of channels in WAV file")
                if wav_file.getsampwidth() not in [1, 2, 3, 4]:
                    errors.append("Invalid sample width in WAV file")
                if wav_file.getframerate() <= 0:
                    errors.append("Invalid sample rate in WAV file")
        except Exception as e:
            errors.append(f"WAV validation failed: {str(e)}")
            return False, errors

        return len(errors) == 0, errors

    @classmethod
    def _validate_mp3_file(cls, file_path: Path) -> Tuple[bool, list[str]]:
        """Validate MP3 file structure."""
        errors = []
        try:
            # Use ffprobe to validate MP3
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(file_path)],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                errors.append(f"MP3 validation failed: {result.stderr}")
                return False, errors

        except subprocess.TimeoutExpired:
            errors.append("MP3 validation timed out")
            return False, errors
        except FileNotFoundError:
            # ffprobe not available, skip detailed validation
            logger.warning("ffprobe not available for MP3 validation")
        except Exception as e:
            errors.append(f"MP3 validation error: {str(e)}")
            return False, errors

        return len(errors) == 0, errors

    @classmethod
    def _validate_m4a_file(cls, file_path: Path) -> Tuple[bool, list[str]]:
        """Validate M4A file structure."""
        errors = []
        try:
            # Use ffprobe to validate M4A
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(file_path)],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                errors.append(f"M4A validation failed: {result.stderr}")
                return False, errors

        except subprocess.TimeoutExpired:
            errors.append("M4A validation timed out")
            return False, errors
        except FileNotFoundError:
            logger.warning("ffprobe not available for M4A validation")
        except Exception as e:
            errors.append(f"M4A validation error: {str(e)}")
            return False, errors

        return len(errors) == 0, errors

    @classmethod
    def _validate_ogg_file(cls, file_path: Path) -> Tuple[bool, list[str]]:
        """Validate OGG file structure."""
        errors = []
        try:
            # Use ffprobe to validate OGG
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(file_path)],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                errors.append(f"OGG validation failed: {result.stderr}")
                return False, errors

        except subprocess.TimeoutExpired:
            errors.append("OGG validation timed out")
            return False, errors
        except FileNotFoundError:
            logger.warning("ffprobe not available for OGG validation")
        except Exception as e:
            errors.append(f"OGG validation error: {str(e)}")
            return False, errors

        return len(errors) == 0, errors

    @classmethod
    def _validate_flac_file(cls, file_path: Path) -> Tuple[bool, list[str]]:
        """Validate FLAC file structure."""
        errors = []
        try:
            # Use ffprobe to validate FLAC
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(file_path)],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                errors.append(f"FLAC validation failed: {result.stderr}")
                return False, errors

        except subprocess.TimeoutExpired:
            errors.append("FLAC validation timed out")
            return False, errors
        except FileNotFoundError:
            logger.warning("ffprobe not available for FLAC validation")
        except Exception as e:
            errors.append(f"FLAC validation error: {str(e)}")
            return False, errors

        return len(errors) == 0, errors

    @classmethod
    def _extract_audio_metadata(cls, file_path: Path, format_type: str) -> Optional[AudioMetadata]:
        """Extract comprehensive audio metadata."""
        try:
            # Use ffprobe for detailed metadata extraction
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', str(file_path)],
                capture_output=True, text=True, timeout=15
            )

            if result.returncode != 0:
                logger.warning(f"ffprobe failed for metadata extraction: {result.stderr}")
                return cls._extract_basic_metadata(file_path, format_type)

            import json
            data = json.loads(result.stdout)

            # Extract from ffprobe output
            if 'streams' in data and len(data['streams']) > 0:
                stream = data['streams'][0]  # Get first audio stream

                metadata = AudioMetadata(
                    format=format_type,
                    sample_rate=int(stream.get('sample_rate', 0)),
                    channels=int(stream.get('channels', 0)),
                    duration=float(data.get('format', {}).get('duration', 0)),
                    bit_rate=int(data.get('format', {}).get('bit_rate', 0)) if data.get('format', {}).get('bit_rate') else None,
                    codec=stream.get('codec_name'),
                    file_size=file_path.stat().st_size
                )

                return metadata

        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Error extracting metadata with ffprobe: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error in metadata extraction: {e}")

        # Fallback to basic metadata extraction
        return cls._extract_basic_metadata(file_path, format_type)

    @classmethod
    def _extract_basic_metadata(cls, file_path: Path, format_type: str) -> Optional[AudioMetadata]:
        """Extract basic metadata when ffprobe is not available."""
        try:
            file_size = file_path.stat().st_size

            # Basic metadata with defaults
            metadata = AudioMetadata(
                format=format_type,
                sample_rate=44100,  # Default sample rate
                channels=2,         # Default stereo
                duration=0.0,       # Unknown duration
                file_size=file_size
            )

            # Try to extract from WAV files
            if format_type == 'wav':
                try:
                    with wave.open(str(file_path), 'rb') as wav_file:
                        metadata.sample_rate = wav_file.getframerate()
                        metadata.channels = wav_file.getnchannels()
                        metadata.duration = wav_file.getnframes() / wav_file.getframerate()
                        metadata.bit_depth = wav_file.getsampwidth() * 8
                except Exception as e:
                    logger.warning(f"Error extracting WAV metadata: {e}")

            return metadata

        except Exception as e:
            logger.warning(f"Error in basic metadata extraction: {e}")
            return None

    @classmethod
    def _assess_audio_quality(cls, metadata: AudioMetadata, file_path: Path) -> float:
        """Assess audio quality based on metadata and file characteristics."""
        score = 100.0

        # Penalize low sample rates
        if metadata.sample_rate < 8000:
            score -= 50
        elif metadata.sample_rate < 22050:
            score -= 20
        elif metadata.sample_rate < 44100:
            score -= 10

        # Penalize mono audio (unless it's intended for mono)
        if metadata.channels == 1:
            score -= 5

        # Penalize very short files (might be corrupted)
        if metadata.duration > 0 and metadata.duration < 0.1:
            score -= 30

        # Check file size vs duration ratio for potential corruption
        if metadata.duration > 0:
            expected_size = metadata.duration * metadata.sample_rate * metadata.channels * 2  # Assume 16-bit
            size_ratio = file_path.stat().st_size / expected_size if expected_size > 0 else 1

            # If file is much smaller or larger than expected, it might be corrupted
            if size_ratio < 0.5 or size_ratio > 2.0:
                score -= 25

        return max(0.0, score)

    @classmethod
    def detect_partial_upload(cls, file_path: Path, expected_size: Optional[int] = None) -> Tuple[bool, str]:
        """
        Detect if an audio file upload is partial or corrupted.

        Returns:
            Tuple of (is_partial: bool, reason: str)
        """
        try:
            if not file_path.exists():
                return True, "File does not exist"

            current_size = file_path.stat().st_size

            if current_size == 0:
                return True, "File is empty"

            # Validate audio content to check for corruption
            is_valid, metadata, errors = cls.validate_audio_content(file_path)

            if not is_valid:
                return True, f"Audio validation failed: {', '.join(errors)}"

            # Check if file size matches expected size
            if expected_size and abs(current_size - expected_size) > 1024:  # Allow 1KB variance
                return True, f"File size mismatch: expected {expected_size}, got {current_size}"

            # Check for truncated audio (very short duration)
            if metadata and metadata.duration > 0 and metadata.duration < 0.5:
                return True, f"Audio duration too short: {metadata.duration:.2f}s"

            return False, "Upload appears complete"

        except Exception as e:
            return True, f"Detection error: {str(e)}"

    @classmethod
    def recover_partial_upload(cls, file_path: Path) -> Tuple[bool, str]:
        """
        Attempt to recover a partial upload by truncating corrupted data.

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # This is a basic implementation - in practice, you might want more sophisticated recovery
            logger.info(f"Attempting recovery for partial upload: {file_path}")

            # For now, we'll just validate and report status
            is_partial, reason = cls.detect_partial_upload(file_path)

            if is_partial:
                logger.warning(f"Partial upload detected for {file_path}: {reason}")
                return False, f"Cannot recover: {reason}"
            else:
                return True, "Upload is complete and valid"

        except Exception as e:
            return False, f"Recovery failed: {str(e)}"


class UploadProgressTracker:
    """Tracks upload progress and provides real-time monitoring."""

    def __init__(self, file_path: Path, callback: Optional[Callable] = None, async_callback: Optional[Callable] = None):
        self.file_path = file_path
        self.callback = callback  # Synchronous callback
        self.async_callback = async_callback  # Asynchronous callback
        self.start_time = time.time()
        self.last_size = 0
        self.last_check_time = self.start_time
        self.size_history: list[Tuple[float, int]] = []
        self._monitoring = False
        self._monitor_task: Optional[Task] = None

    async def start_tracking(self) -> None:
        """Start progress tracking."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_progress())

    async def stop_tracking(self) -> None:
        """Stop progress tracking."""
        if not self._monitoring:
            return

        self._monitoring = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    async def _monitor_progress(self) -> None:
        """Monitor file size changes and calculate progress metrics."""
        try:
            while self._monitoring and self.file_path.exists():
                current_time = time.time()

                if not self.file_path.exists():
                    break

                try:
                    current_size = self.file_path.stat().st_size
                except (OSError, FileNotFoundError):
                    break

                # Record size history for trend analysis
                self.size_history.append((current_time, current_size))

                # Keep only last 60 seconds of history
                cutoff_time = current_time - 60
                self.size_history = [(t, s) for t, s in self.size_history if t > cutoff_time]

                # Calculate progress metrics
                progress_info = self._calculate_progress(current_size, current_time)

                # Call progress callback if provided
                if progress_info:
                    if self.async_callback:
                        try:
                            # Schedule async callback in event loop
                            asyncio.create_task(self.async_callback(progress_info))
                        except Exception as e:
                            logger.warning(f"Error in async progress callback: {e}")
                    elif self.callback:
                        try:
                            # Call synchronous callback directly
                            self.callback(progress_info)
                        except Exception as e:
                            logger.warning(f"Error in sync progress callback: {e}")

                # Wait before next check
                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in progress monitoring: {e}")

    def _calculate_progress(self, current_size: int, current_time: float) -> Optional[Dict[str, Any]]:
        """Calculate progress metrics."""
        if not self.size_history or len(self.size_history) < 2:
            return None

        # Calculate upload speed (bytes per second)
        recent_history = [(t, s) for t, s in self.size_history if current_time - t <= 5]  # Last 5 seconds

        if len(recent_history) >= 2:
            oldest_time, oldest_size = recent_history[0]
            newest_time, newest_size = recent_history[-1]

            if newest_time > oldest_time:
                time_diff = newest_time - oldest_time
                size_diff = newest_size - oldest_size
                upload_speed = size_diff / time_diff if time_diff > 0 else 0

                # Calculate estimated completion time
                eta = 0
                if upload_speed > 0 and self.last_size > 0:
                    remaining_bytes = current_size - self.last_size
                    eta = remaining_bytes / upload_speed if remaining_bytes > 0 else 0

                return {
                    'file_path': str(self.file_path),
                    'current_size': current_size,
                    'upload_speed': upload_speed,
                    'eta_seconds': eta,
                    'progress_percentage': min(100.0, (current_size / max(current_size, 1)) * 100),
                    'timestamp': current_time,
                    'is_uploading': upload_speed > 0
                }

        return None

    def get_current_progress(self) -> Optional[Dict[str, Any]]:
        """Get current progress information."""
        if not self.file_path.exists():
            return None

        try:
            current_size = self.file_path.stat().st_size
            current_time = time.time()

            return self._calculate_progress(current_size, current_time)
        except (OSError, FileNotFoundError):
            return None


class AudioUploadEventHandler(FileSystemEventHandler):
    """Handles file system events for audio uploads."""

    def __init__(self, monitor: 'AudioUploadMonitor'):
        self.monitor = monitor
        self._file_stability_tasks: Dict[str, Task] = {}
        self._progress_trackers: Dict[str, UploadProgressTracker] = {}
        # Thread-safe queue for events from watchdog thread to main event loop
        self._event_queue = ThreadQueue()

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            self._event_queue.put((event.src_path, FileEventType.CREATED))

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            self._event_queue.put((event.src_path, FileEventType.MODIFIED))

    def on_deleted(self, event):
        """Handle file deletion events."""
        if not event.is_directory:
            self._event_queue.put((event.src_path, FileEventType.DELETED))

    def get_pending_events(self) -> list:
        """Get all pending events from the queue (non-blocking)."""
        events = []
        while True:
            try:
                event = self._event_queue.get_nowait()
                events.append(event)
            except:
                break
        return events

    async def _handle_file_event(self, file_path: str, event_type: FileEventType):
        """Process a file event and validate audio files with enhanced validation."""
        try:
            path = Path(file_path)
            job_id = self._extract_job_id(path)

            if not job_id:
                return

            # Validate if this is an audio file
            is_audio, mime_type = AudioFileValidator.is_audio_file(path)

            if not is_audio and event_type == FileEventType.CREATED:
                # For new files, only process if they're audio files
                logger.debug(f"Non-audio file created: {path}")
                return

            # Get file size
            file_size = path.stat().st_size if path.exists() else 0

            # Create event object with enhanced fields
            audio_event = AudioFileEvent(
                job_id=job_id,
                file_path=path,
                event_type=event_type,
                file_size=file_size,
                mime_type=mime_type,
                is_audio=is_audio
            )

            # Perform enhanced validation for audio files
            if is_audio:
                await self._perform_enhanced_validation(audio_event)

                # Set up progress tracking for audio files
                await self._setup_progress_tracking(audio_event)

                # Check stability before processing
                await self._ensure_file_stability(audio_event)
            else:
                # For non-audio files, process immediately
                await self.monitor._process_audio_event(audio_event)

        except Exception as e:
            logger.error(f"Error handling file event {file_path}: {e}")

    def _extract_job_id(self, file_path: Path) -> Optional[str]:
        """Extract job ID from file path."""
        try:
            # Assume the file is in a job directory: .../uploads/{job_id}/...
            parts = file_path.parts
            uploads_index = None

            for i, part in enumerate(parts):
                if part == 'uploads':
                    uploads_index = i
                    break

            if uploads_index is not None and uploads_index + 1 < len(parts):
                return parts[uploads_index + 1]

            return None
        except Exception:
            return None

    async def _ensure_file_stability(self, audio_event: AudioFileEvent):
        """Ensure audio file is stable before processing."""
        file_path_str = str(audio_event.file_path)

        # Cancel any existing stability check for this file
        if file_path_str in self._file_stability_tasks:
            self._file_stability_tasks[file_path_str].cancel()

        # Create new stability check task
        task = asyncio.create_task(self._check_stability_and_process(audio_event))
        self._file_stability_tasks[file_path_str] = task

    async def _check_stability_and_process(self, audio_event: AudioFileEvent):
        """Check file stability and process when ready."""
        try:
            max_checks = 10
            check_interval = 0.5

            for check_num in range(max_checks):
                is_stable, current_size = AudioFileValidator.check_file_stability(
                    audio_event.file_path, min_checks=2
                )

                if is_stable:
                    audio_event.is_stable = True
                    audio_event.file_size = current_size
                    audio_event.stability_checks = check_num + 1
                    logger.info(f"Audio file {audio_event.file_path} is stable after {check_num + 1} checks ({current_size} bytes)")
                    await self.monitor._process_audio_event(audio_event)
                    return

                if check_num == max_checks - 1:
                    # File is still not stable after max checks, process anyway
                    logger.warning(f"Processing audio file {audio_event.file_path} after {max_checks} checks (still unstable)")
                    audio_event.stability_checks = check_num + 1
                    await self.monitor._process_audio_event(audio_event)
                    return

                await asyncio.sleep(check_interval)

        except asyncio.CancelledError:
            logger.debug(f"Stability check cancelled for {audio_event.file_path}")
        except Exception as e:
            logger.error(f"Error in stability check for {audio_event.file_path}: {e}")
        finally:
            # Clean up task reference
            file_path_str = str(audio_event.file_path)
            if file_path_str in self._file_stability_tasks:
                del self._file_stability_tasks[file_path_str]

    async def _perform_enhanced_validation(self, audio_event: AudioFileEvent):
        """Perform comprehensive audio validation."""
        try:
            # Perform content validation
            is_valid, metadata, errors = AudioContentValidator.validate_audio_content(audio_event.file_path)

            if is_valid and metadata:
                # Convert metadata to dictionary for storage
                audio_event.audio_metadata = {
                    'format': metadata.format,
                    'sample_rate': metadata.sample_rate,
                    'channels': metadata.channels,
                    'duration': metadata.duration,
                    'bit_rate': metadata.bit_rate,
                    'bit_depth': metadata.bit_depth,
                    'codec': metadata.codec,
                    'file_size': metadata.file_size,
                    'quality_score': metadata.quality_score
                }
                audio_event.quality_score = metadata.quality_score
                logger.info(
                    f"Audio validation successful: {audio_event.file_path.name} - "
                    f"Format: {metadata.format}, Sample Rate: {metadata.sample_rate}Hz, "
                    f"Channels: {metadata.channels}, Duration: {metadata.duration:.2f}s, "
                    f"Quality Score: {metadata.quality_score:.1f}%"
                )
            else:
                audio_event.validation_errors = errors
                logger.warning(
                    f"Audio validation failed for {audio_event.file_path.name}: {', '.join(errors)}"
                )

        except Exception as e:
            error_msg = f"Enhanced validation error: {str(e)}"
            audio_event.validation_errors.append(error_msg)
            logger.error(f"Error in enhanced validation for {audio_event.file_path}: {e}")

    async def _setup_progress_tracking(self, audio_event: AudioFileEvent):
        """Set up progress tracking for the audio file."""
        try:
            file_path_str = str(audio_event.file_path)

            # Stop existing tracker if any
            if file_path_str in self._progress_trackers:
                await self._progress_trackers[file_path_str].stop_tracking()

            # Create new progress tracker
                progress_tracker = UploadProgressTracker(
                    audio_event.file_path,
                    async_callback=self._progress_callback
                )

            self._progress_trackers[file_path_str] = progress_tracker
            await progress_tracker.start_tracking()

            logger.debug(f"Progress tracking started for {audio_event.file_path}")

        except Exception as e:
            logger.error(f"Error setting up progress tracking for {audio_event.file_path}: {e}")

    def _progress_callback(self, progress_info: Dict[str, Any]):
        """Handle progress updates."""
        try:
            # Update the corresponding audio event with progress info
            file_path_str = progress_info['file_path']
            job_id = self._extract_job_id(Path(file_path_str))

            if job_id:
                # Schedule async update in the event loop
                asyncio.create_task(self.monitor._update_audio_event_progress(job_id, file_path_str, progress_info))

        except Exception as e:
            logger.error(f"Error in progress callback: {e}")

    def cleanup_progress_trackers(self):
        """Clean up all progress trackers."""
        for file_path, tracker in self._progress_trackers.items():
            try:
                asyncio.create_task(tracker.stop_tracking())
            except Exception as e:
                logger.warning(f"Error stopping progress tracker for {file_path}: {e}")

        self._progress_trackers.clear()


class AudioUploadMonitor:
    """
    Event-driven audio upload monitor that replaces polling-based detection.

    Uses file system events to immediately detect when audio files are ready for processing,
    eliminating the need for polling loops and providing better performance and responsiveness.
    """

    def __init__(self, uploads_dir: Path, event_callback: Optional[Callable] = None):
        """
        Initialize the audio upload monitor.

        Args:
            uploads_dir: Directory to monitor for audio uploads
            event_callback: Optional callback function for processing audio events
        """
        self.uploads_dir = Path(uploads_dir)
        self.event_callback = event_callback
        self.observer = Observer()
        self.event_handler = AudioUploadEventHandler(self)
        self.event_queue: Queue = Queue(maxsize=1000)
        self._monitoring = False
        self._processor_task: Optional[Task] = None

        # Track active jobs and their audio files
        self.active_jobs: Dict[str, Dict[str, Any]] = {}

        # Ensure uploads directory exists
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"AudioUploadMonitor initialized for directory: {self.uploads_dir}")

    async def start_monitoring(self) -> None:
        """Start monitoring the uploads directory for file system events."""
        if self._monitoring:
            logger.warning("AudioUploadMonitor is already running")
            return

        try:
            # Start the file system observer
            self.observer.schedule(self.event_handler, str(self.uploads_dir), recursive=True)
            self.observer.start()
            self._monitoring = True

            # Start the event processor
            self._processor_task = asyncio.create_task(self._process_events())

            logger.info(f"AudioUploadMonitor started monitoring: {self.uploads_dir}")

        except Exception as e:
            logger.error(f"Failed to start AudioUploadMonitor: {e}")
            raise

    async def stop_monitoring(self) -> None:
        """Stop monitoring and cleanup resources."""
        if not self._monitoring:
            return

        logger.info("Stopping AudioUploadMonitor...")
        self._monitoring = False

        # Cancel all stability check tasks
        if hasattr(self.event_handler, '_file_stability_tasks'):
            for task in self.event_handler._file_stability_tasks.values():
                task.cancel()
            self.event_handler._file_stability_tasks.clear()

        # Clean up progress trackers
        if hasattr(self.event_handler, 'cleanup_progress_trackers'):
            self.event_handler.cleanup_progress_trackers()

        # Stop event processor
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error stopping event processor: {e}")
            finally:
                self._processor_task = None

        # Stop file system observer with timeout
        try:
            self.observer.stop()
            # Wait for observer to stop with timeout
            self.observer.join(timeout=5.0)
        except Exception as e:
            logger.error(f"Error stopping file system observer: {e}")
            # Force stop if normal stop fails
            try:
                if self.observer.is_alive():
                    logger.warning("Force stopping observer...")
                    # Note: In a real scenario, you might need more aggressive cleanup
            except:
                pass

        # Clear event queue
        try:
            while not self.event_queue.empty():
                try:
                    self.event_queue.get_nowait()
                except:
                    break
        except Exception as e:
            logger.error(f"Error clearing event queue: {e}")

        # Cleanup old jobs
        cleaned_count = self.cleanup_old_jobs(max_age_seconds=60)  # 1 minute for immediate cleanup
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old jobs during shutdown")

        logger.info("AudioUploadMonitor stopped successfully")

    async def _process_events(self) -> None:
        """Process audio events from both queues."""
        try:
            while self._monitoring:
                # Check for events from watchdog thread (non-blocking)
                watchdog_events = self.event_handler.get_pending_events()
                for file_path, event_type in watchdog_events:
                    try:
                        await self._handle_file_event(file_path, event_type)
                    except Exception as e:
                        logger.error(f"Error handling file event {file_path}: {e}")

                # Check asyncio queue for processed events
                try:
                    audio_event = self.event_queue.get_nowait()
                    await self._process_audio_event(audio_event)
                except:
                    pass  # No events in asyncio queue

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.debug("Event processor cancelled")
        except Exception as e:
            logger.error(f"Fatal error in event processor: {e}")

    async def _handle_file_event(self, file_path: str, event_type: FileEventType) -> None:
        """Handle a file event and validate audio files."""
        try:
            path = Path(file_path)
            job_id = self._extract_job_id(path)

            if not job_id:
                return

            # Validate if this is an audio file
            is_audio, mime_type = AudioFileValidator.is_audio_file(path)

            if not is_audio and event_type == FileEventType.CREATED:
                # For new files, only process if they're audio files
                logger.debug(f"Non-audio file created: {path}")
                return

            # Get file size
            file_size = path.stat().st_size if path.exists() else 0

            # Create event object
            audio_event = AudioFileEvent(
                job_id=job_id,
                file_path=path,
                event_type=event_type,
                file_size=file_size,
                mime_type=mime_type,
                is_audio=is_audio
            )

            # For audio files, check stability before processing
            if is_audio:
                await self._ensure_file_stability(audio_event)
            else:
                # For non-audio files, process immediately
                await self._process_audio_event(audio_event)

        except Exception as e:
            logger.error(f"Error handling file event {file_path}: {e}")

    def _extract_job_id(self, file_path: Path) -> Optional[str]:
        """Extract job ID from file path."""
        try:
            # Assume the file is in a job directory: .../uploads/{job_id}/...
            parts = file_path.parts
            uploads_index = None

            for i, part in enumerate(parts):
                if part == 'uploads':
                    uploads_index = i
                    break

            if uploads_index is not None and uploads_index + 1 < len(parts):
                return parts[uploads_index + 1]

            return None
        except Exception:
            return None

    async def _ensure_file_stability(self, audio_event: AudioFileEvent) -> None:
        """Ensure audio file is stable before processing."""
        file_path_str = str(audio_event.file_path)

        # Cancel any existing stability check for this file
        if file_path_str in self.event_handler._file_stability_tasks:
            self.event_handler._file_stability_tasks[file_path_str].cancel()

        # Create new stability check task
        task = asyncio.create_task(self._check_stability_and_process(audio_event))
        self.event_handler._file_stability_tasks[file_path_str] = task

    async def _check_stability_and_process(self, audio_event: AudioFileEvent) -> None:
        """Check file stability and process when ready."""
        try:
            max_checks = 10
            check_interval = 0.5

            for check_num in range(max_checks):
                is_stable, current_size = AudioFileValidator.check_file_stability(
                    audio_event.file_path, min_checks=2
                )

                if is_stable:
                    audio_event.is_stable = True
                    audio_event.file_size = current_size
                    audio_event.stability_checks = check_num + 1
                    logger.info(f"Audio file {audio_event.file_path} is stable after {check_num + 1} checks ({current_size} bytes)")
                    await self._process_audio_event(audio_event)
                    return

                if check_num == max_checks - 1:
                    # File is still not stable after max checks, process anyway
                    logger.warning(f"Processing audio file {audio_event.file_path} after {max_checks} checks (still unstable)")
                    audio_event.stability_checks = check_num + 1
                    await self._process_audio_event(audio_event)
                    return

                await asyncio.sleep(check_interval)

        except asyncio.CancelledError:
            logger.debug(f"Stability check cancelled for {audio_event.file_path}")
        except Exception as e:
            logger.error(f"Error in stability check for {audio_event.file_path}: {e}")
        finally:
            # Clean up task reference
            file_path_str = str(audio_event.file_path)
            if file_path_str in self.event_handler._file_stability_tasks:
                del self.event_handler._file_stability_tasks[file_path_str]

    async def _process_audio_event(self, audio_event: AudioFileEvent) -> None:
        """Process an audio file event."""
        try:
            job_id = audio_event.job_id

            # Update job tracking
            if job_id not in self.active_jobs:
                self.active_jobs[job_id] = {
                    'audio_files': {},
                    'last_activity': time.time(),
                    'status': 'active'
                }

            job_info = self.active_jobs[job_id]

            # Update audio file information with enhanced validation data
            file_key = str(audio_event.file_path)
            file_info = {
                'path': audio_event.file_path,
                'size': audio_event.file_size,
                'mime_type': audio_event.mime_type,
                'is_audio': audio_event.is_audio,
                'is_stable': audio_event.is_stable,
                'last_event': audio_event.event_type.value,
                'timestamp': audio_event.timestamp,
                'stability_checks': audio_event.stability_checks
            }

            # Add enhanced validation information if available
            if audio_event.audio_metadata:
                file_info['audio_metadata'] = audio_event.audio_metadata
            if audio_event.quality_score is not None:
                file_info['quality_score'] = audio_event.quality_score
            if audio_event.validation_errors:
                file_info['validation_errors'] = audio_event.validation_errors
            if audio_event.upload_progress:
                file_info['upload_progress'] = audio_event.upload_progress

            job_info['audio_files'][file_key] = file_info

            job_info['last_activity'] = time.time()

            # Log the event
            logger.info(
                f"Audio event: {audio_event.event_type.value} - "
                f"Job: {job_id}, File: {audio_event.file_path.name}, "
                f"Size: {audio_event.file_size} bytes, "
                f"Stable: {audio_event.is_stable}"
            )

            # Call event callback if provided
            if self.event_callback:
                try:
                    await self.event_callback(audio_event)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")

        except Exception as e:
            logger.error(f"Error processing audio event: {e}")

    async def _update_audio_event_progress(self, job_id: str, file_path: str, progress_info: Dict[str, Any]):
        """Update audio event with progress information."""
        try:
            job_info = self.active_jobs.get(job_id)
            if not job_info:
                return

            # Update progress info for the specific file
            if 'audio_files' in job_info and file_path in job_info['audio_files']:
                job_info['audio_files'][file_path]['upload_progress'] = progress_info
                job_info['last_activity'] = time.time()

                logger.debug(
                    f"Updated progress for {Path(file_path).name}: "
                    f"{progress_info.get('progress_percentage', 0):.1f}% complete, "
                    f"Speed: {progress_info.get('upload_speed', 0)/1024:.1f} KB/s"
                )

        except Exception as e:
            logger.error(f"Error updating audio event progress: {e}")

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific job."""
        return self.active_jobs.get(job_id)

    def get_audio_file_info(self, job_id: str, file_path: Path) -> Optional[Dict[str, Any]]:
        """Get information about a specific audio file in a job."""
        job_info = self.active_jobs.get(job_id)
        if not job_info:
            return None

        file_key = str(file_path)
        return job_info['audio_files'].get(file_key)

    def list_active_jobs(self) -> Dict[str, Dict[str, Any]]:
        """List all active jobs being monitored."""
        return self.active_jobs.copy()

    def cleanup_old_jobs(self, max_age_seconds: float = 3600) -> int:
        """Clean up old job entries that haven't had activity."""
        current_time = time.time()
        jobs_to_remove = []

        for job_id, job_info in self.active_jobs.items():
            if current_time - job_info['last_activity'] > max_age_seconds:
                jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self.active_jobs[job_id]
            logger.debug(f"Cleaned up old job: {job_id}")

        return len(jobs_to_remove)

    async def handle_error(self, error: Exception, context: str = "") -> None:
        """Handle errors with appropriate logging and recovery."""
        error_msg = f"AudioUploadMonitor error {context}: {error}"
        logger.error(error_msg)

        # Check if this is a critical error that requires restart
        critical_errors = (
            OSError,  # File system errors
            RuntimeError,  # Runtime errors
            asyncio.CancelledError,  # Cancellation errors
        )

        if isinstance(error, critical_errors):
            logger.critical(f"Critical error detected: {error}")
            # In a production system, you might want to implement automatic restart logic here
            # For now, we'll just log and continue

        # Attempt to recover from non-critical errors
        if not isinstance(error, asyncio.CancelledError):
            try:
                # Wait a bit before attempting recovery
                await asyncio.sleep(1.0)
                logger.info("Attempting error recovery...")
                # Recovery logic could be added here if needed
            except Exception as recovery_error:
                logger.error(f"Error recovery failed: {recovery_error}")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get detailed monitoring status for debugging."""
        return {
            'monitoring': self._monitoring,
            'uploads_dir': str(self.uploads_dir),
            'active_jobs_count': len(self.active_jobs),
            'event_queue_size': self.event_queue.qsize(),
            'observer_running': self.observer.is_alive() if hasattr(self.observer, 'is_alive') else False,
            'processor_task_running': self._processor_task is not None and not self._processor_task.done(),
            'stability_tasks_count': len(getattr(self.event_handler, '_file_stability_tasks', {})),
            'progress_trackers_count': len(getattr(self.event_handler, '_progress_trackers', {})),
            'active_jobs': list(self.active_jobs.keys())
        }

    def get_enhanced_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get enhanced job status with validation and progress details."""
        job_info = self.active_jobs.get(job_id)
        if not job_info:
            return None

        enhanced_status = dict(job_info)  # Copy existing job info

        # Add validation summary
        audio_files = job_info.get('audio_files', {})
        validation_summary = {
            'total_files': len(audio_files),
            'valid_files': 0,
            'files_with_errors': 0,
            'files_with_metadata': 0,
            'average_quality_score': 0.0,
            'files_with_progress': 0
        }

        quality_scores = []
        for file_info in audio_files.values():
            if file_info.get('is_audio', False):
                if not file_info.get('validation_errors'):
                    validation_summary['valid_files'] += 1
                else:
                    validation_summary['files_with_errors'] += 1

                if file_info.get('audio_metadata'):
                    validation_summary['files_with_metadata'] += 1
                    quality_score = file_info.get('quality_score')
                    if quality_score is not None:
                        quality_scores.append(quality_score)

                if file_info.get('upload_progress'):
                    validation_summary['files_with_progress'] += 1

        if quality_scores:
            validation_summary['average_quality_score'] = sum(quality_scores) / len(quality_scores)

        enhanced_status['validation_summary'] = validation_summary

        # Add recent errors
        recent_errors = []
        for file_info in audio_files.values():
            if file_info.get('validation_errors'):
                recent_errors.extend(file_info['validation_errors'])

        enhanced_status['recent_errors'] = recent_errors[-10:]  # Last 10 errors

        return enhanced_status

    def get_validation_report(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Generate a detailed validation report for a job."""
        job_info = self.active_jobs.get(job_id)
        if not job_info:
            return None

        report = {
            'job_id': job_id,
            'timestamp': time.time(),
            'files_validated': 0,
            'files_with_issues': 0,
            'validation_details': []
        }

        for file_path, file_info in job_info.get('audio_files', {}).items():
            if file_info.get('is_audio', False):
                file_report = {
                    'file_path': str(file_info['path']),
                    'file_size': file_info['size'],
                    'is_valid': not file_info.get('validation_errors', []),
                    'quality_score': file_info.get('quality_score'),
                    'metadata': file_info.get('audio_metadata'),
                    'errors': file_info.get('validation_errors', []),
                    'progress': file_info.get('upload_progress')
                }

                report['validation_details'].append(file_report)

                if file_report['is_valid']:
                    report['files_validated'] += 1
                else:
                    report['files_with_issues'] += 1

        return report


# Global monitor instance
_monitor_instance: Optional[AudioUploadMonitor] = None


def get_monitor(uploads_dir: Optional[Path] = None) -> AudioUploadMonitor:
    """Get or create the global monitor instance."""
    global _monitor_instance

    if _monitor_instance is None:
        if uploads_dir is None:
            # Import here to avoid circular imports
            from .worker_tasks import UPLOADS_DIR
            uploads_dir = UPLOADS_DIR

        _monitor_instance = AudioUploadMonitor(uploads_dir)

    return _monitor_instance


async def start_global_monitor(event_callback: Optional[Callable] = None) -> AudioUploadMonitor:
    """Start the global audio upload monitor."""
    monitor = get_monitor()
    if event_callback:
        monitor.event_callback = event_callback

    await monitor.start_monitoring()
    return monitor


async def stop_global_monitor() -> None:
    """Stop the global audio upload monitor."""
    if _monitor_instance:
        await _monitor_instance.stop_monitoring()


# Enhanced utility functions for the new features

def create_validation_summary(audio_event: AudioFileEvent) -> Dict[str, Any]:
    """Create a summary of validation results for an audio event."""
    summary = {
        'file_path': str(audio_event.file_path),
        'file_size': audio_event.file_size,
        'is_valid': len(audio_event.validation_errors) == 0,
        'quality_score': audio_event.quality_score,
        'has_metadata': audio_event.audio_metadata is not None,
        'validation_errors': audio_event.validation_errors.copy()
    }

    if audio_event.audio_metadata:
        summary['audio_format'] = audio_event.audio_metadata.get('format')
        summary['sample_rate'] = audio_event.audio_metadata.get('sample_rate')
        summary['channels'] = audio_event.audio_metadata.get('channels')
        summary['duration'] = audio_event.audio_metadata.get('duration')

    if audio_event.upload_progress:
        summary['upload_speed'] = audio_event.upload_progress.get('upload_speed')
        summary['progress_percentage'] = audio_event.upload_progress.get('progress_percentage')
        summary['eta_seconds'] = audio_event.upload_progress.get('eta_seconds')

    return summary


def format_validation_error_message(audio_event: AudioFileEvent) -> str:
    """Format a detailed error message for validation failures."""
    if not audio_event.validation_errors:
        return "Audio file validation successful"

    error_parts = [
        f"Audio validation failed for {audio_event.file_path.name}:",
        f"File size: {audio_event.file_size} bytes",
    ]

    if audio_event.audio_metadata:
        metadata = audio_event.audio_metadata
        error_parts.append(f"Detected format: {metadata.get('format', 'unknown')}")
        error_parts.append(f"Sample rate: {metadata.get('sample_rate', 'unknown')} Hz")
        error_parts.append(f"Channels: {metadata.get('channels', 'unknown')}")

    if audio_event.quality_score is not None:
        error_parts.append(f"Quality score: {audio_event.quality_score:.1f}%")

    error_parts.extend([f"Error: {error}" for error in audio_event.validation_errors])

    return "\n".join(error_parts)


def check_system_requirements() -> Tuple[bool, list[str]]:
    """Check if system requirements for enhanced validation are met."""
    warnings = []

    # Check for ffprobe (optional but recommended)
    try:
        result = subprocess.run(['ffprobe', '-version'],
                              capture_output=True, timeout=5)
        if result.returncode != 0:
            warnings.append("ffprobe not found - some advanced audio validation features will be limited")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        warnings.append("ffprobe not available - using basic validation only")

    return len(warnings) == 0, warnings