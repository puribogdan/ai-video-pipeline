# app/worker_tasks.py
import os
import shutil
import subprocess
import sys
import mimetypes
import time
import asyncio
from pathlib import Path
from typing import Dict, Optional
from collections import deque
import logging
import stat

from dotenv import load_dotenv
from .email_utils import send_link_email
from .audio_monitor import AudioUploadMonitor, AudioFileEvent, FileEventType, get_monitor
from .monitoring import get_monitor as get_pipeline_monitor, monitoring_context

import boto3
from botocore.exceptions import ClientError

import json
import tempfile
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

APP_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SRC = APP_ROOT / "pipeline"
MEDIA_DIR = APP_ROOT / "media"

# Enhanced debugging for uploads directory configuration (Render server compatible)
print(f"[worker] [DEBUG] APP_ROOT: {APP_ROOT}", flush=True)
UPLOADS_DIR_DEFAULT = str(APP_ROOT / "uploads")
UPLOADS_DIR_ENV = os.getenv("UPLOADS_DIR")
print(f"[worker] [DEBUG] UPLOADS_DIR from env: {UPLOADS_DIR_ENV}", flush=True)
print(f"[worker] [DEBUG] UPLOADS_DIR default: {UPLOADS_DIR_DEFAULT}", flush=True)

UPLOADS_DIR = Path(UPLOADS_DIR_ENV) if UPLOADS_DIR_ENV else Path(UPLOADS_DIR_DEFAULT)
print(f"[worker] [DEBUG] UPLOADS_DIR configured as: {UPLOADS_DIR}", flush=True)
print(f"[worker] [DEBUG] UPLOADS_DIR exists: {UPLOADS_DIR.exists()}", flush=True)
print(f"[worker] [DEBUG] UPLOADS_DIR parent: {UPLOADS_DIR.parent}", flush=True)
print(f"[worker] [DEBUG] UPLOADS_DIR parent exists: {UPLOADS_DIR.parent.exists()}", flush=True)

# Try to create the directory if it doesn't exist
try:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[worker] [DEBUG] UPLOADS_DIR created/verified: {UPLOADS_DIR}", flush=True)
except Exception as e:
    print(f"[worker] [ERROR] Failed to create UPLOADS_DIR {UPLOADS_DIR}: {e}", flush=True)
    print(f"[worker] [ERROR] Exception type: {type(e).__name__}", flush=True)

BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000")

REQUIRED_ENV = ["OPENAI_API_KEY", "REPLICATE_API_TOKEN"]


def log(msg: str) -> None:
    print(f"[worker] {msg}", flush=True)


def cleanup_job_resources(job_dir: Path) -> None:
    """Force cleanup of all job resources with retry logic"""
    try:
        # Force garbage collection to release file handles
        import gc
        gc.collect()

        # Remove pipeline directory with retry
        pipeline_dir = job_dir / "pipeline"
        if pipeline_dir.exists():
            for attempt in range(5):
                try:
                    shutil.rmtree(pipeline_dir)
                    log(f"[CLEANUP] Successfully removed pipeline directory (attempt {attempt + 1})")
                    break
                except OSError as e:
                    log(f"[CLEANUP] Attempt {attempt + 1} failed: {e}")
                    if attempt < 4:
                        time.sleep(1)
                    else:
                        # Force cleanup with system command
                        try:
                            import subprocess
                            subprocess.run(['rm', '-rf', str(pipeline_dir)], check=True, timeout=10)
                            log("[CLEANUP] Force cleanup successful")
                        except Exception as force_e:
                            log(f"[WARNING] Force cleanup failed: {force_e}")

        # Clean up any remaining temp files
        cleanup_patterns = ["*.tmp", "*.temp", "*.lock", "*.pid", ".rq_job_id"]
        for pattern in cleanup_patterns:
            for f in job_dir.glob(pattern):
                try:
                    f.unlink()
                    log(f"[CLEANUP] Removed temp file: {f.name}")
                except Exception as e:
                    log(f"[WARNING] Failed to remove temp file {f.name}: {e}")

    except Exception as e:
        log(f"[WARNING] Cleanup failed for {job_dir}: {e}")


def _copy_pipeline_to(job_dir: Path) -> None:
    if not PIPELINE_SRC.exists():
        raise RuntimeError(f"Pipeline folder not found: {PIPELINE_SRC}")
    target = job_dir / "pipeline"

    # More robust cleanup with retry
    if target.exists():
        for attempt in range(5):
            try:
                shutil.rmtree(target)
                log(f"[PIPELINE] Cleaned existing directory (attempt {attempt + 1})")
                break
            except OSError as e:
                log(f"[PIPELINE] Cleanup attempt {attempt + 1} failed: {e}")
                if attempt == 4:
                    # Last attempt - try system command
                    try:
                        import subprocess
                        subprocess.run(['rm', '-rf', str(target)], check=True, timeout=10)
                        log("[PIPELINE] Force cleanup successful")
                    except Exception as force_e:
                        log(f"[ERROR] Force cleanup failed: {force_e}")
                        raise RuntimeError(f"Failed to cleanup pipeline directory: {force_e}")
                time.sleep(0.5)

    # Ensure parent directory exists
    target.parent.mkdir(parents=True, exist_ok=True)

    # Copy with verification
    try:
        shutil.copytree(PIPELINE_SRC, target)
        log(f"[PIPELINE] Successfully copied to {target}")

        # Verify copy
        if not target.exists() or not any(target.iterdir()):
            raise RuntimeError("Pipeline copy verification failed - directory empty or missing")

    except Exception as e:
        log(f"[ERROR] Pipeline copy failed: {e}")
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=5),
    retry=retry_if_exception_type((subprocess.CalledProcessError, FileNotFoundError, Exception)),
    reraise=True
)
def ensure_mp3(src_path: Path) -> Path:
    try:
        mt, _ = mimetypes.guess_type(str(src_path))
        is_mp3 = (src_path.suffix.lower() == ".mp3") or (mt == "audio/mpeg")
        if is_mp3:
            return src_path
        out_path = src_path.with_suffix(".mp3")
        cmd = ["ffmpeg", "-y", "-i", str(src_path), "-vn", "-acodec", "libmp3lame", "-b:a", "192k", "-ar", "44100", "-ac", "2", str(out_path)]
        log(f"[LOG] ensure_mp3 input: {src_path} (ext: {src_path.suffix}, size: {src_path.stat().st_size}), cmd: {' '.join(cmd)}")

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"[ERROR] ffmpeg failed with return code {result.returncode}: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)

        out_size = out_path.stat().st_size if out_path.exists() else 0
        log(f"[LOG] ensure_mp3 output: {out_path} (size: {out_size})")
        if out_size == 0:
            raise RuntimeError("Conversion produced empty file")
        return out_path
    except Exception as e:
        log(f"[ERROR] ensure_mp3 failed for {src_path}: {e}")
        raise


def run_with_live_output(cmd: list[str], cwd: Path, env: dict, timeout: int = 5400) -> None:
    log(f"RUN (cwd={cwd}, timeout={timeout}s): {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    assert proc.stdout is not None
    tail = deque(maxlen=200)

    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            tail.append(line.rstrip())

        ret = proc.wait(timeout=timeout)
        if ret != 0:
            raise RuntimeError("make_video.py failed (exit %d)\n--- tail ---\n%s" % (ret, "\n".join(tail)))
    except subprocess.TimeoutExpired:
        log(f"Process timed out after {timeout} seconds, terminating...")
        proc.kill()
        proc.wait()
        raise RuntimeError(f"make_video.py timed out after {timeout} seconds")


def _listdir_safe(p: Path) -> list[str]:
    try:
        return sorted([x.name for x in p.iterdir()])
    except Exception:
        return []


def _find_any_audio(job_dir: Path) -> Optional[Path]:
    log(f"[DEBUG] _find_any_audio searching in: {job_dir}")
    audio_exts = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma", ".webm")
    found_files = []

    # First pass: look for files with audio extensions
    for ext in audio_exts:
        pattern = f"*{ext}"
        log(f"[DEBUG] Searching for pattern: {pattern}")
        try:
            for f in sorted(job_dir.glob(pattern)):
                log(f"[DEBUG] Found potential audio file: {f.name}")
                if f.is_file():
                    try:
                        size = f.stat().st_size
                        log(f"[DEBUG] File {f.name}: size={size}, is_file={f.is_file()}")
                        if size > 0:
                            found_files.append(f)
                            log(f"[DEBUG] Added valid audio file: {f.name} ({size} bytes)")
                    except Exception as e:
                        log(f"[DEBUG] Error checking file {f.name}: {e}")
        except Exception as e:
            log(f"[DEBUG] Error searching for {pattern}: {e}")

    if found_files:
        # Return the first valid audio file
        selected = found_files[0]
        log(f"[DEBUG] Selected audio file: {selected.name}")
        return selected

    # Second pass: look for any file with audio mime type
    log(f"[DEBUG] No audio files found by extension, checking mime types...")
    try:
        for f in sorted(job_dir.glob("*")):
            if f.is_file():
                try:
                    mt, _ = mimetypes.guess_type(str(f))
                    size = f.stat().st_size
                    log(f"[DEBUG] Checking file {f.name}: size={size}, mime={mt}")
                    if mt and mt.startswith("audio/") and size > 0:
                        log(f"[DEBUG] Found audio file by mime type: {f.name}")
                        return f
                except Exception as e:
                    log(f"[DEBUG] Error checking mime type for {f.name}: {e}")
    except Exception as e:
        log(f"[DEBUG] Error in mime type search: {e}")

    log(f"[DEBUG] No audio files found in {job_dir}")
    return None


async def _wait_for_audio_event_driven(job_id: str, hint_path: Optional[Path], timeout_s: float = 180.0) -> Path:
    """
    Event-driven audio file waiting using the AudioUploadMonitor.

    This function replaces the polling-based approach with event-driven monitoring
    for better performance and immediate response to file uploads.
    """
    log(f"[DEBUG] _wait_for_audio_event_driven: job_id={job_id}, hint_path={hint_path}, timeout={timeout_s}s")

    # Get the global monitor instance
    monitor = get_monitor()

    # Check if monitor is running, start it if not
    if not monitor._monitoring:
        log(f"[INFO] Starting AudioUploadMonitor for event-driven audio detection")
        await monitor.start_monitoring()

    # Check if we already have a stable audio file for this job
    job_status = monitor.get_job_status(job_id)
    if job_status:
        for file_key, file_info in job_status.get('audio_files', {}).items():
            if (file_info.get('is_audio') and
                file_info.get('is_stable') and
                file_info.get('file_size', 0) > 0):

                file_path = Path(file_key)
                if file_path.exists() and file_path.stat().st_size > 0:
                    log(f"[INFO] Found existing stable audio file via monitor: {file_path}")
                    return file_path

    # Set up event-driven waiting
    audio_file_ready = asyncio.Event()
    found_audio_path = None

    # Event callback to handle audio file events
    async def audio_event_callback(event: AudioFileEvent):
        nonlocal found_audio_path, audio_file_ready

        if event.job_id == job_id and event.is_audio and event.is_stable:
            log(f"[INFO] Audio file ready via event: {event.file_path} ({event.file_size} bytes)")
            found_audio_path = event.file_path
            audio_file_ready.set()

    # Store original callback and set our temporary one
    original_callback = monitor.event_callback
    monitor.event_callback = audio_event_callback

    try:
        # Also check hint path if provided
        if hint_path and hint_path.exists():
            size = hint_path.stat().st_size
            if size > 0:
                # Validate it's an audio file
                mt, _ = mimetypes.guess_type(str(hint_path))
                is_audio = (hint_path.suffix.lower() in ('.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma', '.webm') or
                            (mt and mt.startswith('audio/')))

                if is_audio:
                    # Check stability
                    time.sleep(0.5)
                    size2 = hint_path.stat().st_size
                    if size == size2:
                        log(f"[INFO] Using hint path audio file: {hint_path}")
                        return hint_path

        # Wait for audio file event or timeout
        try:
            await asyncio.wait_for(audio_file_ready.wait(), timeout=timeout_s)
            if found_audio_path and found_audio_path.exists():
                return found_audio_path
        except asyncio.TimeoutError:
            pass

        # Fallback to polling if event-driven approach doesn't find anything
        log(f"[WARNING] Event-driven approach timed out, falling back to polling")
        return await _wait_for_any_audio_polling_fallback(job_id, hint_path, timeout_s)

    finally:
        # Restore original callback
        monitor.event_callback = original_callback


async def _wait_for_any_audio_polling_fallback(job_id: str, hint_path: Optional[Path], timeout_s: float = 180.0) -> Path:
    """
    Fallback polling-based audio file detection.

    This is used when the event-driven approach fails or times out,
    maintaining backward compatibility with the existing system.
    """
    log(f"[DEBUG] Using polling fallback for job_id={job_id}")

    # Use the original polling logic but with async wrapper
    loop = asyncio.get_event_loop()

    # Run the original synchronous function in a thread pool
    result = await loop.run_in_executor(None, _wait_for_any_audio_sync, job_id, hint_path, timeout_s)
    return result


def _run_async_audio_detection(job_id: str, hint_path: Optional[Path]) -> Path:
    """
    Helper function to run async audio detection in a separate event loop.
    Used when we need to run async code from a sync context that already has an event loop.

    Includes proper error handling for cases where event loop operations fail.
    """
    try:
        # Create new event loop for this specific operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                _wait_for_audio_event_driven(job_id=job_id, hint_path=hint_path)
            )
        finally:
            try:
                # Cancel any remaining tasks before closing
                pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
                for task in pending_tasks:
                    task.cancel()

                # Wait for tasks to complete cancellation
                if pending_tasks:
                    loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
            except Exception as cleanup_error:
                log(f"[WARNING] Error during event loop cleanup: {cleanup_error}")
            finally:
                loop.close()
    except Exception as e:
        log(f"[ERROR] Async audio detection failed: {e}")
        log(f"[ERROR] Falling back to polling-based detection due to event loop issues")
        # Don't re-raise - let the caller handle fallback to polling
        raise RuntimeError(f"Event loop audio detection failed: {e}")


def _run_job_in_thread_pool(job_id: str, email: str, upload_path: str, style: str) -> Dict[str, str]:
    """
    Helper function to run async job processing in a separate event loop.
    Used when called from a sync context that already has an event loop running (like RQ workers).

    Includes proper error handling and fallback mechanisms.
    """
    try:
        # Create new event loop for this specific job
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                _process_job_async(job_id, email, upload_path, style)
            )
        finally:
            try:
                # Cancel any remaining tasks before closing
                pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
                for task in pending_tasks:
                    task.cancel()

                # Wait for tasks to complete cancellation
                if pending_tasks:
                    loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
            except Exception as cleanup_error:
                log(f"[WARNING] Error during job event loop cleanup: {cleanup_error}")
            finally:
                loop.close()
    except Exception as e:
        log(f"[ERROR] Job processing in thread pool failed: {e}")
        import traceback
        log(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
        log(f"[ERROR] This may be due to event loop conflicts in RQ worker environment")
        raise RuntimeError(f"Async job processing failed: {e}")


def _wait_for_any_audio_sync(job_id: str, hint_path: Optional[Path], timeout_s: float = 180.0) -> Path:
    """
    Synchronous version of the original polling-based audio detection.

    This maintains the exact same logic as the original _wait_for_any_audio function
    but can be called from async contexts via run_in_executor.
    """
    t0 = time.time()
    job_dir = UPLOADS_DIR / job_id
    log(f"[DEBUG] _wait_for_any_audio_sync: job_dir={job_dir}, hint={hint_path}, timeout={timeout_s}s")

    # Debug: Check job directory state
    log(f"[DEBUG] Job directory exists: {job_dir.exists()}")
    if job_dir.exists():
        try:
            contents = list(job_dir.iterdir())
            log(f"[DEBUG] Job directory contents: {[p.name for p in contents]}")
            for p in contents:
                if p.is_file():
                    log(f"[DEBUG] File {p.name}: size={p.stat().st_size}, mime={mimetypes.guess_type(str(p))}")
        except Exception as e:
            log(f"[DEBUG] Error listing directory: {e}")

    while time.time() - t0 < timeout_s:
        # Check hint path first with enhanced validation
        if hint_path:
            log(f"[DEBUG] Checking hint path: {hint_path}")
            log(f"[DEBUG] Hint path exists: {hint_path.exists()}")
            if hint_path.exists():
                try:
                    size = hint_path.stat().st_size
                    log(f"[DEBUG] Hint path size: {size} bytes")
                    if size > 0:
                        # Additional stability check for hint path
                        time.sleep(0.5)
                        size2 = hint_path.stat().st_size
                        if size == size2:
                            log(f"[DEBUG] Found stable hint {hint_path} ({size} bytes)")
                            return hint_path
                        else:
                            log(f"[DEBUG] Hint path size changed ({size} -> {size2}), still being written...")
                    else:
                        log(f"[DEBUG] Hint path exists but is empty (0 bytes)")
                except Exception as e:
                    log(f"[DEBUG] Error checking hint path: {e}")

        # Look for any audio file in job directory
        if job_dir.exists():
            found = _find_any_audio(job_dir)
            if found:
                try:
                    # Validate file stability
                    size1 = found.stat().st_size
                    time.sleep(0.5)
                    size2 = found.stat().st_size

                    if size1 == size2 and size1 > 0:
                        log(f"[DEBUG] Discovered stable audio file: {found} ({size1} bytes)")
                        return found
                    else:
                        log(f"[DEBUG] Audio file size changed ({size1} -> {size2}), still being written...")
                except Exception as e:
                    log(f"[DEBUG] Error checking found audio file: {e}")
            else:
                log(f"[DEBUG] No audio files found in job directory")
        else:
            log(f"[DEBUG] Job directory does not exist")

        if int(time.time() - t0) % 5 == 0:  # Reduced frequency for less noise
            elapsed = time.time() - t0
            listing = _listdir_safe(job_dir)
            log(f"[DEBUG] Still waiting after {elapsed:.1f}s... listing={listing}")

        time.sleep(0.5)  # Increased from 0.25 to 0.5 for better stability

    # Timeout reached - provide better error message
    job_listing = _listdir_safe(job_dir)
    if not job_dir.exists():
        raise RuntimeError(f"Job directory does not exist: {job_dir}")
    elif not job_listing:
        raise RuntimeError(f"Job directory is empty: {job_dir}")
    else:
        raise RuntimeError(f"Audio not found in {job_dir} after {timeout_s}s. Listing: {job_listing}")


def _wait_for_any_audio(job_id: str, hint_path: Optional[Path], timeout_s: float = 180.0) -> Path:
    t0 = time.time()
    job_dir = UPLOADS_DIR / job_id
    log(f"[DEBUG] _wait_for_any_audio: job_dir={job_dir}, hint={hint_path}, timeout={timeout_s}s")

    # Debug: Check job directory state
    log(f"[DEBUG] Job directory exists: {job_dir.exists()}")
    if job_dir.exists():
        try:
            contents = list(job_dir.iterdir())
            log(f"[DEBUG] Job directory contents: {[p.name for p in contents]}")
            for p in contents:
                if p.is_file():
                    log(f"[DEBUG] File {p.name}: size={p.stat().st_size}, mime={mimetypes.guess_type(str(p))}")
        except Exception as e:
            log(f"[DEBUG] Error listing directory: {e}")

    while time.time() - t0 < timeout_s:
        # Check hint path first with enhanced validation
        if hint_path:
            log(f"[DEBUG] Checking hint path: {hint_path}")
            log(f"[DEBUG] Hint path exists: {hint_path.exists()}")
            if hint_path.exists():
                try:
                    size = hint_path.stat().st_size
                    log(f"[DEBUG] Hint path size: {size} bytes")
                    if size > 0:
                        # Additional stability check for hint path
                        time.sleep(0.5)
                        size2 = hint_path.stat().st_size
                        if size == size2:
                            log(f"[DEBUG] Found stable hint {hint_path} ({size} bytes)")
                            return hint_path
                        else:
                            log(f"[DEBUG] Hint path size changed ({size} -> {size2}), still being written...")
                    else:
                        log(f"[DEBUG] Hint path exists but is empty (0 bytes)")
                except Exception as e:
                    log(f"[DEBUG] Error checking hint path: {e}")

        # Look for any audio file in job directory
        if job_dir.exists():
            found = _find_any_audio(job_dir)
            if found:
                try:
                    # Validate file stability
                    size1 = found.stat().st_size
                    time.sleep(0.5)
                    size2 = found.stat().st_size

                    if size1 == size2 and size1 > 0:
                        log(f"[DEBUG] Discovered stable audio file: {found} ({size1} bytes)")
                        return found
                    else:
                        log(f"[DEBUG] Audio file size changed ({size1} -> {size2}), still being written...")
                except Exception as e:
                    log(f"[DEBUG] Error checking found audio file: {e}")
            else:
                log(f"[DEBUG] No audio files found in job directory")
        else:
            log(f"[DEBUG] Job directory does not exist")

        if int(time.time() - t0) % 5 == 0:  # Reduced frequency for less noise
            elapsed = time.time() - t0
            listing = _listdir_safe(job_dir)
            log(f"[DEBUG] Still waiting after {elapsed:.1f}s... listing={listing}")

        time.sleep(0.5)  # Increased from 0.25 to 0.5 for better stability

    # Timeout reached - provide better error message
    job_listing = _listdir_safe(job_dir)
    if not job_dir.exists():
        raise RuntimeError(f"Job directory does not exist: {job_dir}")
    elif not job_listing:
        raise RuntimeError(f"Job directory is empty: {job_dir}")
    else:
        raise RuntimeError(f"Audio not found in {job_dir} after {timeout_s}s. Listing: {job_listing}")


def _find_portrait_file(job_dir: Path) -> Optional[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    for p in sorted(job_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts and p.stat().st_size > 0:
            log(f"portrait pick (ext): {p.resolve()} ({p.stat().st_size} bytes)")
            return p
    for p in sorted(job_dir.iterdir()):
        if not p.is_file():
            continue
        mt, _ = mimetypes.guess_type(str(p))
        if mt and mt.startswith("image/") and p.stat().st_size > 0:
            log(f"portrait pick (mime): {p.resolve()} ({p.stat().st_size} bytes)")
            return p
    return None


def _run_make_video(job_dir: Path, hint_audio: Optional[Path], style: str) -> Path:
    log(f"[DEBUG] _run_make_video called with job_dir={job_dir}, hint_audio={hint_audio}")

    # Debug: List all files in job directory before waiting
    if job_dir.exists():
        all_files = [p.name for p in job_dir.iterdir()]
        log(f"[DEBUG] Job directory contents before audio wait: {all_files}")
    else:
        log(f"[DEBUG] Job directory does not exist: {job_dir}")

    # Debug: Check if hint_audio exists and what it contains
    if hint_audio:
        log(f"[DEBUG] Hint audio path: {hint_audio}")
        log(f"[DEBUG] Hint audio exists: {hint_audio.exists()}")
        if hint_audio.exists():
            log(f"[DEBUG] Hint audio size: {hint_audio.stat().st_size} bytes")
            log(f"[DEBUG] Hint audio mime type: {mimetypes.guess_type(str(hint_audio))}")

    # Use event-driven audio detection instead of polling
    try:
        # Try to use existing event loop first
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, create task instead of using run_until_complete
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_run_async_audio_detection, job_id=job_dir.name, hint_path=hint_audio)
                audio_src = future.result(timeout=180.0)  # 3 minute timeout
        except RuntimeError:
            # No event loop running, create new one (for backward compatibility)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                audio_src = loop.run_until_complete(
                    _wait_for_audio_event_driven(job_id=job_dir.name, hint_path=hint_audio)
                )
            finally:
                loop.close()

        log(f"[INFO] Event-driven audio detection successful: {audio_src}")
    except Exception as e:
        log(f"[WARNING] Event-driven audio detection failed: {e}")
        log(f"[WARNING] This may be due to event loop conflicts in RQ worker environment")
        log(f"[INFO] Falling back to polling-based detection for backward compatibility")
        try:
            # Fallback to original polling method for backward compatibility
            audio_src = _wait_for_any_audio(job_id=job_dir.name, hint_path=hint_audio)
            log(f"[INFO] Polling fallback successful: {audio_src}")
        except Exception as polling_error:
            log(f"[ERROR] Both event-driven and polling detection failed: {polling_error}")
            log(f"[ERROR] Event-driven error: {e}")
            # Try one more time with a simpler approach
            try:
                audio_src = _wait_for_any_audio_sync(job_id=job_dir.name, hint_path=hint_audio, timeout_s=60.0)
                log(f"[INFO] Synchronous fallback successful: {audio_src}")
            except Exception as sync_error:
                log(f"[CRITICAL] All audio detection methods failed for job {job_dir.name}")
                raise RuntimeError(f"All audio detection methods failed. Event-driven: {e}, Polling: {polling_error}, Sync: {sync_error}")

    _copy_pipeline_to(job_dir)
    pipe_dir = job_dir / "pipeline"
    audio_input_dir = pipe_dir / "audio_input"
    audio_input_dir.mkdir(parents=True, exist_ok=True)

    audio_for_pipeline = ensure_mp3(audio_src)
    target_audio = audio_input_dir / "input.mp3"
    shutil.copy2(audio_for_pipeline, target_audio)
    log(f"Staged audio -> {target_audio} ({target_audio.stat().st_size} bytes)")

    env = os.environ.copy()
    for key in REQUIRED_ENV:
        if not env.get(key):
            raise RuntimeError(f"Missing required env: {key}")

    portrait = _find_portrait_file(job_dir)
    if portrait:
        env["PORTRAIT_PATH"] = str(portrait)
        log(f"Staging portrait via PORTRAIT_PATH={portrait}")

    # Forward selected style
    env["STYLE_CHOICE"] = style  # Uses style keys from STYLE_LIBRARY (e.g., kid_friendly_cartoon, japanese_kawaii, etc.)

    trim_extra = os.getenv("TRIM_EXTRA_ARGS", "").strip()
    if trim_extra:
        env["TRIM_EXTRA_ARGS"] = trim_extra

    cmd = [sys.executable, "make_video.py", "--job-id", job_dir.name]
    run_with_live_output(cmd, cwd=pipe_dir, env=env, timeout=5400)  # 90 minute timeout

    final_video = pipe_dir / "final_video.mp4"
    if not final_video.exists():
        raise RuntimeError("final_video.mp4 not produced by pipeline")
    return final_video



@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ClientError, Exception)),
    reraise=True
)
def upload_to_b2(job_id: str, video_path: Path, job_dir: Optional[Path] = None) -> Optional[str]:
    """Upload video, scripts, and portrait images to Backblaze B2 (S3-compatible) and return public URL, or None on failure."""
    bucket_name = os.getenv("B2_BUCKET_NAME")
    key_id = os.getenv("B2_KEY_ID")
    app_key = os.getenv("B2_APPLICATION_KEY")
    if bucket_name is None or key_id is None or app_key is None:
        log("B2 env vars not set; skipping upload.")
        return None

    # Safe logging
    log(f"[DEBUG] B2_KEY_ID length: {len(key_id)}")
    log(f"[DEBUG] B2_APP_KEY length: {len(app_key)}")
    log(f"[DEBUG] B2_KEY_ID prefix: {key_id[:5]}...")
    log(f"[DEBUG] B2_APP_KEY prefix: {app_key[:5]}...")

    local_file = video_path.absolute()
    file_name = f"exports/{job_id}/final_video.mp4"
    log(f"[DEBUG] local_file absolute path: {local_file}")
    log(f"[DEBUG] file_name in B2: {file_name}")
    log(f"[DEBUG] file_size: {video_path.stat().st_size} bytes")

    region = os.getenv("B2_REGION", "eu-central-003")
    endpoint = f'https://s3.{region}.backblazeb2.com'
    log(f"[DEBUG] Using endpoint: {endpoint}, region: {region}")

    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=key_id,
            aws_secret_access_key=app_key,
            endpoint_url=endpoint,
            region_name=region
        )
        log("Using S3-compatible boto3 client for Backblaze B2.")
        log("[DEBUG] S3 client created successfully.")

        # Upload using S3 client (S3-compatible)
        log(f"[DEBUG] Starting upload...")
        s3.upload_file(
            Filename=str(local_file),
            Bucket=bucket_name,
            Key=file_name,
            ExtraArgs={'ContentType': 'video/mp4'}
        )
        log("[DEBUG] Upload call completed.")

        # Upload input_script.json and prompt.json if job_dir provided
        if job_dir:
            # Upload input_script.json
            script_path = job_dir / "pipeline" / "scripts" / "input_script.json"
            if script_path.exists():
                script_key = f"exports/{job_id}/input_script.json"
                s3.upload_file(
                    str(script_path),
                    bucket_name,
                    script_key,
                    ExtraArgs={'ContentType': 'application/json'}
                )
                log(f"Uploaded input_script.json to: {script_key}")

            # Upload existing prompt.json
            prompt_path = job_dir / "pipeline" / "scenes" / "prompt.json"
            if prompt_path.exists():
                prompt_key = f"exports/{job_id}/prompt.json"
                s3.upload_file(
                    str(prompt_path),
                    bucket_name,
                    prompt_key,
                    ExtraArgs={'ContentType': 'application/json'}
                )
                log(f"Uploaded prompt.json to: {prompt_key}")

            # Upload video_prompts.json
            video_prompts_path = job_dir / "pipeline" / "scenes" / "video_prompts.json"
            if video_prompts_path.exists():
                video_prompts_key = f"exports/{job_id}/video_prompts.json"
                s3.upload_file(
                    str(video_prompts_path),
                    bucket_name,
                    video_prompts_key,
                    ExtraArgs={'ContentType': 'application/json'}
                )
                log(f"Uploaded video_prompts.json to: {video_prompts_key}")

            # Upload portrait images (both original and background-removed) if they exist
            portrait_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
            portrait_uploaded = False


            # Upload background-removed portrait image
            no_bg_portrait_path = job_dir / "pipeline" / "scenes" / "portrait_ref_no_bg.png"
            if no_bg_portrait_path.exists():
                portrait_no_bg_key = f"exports/{job_id}/portrait_ref_no_bg.png"
                s3.upload_file(
                    str(no_bg_portrait_path),
                    bucket_name,
                    portrait_no_bg_key,
                    ExtraArgs={'ContentType': 'image/png'}
                )
                log(f"Uploaded background-removed portrait image to: {portrait_no_bg_key}")
                portrait_uploaded = True

            # Also check for other portrait formats (fallback for compatibility)
            for ext in portrait_extensions:
                potential_portrait = job_dir / "pipeline" / "scenes" / f"portrait_ref{ext}"
                if potential_portrait.exists():
                    portrait_key = f"exports/{job_id}/portrait_ref{ext}"
                    # Determine content type based on file extension
                    content_type = 'image/png'  # default
                    if ext.lower() == '.jpg' or ext.lower() == '.jpeg':
                        content_type = 'image/jpeg'
                    elif ext.lower() == '.webp':
                        content_type = 'image/webp'
                    elif ext.lower() == '.bmp':
                        content_type = 'image/bmp'

                    s3.upload_file(
                        str(potential_portrait),
                        bucket_name,
                        portrait_key,
                        ExtraArgs={'ContentType': content_type}
                    )
                    log(f"Uploaded portrait image to: {portrait_key}")
                    portrait_uploaded = True
                    break

            # Upload audio files if job_dir provided
            if job_dir:
                # Upload original input audio
                audio_input_dir = job_dir / "pipeline" / "audio_input"
                original_audio_path = audio_input_dir / "input.mp3"
                if original_audio_path.exists():
                    audio_key = f"exports/{job_id}/audio_input.mp3"
                    s3.upload_file(
                        str(original_audio_path),
                        bucket_name,
                        audio_key,
                        ExtraArgs={'ContentType': 'audio/mpeg'}
                    )
                    log(f"Uploaded original input audio to: {audio_key}")

                # Upload trimmed audio if it exists
                trimmed_audio_path = audio_input_dir / "input_trimmed.mp3"
                if trimmed_audio_path.exists():
                    trimmed_audio_key = f"exports/{job_id}/audio_input_trimmed.mp3"
                    s3.upload_file(
                        str(trimmed_audio_path),
                        bucket_name,
                        trimmed_audio_key,
                        ExtraArgs={'ContentType': 'audio/mpeg'}
                    )
                    log(f"Uploaded trimmed input audio to: {trimmed_audio_key}")

        # Get ETag from last response (or head for verification)
        try:
            head = s3.head_object(Bucket=bucket_name, Key=file_name)
            etag = head.get('ETag', 'unknown')
            content_type = head.get('ContentType', 'unknown')
            size = head['ContentLength']
            log(f"[DEBUG] Upload verified: ETag={etag}, size={size} bytes, content_type={content_type}")
        except ClientError as head_e:
            log(f"[DEBUG] Head verification failed: {head_e.response['Error']['Message']}")

        log("Upload successful.")

        # Construct public URL (assuming public bucket)
        video_url = f"https://{bucket_name}.s3.{region}.backblazeb2.com/exports/{job_id}/final_video.mp4"
        log(f"Uploaded to B2: {video_url}")
        return video_url

    except Exception as e:
        import traceback
        err_msg = str(e)
        if isinstance(e, ClientError):
            err_code = e.response['Error']['Code']
            err_message = e.response['Error']['Message']
            err_msg += f" | Server: {err_code} - {err_message}"
        log(f"B2 upload failed: {type(e).__name__}: {err_msg}")
        log(f"Full traceback:\n{traceback.format_exc()}")
        return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ClientError, Exception)),
    reraise=True
)
def upload_images_to_b2(job_id: str, images_dir: Path) -> Optional[Dict[str, str]]:
    """Upload generated images to Backblaze B2 and return a dictionary of image URLs, or None on failure."""
    bucket_name = os.getenv("B2_BUCKET_NAME")
    key_id = os.getenv("B2_KEY_ID")
    app_key = os.getenv("B2_APPLICATION_KEY")

    log(f"[DEBUG] B2 Image Upload - Checking environment variables...")
    log(f"[DEBUG] B2_BUCKET_NAME: {'Set' if bucket_name else 'Not set'}")
    log(f"[DEBUG] B2_KEY_ID: {'Set' if key_id else 'Not set'}")
    log(f"[DEBUG] B2_APPLICATION_KEY: {'Set' if app_key else 'Not set'}")

    if not bucket_name or not key_id or not app_key:
        log("❌ B2 Image Upload - B2 credentials not properly configured. Please check environment variables.")
        log(f"[DEBUG] B2_BUCKET_NAME: {bucket_name}")
        log(f"[DEBUG] B2_KEY_ID: {'[SET]' if key_id else '[NOT SET]'}")
        log(f"[DEBUG] B2_APPLICATION_KEY: {'[SET]' if app_key else '[NOT SET]'}")
        return None

    region = os.getenv("B2_REGION", "eu-central-003")
    endpoint = f'https://s3.{region}.backblazeb2.com'
    log(f"[DEBUG] B2 Image Upload - Configuration: bucket={bucket_name}, region={region}")
    log(f"[DEBUG] B2 Image Upload - Images directory: {images_dir}")
    log(f"[DEBUG] B2 Image Upload - Images directory exists: {images_dir.exists()}")

    # Test B2 connection before proceeding
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=key_id,
            aws_secret_access_key=app_key,
            endpoint_url=endpoint,
            region_name=region
        )
        log("✅ B2 Image Upload - S3 client created successfully for Backblaze B2")

        # Test the connection with a simple head_bucket call
        test_response = s3.head_bucket(Bucket=bucket_name)
        log(f"✅ B2 Image Upload - Successfully connected to bucket: {bucket_name}")

        image_urls = {}

        # Upload all images from the images directory
        if images_dir.exists() and images_dir.is_dir():
            image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']

            # List all files in the images directory first
            try:
                all_files = list(images_dir.iterdir())
                log(f"[DEBUG] B2 Image Upload - Found {len(all_files)} files in images directory")
                for f in all_files:
                    log(f"[DEBUG] B2 Image Upload - File: {f.name} (size: {f.stat().st_size} bytes)")
            except Exception as e:
                log(f"[WARNING] B2 Image Upload - Error listing images directory: {e}")

            images_uploaded = 0

            for image_file in sorted(images_dir.iterdir()):
                if image_file.is_file() and image_file.suffix.lower() in [ext.lower() for ext in image_extensions]:
                    try:
                        file_size = image_file.stat().st_size
                        log(f"[DEBUG] B2 Image Upload - Processing image: {image_file.name} ({file_size} bytes)")

                        # Determine content type based on file extension
                        _, ext = os.path.splitext(str(image_file))
                        content_type = 'image/png'  # default
                        if ext.lower() == '.jpg' or ext.lower() == '.jpeg':
                            content_type = 'image/jpeg'
                        elif ext.lower() == '.webp':
                            content_type = 'image/webp'
                        elif ext.lower() == '.bmp':
                            content_type = 'image/bmp'

                        # Create B2 key for the image
                        image_key = f"exports/{job_id}/images/{image_file.name}"
                        log(f"[DEBUG] B2 Image Upload - Uploading to B2 key: {image_key}")

                        # Upload image to B2
                        s3.upload_file(
                            Filename=str(image_file),
                            Bucket=bucket_name,
                            Key=image_key,
                            ExtraArgs={'ContentType': content_type}
                        )

                        # Create public URL for the image
                        image_url = f"https://{bucket_name}.s3.{region}.backblazeb2.com/{image_key}"
                        image_urls[image_file.name] = image_url
                        images_uploaded += 1
                        log(f"✅ B2 Image Upload - Successfully uploaded: {image_key}")

                    except Exception as e:
                        log(f"❌ B2 Image Upload - Failed to upload image {image_file.name}: {e}")
                        continue

            log(f"✅ B2 Image Upload - Successfully uploaded {images_uploaded} images to B2")
            return image_urls if image_urls else None
        else:
            log(f"❌ B2 Image Upload - Images directory does not exist or is not a directory: {images_dir}")
            return None

    except Exception as e:
        import traceback
        err_msg = str(e)
        if isinstance(e, ClientError):
            err_code = e.response.get('Error', {}).get('Code', 'Unknown')
            err_message = e.response.get('Error', {}).get('Message', 'Unknown error')
            err_msg += f" | Server: {err_code} - {err_message}"
        log(f"❌ B2 Image Upload - B2 image upload failed: {type(e).__name__}: {err_msg}")
        log(f"B2 Image Upload - Full traceback:\n{traceback.format_exc()}")
        return None


def process_job(job_id: str, email: str, upload_path: str, style: str) -> Dict[str, str]:
    """
    Synchronous wrapper for async job processing.

    This function maintains backward compatibility with RQ while using
    the new event-driven audio monitoring system internally.

    Handles both scenarios: when called from RQ workers (with existing event loop)
    and when called from contexts without an event loop.
    """
    try:
        # Check if we're already in an async context (like RQ worker)
        try:
            # Try to get the current running loop
            loop = asyncio.get_running_loop()
            # We're in an async context, use thread pool to run async function
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_run_job_in_thread_pool, job_id, email, upload_path, style)
                result = future.result(timeout=3600.0)  # 1 hour timeout for video processing
                return result

        except RuntimeError:
            # No event loop running, create new one (backward compatibility)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Run the async version
                result = loop.run_until_complete(
                    _process_job_async(job_id, email, upload_path, style)
                )
                return result
            finally:
                loop.close()

    except Exception as e:
        log(f"[ERROR] Job {job_id} failed in sync wrapper: {e}")
        import traceback
        log(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
        raise


async def _process_job_async(job_id: str, email: str, upload_path: str, style: str) -> Dict[str, str]:
    """
    Async implementation of video job processing with event-driven audio detection.

    This function contains the main job processing logic using the new
    AudioUploadMonitor for immediate file system event detection.

    This function is designed to work within existing event loops (like RQ workers)
    and handles the case where no event loop is running.
    """
    job_dir = None
    monitor = get_pipeline_monitor()
    correlation_id = monitor.start_job_tracking(job_id)

    try:
        log(f"[DEBUG] ===== WORKER JOB DEBUGGING =====")
        log(f"[DEBUG] process_job called with job_id={job_id}, email={email}, upload_path={upload_path}, style={style}")
        log(f"Starting job processing: {job_id} (email: {email}, style: {style})")
        log(f"[DEBUG] Upload path type: {type(upload_path)}")
        log(f"[DEBUG] Upload path string: {str(upload_path)}")
        log(f"[DEBUG] Upload path length: {len(upload_path)}")
        log(f"[INFO] Worker is now processing job {job_id} - this confirms the job was dequeued successfully")

        # Record initial job metrics
        if upload_path:
            upload_file_path = Path(upload_path)
            if upload_file_path.exists():
                file_size = upload_file_path.stat().st_size
                monitor.record_upload_metrics(file_size, 0.1, True)  # Assume upload was successful

        # Enhanced debugging for upload path issues
        if upload_path:
            upload_path_obj = Path(upload_path)
            log(f"[DEBUG] Upload path as Path object: {upload_path_obj}")
            log(f"[DEBUG] Upload path exists: {upload_path_obj.exists()}")
            log(f"[DEBUG] Upload path is absolute: {upload_path_obj.is_absolute()}")
            log(f"[DEBUG] Upload path parent: {upload_path_obj.parent}")
            log(f"[DEBUG] Upload path parent exists: {upload_path_obj.parent.exists() if upload_path_obj.parent else 'N/A'}")
            log(f"[DEBUG] Upload path name: {upload_path_obj.name}")
            log(f"[DEBUG] Upload path suffix: {upload_path_obj.suffix}")

            # Check if this is the hardcoded filename issue
            if 'input_cut.mp3' in upload_path:
                log(f"[WARNING] CONFIRMED HARDCODED FILENAME ISSUE: upload_path contains 'input_cut.mp3'")
                log(f"[DEBUG] This indicates the client or middleware is sending a hardcoded filename")
            else:
                log(f"[DEBUG] Upload path does not contain hardcoded 'input_cut.mp3'")

        log(f"[DEBUG] ================================")

        # Flexible audio file verification - accept any audio filename
        log(f"[DEBUG] Looking for any audio file in job directory...")
        audio_file_verified = False
        final_audio_path = None

        # Create job directory if it doesn't exist
        job_dir = UPLOADS_DIR / job_id
        try:
            job_dir.mkdir(parents=True, exist_ok=True)
            log(f"[DEBUG] Job directory ready: {job_dir}")
        except Exception as e:
            log(f"[ERROR] Failed to create job directory {job_dir}: {e}")
            raise RuntimeError(f"Failed to create job directory: {e}")

        # Implement retry logic with exponential backoff for file access
        # This handles race conditions where file upload completes after job starts
        max_file_wait_retries = 10
        retry_delay = 0.5  # Start with 0.5 seconds
        
        for retry_attempt in range(max_file_wait_retries):
            # First, try the provided upload path if it exists
            if upload_path:
                upload_path_obj = Path(upload_path)
                log(f"[DEBUG] Checking provided upload path (attempt {retry_attempt + 1}/{max_file_wait_retries}): {upload_path_obj}")

                if upload_path_obj.exists() and upload_path_obj.is_file():
                    try:
                        # Verify it's actually an audio file
                        size = upload_path_obj.stat().st_size
                        if size > 0:
                            # Check if it's an audio file by extension or mime type
                            mt, _ = mimetypes.guess_type(str(upload_path_obj))
                            is_audio = (upload_path_obj.suffix.lower() in ('.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma', '.webm') or
                                       (mt and mt.startswith('audio/')))

                            if is_audio:
                                # Verify file stability (not still being written)
                                time.sleep(0.3)
                                size2 = upload_path_obj.stat().st_size
                                if size == size2:
                                    log(f"[INFO] Using provided audio file: {upload_path_obj} ({size} bytes)")
                                    final_audio_path = upload_path_obj
                                    audio_file_verified = True
                                    break
                                else:
                                    log(f"[DEBUG] File size changed ({size} -> {size2}), still being written...")
                            else:
                                log(f"[WARNING] Provided file is not an audio file: {upload_path_obj} (mime: {mt})")
                        else:
                            log(f"[DEBUG] Provided file is empty (attempt {retry_attempt + 1})")
                    except Exception as e:
                        log(f"[DEBUG] Error checking provided upload path (attempt {retry_attempt + 1}): {e}")
                else:
                    log(f"[DEBUG] Provided upload path does not exist or is not a file (attempt {retry_attempt + 1})")

            # If upload path didn't work, look for ANY audio file in job directory
            if not audio_file_verified:
                log(f"[DEBUG] Searching for any audio file in job directory (attempt {retry_attempt + 1})...")
                found_audio = _find_any_audio(job_dir)

                if found_audio:
                    try:
                        size1 = found_audio.stat().st_size
                        if size1 > 0:
                            # Verify file stability
                            time.sleep(0.3)
                            size2 = found_audio.stat().st_size
                            if size1 == size2:
                                log(f"[INFO] Found stable audio file in job directory: {found_audio.name} ({size1} bytes)")
                                final_audio_path = found_audio
                                audio_file_verified = True
                                break
                            else:
                                log(f"[DEBUG] Audio file size changed ({size1} -> {size2}), still being written...")
                    except Exception as e:
                        log(f"[DEBUG] Error checking found audio file: {e}")
                else:
                    log(f"[DEBUG] No audio files found in job directory (attempt {retry_attempt + 1})")
            
            # If file found and verified, break out of retry loop
            if audio_file_verified:
                break
            
            # Wait before next retry with exponential backoff
            if retry_attempt < max_file_wait_retries - 1:
                wait_time = retry_delay * (2 ** retry_attempt)
                log(f"[DEBUG] Waiting {wait_time:.1f}s before retry {retry_attempt + 2}/{max_file_wait_retries}...")
                time.sleep(wait_time)

        # Final verification - ensure we have a valid audio file
        if audio_file_verified and final_audio_path:
            try:
                # Final validation of the audio file
                audio_size = final_audio_path.stat().st_size
                if audio_size == 0:
                    raise RuntimeError(f"Audio file exists but is empty (0 bytes): {final_audio_path}")

                # Verify file is readable
                with open(final_audio_path, 'rb') as f:
                    header = f.read(64)
                    if len(header) < 64:
                        raise RuntimeError(f"Audio file appears truncated: only {len(header)} bytes readable")

                log(f"[INFO] Audio file validated: {final_audio_path} ({audio_size} bytes)")
                log(f"[DEBUG] Audio file is ready for processing")

            except Exception as e:
                log(f"[ERROR] Audio file validation failed: {e}")
                # Try to find alternative audio file if validation fails
                log(f"[DEBUG] Attempting to find alternative audio file...")
                alternative_audio = _find_any_audio(job_dir)
                if alternative_audio and alternative_audio != final_audio_path:
                    log(f"[INFO] Found alternative audio file: {alternative_audio}")
                    final_audio_path = alternative_audio
                    audio_file_verified = True
                    log(f"[INFO] Using alternative audio file: {final_audio_path}")
                else:
                    audio_file_verified = False

        # Mark job as active if audio file is verified
        if audio_file_verified:
            try:
                from app.main import _save_job_completion
                _save_job_completion(job_id, "active", {"message": "Job started - initializing video processing pipeline"})
                log(f"[INFO] Job status updated to 'active' for job_id: {job_id}")
            except Exception as e:
                log(f"[WARNING] Failed to save initial status: {e}")
        else:
            log(f"[ERROR] No valid audio file found for job {job_id}")
            # Enhanced error message that doesn't assume hardcoded filenames
            job_files = []
            if job_dir.exists():
                try:
                    job_files = [p.name for p in job_dir.iterdir() if p.is_file()]
                except Exception as e:
                    job_files = [f"error_listing_dir: {e}"]

            raise RuntimeError(
                f"No valid audio file found. Upload path: {upload_path}, "
                f"Job directory: {job_dir}, Available files: {job_files}. "
                f"System now accepts any audio filename - ensure file is fully uploaded and accessible."
            )

        if os.getenv("DEV_FAKE_PIPELINE", "0") == "1":
            log("Using fake pipeline mode")
            MEDIA_DIR.mkdir(parents=True, exist_ok=True)
            placeholder = APP_ROOT / "app" / "static" / "sample.mp4"
            if not placeholder.exists():
                placeholder.parent.mkdir(parents=True, exist_ok=True)
                with open(placeholder, "wb") as f:
                    f.write(b"\x00")
            out = MEDIA_DIR / f"{job_id}.mp4"
            shutil.copy2(placeholder, out)
            video_url = f"{BASE_URL}/media/{job_id}.mp4"
            send_link_email(email, video_url, job_id)
            
            # Cleanup even in fake mode
            job_dir = UPLOADS_DIR / job_id
            cleanup_job_resources(job_dir)
            
            return {"status": "done", "video_url": video_url}

        # Create job directory
        job_dir = UPLOADS_DIR / job_id
        try:
            job_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log(f"[ERROR] Failed to create job directory {job_dir}: {e}")
            raise RuntimeError(f"Failed to create job directory: {e}")

        # Use the verified audio file path
        hint_audio = final_audio_path
        log(f"[DEBUG] Using verified audio file: {hint_audio}")
        log(f"[DEBUG] Audio file exists: {hint_audio.exists() if hint_audio else 'N/A'}")
        log(f"[DEBUG] Audio file size: {hint_audio.stat().st_size if hint_audio else 'N/A'} bytes")

        # Enhanced debugging for the selected audio file
        if hint_audio:
            log(f"[DEBUG] Selected audio filename: {hint_audio.name}")
            log(f"[DEBUG] Selected audio extension: {hint_audio.suffix}")
            log(f"[DEBUG] Selected audio path absolute: {hint_audio.absolute()}")
            log(f"[DEBUG] Selected audio path parent: {hint_audio.parent}")
            log(f"[DEBUG] Selected audio path parent exists: {hint_audio.parent.exists()}")

        # System now accepts any audio filename - no hardcoded expectations
        log(f"[INFO] System configured to accept any audio filename for maximum flexibility")

        initial_listing = sorted([p.name for p in job_dir.iterdir()]) if job_dir.exists() else []
        log(f"[DEBUG] Job dir initial listing: {initial_listing}")

        # Check if completion_status.json exists without audio files (potential race condition)
        if "completion_status.json" in initial_listing and not any(f.endswith(('.mp3', '.wav', '.m4a', '.aac', '.ogg')) for f in initial_listing):
            log(f"[WARNING] Found completion_status.json but no audio files - possible premature status save")
            log(f"[DEBUG] completion_status.json size: {Path(job_dir / 'completion_status.json').stat().st_size if (job_dir / 'completion_status.json').exists() else 'N/A'}")
            log(f"[DEBUG] completion_status.json modification time: {(job_dir / 'completion_status.json').stat().st_mtime if (job_dir / 'completion_status.json').exists() else 'N/A'}")

        # Enhanced: Always look for any audio file in the job directory first (most robust approach)
        # This makes the system flexible and accepts any audio filename, not just specific ones
        log(f"[INFO] System prioritizes finding any audio file in job directory for maximum compatibility")
        log(f"[DEBUG] Looking for any audio file in job directory...")
        found_audio = _find_any_audio(job_dir)

        if found_audio:
            log(f"[INFO] Found audio file in job directory: {found_audio.name}")
            log(f"[DEBUG] Found audio file size: {found_audio.stat().st_size} bytes")
            log(f"[DEBUG] Found audio file mime type: {mimetypes.guess_type(str(found_audio))}")
            log(f"[DEBUG] Found audio file path: {found_audio.absolute()}")
            hint_audio = found_audio
        elif hint_audio and hint_audio.exists():
            log(f"[INFO] Using provided hint audio path: {hint_audio}")
            log(f"[DEBUG] Hint audio size: {hint_audio.stat().st_size} bytes")
            log(f"[DEBUG] Hint audio mime type: {mimetypes.guess_type(str(hint_audio))}")
        else:
            log(f"[WARNING] No audio file found via directory scan or hint path")
            log(f"[DEBUG] Upload path details: {upload_path}")
            log(f"[DEBUG] Upload path exists: {Path(upload_path).exists() if upload_path else 'N/A'}")
            log(f"[DEBUG] Upload path parent: {Path(upload_path).parent if upload_path else 'N/A'}")
            log(f"[DEBUG] Upload path parent exists: {Path(upload_path).parent.exists() if upload_path else 'N/A'}")

            # Final attempt: try to find any audio file again (in case of timing issues)
            found_audio = _find_any_audio(job_dir)
            if found_audio:
                log(f"[INFO] Found audio file on second attempt: {found_audio}")
                hint_audio = found_audio
            else:
                # No audio file found - implement robust retry logic for race condition
                # Enhanced for multiple simultaneous jobs
                max_retries = 8  # Increased retries for better race condition handling
                base_wait_time = 2  # Reduced base wait time for faster initial retries

                for attempt in range(max_retries):
                    # Use exponential backoff with jitter for multiple jobs
                    import random
                    base_wait = base_wait_time * (2 ** attempt)
                    jitter = random.uniform(0.5, 1.5)  # Add jitter to prevent thundering herd
                    wait_time = base_wait * jitter

                    log(f"[WARNING] No audio file found, waiting {wait_time:.1f} seconds and retrying (attempt {attempt + 1}/{max_retries})...")
                    log(f"[DEBUG] Job directory before wait: {job_dir}")
                    log(f"[DEBUG] Job directory exists: {job_dir.exists()}")

                    if job_dir.exists():
                        log(f"[DEBUG] Job directory permissions: {oct(job_dir.stat().st_mode)}")
                        try:
                            contents_before = [p.name for p in job_dir.iterdir()]
                            log(f"[DEBUG] Job directory contents before wait: {contents_before}")

                            # Enhanced debugging for file type detection
                            for file_path in job_dir.iterdir():
                                if file_path.is_file():
                                    try:
                                        size = file_path.stat().st_size
                                        mtime = file_path.stat().st_mtime
                                        mime_type, _ = mimetypes.guess_type(str(file_path))
                                        log(f"[DEBUG] File analysis - {file_path.name}: size={size}, mtime={mtime}, mime={mime_type}")
                                    except Exception as e:
                                        log(f"[DEBUG] Error analyzing file {file_path.name}: {e}")

                            # Check if upload path exists and monitor its size (but don't rely on it)
                                if upload_path:
                                    upload_path_obj = Path(upload_path)
                                    log(f"[DEBUG] Upload path object: {upload_path_obj}")
                                    log(f"[DEBUG] Upload path absolute: {upload_path_obj.absolute()}")
                                    if upload_path_obj.exists():
                                        size_before = upload_path_obj.stat().st_size
                                        mtime_before = upload_path_obj.stat().st_mtime
                                        log(f"[DEBUG] Upload path exists, size before wait: {size_before} bytes, mtime: {mtime_before}")
                                        mime_type, _ = mimetypes.guess_type(str(upload_path_obj))
                                        log(f"[DEBUG] Upload path mime type: {mime_type}")
                                    else:
                                        log(f"[DEBUG] Upload path does not exist - will rely on finding any audio file in job directory")
                        except Exception as e:
                            log(f"[DEBUG] Error checking directory contents: {e}")

                    time.sleep(wait_time)

                    # Check again after waiting
                    log(f"[DEBUG] Checking for audio files after wait...")
                    found_audio = _find_any_audio(job_dir)

                    if found_audio:
                        # Validate file is not still being written (check size stability)
                        try:
                            size1 = found_audio.stat().st_size
                            time.sleep(2)  # Wait 2 seconds for better stability
                            size2 = found_audio.stat().st_size

                            if size1 == size2 and size1 > 0:
                                log(f"[INFO] Found stable audio file after {attempt + 1} attempts: {found_audio}")
                                log(f"[DEBUG] Audio file size: {size1} bytes (stable)")
                                log(f"[DEBUG] Audio file mime type: {mimetypes.guess_type(str(found_audio))}")
                                log(f"[DEBUG] Audio file absolute path: {found_audio.absolute()}")
                                hint_audio = found_audio
                                break
                            else:
                                log(f"[WARNING] Audio file size changed during check ({size1} -> {size2}), still being written...")
                                if attempt == max_retries - 1:  # Last attempt
                                    log(f"[INFO] Using file despite size change (last attempt): {found_audio}")
                                    log(f"[DEBUG] Final file size: {size2} bytes")
                                    hint_audio = found_audio
                                    break
                        except Exception as e:
                            log(f"[DEBUG] Error checking file stability: {e}")
                            if attempt == max_retries - 1:  # Last attempt
                                log(f"[INFO] Using found audio file (last attempt): {found_audio}")
                                hint_audio = found_audio
                                break
                    else:
                        log(f"[DEBUG] Still no audio file found after attempt {attempt + 1}")

                        # On last attempt, check if upload path exists (but prefer any audio file found)
                        if attempt == max_retries - 1 and upload_path:
                            upload_path_obj = Path(upload_path)
                            if upload_path_obj.exists():
                                log(f"[DEBUG] Upload path exists on final attempt: {upload_path_obj}")
                                log(f"[DEBUG] Upload path size: {upload_path_obj.stat().st_size} bytes")
                                if upload_path_obj.stat().st_size > 0:
                                    log(f"[INFO] Using upload path as audio file: {upload_path_obj}")
                                    hint_audio = upload_path_obj
                                    break
                    # Enhanced debugging for final attempt
                    available_files = [p.name for p in job_dir.iterdir()] if job_dir.exists() else []
                    log(f"[ERROR] Audio file still not found after retry.")
                    log(f"[DEBUG] Final attempt debugging:")
                    log(f"[DEBUG] Upload path: {upload_path}")
                    log(f"[DEBUG] Job directory: {job_dir}")
                    log(f"[DEBUG] Available files: {available_files}")
                    log(f"[DEBUG] Job directory exists: {job_dir.exists()}")

                    # Check if upload path is different from job directory
                    if upload_path and job_dir.exists():
                        upload_parent = Path(upload_path).parent
                        log(f"[DEBUG] Upload path parent: {upload_parent}")
                        log(f"[DEBUG] Upload path parent exists: {upload_parent.exists()}")
                        if upload_parent.exists():
                            try:
                                parent_contents = [p.name for p in upload_parent.iterdir()]
                                log(f"[DEBUG] Upload path parent contents: {parent_contents}")
                            except Exception as e:
                                log(f"[DEBUG] Error listing upload path parent: {e}")

                    # Additional debugging info
                    if job_dir.exists():
                        log(f"[DEBUG] Job directory permissions: {oct(job_dir.stat().st_mode)}")
                        try:
                            for root, dirs, files in os.walk(job_dir):
                                level = root.replace(str(job_dir), '').count(os.sep)
                                indent = ' ' * 2 * level
                                log(f"[DEBUG] {indent}{os.path.basename(root)}/")
                                subindent = ' ' * 2 * (level + 1)
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    try:
                                        file_stat = os.stat(file_path)
                                        log(f"[DEBUG] {subindent}{file} (size: {file_stat.st_size}, mode: {oct(file_stat.st_mode)})")
                                    except Exception as e:
                                        log(f"[DEBUG] {subindent}{file} (error: {e})")
                        except Exception as e:
                            log(f"[DEBUG] Error walking job directory: {e}")

                    # Check if the upload path exists at all
                    if upload_path:
                        upload_path_obj = Path(upload_path)
                        log(f"[DEBUG] Upload path exists: {upload_path_obj.exists()}")
                        if upload_path_obj.exists():
                            log(f"[DEBUG] Upload path size: {upload_path_obj.stat().st_size} bytes")
                            log(f"[DEBUG] Upload path is file: {upload_path_obj.is_file()}")
                        else:
                            log(f"[DEBUG] Upload path parent exists: {upload_path_obj.parent.exists()}")
                            if upload_path_obj.parent.exists():
                                try:
                                    log(f"[DEBUG] Upload path parent contents: {[p.name for p in upload_path_obj.parent.iterdir()]}")
                                except Exception as e:
                                    log(f"[DEBUG] Error listing upload path parent: {e}")


                    # More flexible error message since system now accepts any audio file
                    audio_files = [f for f in available_files if f.endswith(('.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma', '.webm'))]
                    if audio_files:
                        log(f"[INFO] Audio files found in directory: {audio_files} - possible file access issue")
                        raise RuntimeError(
                            f"Audio file not accessible. Upload path: {upload_path}, "
                            f"Job directory: {job_dir}, Available audio files: {audio_files}"
                        )
                    else:
                        raise RuntimeError(
                            f"No audio file found. Upload path: {upload_path}, "
                            f"Job directory: {job_dir}, Available files: {available_files}"
                        )

        # Validate audio file is not empty and still accessible
        if hint_audio is None:
            raise RuntimeError("Audio file is None - this should not happen")

        # Additional validation before processing
        try:
            audio_size = hint_audio.stat().st_size
            if audio_size == 0:
                raise RuntimeError(f"Audio file exists but is empty (0 bytes): {hint_audio}")

            # Verify file is still readable
            with open(hint_audio, 'rb') as f:
                header = f.read(64)
                if len(header) < 64:
                    raise RuntimeError(f"Audio file appears truncated: only {len(header)} bytes readable")

            log(f"[INFO] Audio file validated: {hint_audio} ({audio_size} bytes)")
        except Exception as e:
            log(f"[ERROR] Audio file validation failed: {e}")
            # Try to find alternative audio file
            log(f"[DEBUG] Attempting to find alternative audio file...")
            alternative_audio = _find_any_audio(job_dir)
            if alternative_audio and alternative_audio != hint_audio:
                log(f"[INFO] Found alternative audio file: {alternative_audio}")
                hint_audio = alternative_audio
                audio_size = hint_audio.stat().st_size
                log(f"[INFO] Using alternative audio file: {hint_audio} ({audio_size} bytes)")
            else:
                raise RuntimeError(f"Audio file validation failed and no alternative found: {e}")

        # Run the main video processing pipeline
        try:
            # Update status: processing
            try:
                from app.main import _save_job_completion
                _save_job_completion(job_id, "processing", {"message": "Processing audio, generating script, and creating video frames"})
            except Exception as e:
                log(f"[WARNING] Failed to save processing status: {e}")

            # Record processing stage start
            processing_start = time.time()
            final_video = _run_make_video(job_dir, hint_audio, style)
            processing_time = time.time() - processing_start

            # Record audio processing metrics
            if hint_audio:
                audio_size = hint_audio.stat().st_size
                monitor.record_audio_processing_metrics(job_id, audio_size, processing_time, True)

            # Record job stage completion
            monitor.record_job_stage(job_id, "video_processing", processing_time, {"style": style})

            # Update status: processing
            try:
                from app.main import _save_job_completion
                _save_job_completion(job_id, "processing", {"message": "Finalizing video - adding audio and preparing for upload"})
            except Exception as e:
                log(f"[WARNING] Failed to save completion status: {e}")

        except Exception as e:
            log(f"[ERROR] Video processing failed for job {job_id}: {e}")
            import traceback
            log(f"[ERROR] Full traceback:\n{traceback.format_exc()}")

            # Record failed processing metrics
            if hint_audio:
                audio_size = hint_audio.stat().st_size
                monitor.record_audio_processing_metrics(job_id, audio_size, 0, False)

            monitor.complete_job_tracking(job_id, "failed", str(e))
            raise RuntimeError(f"Video processing failed: {e}")

        # Copy to public media directory
        try:
            MEDIA_DIR.mkdir(parents=True, exist_ok=True)
            public_path = MEDIA_DIR / f"{job_id}.mp4"
            shutil.copy2(final_video, public_path)
            log(f"Video copied to public path: {public_path}")
        except Exception as e:
            log(f"[ERROR] Failed to copy video to public directory: {e}")
            raise RuntimeError(f"Failed to copy video to public directory: {e}")

        # Upload to Backblaze B2 (with fallback to local)
        upload_start = time.time()
        try:
            b2_url = upload_to_b2(job_id, public_path, job_dir)
            video_url = b2_url or f"{BASE_URL}/media/{job_id}.mp4"
            upload_time = time.time() - upload_start

            # Record upload metrics
            if public_path.exists():
                video_size = public_path.stat().st_size
                monitor.record_upload_metrics(video_size, upload_time, b2_url is not None)

            log(f"Using video URL: {video_url}")

            # Upload generated images to B2
            images_dir = job_dir / "pipeline" / "images"
            if images_dir.exists() and images_dir.is_dir():
                log(f"Uploading images from: {images_dir}")
                image_urls = upload_images_to_b2(job_id, images_dir)
                if image_urls:
                    log(f"✅ Successfully uploaded {len(image_urls)} images to B2")
                    # Save image URLs to manifest for reference
                    manifest_path = job_dir / "pipeline" / "scenes" / "manifest.json"
                    if manifest_path.exists():
                        try:
                            import json
                            manifest = json.loads(manifest_path.read_text())
                            manifest["b2_image_urls"] = image_urls
                            manifest_path.write_text(json.dumps(manifest, indent=2))
                            log(f"Updated manifest with B2 image URLs")
                        except Exception as e:
                            log(f"[WARNING] Failed to update manifest with image URLs: {e}")
                else:
                    log(f"⚠️ No images uploaded to B2 or upload failed")
            else:
                log(f"[WARNING] Images directory not found: {images_dir}")

        except Exception as e:
            log(f"[WARNING] B2 upload failed, falling back to local URL: {e}")
            video_url = f"{BASE_URL}/media/{job_id}.mp4"
            upload_time = time.time() - upload_start

            # Record failed upload metrics
            if public_path.exists():
                video_size = public_path.stat().st_size
                monitor.record_upload_metrics(video_size, upload_time, False)

        # Send email notification
        try:
            send_link_email(email, video_url, job_id)
            log(f"Email notification sent to {email}")
        except Exception as e:
            log(f"[WARNING] Failed to send email notification: {e}")
            # Don't fail the job if email fails

        log(f"Job {job_id} completed successfully")

        # Record successful completion
        monitor.complete_job_tracking(job_id, "success")

        # Get video duration from the merge script output
        video_duration = None
        try:
            # Check if merge script printed VIDEO_DURATION: <seconds>
            # This would be captured from the subprocess output, but for now we'll get it directly
            import subprocess
            result = subprocess.run([
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(public_path)
            ], capture_output=True, text=True)
            if result.returncode == 0:
                duration_float = float(result.stdout.strip())
                video_duration = int(round(duration_float))
                log(f"[INFO] Video duration: {video_duration} seconds")
        except Exception as e:
            log(f"[WARNING] Could not get video duration: {e}")

        # Save completion status for persistent tracking
        try:
            from app.main import _save_job_completion
            completion_data = {"status": "done", "video_url": video_url}

            # Add video_duration if available
            if video_duration is not None:
                completion_data["video_duration"] = video_duration

            # Add thumbnail_url if images were uploaded and scene_001.png exists
            if image_urls and 'scene_001.png' in image_urls:
                completion_data["thumbnail_url"] = image_urls['scene_001.png']
                log(f"[INFO] Added thumbnail_url to completion data: {completion_data['thumbnail_url']}")

            _save_job_completion(job_id, "done", completion_data)
        except Exception as e:
            log(f"[WARNING] Failed to save completion status: {e}")

        # Cleanup job resources after successful completion
        cleanup_job_resources(job_dir)
        log(f"[CLEANUP] Job {job_id} cleanup completed")

        return {"status": "done", "video_url": video_url}

    except Exception as e:
        log(f"[ERROR] Job {job_id} failed completely: {e}")
        import traceback
        log(f"[ERROR] Full traceback:\n{traceback.format_exc()}")

        # Cleanup even on failure
        if job_dir:
            cleanup_job_resources(job_dir)
            log(f"[CLEANUP] Job {job_id} cleanup completed after failure")

        # Save failure status for persistent tracking
        try:
            from app.main import _save_job_completion
            _save_job_completion(job_id, "failed", {"error": str(e)})
        except Exception as save_error:
            log(f"[WARNING] Failed to save failure status: {save_error}")

        raise
