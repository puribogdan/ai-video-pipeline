# app/main.py
import os
import time
import uuid
import subprocess
import shutil
from pathlib import Path
import json
from datetime import timedelta
from fastapi import Header, Query, HTTPException
from fastapi.responses import FileResponse

from fastapi import FastAPI, Request, UploadFile, Form, File, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from redis import Redis
from rq import Queue, Retry

# Import here to avoid circular imports
from .worker_tasks import process_job

load_dotenv()

APP_ROOT = Path(__file__).resolve().parents[1]
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "/app/uploads"))  # Shared persistent volume
MEDIA_DIR = APP_ROOT / "media"
TEMPLATES = Jinja2Templates(directory=str(APP_ROOT / "app" / "templates"))

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Redis key configuration
QUEUE_NAME = os.environ.get("QUEUE_NAME", "video-jobs")
KEY_PREFIX = os.environ.get("KEY_PREFIX", "")
QUEUE_KEY = f"{KEY_PREFIX}{QUEUE_NAME}"

MEDIA_DIR.mkdir(parents=True, exist_ok=True)

# Enhanced debugging for uploads directory creation (Render server compatible)
print(f"[DEBUG] APP_ROOT: {APP_ROOT}", flush=True)
print(f"[DEBUG] UPLOADS_DIR configured as: {UPLOADS_DIR}", flush=True)
print(f"[DEBUG] UPLOADS_DIR exists before mkdir: {UPLOADS_DIR.exists()}", flush=True)
print(f"[DEBUG] UPLOADS_DIR parent: {UPLOADS_DIR.parent}", flush=True)
print(f"[DEBUG] UPLOADS_DIR parent exists: {UPLOADS_DIR.parent.exists()}", flush=True)

try:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] UPLOADS_DIR created successfully: {UPLOADS_DIR}", flush=True)
    print(f"[DEBUG] UPLOADS_DIR exists after mkdir: {UPLOADS_DIR.exists()}", flush=True)
    print(f"[DEBUG] UPLOADS_DIR is writable: {UPLOADS_DIR.exists() and os.access(UPLOADS_DIR, os.W_OK)}", flush=True)
except Exception as e:
    print(f"[ERROR] Failed to create UPLOADS_DIR {UPLOADS_DIR}: {e}", flush=True)
    print(f"[ERROR] Exception type: {type(e).__name__}", flush=True)
    raise HTTPException(status_code=500, detail=f"Cannot create uploads directory: {e}")

STATIC_DIR = APP_ROOT / "app" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

redis = Redis.from_url(REDIS_URL)
queue = Queue(QUEUE_NAME, connection=redis, default_timeout=5400)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse("upload.html", {"request": request})


def _safe_name(name: str) -> str:
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    cleaned = "".join(ch if ch in keep else "_" for ch in (name or "upload"))
    if not cleaned or cleaned.startswith("."):
        cleaned = f"file_{uuid.uuid4().hex}"
    return cleaned[:128]


_ALLOWED_STYLES = {"kid_friendly_cartoon", "japanese_kawaii", "storybook_illustrated", "watercolor_storybook", "paper_cutout", "cutout_collage", "realistic_3d", "claymation", "needle_felted", "stop_motion_felt_clay", "hybrid_mix", "japanese_anime", "pixel_art", "van_gogh", "impressionism", "art_deco", "cubism", "graphic_novel", "motion_comic", "comic_book", "gothic", "silhouette", "fantasy_magic_glow", "surrealism_hybrid", "ink_parchment", "japanese_woodblock", "ink_wash", "japanese_gold_screen", "japanese_scroll", "japanese_court"}


@app.post("/submit", response_class=HTMLResponse)
async def submit(
    request: Request,
    email: str = Form(...),
    audio: UploadFile = File(...),
    portrait: UploadFile | None = File(default=None),
    style: str = Form("kid"),
):
    # validate style
    style = (style or "kid").lower().strip()
    if style not in _ALLOWED_STYLES:
        raise HTTPException(status_code=400, detail=f"Invalid style '{style}'. Choose one of: {sorted(_ALLOWED_STYLES)}")

    allowed_mimes = {
        "audio/mpeg", "audio/wav", "audio/x-wav",
        "audio/m4a", "audio/x-m4a", "audio/mp4",
        "audio/aac", "audio/ogg",
    }
    if audio.content_type not in allowed_mimes:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {audio.content_type}")

    data = await audio.read()
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")

    job_id = str(uuid.uuid4())
    user_dir = UPLOADS_DIR / job_id

    # Ensure uploads directory is writable
    try:
        test_file = user_dir / ".write_test"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_bytes(b"test")
        test_file.unlink()  # Clean up test file
        print(f"[DEBUG] Uploads directory is writable: {user_dir}", flush=True)
    except Exception as e:
        print(f"[ERROR] Uploads directory not writable: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Upload directory not writable: {e}")

    user_dir.mkdir(parents=True, exist_ok=True)

    # Save audio with original (sanitized) name
    orig_audio_name = _safe_name(audio.filename or "audio_upload")
    audio_path = user_dir / orig_audio_name
    print(f"[DEBUG] Saving audio file as: {audio_path}", flush=True)
    print(f"[DEBUG] Original filename: {audio.filename}", flush=True)
    print(f"[DEBUG] Sanitized filename: {orig_audio_name}", flush=True)
    print(f"[DEBUG] Audio file size to save: {len(data)} bytes", flush=True)
    print(f"[DEBUG] Job directory before save: {user_dir}", flush=True)
    print(f"[DEBUG] Job directory exists: {user_dir.exists()}", flush=True)

    with open(audio_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())  # Force write to disk

    print(f"[DEBUG] File write completed for: {audio_path}", flush=True)

    # Additional verification that file was written correctly
    if not audio_path.exists():
        raise HTTPException(status_code=500, detail=f"Failed to create audio file: {audio_path}")

    actual_size = audio_path.stat().st_size
    if actual_size != len(data):
        audio_path.unlink()  # Remove corrupted file
        raise HTTPException(status_code=500, detail=f"Audio file size mismatch: expected {len(data)} bytes, got {actual_size} bytes")

    # Optional portrait
    if portrait and portrait.filename:
        img = await portrait.read()
        if img:
            portrait_name = _safe_name(portrait.filename)
            (user_dir / portrait_name).write_bytes(img)

    # verify write with more robust checking
    print(f"[DEBUG] Verifying audio file write to {audio_path}", flush=True)
    deadline = time.time() + 10
    verification_attempts = 0
    while time.time() < deadline:
        try:
            verification_attempts += 1
            if audio_path.exists():
                saved_size = audio_path.stat().st_size
                print(f"[DEBUG] Verification attempt {verification_attempts}: file exists, size={saved_size}, expected={len(data)}", flush=True)
                if saved_size == len(data) and saved_size > 0:
                    print(f"[DEBUG] Audio file verification successful", flush=True)
                    break
            else:
                print(f"[DEBUG] Verification attempt {verification_attempts}: file does not exist yet", flush=True)
        except FileNotFoundError as e:
            print(f"[DEBUG] Verification attempt {verification_attempts}: FileNotFoundError: {e}", flush=True)
        except Exception as e:
            print(f"[DEBUG] Verification attempt {verification_attempts}: Unexpected error: {e}", flush=True)
        time.sleep(0.1)

    # Final verification
    if not audio_path.exists():
        raise HTTPException(status_code=500, detail=f"Audio file was not created: {audio_path}")

    final_size = audio_path.stat().st_size
    if final_size != len(data):
        raise HTTPException(status_code=500, detail=f"Audio file size mismatch: expected {len(data)} bytes, got {final_size} bytes")

    if final_size == 0:
        raise HTTPException(status_code=500, detail="Audio file was created but is empty (0 bytes)")

    print(f"[web] saved upload -> {audio_path} ({final_size} bytes)", flush=True)
    try:
        dir_contents = [p.name for p in user_dir.glob('*')]
        print(f"[web] dir listing {user_dir}: {dir_contents}", flush=True)
        print(f"[web] audio file in directory: {audio_path.name in dir_contents}", flush=True)
        print(f"[web] audio path exists: {audio_path.exists()}", flush=True)
        print(f"[web] audio path size: {audio_path.stat().st_size if audio_path.exists() else 'N/A'}", flush=True)

        # Additional verification - check if file is actually readable
        try:
            with open(audio_path, 'rb') as f:
                chunk = f.read(1024)  # Read first 1KB
                print(f"[web] audio file readable, first chunk size: {len(chunk)} bytes", flush=True)
        except Exception as e:
            print(f"[web] ERROR: Cannot read audio file: {e}", flush=True)
            raise HTTPException(status_code=500, detail=f"Audio file not readable after save: {e}")
    except Exception as e:
        print(f"[web] ERROR: Cannot list directory: {e}", flush=True)

    # Enhanced debugging for audio file issues
    print(f"[DEBUG] ===== AUDIO FILE DEBUGGING =====", flush=True)
    print(f"[DEBUG] Original uploaded filename: {audio.filename}", flush=True)
    print(f"[DEBUG] Sanitized filename: {orig_audio_name}", flush=True)
    print(f"[DEBUG] Final audio path: {audio_path}", flush=True)
    print(f"[DEBUG] Audio file exists: {audio_path.exists()}", flush=True)

    if audio_path.exists():
        audio_size = audio_path.stat().st_size
        print(f"[DEBUG] Audio file size: {audio_size} bytes", flush=True)
        print(f"[DEBUG] Audio file path: {audio_path}", flush=True)
        print(f"[DEBUG] Audio file absolute path: {audio_path.absolute()}", flush=True)
        print(f"[DEBUG] Audio file parent directory: {audio_path.parent}", flush=True)
        print(f"[DEBUG] Audio file parent exists: {audio_path.parent.exists()}", flush=True)

        # Check if file is readable
        try:
            with open(audio_path, 'rb') as f:
                header = f.read(64)  # Read first 64 bytes
                print(f"[DEBUG] Audio file header (first 64 bytes): {header[:32]}...", flush=True)
                print(f"[DEBUG] Audio file is readable: True", flush=True)
        except Exception as e:
            print(f"[ERROR] Audio file not readable: {e}", flush=True)
    else:
        print(f"[ERROR] Audio file does not exist at path: {audio_path}", flush=True)

    print(f"[DEBUG] Job directory contents before enqueue: {[p.name for p in user_dir.glob('*')]}", flush=True)
    print(f"[DEBUG] Job directory path: {user_dir}", flush=True)
    print(f"[DEBUG] Upload path string that will be passed to worker: {str(audio_path)}", flush=True)

    # Check for hardcoded filename issue
    if 'input_cut.mp3' in str(audio_path):
        print(f"[WARNING] DETECTED HARDCODED FILENAME in upload path: {audio_path}", flush=True)
    else:
        print(f"[DEBUG] Upload path does not contain hardcoded filename", flush=True)

    print(f"[DEBUG] ================================", flush=True)

    # Force filesystem sync to ensure file is visible to worker (Unix-like systems only)
    try:
        if hasattr(os, 'sync'):
            os.sync()
            print(f"[DEBUG] Filesystem sync completed", flush=True)
        else:
            print(f"[DEBUG] os.sync() not available on this platform (Windows), skipping", flush=True)
    except Exception as e:
        print(f"[WARNING] Filesystem sync failed: {e}", flush=True)

    # Final verification before enqueuing - ensure file is stable and readable
    print(f"[DEBUG] Final verification before enqueuing job...", flush=True)
    verification_start = time.time()
    max_verification_time = 5  # seconds
    
    while time.time() - verification_start < max_verification_time:
        try:
            if audio_path.exists() and audio_path.is_file():
                current_size = audio_path.stat().st_size
                # Verify file is readable
                with open(audio_path, 'rb') as f:
                    f.read(1024)  # Read first 1KB to verify accessibility
                
                # Check size stability
                time.sleep(0.2)
                stable_size = audio_path.stat().st_size
                
                if current_size == stable_size and current_size > 0:
                    print(f"[DEBUG] File verification successful: {audio_path} ({current_size} bytes)", flush=True)
                    break
                else:
                    print(f"[DEBUG] File size changed ({current_size} -> {stable_size}), waiting...", flush=True)
            else:
                print(f"[DEBUG] File not yet accessible, waiting...", flush=True)
        except Exception as e:
            print(f"[DEBUG] Verification attempt failed: {e}", flush=True)
        
        time.sleep(0.1)
    
    # Final check
    if not audio_path.exists():
        raise HTTPException(status_code=500, detail=f"File verification failed: {audio_path} does not exist")
    
    final_size = audio_path.stat().st_size
    if final_size == 0:
        raise HTTPException(status_code=500, detail=f"File verification failed: {audio_path} is empty")
    
    print(f"[DEBUG] File ready for worker: {audio_path} ({final_size} bytes)", flush=True)

    # Enqueue job immediately - file verification above ensures it's ready
    # The worker retry logic will handle any remaining race conditions
    rq_job = queue.enqueue(
        process_job,           # Function to execute
        job_id,                # First argument to process_job
        email,                 # Second argument to process_job
        str(audio_path),       # Third argument to process_job
        style,                 # Fourth argument to process_job
        retry=Retry(max=3, interval=[15, 30, 60]),
        job_timeout=5400,
    )

    print(f"[DEBUG] Job enqueued successfully with ID: {rq_job.id}", flush=True)

    # Store the RQ job ID mapping for status checks
    rq_job_mapping = user_dir / ".rq_job_id"
    rq_job_mapping.write_text(rq_job.id)
    print(f"[DEBUG] Stored RQ job ID mapping: {job_id} -> {rq_job.id}", flush=True)

    return TEMPLATES.TemplateResponse("upload.html", {"request": request, "job_id": job_id})


@app.get("/status/{job_id}")
async def status(job_id: str):
    """Get job status from file-based tracking (primary) and Redis (fallback for active jobs)"""
    from rq.job import Job

    # First, check if job directory exists
    job_dir = UPLOADS_DIR / job_id
    if not job_dir.exists():
        return {"state": "unknown", "error": f"Job directory not found: {job_id}"}

    # Check for completion file first (most reliable method)
    completion_file = job_dir / "completion_status.json"
    if completion_file.exists():
        try:
            with open(completion_file, 'r') as f:
                completion_data = json.load(f)
            # Map old state names to new ones for backward compatibility
            if completion_data.get("state") == "finished":
                completion_data["state"] = "done"
            return completion_data
        except Exception as file_error:
            print(f"[WARNING] Failed to read completion file for job {job_id}: {file_error}", flush=True)

    # If no completion file, try Redis for active jobs
    rq_job_id = None

    # First, try to get the actual RQ job ID from the mapping file
    rq_job_mapping = job_dir / ".rq_job_id"
    if rq_job_mapping.exists():
        try:
            rq_job_id = rq_job_mapping.read_text().strip()
            print(f"[DEBUG] Found RQ job ID mapping: {job_id} -> {rq_job_id}", flush=True)
        except Exception as e:
            print(f"[WARNING] Failed to read RQ job ID mapping: {e}", flush=True)

    # Try to fetch job using RQ job ID if available, otherwise use directory job_id
    lookup_job_id = rq_job_id if rq_job_id else job_id

    try:
        job = Job.fetch(lookup_job_id, connection=redis)
        meta = job.meta or {}

        if job.is_finished:
            # Job completed successfully - save to file for persistence
            _save_job_completion(job_id, "done", job.result)
            return {"state": "done", "result": job.result}
        elif job.is_failed:
            # Job failed - save to file for persistence
            _save_job_completion(job_id, "failed", {"error": str(job.exc_info)[:2000]})
            return {"state": "failed", "exc": str(job.exc_info)[:2000]}
        else:
            # Job still in progress - map RQ states to desired states
            rq_state = job.get_status(refresh=True)
            mapped_state = _map_rq_state_to_custom(rq_state)
            return {"state": mapped_state, "meta": meta}

    except Exception as e:
        # Job not found in Redis - this is normal for completed jobs that have been cleaned up
        print(f"[DEBUG] Job {lookup_job_id} not found in Redis (may be completed): {str(e)}", flush=True)

    # No completion file and no active job - job is likely queued or processing
    # Check if job directory has any content (indicates job was created)
    try:
        dir_contents = list(job_dir.iterdir())
        if dir_contents:
            # Job directory exists with content - job is likely queued or processing
            return {"state": "queue", "message": "Job is queued or processing"}
        else:
            # Empty job directory - job may have been cleaned up
            return {"state": "unknown", "error": f"Job directory exists but is empty: {job_id}"}
    except Exception as e:
        return {"state": "unknown", "error": f"Error checking job directory: {str(e)}"}


def _map_rq_state_to_custom(rq_state: str) -> str:
    """Map RQ job states to custom frontend states"""
    state_mapping = {
        "queued": "queue",
        "started": "active",
        "deferred": "queue",  # Treat deferred jobs as queued
        "processing": "processing",  # Custom state from worker
    }
    return state_mapping.get(rq_state, rq_state)  # Return original state if no mapping found


def _save_job_completion(job_id: str, state: str, result: dict) -> None:
    """Save job completion status to a file for persistence beyond Redis cleanup"""
    try:
        job_dir = UPLOADS_DIR / job_id
        completion_file = job_dir / "completion_status.json"

        completion_data = {
            "state": state,
            "result": result,
            "completed_at": time.time()
        }

        # Ensure directory exists
        completion_file.parent.mkdir(parents=True, exist_ok=True)

        # Write completion status
        with open(completion_file, 'w') as f:
            json.dump(completion_data, f, indent=2)

    except Exception as e:
        # Log error but don't fail the job
        print(f"[WARNING] Failed to save completion status for job {job_id}: {e}", flush=True)


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/health")
async def detailed_health():
    """Comprehensive health check for all dependencies"""
    import shutil

    health_status = {
        "status": "healthy",
        "checks": {},
        "timestamp": time.time()
    }

    # Check Redis connection
    try:
        redis.ping()
        health_status["checks"]["redis"] = {"status": "ok", "message": "Connected successfully"}
    except Exception as e:
        health_status["checks"]["redis"] = {"status": "error", "message": str(e)}
        health_status["status"] = "degraded"

    # Check required environment variables
    required_env_vars = ["OPENAI_API_KEY", "REPLICATE_API_TOKEN"]
    for var in required_env_vars:
        if os.getenv(var):
            health_status["checks"][var] = {"status": "ok", "message": "Set"}
        else:
            health_status["checks"][var] = {"status": "error", "message": "Missing"}
            health_status["status"] = "unhealthy"

    # Check B2 credentials
    b2_vars = ["B2_BUCKET_NAME", "B2_KEY_ID", "B2_APPLICATION_KEY"]
    b2_configured = all(os.getenv(var) for var in b2_vars)
    if b2_configured:
        health_status["checks"]["b2_storage"] = {"status": "ok", "message": "Configured"}
    else:
        health_status["checks"]["b2_storage"] = {"status": "warning", "message": "Not configured (optional)"}

    # Check disk space
    try:
        disk_usage = shutil.disk_usage(UPLOADS_DIR)
        free_gb = disk_usage.free / (1024**3)
        if free_gb < 1:  # Less than 1GB free
            health_status["checks"]["disk_space"] = {"status": "warning", "message": f"Low space: {free_gb:.1f}GB free"}
            health_status["status"] = "degraded"
        else:
            health_status["checks"]["disk_space"] = {"status": "ok", "message": f"{free_gb:.1f}GB free"}
    except Exception as e:
        health_status["checks"]["disk_space"] = {"status": "error", "message": str(e)}
        health_status["status"] = "degraded"

    # Check ffmpeg availability
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            health_status["checks"]["ffmpeg"] = {"status": "ok", "message": "Available"}
        else:
            health_status["checks"]["ffmpeg"] = {"status": "error", "message": "Not available"}
            health_status["status"] = "unhealthy"
    except FileNotFoundError:
        health_status["checks"]["ffmpeg"] = {"status": "error", "message": "Not found"}
        health_status["status"] = "unhealthy"

    return health_status


@app.get("/health/monitoring")
async def monitoring_health():
    """Get comprehensive monitoring system health"""
    try:
        from .monitoring import get_monitor

        monitor = get_monitor()
        status = monitor.get_system_status()

        # Determine overall status
        overall_status = "healthy"
        if status["active_alerts"]:
            if any(alert["severity"] == "critical" for alert in status["active_alerts"]):
                overall_status = "unhealthy"
            elif any(alert["severity"] == "warning" for alert in status["active_alerts"]):
                overall_status = "degraded"

        # Check for unhealthy health checks
        for check_name, check_info in status["health_checks"].items():
            if check_info["status"] == "unhealthy":
                overall_status = "unhealthy"
            elif check_info["status"] == "degraded" and overall_status == "healthy":
                overall_status = "degraded"

        return {
            "status": overall_status,
            "monitoring": status,
            "timestamp": time.time()
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": f"Monitoring health check failed: {str(e)}",
            "timestamp": time.time()
        }


@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics"""
    try:
        from .monitoring import get_monitor

        monitor = get_monitor()
        return monitor.metrics.get_prometheus_metrics()

    except Exception as e:
        return {"error": f"Failed to get metrics: {str(e)}"}


@app.get("/monitoring/status")
async def monitoring_status():
    """Get detailed monitoring status"""
    try:
        from .monitoring import get_monitor

        monitor = get_monitor()
        return monitor.get_system_status()

    except Exception as e:
        return {"error": f"Failed to get monitoring status: {str(e)}"}


@app.get("/monitoring/alerts")
async def get_alerts():
    """Get active alerts"""
    try:
        from .monitoring import get_monitor

        monitor = get_monitor()
        alerts = monitor.alert_manager.get_active_alerts()

        return {
            "alerts": [
                {
                    "id": alert.id,
                    "severity": alert.severity,
                    "title": alert.title,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved
                }
                for alert in alerts
            ],
            "total": len(alerts)
        }

    except Exception as e:
        return {"error": f"Failed to get alerts: {str(e)}"}


@app.post("/monitoring/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve an alert"""
    try:
        from .monitoring import get_monitor

        monitor = get_monitor()
        monitor.alert_manager.resolve_alert(alert_id)

        return {"status": "resolved", "alert_id": alert_id}

    except Exception as e:
        return {"error": f"Failed to resolve alert: {str(e)}"}


@app.get("/error-recovery/status")
async def get_error_recovery_status():
    """Get comprehensive error recovery system status"""
    try:
        from .error_recovery import get_error_recovery_manager
        from .automated_recovery import get_automated_recovery, get_predictive_failure_detector

        error_recovery = get_error_recovery_manager()
        automated_recovery = get_automated_recovery()
        failure_detector = get_predictive_failure_detector()

        return {
            "error_recovery": error_recovery.get_recovery_stats(),
            "automated_workflows": automated_recovery.get_workflow_statistics(),
            "failure_predictions": failure_detector.get_failure_predictions(),
            "timestamp": time.time()
        }

    except Exception as e:
        return {"error": f"Failed to get error recovery status: {str(e)}"}


@app.post("/error-recovery/test/{component}")
async def test_error_recovery(component: str):
    """Test error recovery for a specific component"""
    try:
        from .error_recovery import get_error_recovery_manager

        error_recovery = get_error_recovery_manager()

        # Create a test error based on component
        if component == "network":
            test_error = ConnectionError("Test network error")
        elif component == "storage":
            test_error = FileNotFoundError("Test storage error")
        elif component == "external_service":
            from botocore.exceptions import ClientError
            test_error = ClientError({"Error": {"Code": "TestError", "Message": "Test external service error"}}, "TestOperation")
        else:
            test_error = RuntimeError(f"Test error for component: {component}")

        # Create error context
        context = error_recovery.create_error_context(
            error=test_error,
            component=component,
            operation="test",
            metadata={"test": True}
        )

        # Handle the error with recovery
        result = await error_recovery.handle_error(test_error, context)

        return {
            "success": True,
            "test_component": component,
            "recovery_result": result,
            "recovery_stats": error_recovery.get_recovery_stats()
        }

    except Exception as e:
        return {"error": f"Error recovery test failed: {str(e)}"}


@app.get("/error-recovery/workflows")
async def list_recovery_workflows():
    """List all automated recovery workflows"""
    try:
        from .automated_recovery import get_automated_recovery

        automated_recovery = get_automated_recovery()
        workflows = automated_recovery.list_workflows()

        return {
            "workflows": [
                {
                    "name": w.name,
                    "description": w.description,
                    "enabled": w.enabled,
                    "execution_count": w.execution_count,
                    "last_executed": w.last_executed.isoformat() if w.last_executed else None,
                    "cooldown_period": w.cooldown_period,
                    "max_executions": w.max_executions
                }
                for w in workflows
            ],
            "total_workflows": len(workflows)
        }

    except Exception as e:
        return {"error": f"Failed to list workflows: {str(e)}"}


@app.post("/error-recovery/workflows/{workflow_name}/execute")
async def execute_recovery_workflow(workflow_name: str):
    """Manually execute a recovery workflow"""
    try:
        from .automated_recovery import get_automated_recovery

        automated_recovery = get_automated_recovery()
        workflow = automated_recovery.get_workflow(workflow_name)

        if not workflow:
            return {"error": f"Workflow not found: {workflow_name}"}

        # Execute the workflow manually
        await automated_recovery._execute_workflow(workflow)

        return {
            "success": True,
            "workflow_name": workflow_name,
            "execution_count": workflow.execution_count
        }

    except Exception as e:
        return {"error": f"Failed to execute workflow: {str(e)}"}


@app.websocket("/ws/monitoring")
async def monitoring_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring updates"""
    try:
        from .monitoring import get_monitor

        monitor = get_monitor()
        client_id = str(uuid.uuid4())

        # Accept the connection
        await monitor.websocket_manager.connect(client_id, websocket)

        try:
            while True:
                # Keep the connection alive and listen for client messages
                data = await websocket.receive_text()

                # Client can send ping or other control messages
                if data == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})
                elif data == "get_status":
                    # Send current status to client
                    status = monitor.get_system_status()
                    await websocket.send_json({
                        "type": "status_update",
                        "data": status,
                        "timestamp": time.time()
                    })

        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            monitor.websocket_manager.disconnect(client_id, websocket)

    except Exception as e:
        print(f"WebSocket connection error: {e}")
        await websocket.close()


@app.on_event("startup")
async def startup_monitoring():
    """Start the monitoring system when the application starts"""
    try:
        from .monitoring import get_monitor

        monitor = get_monitor()
        await monitor.start_monitoring()
        print("✅ Audio Pipeline Monitoring System Started")

    except Exception as e:
        print(f"❌ Failed to start monitoring system: {e}")


@app.on_event("shutdown")
async def shutdown_monitoring():
    """Stop the monitoring system when the application shuts down"""
    try:
        from .monitoring import get_monitor

        monitor = get_monitor()
        await monitor.stop_monitoring()
        print("✅ Audio Pipeline Monitoring System Stopped")

    except Exception as e:
        print(f"❌ Error stopping monitoring system: {e}")


@app.on_event("startup")
async def startup_error_recovery():
    """Start the error recovery system when the application starts"""
    try:
        from .error_recovery import get_error_recovery_manager
        from .automated_recovery import start_automated_recovery

        # Initialize error recovery manager
        error_recovery = get_error_recovery_manager()
        print("✅ Error Recovery System Initialized")

        # Start automated recovery workflows
        await start_automated_recovery()
        print("✅ Automated Recovery Workflows Started")

    except Exception as e:
        print(f"❌ Failed to start error recovery system: {e}")


@app.on_event("shutdown")
async def shutdown_error_recovery():
    """Stop the error recovery system when the application shuts down"""
    try:
        from .automated_recovery import stop_automated_recovery

        # Stop automated recovery workflows
        await stop_automated_recovery()
        print("✅ Automated Recovery Workflows Stopped")

    except Exception as e:
        print(f"❌ Error stopping error recovery system: {e}")


@app.get("/monitoring/dashboard")
async def monitoring_dashboard():
    """Serve the monitoring dashboard HTML"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Audio Pipeline Monitoring Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .header {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .status-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            .status-card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .status-healthy { background: #10b981; }
            .status-degraded { background: #f59e0b; }
            .status-unhealthy { background: #ef4444; }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
            }
            .chart-container {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .alerts {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .alert-item {
                padding: 10px;
                margin: 5px 0;
                border-radius: 4px;
                border-left: 4px solid;
            }
            .alert-warning { border-left-color: #f59e0b; background: #fef3c7; }
            .alert-error { border-left-color: #ef4444; background: #fee2e2; }
            .alert-critical { border-left-color: #dc2626; background: #fecaca; }
            .connection-status {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 10px;
                border-radius: 4px;
                font-size: 12px;
            }
            .connected { background: #10b981; color: white; }
            .disconnected { background: #ef4444; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎵 Audio Pipeline Monitoring Dashboard</h1>
                <p>Real-time monitoring and metrics for the audio processing pipeline</p>
                <div id="connection-status" class="connection-status disconnected">
                    Disconnected
                </div>
            </div>

            <div class="status-grid">
                <div class="status-card">
                    <h3>System Health</h3>
                    <div id="health-checks">
                        <p>Loading...</p>
                    </div>
                </div>

                <div class="status-card">
                    <h3>Active Alerts</h3>
                    <div id="active-alerts">
                        <p>Loading...</p>
                    </div>
                </div>

                <div class="status-card">
                    <h3>System Metrics</h3>
                    <div id="system-metrics">
                        <p>Loading...</p>
                    </div>
                </div>
            </div>

            <div class="metrics-grid">
                <div class="chart-container">
                    <h3>Job Processing Times</h3>
                    <canvas id="processing-chart" width="400" height="200"></canvas>
                </div>

                <div class="chart-container">
                    <h3>System Resource Usage</h3>
                    <canvas id="resource-chart" width="400" height="200"></canvas>
                </div>
            </div>

            <div class="alerts">
                <h3>Recent Alerts</h3>
                <div id="alerts-list">
                    <p>Loading...</p>
                </div>
            </div>
        </div>

        <script>
            let socket;
            let processingChart;
            let resourceChart;
            let statusInterval;

            function connect() {
                socket = new WebSocket('ws://' + window.location.host + '/ws/monitoring');

                socket.onopen = function(e) {
                    document.getElementById('connection-status').textContent = 'Connected';
                    document.getElementById('connection-status').className = 'connection-status connected';

                    // Request initial status
                    socket.send('get_status');

                    // Start periodic status updates
                    statusInterval = setInterval(() => {
                        socket.send('get_status');
                    }, 30000);
                };

                socket.onmessage = function(event) {
                    const data = JSON.parse(event.data);

                    if (data.type === 'monitoring_update') {
                        updateDashboard(data);
                    } else if (data.type === 'status_update') {
                        updateStatus(data.data);
                    } else if (data.type === 'pong') {
                        // Keep connection alive
                    }
                };

                socket.onclose = function(event) {
                    document.getElementById('connection-status').textContent = 'Disconnected';
                    document.getElementById('connection-status').className = 'connection-status disconnected';

                    clearInterval(statusInterval);

                    // Attempt to reconnect after 5 seconds
                    setTimeout(connect, 5000);
                };

                socket.onerror = function(error) {
                    console.error('WebSocket error:', error);
                };
            }

            function updateDashboard(data) {
                updateHealthChecks(data.health_checks);
                updateAlerts(data.alerts);
            }

            function updateStatus(data) {
                updateHealthChecks(data.health_checks);
                updateSystemMetrics(data.system_metrics);
                updateAlertsFromStatus(data.active_alerts);
            }

            function updateHealthChecks(healthChecks) {
                const container = document.getElementById('health-checks');
                let html = '';

                for (const [name, check] of Object.entries(healthChecks)) {
                    const statusClass = `status-${check.status}`;
                    html += `
                        <div style="margin: 10px 0;">
                            <span class="status-indicator ${statusClass}"></span>
                            <strong>${name}:</strong> ${check.message}
                        </div>
                    `;
                }

                container.innerHTML = html;
            }

            function updateSystemMetrics(metrics) {
                const container = document.getElementById('system-metrics');
                container.innerHTML = `
                    <div><strong>Active Jobs:</strong> ${metrics.active_jobs}</div>
                    <div><strong>Tracked Jobs:</strong> ${metrics.tracked_jobs}</div>
                    <div><strong>WebSocket Connections:</strong> ${metrics.websocket_connections}</div>
                `;
            }

            function updateAlerts(alerts) {
                updateAlertsFromStatus(alerts);
            }

            function updateAlertsFromStatus(alerts) {
                const alertsContainer = document.getElementById('active-alerts');
                const alertsList = document.getElementById('alerts-list');

                if (alerts.length === 0) {
                    alertsContainer.innerHTML = '<p>No active alerts</p>';
                    alertsList.innerHTML = '<p>No recent alerts</p>';
                    return;
                }

                // Update active alerts count
                alertsContainer.innerHTML = `<p><strong>${alerts.length}</strong> active alerts</p>`;

                // Update alerts list
                let html = '';
                for (const alert of alerts) {
                    const alertClass = `alert-${alert.severity}`;
                    html += `
                        <div class="alert-item ${alertClass}">
                            <div><strong>${alert.title}</strong></div>
                            <div>${alert.message}</div>
                            <div style="font-size: 12px; color: #666;">
                                ${new Date(alert.timestamp).toLocaleString()}
                            </div>
                        </div>
                    `;
                }

                alertsList.innerHTML = html;
            }

            function initCharts() {
                // Initialize processing time chart
                const processingCtx = document.getElementById('processing-chart').getContext('2d');
                processingChart = new Chart(processingCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Processing Time (seconds)',
                            data: [],
                            borderColor: 'rgb(75, 192, 192)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });

                // Initialize resource usage chart
                const resourceCtx = document.getElementById('resource-chart').getContext('2d');
                resourceChart = new Chart(resourceCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [
                            {
                                label: 'CPU Usage (%)',
                                data: [],
                                borderColor: 'rgb(255, 99, 132)',
                                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                                tension: 0.1
                            },
                            {
                                label: 'Memory Usage (%)',
                                data: [],
                                borderColor: 'rgb(54, 162, 235)',
                                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                                tension: 0.1
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 100
                            }
                        }
                    }
                });
            }

            // Initialize dashboard when page loads
            window.onload = function() {
                initCharts();
                connect();
            };

            // Send ping every 30 seconds to keep connection alive
            setInterval(() => {
                if (socket && socket.readyState === WebSocket.OPEN) {
                    socket.send('ping');
                }
            }, 30000);
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()


@app.get("/admin/prompts/{job_id}")
async def get_prompt_json(
    job_id: str,
    x_admin_key: str | None = Header(default=None),
    key: str | None = Query(default=None),
):
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=500, detail="ADMIN_TOKEN not set on server")
    supplied = x_admin_key or key
    if supplied != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    p = UPLOADS_DIR / job_id / "pipeline" / "scenes" / "prompt.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="prompt.json not found")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read prompt.json: {e}")
    return JSONResponse(data)


@app.get("/admin/script/{job_id}")
async def admin_download_script(
    job_id: str,
    key: str | None = Query(default=None),
    x_admin_key: str | None = Header(default=None),
):
    admin_token = os.getenv("ADMIN_TOKEN", "")
    if not admin_token or (key != admin_token and x_admin_key != admin_token):
        raise HTTPException(status_code=403, detail="Forbidden")

    path = UPLOADS_DIR / job_id / "pipeline" / "scripts" / "input_script.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="script not found")

    return FileResponse(path, media_type="application/json", filename=f"{job_id}_script.json")
