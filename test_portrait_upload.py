import os
from pathlib import Path
from dotenv import load_dotenv
from app.worker_tasks import upload_to_b2, log

load_dotenv()

# Create a dummy portrait image file
dummy_portrait_path = Path("dummy_portrait.png")
if not dummy_portrait_path.exists():
    # Create a simple 1x1 pixel PNG
    with open(dummy_portrait_path, "wb") as f:
        # Minimal PNG header
        f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\fIDATx\x9cc\xf8\x0f\x00\x00\x01\x00\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82')

# Create a dummy video file
dummy_video_path = Path("dummy_video.mp4")
if not dummy_video_path.exists():
    with open(dummy_video_path, "wb") as f:
        f.write(b"dummy video content")

# Create a job directory structure
job_id = "test-portrait-upload"
job_dir = Path(f"test_job_{job_id}")
job_dir.mkdir(exist_ok=True)

# Copy portrait to the expected location
scenes_dir = job_dir / "pipeline" / "scenes"
scenes_dir.mkdir(parents=True, exist_ok=True)
portrait_dest = scenes_dir / "portrait_ref.png"
if not portrait_dest.exists():
    import shutil
    shutil.copy2(dummy_portrait_path, portrait_dest)

# Test the upload
log(f"Testing portrait upload for job {job_id}")
url = upload_to_b2(job_id, dummy_video_path, job_dir)

print(f"Upload result: {url}")

# Cleanup
if dummy_portrait_path.exists():
    dummy_portrait_path.unlink()
if dummy_video_path.exists():
    dummy_video_path.unlink()
if job_dir.exists():
    import shutil
    shutil.rmtree(job_dir)