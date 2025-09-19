import os
from pathlib import Path
from dotenv import load_dotenv
from app.worker_tasks import upload_to_b2, log

load_dotenv()

# Create a dummy small video file
dummy_path = Path("dummy.mp4")
if not dummy_path.exists():
    with open(dummy_path, "wb") as f:
        f.write(b"dummy video content")  # Small dummy

job_id = "test-upload"
url = upload_to_b2(job_id, dummy_path)
print(f"Upload result: {url}")

if dummy_path.exists():
    dummy_path.unlink()