import os
BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000")


REQUIRED_ENV = ["OPENAI_API_KEY", "REPLICATE_API_TOKEN"]




def _copy_pipeline_to(job_dir: Path) -> None:
if not PIPELINE_SRC.exists():
raise RuntimeError(f"Pipeline folder not found: {PIPELINE_SRC}")
# Copy the pipeline folder fresh for isolation per job
shutil.copytree(PIPELINE_SRC, job_dir / "pipeline")




def _run_make_video(job_dir: Path, audio_src: Path) -> Path:
"""Run make_video.py inside an isolated job workspace.
Returns path to final_video.mp4
"""
pipe_dir = job_dir / "pipeline"
audio_input_dir = pipe_dir / "audio_input"
audio_input_dir.mkdir(parents=True, exist_ok=True)


# Put the uploaded audio where the pipeline expects it
target_audio = audio_input_dir / "input.mp3"
shutil.copy2(audio_src, target_audio)


# Ensure env for the pipeline
env = os.environ.copy()
for key in REQUIRED_ENV:
if not env.get(key):
raise RuntimeError(f"Missing required env: {key}")


# Run the pipeline (uses your non-overwriting make_video.py)
cmd = [sys.executable, "make_video.py", "--job-id", job_dir.name]
subprocess.run(cmd, cwd=str(pipe_dir), check=True)


final_video = pipe_dir / "final_video.mp4"
if not final_video.exists():
raise RuntimeError("final_video.mp4 not produced by pipeline")
return final_video




def process_job(job_id: str, email: str, upload_path: str) -> Dict[str, str]:
"""Background task executed by RQ worker."""
job_dir = (UPLOADS_DIR / job_id)
job_dir.mkdir(parents=True, exist_ok=True)


# Keep original upload
src_audio = Path(upload_path)
if not src_audio.exists():
raise RuntimeError("Uploaded audio missing")


# Copy pipeline and run
_copy_pipeline_to(job_dir)
final_video = _run_make_video(job_dir, src_audio)


# Move the final video into public media and construct a link
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
public_name = f"{job_id}.mp4"
public_path = MEDIA_DIR / public_name
shutil.copy2(final_video, public_path)


video_url = f"{BASE_URL}/media/{public_name}"


# Email the user
send_link_email(email, video_url, job_id)


return {"status": "done", "video_url": video_url}