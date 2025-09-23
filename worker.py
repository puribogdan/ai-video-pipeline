#!/usr/bin/env python3
# worker.py - Custom worker with idle logging
import os
import sys
import time
import json
from pathlib import Path
from dotenv import load_dotenv
from redis import Redis

# Load environment variables
load_dotenv()

# Add app directory to path so we can import worker_tasks
sys.path.insert(0, str(Path(__file__).parent))

from app.worker_tasks import process_job, log

# Redis configuration (same as main.py)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.environ.get("QUEUE_NAME", "video-jobs")
KEY_PREFIX = os.environ.get("KEY_PREFIX", "")
QUEUE_KEY = f"{KEY_PREFIX}{QUEUE_NAME}"

# Worker configuration
IDLE_TIMEOUT = int(os.getenv("WORKER_IDLE_TIMEOUT", "10"))  # seconds
MAX_RETRIES = int(os.getenv("WORKER_MAX_RETRIES", "3"))

def main():
    """Main worker loop with idle logging"""
    log(f"Starting worker for queue: {QUEUE_KEY}")
    log(f"Idle timeout: {IDLE_TIMEOUT}s, Max retries: {MAX_RETRIES}")

    redis_conn = Redis.from_url(REDIS_URL)
    retry_count = 0

    # Worker loop
    while True:
        try:
            # Use BRPOP with timeout to wait for jobs
            # RQ stores jobs as a list of job IDs in the queue
            job_data = redis_conn.brpop([QUEUE_KEY], timeout=IDLE_TIMEOUT)

            if job_data is None:
                # Queue is idle - log and continue
                print(f"[WAIT] queue={QUEUE_KEY} idle={IDLE_TIMEOUT}s", flush=True)
                retry_count = 0  # Reset retry count on successful idle
                continue

            # We got a job! Reset retry count
            retry_count = 0

            # Parse job data: (queue_key, job_id)
            queue_key, job_id = job_data
            job_id = job_id.decode('utf-8')

            log(f"Received job ID: {job_id}")

            # Get job data from RQ's job hash
            job_hash_key = f"rq:job:{job_id}"
            job_data_dict = redis_conn.hgetall(job_hash_key)

            if not job_data_dict:
                log(f"Warning: No job data found for {job_id}")
                continue

            # Parse job arguments from RQ job data
            # RQ stores function arguments as JSON in the 'data' field
            try:
                job_args_data = job_data_dict.get(b'data', b'')
                if job_args_data:
                    # RQ stores args as JSON array
                    args = json.loads(job_args_data.decode('utf-8'))
                    if len(args) >= 4:
                        job_id_arg = args[0]  # First arg is job_id
                        email = args[1]       # Second arg is email
                        upload_path = args[2] # Third arg is upload_path
                        style = args[3]       # Fourth arg is style

                        log(f"Processing job: {job_id_arg} (email: {email}, style: {style})")

                        # Pickup logging - job started processing
                        print(f"[PROCESSING] job_id={job_id_arg} queue={QUEUE_KEY}", flush=True)

                        try:
                            # Process the job using existing worker_tasks function
                            result = process_job(job_id_arg, email, upload_path, style)
                            log(f"Job {job_id_arg} completed successfully: {result}")

                            # Remove the job from the queue after successful processing
                            redis_conn.lrem(QUEUE_KEY, 0, job_id)

                        except Exception as e:
                            log(f"Job {job_id_arg} failed: {e}")
                            import traceback
                            log(f"Full traceback:\n{traceback.format_exc()}")

                            # Remove failed job from queue
                            redis_conn.lrem(QUEUE_KEY, 0, job_id)
                    else:
                        log(f"Invalid job arguments for {job_id}: {args}")
                        redis_conn.lrem(QUEUE_KEY, 0, job_id)
                else:
                    log(f"No job data found for {job_id}")
                    redis_conn.lrem(QUEUE_KEY, 0, job_id)

            except json.JSONDecodeError as e:
                log(f"Failed to parse job data for {job_id}: {e}")
                redis_conn.lrem(QUEUE_KEY, 0, job_id)
            except Exception as e:
                log(f"Error processing job {job_id}: {e}")
                redis_conn.lrem(QUEUE_KEY, 0, job_id)

        except Exception as e:
            retry_count += 1
            log(f"Worker error (attempt {retry_count}/{MAX_RETRIES}): {e}")

            if retry_count >= MAX_RETRIES:
                log(f"Max retries reached, exiting worker")
                sys.exit(1)

            # Wait before retrying
            wait_time = min(2 ** retry_count, 30)  # Exponential backoff, max 30s
            log(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)

if __name__ == "__main__":
    main()