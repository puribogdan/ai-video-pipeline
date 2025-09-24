#!/usr/bin/env python3
# check_failed_jobs.py - Check details of failed jobs
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from redis import Redis
from rq import Queue

# Load environment variables
load_dotenv()

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.environ.get("QUEUE_NAME", "video-jobs")

def log(msg: str) -> None:
    print(f"[FAILED] {msg}", flush=True)

def main():
    """Check failed jobs details"""
    try:
        log(f"Connecting to Redis at {REDIS_URL}")

        # Connect to Redis
        redis_conn = Redis.from_url(REDIS_URL)
        redis_conn.ping()
        log("Redis connection successful")

        # Get queue
        queue = Queue(QUEUE_NAME, connection=redis_conn)

        # Check failed jobs
        failed_jobs = queue.failed_job_registry.get_job_ids()
        log(f"Found {len(failed_jobs)} failed jobs")

        for i, job_id in enumerate(failed_jobs):
            try:
                job = queue.fetch_job(job_id)
                if job:
                    log(f"\n--- Failed Job {i+1}: {job_id} ---")
                    log(f"Status: {job.get_status()}")
                    log(f"Created: {job.created_at}")
                    log(f"Started: {getattr(job, 'started_at', 'Never')}")
                    log(f"Ended: {getattr(job, 'ended_at', 'Never')}")

                    if hasattr(job, 'exc_info') and job.exc_info:
                        log(f"Exception: {job.exc_info}")

                    # Check job metadata
                    if hasattr(job, 'meta') and job.meta:
                        log(f"Metadata: {job.meta}")

                    # Check job args
                    if hasattr(job, 'args') and job.args:
                        log(f"Arguments: {job.args}")

                else:
                    log(f"Job {job_id}: Not found")

            except Exception as e:
                log(f"Error checking job {job_id}: {e}")

        log("Failed job analysis completed")

    except Exception as e:
        log(f"Error during analysis: {e}")
        import traceback
        log(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()