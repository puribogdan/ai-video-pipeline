#!/usr/bin/env python3
# check_queue_status.py - Diagnostic script for queue issues
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from redis import Redis
from rq import Queue, Worker

# Load environment variables
load_dotenv()

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.environ.get("QUEUE_NAME", "video-jobs")

def log(msg: str) -> None:
    print(f"[DIAG] {msg}", flush=True)

def success(msg: str) -> None:
    print(f"[DIAG] [OK] {msg}", flush=True)

def error(msg: str) -> None:
    print(f"[DIAG] [ERROR] {msg}", flush=True)

def main():
    """Check queue and worker status"""
    try:
        log(f"Connecting to Redis at {REDIS_URL}")

        # Connect to Redis
        redis_conn = Redis.from_url(REDIS_URL)

        # Test connection
        redis_conn.ping()
        success("Redis connection successful")

        # Check queue length
        queue = Queue(QUEUE_NAME, connection=redis_conn)
        queue_length = len(queue)
        success(f"Queue '{QUEUE_NAME}' length: {queue_length}")

        # Check for jobs in queue
        if queue_length > 0:
            log("Jobs in queue:")
            for i, job_id in enumerate(queue.job_ids):
                try:
                    job = queue.fetch_job(job_id)
                    if job:
                        status = job.get_status()
                        log(f"  {i+1}. Job {job_id}: {status}")
                        if hasattr(job, 'exc_info') and job.exc_info:
                            log(f"     Error: {str(job.exc_info)[:100]}...")
                    else:
                        log(f"  {i+1}. Job {job_id}: Not found")
                except Exception as e:
                    log(f"  {i+1}. Job {job_id}: Error fetching - {e}")
        else:
            log("No jobs in queue")

        # Check for active workers
        workers = Worker.all(connection=redis_conn)
        log(f"Active workers: {len(workers)}")

        for worker in workers:
            log(f"  Worker {worker.name}: {worker.get_state()}")
            # Note: current_job attribute may not exist in all RQ versions
            current_job = getattr(worker, 'current_job', None)
            if current_job:
                log(f"    Current job: {current_job.id}")

        # Check failed jobs
        failed_jobs = queue.failed_job_registry.get_job_ids()
        log(f"Failed jobs: {len(failed_jobs)}")

        if failed_jobs:
            log("Failed job IDs:")
            for job_id in failed_jobs[:5]:  # Show first 5
                log(f"  {job_id}")

        # Check deferred jobs
        deferred_jobs = queue.deferred_job_registry.get_job_ids()
        log(f"Deferred jobs: {len(deferred_jobs)}")

        if deferred_jobs:
            log("Deferred job IDs:")
            for job_id in deferred_jobs[:5]:  # Show first 5
                log(f"  {job_id}")

        success("Queue diagnostic completed successfully")

    except Exception as e:
        error(f"Error during diagnosis: {e}")
        import traceback
        log(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()