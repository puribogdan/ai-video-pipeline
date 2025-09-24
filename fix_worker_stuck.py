#!/usr/bin/env python3
# fix_worker_stuck.py - Fix stuck worker by clearing failed jobs and restarting
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
    print(f"[FIX] {msg}", flush=True)

def main():
    """Fix stuck worker by clearing failed jobs"""
    try:
        log(f"Connecting to Redis at {REDIS_URL}")

        # Connect to Redis
        redis_conn = Redis.from_url(REDIS_URL)
        redis_conn.ping()
        log("Redis connection successful")

        # Get queue
        queue = Queue(QUEUE_NAME, connection=redis_conn)

        # Clear failed jobs
        failed_jobs = queue.failed_job_registry.get_job_ids()
        log(f"Found {len(failed_jobs)} failed jobs to clear")

        for job_id in failed_jobs:
            try:
                job = queue.fetch_job(job_id)
                if job:
                    queue.failed_job_registry.remove(job)
                    log(f"Cleared failed job: {job_id}")
                else:
                    log(f"Job {job_id} not found, skipping")
            except Exception as e:
                log(f"Error clearing job {job_id}: {e}")

        # Check if queue is empty now
        queue_length = len(queue)
        log(f"Queue length after clearing failed jobs: {queue_length}")

        if queue_length == 0:
            log("Queue is now empty. Worker should become idle.")
        else:
            log("Jobs still in queue. Worker should process them.")

        # Check worker status
        from rq import Worker
        workers = Worker.all(connection=redis_conn)
        log(f"Active workers: {len(workers)}")

        for worker in workers:
            log(f"Worker {worker.name}: {worker.get_state()}")

        log("Worker fix completed")

    except Exception as e:
        log(f"Error during fix: {e}")
        import traceback
        log(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()