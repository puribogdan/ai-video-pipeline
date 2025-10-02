#!/usr/bin/env python3
# rq_worker.py - Custom RQ worker with enhanced logging
import os
import sys
import time
import threading
from pathlib import Path
from dotenv import load_dotenv
from redis import Redis
from rq import Worker, Queue

# Load environment variables
load_dotenv()

# Add app directory to path so we can import worker_tasks
sys.path.insert(0, str(Path(__file__).parent))

try:
    from app import process_job
    from app.worker_tasks import log
    print(f"[DEBUG] Successfully imported process_job and log functions", flush=True)
except Exception as e:
    print(f"[ERROR] Failed to import functions: {e}", flush=True)
    import traceback
    print(f"[ERROR] Import traceback: {traceback.format_exc()}", flush=True)
    # Define simple functions if import fails
    def log(msg):
        print(f"[worker] {msg}", flush=True)

    def process_job(*args, **kwargs):
        print(f"[ERROR] process_job function not available: {e}", flush=True)
        raise RuntimeError(f"process_job function not imported: {e}")

# Redis configuration (same as main.py)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.environ.get("QUEUE_NAME", "video-jobs")
KEY_PREFIX = os.environ.get("KEY_PREFIX", "")
QUEUE_KEY = f"{KEY_PREFIX}{QUEUE_NAME}"

# Worker configuration
IDLE_TIMEOUT = int(os.getenv("WORKER_IDLE_TIMEOUT", "10"))  # seconds
HEARTBEAT_INTERVAL = int(os.getenv("WORKER_HEARTBEAT_INTERVAL", "30"))  # seconds

class LoggingWorker(Worker):
    """Custom RQ Worker with enhanced logging capabilities"""

    def __init__(self, queues, connection=None, **kwargs):
        try:
            # Boot logging BEFORE calling super().__init__
            print(f"[BOOT] Starting RQ worker for queue={QUEUE_KEY}", flush=True)

            print(f"[DEBUG] Calling super().__init__ with queues={len(queues)} queues", flush=True)
            super().__init__(queues, connection=connection, **kwargs)
            print(f"[DEBUG] super().__init__ completed successfully", flush=True)

            # Get Redis connection info
            redis_conn = self.connection or Redis.from_url(REDIS_URL)
            try:
                connection = redis_conn.connection_pool.get_connection('_')
                redis_host = getattr(connection, 'host', 'localhost')
                redis_port = getattr(connection, 'port', 6379)
                redis_db = getattr(redis_conn, 'db', 0)
            except Exception as e:
                print(f"[DEBUG] Error getting Redis connection info: {e}", flush=True)
                redis_host = 'localhost'
                redis_port = 6379
                redis_db = 0

            print(f"[BOOT] service=worker queue={QUEUE_KEY} redis={redis_host}:{redis_port}/{redis_db}", flush=True)
            print(f"[BOOT] worker_name={self.name} heartbeat={HEARTBEAT_INTERVAL}s idle_timeout={IDLE_TIMEOUT}s", flush=True)

            # Idle detection
            self.last_job_time = time.time()
            self.idle_timer = None
            self.start_idle_timer()

            # Heartbeat thread
            self.heartbeat_thread = None
            self.start_heartbeat()

            print(f"[DEBUG] Worker initialization completed successfully", flush=True)

        except Exception as e:
            print(f"[ERROR] Worker initialization failed: {e}", flush=True)
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}", flush=True)
            # Don't re-raise the exception, let the idle timer still work for debugging
            raise  # Re-raise to prevent silent failures

    def start_idle_timer(self):
        """Start the idle timer that logs when no jobs are processed"""
        def idle_check():
            while True:
                time.sleep(IDLE_TIMEOUT)
                current_time = time.time()
                if current_time - self.last_job_time >= IDLE_TIMEOUT:
                    print(f"[WAIT] queue={QUEUE_KEY} idle={IDLE_TIMEOUT}s", flush=True)

        self.idle_timer = threading.Thread(target=idle_check, daemon=True)
        self.idle_timer.start()

    def start_heartbeat(self):
        """Start the heartbeat thread for periodic status logging"""
        def heartbeat():
            while True:
                time.sleep(HEARTBEAT_INTERVAL)
                try:
                    status = "active" if self.get_state() == "busy" else "idle"
                    print(f"[HEARTBEAT] worker={self.name} status={status} queue={QUEUE_KEY}", flush=True)
                except:
                    print(f"[HEARTBEAT] worker={self.name} status=unknown queue={QUEUE_KEY}", flush=True)

        self.heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        self.heartbeat_thread.start()

    def perform_job(self, job, queue):
        """Override perform_job to add processing and completion logging"""
        job_id = job.id

        # Processing log
        print(f"[PROCESSING] job_id={job_id} queue={QUEUE_KEY}", flush=True)

        # Update last job time to reset idle timer
        self.last_job_time = time.time()

        try:
            # Call the original perform_job method
            result = super().perform_job(job, queue)

            # Done log
            print(f"[DONE] job_id={job_id} queue={QUEUE_KEY}", flush=True)

            return result

        except Exception as e:
            # Error log
            print(f"[ERROR] job_id={job_id} queue={QUEUE_KEY} error={str(e)}", flush=True)
            raise

    def handle_exception(self, job, *exc_info):
        """Override to add error logging"""
        job_id = job.id if job else "unknown"
        print(f"[ERROR] job_id={job_id} queue={QUEUE_KEY} exc_info={exc_info[1]}", flush=True)
        return super().handle_exception(job, *exc_info)

def main():
    """Main function to start the custom worker"""
    try:
        print(f"[DEBUG] Starting main function", flush=True)
        print(f"[DEBUG] Connecting to Redis at {REDIS_URL}", flush=True)

        # Connect to Redis
        redis_conn = Redis.from_url(REDIS_URL)
        try:
            redis_conn.ping()  # Test connection
            print(f"[DEBUG] Redis connection successful", flush=True)
        except Exception as e:
            print(f"[ERROR] Redis connection failed: {e}", flush=True)
            print(f"[ERROR] Please check REDIS_URL: {REDIS_URL}", flush=True)
            sys.exit(1)

        # Create queue
        queue = Queue(QUEUE_NAME, connection=redis_conn)
        print(f"[DEBUG] Queue created: {QUEUE_NAME}", flush=True)

        # Create and start custom worker
        print(f"[DEBUG] Creating LoggingWorker...", flush=True)
        worker = LoggingWorker([queue], connection=redis_conn)
        print(f"[DEBUG] Worker created: {worker.name}", flush=True)

        # Start working
        print(f"[DEBUG] Starting worker.work()", flush=True)
        worker.work()
        print(f"[DEBUG] worker.work() completed", flush=True)

    except Exception as e:
        print(f"[ERROR] Failed to start worker: {e}", flush=True)
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()