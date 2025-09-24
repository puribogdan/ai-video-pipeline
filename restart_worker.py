#!/usr/bin/env python3
# restart_worker.py - Restart the RQ worker properly
import os
import sys
import signal
import time
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def log(msg: str) -> None:
    print(f"[RESTART] {msg}", flush=True)

def find_worker_process():
    """Find the RQ worker process"""
    try:
        result = subprocess.run(['pgrep', '-f', 'rq_worker.py'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            return [pid for pid in pids if pid]
        return []
    except Exception as e:
        log(f"Error finding worker process: {e}")
        return []

def kill_worker():
    """Kill existing worker processes"""
    pids = find_worker_process()
    if not pids:
        log("No worker processes found")
        return

    log(f"Found worker processes: {pids}")
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGTERM)
            log(f"Sent SIGTERM to worker {pid}")
        except Exception as e:
            log(f"Error killing worker {pid}: {e}")

    # Wait a bit for graceful shutdown
    time.sleep(2)

    # Force kill if still running
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGKILL)
            log(f"Sent SIGKILL to worker {pid}")
        except:
            pass

def start_worker():
    """Start the RQ worker"""
    log("Starting RQ worker...")
    try:
        # Use subprocess to start worker in background
        proc = subprocess.Popen([sys.executable, 'rq_worker.py'],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              text=True,
                              encoding='utf-8',
                              errors='replace')

        log(f"Worker started with PID {proc.pid}")

        # Monitor startup for a few seconds
        time.sleep(3)

        if proc.poll() is None:
            log("Worker appears to be running successfully")
            return proc
        else:
            log(f"Worker failed to start (exit code: {proc.poll()})")
            return None

    except Exception as e:
        log(f"Error starting worker: {e}")
        return None

def main():
    """Main restart function"""
    log("Restarting RQ worker...")

    # Kill existing workers
    kill_worker()

    # Start new worker
    worker_proc = start_worker()

    if worker_proc:
        log("Worker restart completed successfully")
        log(f"New worker PID: {worker_proc.pid}")
    else:
        log("Worker restart failed")
        sys.exit(1)

if __name__ == "__main__":
    main()