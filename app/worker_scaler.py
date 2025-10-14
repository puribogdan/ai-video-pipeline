# app/worker_scaler.py
import asyncio
import json
import psutil
import subprocess
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import redis
from redis import Redis

from .resource_manager import get_resource_manager, ResourceType
from .job_scheduler import get_scheduler
from .monitoring import get_monitor, StructuredLogger


class WorkerStatus(Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class ScalingStrategy(Enum):
    CONSERVATIVE = "conservative"  # Scale slowly, prioritize stability
    AGGRESSIVE = "aggressive"      # Scale quickly, prioritize throughput
    BALANCED = "balanced"          # Moderate scaling for mixed workloads


@dataclass
class WorkerNode:
    worker_id: str
    process_id: Optional[int]
    status: WorkerStatus
    host: str
    port: int
    started_at: datetime
    last_heartbeat: datetime
    jobs_processed: int
    current_job: Optional[str] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingDecision:
    action: str  # "scale_up", "scale_down", "no_action"
    target_workers: int
    reason: str
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkerManager:
    """Manages a pool of RQ workers with adaptive scaling"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = Redis.from_url(redis_url)
        self.logger = StructuredLogger("worker_manager")

        # Worker tracking
        self.workers: Dict[str, WorkerNode] = {}
        self.worker_processes: Dict[str, subprocess.Popen] = {}

        # Configuration
        self.min_workers = 1
        self.max_workers = 8
        self.target_queue_latency = 30.0  # seconds
        self.scale_up_threshold = 5  # jobs per worker
        self.scale_down_threshold = 1  # jobs per worker
        self.heartbeat_interval = 30  # seconds
        self.worker_timeout = 300  # seconds

        # Scaling configuration
        self.scaling_strategy = ScalingStrategy.BALANCED
        self.scale_cooldown = 60  # seconds between scaling actions
        self.last_scaling_action = 0

        # Resource management
        self.resource_manager = get_resource_manager()
        self.scheduler = get_scheduler()

        # Background tasks
        self._monitor_task = None
        self._heartbeat_task = None
        self._scaling_task = None
        self._running = False

        # Locks
        self._worker_lock = threading.Lock()

        # Performance tracking
        self.scaling_history: deque = deque(maxlen=100)
        self.worker_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

    def start_worker(self, worker_id: Optional[str] = None) -> Optional[str]:
        """Start a new RQ worker"""
        if worker_id is None:
            worker_id = str(uuid.uuid4())

        try:
            # Check if we can start more workers
            if len(self.workers) >= self.max_workers:
                self.logger.warning("Maximum worker count reached",
                                  context={"max_workers": self.max_workers})
                return None

            # Check resource availability
            resource_status = self.resource_manager.get_system_resource_status()
            cpu_usage = resource_status.get("cpu", {}).get("current_usage_percent", 0)
            memory_usage = resource_status.get("memory", {}).get("usage_percent", 0)

            if cpu_usage > 90 or memory_usage > 90:
                self.logger.warning("Insufficient resources for new worker",
                                  context={"cpu_usage": cpu_usage, "memory_usage": memory_usage})
                return None

            # Start worker process
            cmd = ["python", "rq_worker.py"]

            # Set environment variables for the worker
            env = {
                "REDIS_URL": self.redis.connection_pool.connection_kwargs.get("url", "redis://localhost:6379/0"),
                "QUEUE_NAME": "video-jobs",
                "WORKER_IDLE_TIMEOUT": "10",
                "WORKER_HEARTBEAT_INTERVAL": "30"
            }

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**dict(os.environ), **env}
            )

            # Create worker node record
            worker_node = WorkerNode(
                worker_id=worker_id,
                process_id=process.pid,
                status=WorkerStatus.STARTING,
                host="localhost",  # In a distributed setup, this would be different
                port=6379,
                started_at=datetime.utcnow(),
                last_heartbeat=datetime.utcnow(),
                jobs_processed=0
            )

            with self._worker_lock:
                self.workers[worker_id] = worker_node
                self.worker_processes[worker_id] = process

            self.logger.info("Worker started",
                           context={"worker_id": worker_id, "pid": process.pid})

            return worker_id

        except Exception as e:
            self.logger.error("Failed to start worker",
                            context={"error": str(e), "worker_id": worker_id})
            return None

    def stop_worker(self, worker_id: str) -> bool:
        """Stop a specific worker"""
        with self._worker_lock:
            if worker_id not in self.workers:
                return False

            worker = self.workers[worker_id]

            if worker.status == WorkerStatus.STOPPED:
                return True

            # Update status
            worker.status = WorkerStatus.STOPPING

            # Terminate process
            if worker_id in self.worker_processes:
                process = self.worker_processes[worker_id]
                try:
                    process.terminate()

                    # Wait a bit for graceful shutdown
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        # Force kill if it doesn't shut down gracefully
                        process.kill()
                        process.wait()

                    del self.worker_processes[worker_id]

                except Exception as e:
                    self.logger.error("Error stopping worker process",
                                    context={"worker_id": worker_id, "error": str(e)})

            # Update worker status
            worker.status = WorkerStatus.STOPPED

            self.logger.info("Worker stopped",
                           context={"worker_id": worker_id})

            return True

    def get_optimal_worker_count(self) -> Tuple[int, str]:
        """Calculate the optimal number of workers based on current conditions"""
        try:
            # Get queue status
            scheduler_status = self.scheduler.get_scheduler_status()
            queued_jobs = scheduler_status.get("queued_jobs", 0)
            running_jobs = scheduler_status.get("running_jobs", 0)
            total_jobs = queued_jobs + running_jobs

            # Get resource status
            resource_status = self.resource_manager.get_system_resource_status()
            cpu_usage = resource_status.get("cpu", {}).get("current_usage_percent", 0)
            memory_usage = resource_status.get("memory", {}).get("usage_percent", 0)
            disk_usage = resource_status.get("disk", {}).get("usage_percent", 0)

            # Get current worker performance
            active_workers = len([w for w in self.workers.values() if w.status == WorkerStatus.RUNNING])
            avg_jobs_per_worker = active_workers if active_workers > 0 else 1

            # Base calculation on queue length
            if total_jobs == 0:
                optimal = self.min_workers
                reason = "No jobs in queue"
            else:
                # Calculate based on jobs per worker ratio
                target_workers_by_queue = max(self.min_workers, min(self.max_workers,
                    int(total_jobs / self.scale_up_threshold) + 1))

                # Adjust for resource availability
                resource_factor = 1.0
                if cpu_usage > 80 or memory_usage > 85:
                    resource_factor = 0.7  # Reduce target if resources are constrained
                elif cpu_usage < 50 and memory_usage < 60:
                    resource_factor = 1.3  # Increase target if resources are plentiful

                optimal = max(self.min_workers, min(self.max_workers,
                    int(target_workers_by_queue * resource_factor)))

                reason = f"Queue: {total_jobs} jobs, Resources: CPU {cpu_usage}%, Memory {memory_usage}%"

            return optimal, reason

        except Exception as e:
            self.logger.error("Error calculating optimal worker count",
                            context={"error": str(e)})
            return self.min_workers, f"Error: {str(e)}"

    def make_scaling_decision(self) -> ScalingDecision:
        """Make a decision about scaling workers"""
        current_time = time.time()

        # Check cooldown period
        if current_time - self.last_scaling_action < self.scale_cooldown:
            return ScalingDecision(
                action="no_action",
                target_workers=len([w for w in self.workers.values() if w.status == WorkerStatus.RUNNING]),
                reason="Cooldown period active",
                confidence=1.0,
                timestamp=datetime.utcnow()
            )

        # Get current state
        active_workers = len([w for w in self.workers.values() if w.status == WorkerStatus.RUNNING])
        optimal_count, reason = self.get_optimal_worker_count()

        # Make decision based on strategy
        if optimal_count > active_workers:
            action = "scale_up"
            target = min(optimal_count, self.max_workers)
            confidence = min(1.0, (optimal_count - active_workers) / 3.0)  # Higher confidence for larger gaps
            reason = f"{reason} - Need {target} workers, have {active_workers}"

        elif optimal_count < active_workers and active_workers > self.min_workers:
            # Check if we can safely scale down
            idle_workers = self._get_idle_workers()
            if idle_workers:
                action = "scale_down"
                target = max(optimal_count, self.min_workers)
                confidence = 0.8  # High confidence for scaling down idle workers
                reason = f"{reason} - Can scale down {len(idle_workers)} idle workers"
            else:
                action = "no_action"
                target = active_workers
                confidence = 1.0
                reason = f"{reason} - No idle workers to scale down"

        else:
            action = "no_action"
            target = active_workers
            confidence = 1.0
            reason = f"{reason} - Current count {active_workers} is optimal"

        return ScalingDecision(
            action=action,
            target_workers=target,
            reason=reason,
            confidence=confidence,
            timestamp=datetime.utcnow()
        )

    def _get_idle_workers(self) -> List[str]:
        """Get list of idle worker IDs"""
        idle_workers = []

        for worker_id, worker in self.workers.items():
            if worker.status == WorkerStatus.RUNNING:
                # Check if worker has been idle for a while
                time_since_heartbeat = (datetime.utcnow() - worker.last_heartbeat).total_seconds()

                if time_since_heartbeat > self.worker_timeout:
                    idle_workers.append(worker_id)

        return idle_workers

    def execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Execute a scaling decision"""
        try:
            active_workers = len([w for w in self.workers.values() if w.status == WorkerStatus.RUNNING])

            if decision.action == "scale_up":
                workers_to_start = decision.target_workers - active_workers

                for _ in range(workers_to_start):
                    worker_id = self.start_worker()
                    if worker_id:
                        self.logger.info("Started worker for scale up",
                                       context={"worker_id": worker_id, "decision": decision.reason})

            elif decision.action == "scale_down":
                workers_to_stop = active_workers - decision.target_workers
                idle_workers = self._get_idle_workers()

                for worker_id in idle_workers[:workers_to_stop]:
                    if self.stop_worker(worker_id):
                        self.logger.info("Stopped worker for scale down",
                                       context={"worker_id": worker_id, "decision": decision.reason})

            # Record scaling action
            self.last_scaling_action = time.time()
            self.scaling_history.append(decision)

            # Update monitoring
            self._update_scaling_metrics(decision)

            return True

        except Exception as e:
            self.logger.error("Error executing scaling decision",
                            context={"error": str(e), "decision": decision.reason})
            return False

    def _update_scaling_metrics(self, decision: ScalingDecision):
        """Update scaling-related metrics"""
        try:
            active_workers = len([w for w in self.workers.values() if w.status == WorkerStatus.RUNNING])

            self.resource_manager.monitor.metrics.set_gauge("worker_manager_active_workers", active_workers)
            self.resource_manager.monitor.metrics.set_gauge("worker_manager_target_workers", decision.target_workers)
            self.resource_manager.monitor.metrics.set_gauge("worker_manager_scaling_confidence", decision.confidence)

            # Record scaling action
            action_type = 1 if decision.action == "scale_up" else (-1 if decision.action == "scale_down" else 0)
            self.resource_manager.monitor.metrics.increment_counter("worker_scaling_actions_total",
                                                                  labels={"action": decision.action})

        except Exception as e:
            self.logger.error("Error updating scaling metrics",
                            context={"error": str(e)})

    def update_worker_heartbeat(self, worker_id: str, jobs_processed: int = 0,
                              current_job: Optional[str] = None,
                              resource_usage: Optional[Dict[str, float]] = None):
        """Update worker heartbeat information"""
        with self._worker_lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                worker.last_heartbeat = datetime.utcnow()
                worker.jobs_processed = jobs_processed
                worker.current_job = current_job

                if resource_usage:
                    worker.resource_usage.update(resource_usage)

                # Update status if needed
                if worker.status == WorkerStatus.STARTING:
                    worker.status = WorkerStatus.RUNNING

    def cleanup_failed_workers(self):
        """Clean up workers that have failed or stopped unexpectedly"""
        failed_workers = []

        with self._worker_lock:
            for worker_id, worker in self.workers.items():
                # Check if process is still running
                if worker_id in self.worker_processes:
                    process = self.worker_processes[worker_id]
                    if process.poll() is not None:  # Process has terminated
                        failed_workers.append(worker_id)

                # Check for heartbeat timeout
                time_since_heartbeat = (datetime.utcnow() - worker.last_heartbeat).total_seconds()
                if time_since_heartbeat > self.worker_timeout * 2:  # Double timeout for safety
                    failed_workers.append(worker_id)

        # Clean up failed workers
        for worker_id in failed_workers:
            self.logger.warning("Cleaning up failed worker",
                              context={"worker_id": worker_id})

            with self._worker_lock:
                if worker_id in self.workers:
                    worker = self.workers[worker_id]
                    worker.status = WorkerStatus.FAILED

                    # Remove process if it exists
                    if worker_id in self.worker_processes:
                        del self.worker_processes[worker_id]

    def get_worker_status(self) -> Dict[str, Any]:
        """Get comprehensive worker status"""
        with self._worker_lock:
            workers_by_status = defaultdict(list)
            for worker in self.workers.values():
                workers_by_status[worker.status.value].append({
                    "worker_id": worker.worker_id,
                    "process_id": worker.process_id,
                    "host": worker.host,
                    "started_at": worker.started_at.isoformat(),
                    "last_heartbeat": worker.last_heartbeat.isoformat(),
                    "jobs_processed": worker.jobs_processed,
                    "current_job": worker.current_job,
                    "resource_usage": worker.resource_usage
                })

            active_workers = len([w for w in self.workers.values() if w.status == WorkerStatus.RUNNING])

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "total_workers": len(self.workers),
                "active_workers": active_workers,
                "workers_by_status": dict(workers_by_status),
                "scaling_config": {
                    "min_workers": self.min_workers,
                    "max_workers": self.max_workers,
                    "scaling_strategy": self.scaling_strategy.value,
                    "last_scaling_action": self.last_scaling_action
                },
                "performance": {
                    "total_jobs_processed": sum(w.jobs_processed for w in self.workers.values()),
                    "scaling_actions": len(self.scaling_history)
                }
            }

    async def start_worker_management(self):
        """Start the worker management system"""
        if self._running:
            return

        self._running = True

        # Start background tasks
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._scaling_task = asyncio.create_task(self._scaling_loop())

        self.logger.info("Worker management system started")

    async def stop_worker_management(self):
        """Stop the worker management system"""
        if not self._running:
            return

        self._running = False

        # Stop all workers gracefully
        for worker_id in list(self.workers.keys()):
            self.stop_worker(worker_id)

        # Cancel background tasks
        if self._monitor_task:
            self._monitor_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._scaling_task:
            self._scaling_task.cancel()

        self.logger.info("Worker management system stopped")

    async def _monitoring_loop(self):
        """Monitor worker health and performance"""
        while self._running:
            try:
                # Clean up failed workers
                self.cleanup_failed_workers()

                # Update worker metrics
                self._update_worker_metrics()

                # Check for resource pressure affecting workers
                await self._check_worker_resource_pressure()

            except Exception as e:
                self.logger.error("Error in worker monitoring loop",
                                context={"error": str(e)})

            await asyncio.sleep(30)  # Monitor every 30 seconds

    async def _heartbeat_loop(self):
        """Send heartbeat signals to workers"""
        while self._running:
            try:
                # Update heartbeat for all active workers
                current_time = datetime.utcnow()

                with self._worker_lock:
                    for worker in self.workers.values():
                        if worker.status == WorkerStatus.RUNNING:
                            # In a real implementation, this would send actual heartbeat signals
                            # For now, we'll just update the timestamp
                            worker.last_heartbeat = current_time

            except Exception as e:
                self.logger.error("Error in heartbeat loop",
                                context={"error": str(e)})

            await asyncio.sleep(self.heartbeat_interval)

    async def _scaling_loop(self):
        """Main scaling decision loop"""
        while self._running:
            try:
                # Make scaling decision
                decision = self.make_scaling_decision()

                if decision.action != "no_action":
                    # Execute scaling decision
                    success = self.execute_scaling_decision(decision)

                    if success:
                        self.logger.info("Scaling decision executed",
                                       context={
                                           "action": decision.action,
                                           "target_workers": decision.target_workers,
                                           "reason": decision.reason,
                                           "confidence": decision.confidence
                                       })
                    else:
                        self.logger.error("Failed to execute scaling decision",
                                        context={"decision": decision.reason})
                else:
                    # Log no-action decisions periodically
                    if len(self.scaling_history) % 10 == 0:
                        self.logger.debug("No scaling action needed",
                                        context={"reason": decision.reason})

            except Exception as e:
                self.logger.error("Error in scaling loop",
                                context={"error": str(e)})

            await asyncio.sleep(60)  # Make scaling decisions every minute

    def _update_worker_metrics(self):
        """Update worker-related metrics"""
        try:
            active_workers = len([w for w in self.workers.values() if w.status == WorkerStatus.RUNNING])
            total_workers = len(self.workers)

            self.resource_manager.monitor.metrics.set_gauge("workers_active", active_workers)
            self.resource_manager.monitor.metrics.set_gauge("workers_total", total_workers)

            # Calculate average jobs per worker
            if active_workers > 0:
                total_jobs = sum(w.jobs_processed for w in self.workers.values())
                avg_jobs_per_worker = total_jobs / active_workers
                self.resource_manager.monitor.metrics.set_gauge("workers_avg_jobs_per_worker", avg_jobs_per_worker)

            # Worker status distribution
            status_counts = defaultdict(int)
            for worker in self.workers.values():
                status_counts[worker.status.value] += 1

            for status, count in status_counts.items():
                self.resource_manager.monitor.metrics.set_gauge(f"workers_status_{status}", count)

        except Exception as e:
            self.logger.error("Error updating worker metrics",
                            context={"error": str(e)})

    async def _check_worker_resource_pressure(self):
        """Check if workers are under resource pressure"""
        try:
            resource_status = self.resource_manager.get_system_resource_status()

            cpu_usage = resource_status.get("cpu", {}).get("current_usage_percent", 0)
            memory_usage = resource_status.get("memory", {}).get("usage_percent", 0)

            # If system is under heavy load, consider scaling down
            if cpu_usage > 95 or memory_usage > 95:
                self.logger.warning("System under extreme resource pressure, considering scale down",
                                  context={"cpu_usage": cpu_usage, "memory_usage": memory_usage})

                # Scale down by 1 worker if we have more than minimum
                active_workers = len([w for w in self.workers.values() if w.status == WorkerStatus.RUNNING])
                if active_workers > self.min_workers:
                    idle_workers = self._get_idle_workers()
                    if idle_workers:
                        worker_to_stop = idle_workers[0]
                        self.stop_worker(worker_to_stop)
                        self.logger.info("Scaled down worker due to resource pressure",
                                       context={"worker_id": worker_to_stop})

        except Exception as e:
            self.logger.error("Error checking worker resource pressure",
                            context={"error": str(e)})


class LoadBalancer:
    """Load balancer for distributing jobs across multiple worker nodes"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = Redis.from_url(redis_url)
        self.logger = StructuredLogger("load_balancer")

        self.worker_manager = WorkerManager(redis_url)
        self.resource_manager = get_resource_manager()

        # Load balancing strategies
        self.balancing_strategy = "round_robin"  # round_robin, least_loaded, resource_aware
        self.worker_load_scores: Dict[str, float] = {}
        self.last_used_worker = None

    def select_worker_for_job(self, job_requirements) -> Optional[str]:
        """Select the best worker for a job based on current strategy"""
        active_workers = [w for w in self.worker_manager.workers.values()
                         if w.status == WorkerStatus.RUNNING]

        if not active_workers:
            return None

        if self.balancing_strategy == "round_robin":
            return self._round_robin_selection(active_workers)
        elif self.balancing_strategy == "least_loaded":
            return self._least_loaded_selection(active_workers, job_requirements)
        elif self.balancing_strategy == "resource_aware":
            return self._resource_aware_selection(active_workers, job_requirements)
        else:
            return self._round_robin_selection(active_workers)

    def _round_robin_selection(self, workers: List[WorkerNode]) -> Optional[str]:
        """Simple round-robin selection"""
        if not workers:
            return None

        if self.last_used_worker is None:
            self.last_used_worker = workers[0].worker_id
            return self.last_used_worker

        # Find current worker index and get next
        try:
            current_index = next(i for i, w in enumerate(workers) if w.worker_id == self.last_used_worker)
            next_index = (current_index + 1) % len(workers)
            self.last_used_worker = workers[next_index].worker_id
            return self.last_used_worker
        except (StopIteration, IndexError):
            # Fallback to first worker
            self.last_used_worker = workers[0].worker_id
            return self.last_used_worker

    def _least_loaded_selection(self, workers: List[WorkerNode], job_requirements) -> Optional[str]:
        """Select worker with least current load"""
        if not workers:
            return None

        # Calculate load scores for each worker
        worker_scores = []

        for worker in workers:
            # Base score on jobs processed (inverse - fewer jobs = lower score)
            base_score = worker.jobs_processed

            # Adjust for current resource usage if available
            if worker.resource_usage:
                cpu_usage = worker.resource_usage.get("cpu_percent", 0)
                memory_usage = worker.resource_usage.get("memory_percent", 0)
                resource_penalty = (cpu_usage + memory_usage) / 200.0  # Normalize to 0-1 range
                base_score += resource_penalty * 100

            worker_scores.append((worker.worker_id, base_score))

        # Select worker with lowest score
        best_worker = min(worker_scores, key=lambda x: x[1])
        return best_worker[0]

    def _resource_aware_selection(self, workers: List[WorkerNode], job_requirements) -> Optional[str]:
        """Select worker based on resource requirements and availability"""
        if not workers:
            return None

        best_worker = None
        best_score = float('inf')

        for worker in workers:
            # Calculate how well this worker matches job requirements
            score = 0.0

            # Check CPU availability
            worker_cpu = worker.resource_usage.get("cpu_percent", 50)  # Assume 50% if unknown
            required_cpu = job_requirements.cpu_cores * 100 / self.resource_manager.cpu_cores
            cpu_match = abs(worker_cpu - required_cpu)
            score += cpu_match * 0.4

            # Check memory availability
            worker_memory = worker.resource_usage.get("memory_percent", 50)  # Assume 50% if unknown
            required_memory = (job_requirements.memory_mb / 4096) * 100  # Assume 4GB total
            memory_match = abs(worker_memory - required_memory)
            score += memory_match * 0.3

            # Consider current load
            load_penalty = worker.jobs_processed * 0.3
            score += load_penalty

            if score < best_score:
                best_score = score
                best_worker = worker.worker_id

        return best_worker

    def update_worker_load(self, worker_id: str, current_job: Optional[str] = None,
                          resource_usage: Optional[Dict[str, float]] = None):
        """Update load information for a worker"""
        # Update worker manager
        self.worker_manager.update_worker_heartbeat(
            worker_id,
            current_job=current_job,
            resource_usage=resource_usage
        )

        # Update load scores for load balancer
        if resource_usage:
            load_score = (resource_usage.get("cpu_percent", 0) + resource_usage.get("memory_percent", 0)) / 2
            self.worker_load_scores[worker_id] = load_score


# Global instances
_worker_manager_instance = None
_load_balancer_instance = None
_scaler_lock = threading.Lock()


def get_worker_manager() -> WorkerManager:
    """Get the global worker manager instance"""
    global _worker_manager_instance

    if _worker_manager_instance is None:
        with _scaler_lock:
            if _worker_manager_instance is None:
                _worker_manager_instance = WorkerManager()

    return _worker_manager_instance


def get_load_balancer() -> LoadBalancer:
    """Get the global load balancer instance"""
    global _load_balancer_instance

    if _load_balancer_instance is None:
        with _scaler_lock:
            if _load_balancer_instance is None:
                _load_balancer_instance = LoadBalancer()

    return _load_balancer_instance