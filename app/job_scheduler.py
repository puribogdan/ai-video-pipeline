# app/job_scheduler.py
import asyncio
import json
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
import redis
from redis import Redis

from .resource_manager import (
    get_resource_manager, ResourceManager, JobResourceRequirement,
    Priority, ResourceType
)
from .monitoring import get_monitor, StructuredLogger


class JobStatus(Enum):
    QUEUED = "queued"
    WAITING_FOR_RESOURCES = "waiting_for_resources"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PREEMPTED = "preempted"


class QueueType(Enum):
    HIGH_PRIORITY = "high_priority"
    NORMAL = "normal"
    LOW_PRIORITY = "low_priority"
    BATCH = "batch"
    DEPENDENCY = "dependency"


@dataclass
class JobDependency:
    job_id: str
    depends_on: List[str]  # List of job IDs this job depends on
    dependency_type: str = "finish"  # "finish", "success", "data"
    timeout_seconds: float = 3600.0  # 1 hour timeout


@dataclass
class QueuedJob:
    job_id: str
    priority: Priority
    queue_type: QueueType
    resource_requirements: JobResourceRequirement
    submitted_at: datetime
    dependencies: Optional[JobDependency] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    status: JobStatus = JobStatus.QUEUED


@dataclass
class BatchGroup:
    batch_id: str
    jobs: List[QueuedJob]
    max_concurrent_jobs: int
    resource_sharing_enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None


class PriorityQueue:
    """Priority-based job queue with resource awareness"""

    def __init__(self):
        self.queues: Dict[Priority, deque] = {
            Priority.CRITICAL: deque(),
            Priority.HIGH: deque(),
            Priority.NORMAL: deque(),
            Priority.LOW: deque()
        }
        self.lock = threading.Lock()

    def enqueue(self, job: QueuedJob):
        """Add a job to the appropriate priority queue"""
        with self.lock:
            self.queues[job.priority].append(job)

    def dequeue(self) -> Optional[QueuedJob]:
        """Get the next job to process (highest priority first)"""
        with self.lock:
            for priority in [Priority.CRITICAL, Priority.HIGH, Priority.NORMAL, Priority.LOW]:
                if self.queues[priority]:
                    return self.queues[priority].popleft()
            return None

    def peek_next(self) -> Optional[QueuedJob]:
        """Peek at the next job without removing it"""
        with self.lock:
            for priority in [Priority.CRITICAL, Priority.HIGH, Priority.NORMAL, Priority.LOW]:
                if self.queues[priority]:
                    return self.queues[priority][0]
            return None

    def remove_job(self, job_id: str) -> Optional[QueuedJob]:
        """Remove a specific job from the queue"""
        with self.lock:
            for priority_queue in self.queues.values():
                for i, job in enumerate(priority_queue):
                    if job.job_id == job_id:
                        return priority_queue.pop(i)
            return None

    def get_queue_lengths(self) -> Dict[str, int]:
        """Get the length of each priority queue"""
        with self.lock:
            return {
                priority.name: len(queue)
                for priority, queue in self.queues.items()
            }

    def get_total_jobs(self) -> int:
        """Get total number of jobs in all queues"""
        return sum(len(queue) for queue in self.queues.values())


class DependencyManager:
    """Manages job dependencies and complex workflows"""

    def __init__(self):
        self.job_dependencies: Dict[str, JobDependency] = {}
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)  # job_id -> set of jobs that depend on it
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)  # job_id -> set of jobs it depends on
        self.completed_jobs: Set[str] = set()
        self.lock = threading.Lock()

    def add_dependency(self, dependency: JobDependency):
        """Add a job dependency"""
        with self.lock:
            self.job_dependencies[dependency.job_id] = dependency

            # Update dependency graphs
            for dep_job_id in dependency.depends_on:
                self.dependency_graph[dep_job_id].add(dependency.job_id)
                self.reverse_graph[dependency.job_id].add(dep_job_id)

    def mark_job_completed(self, job_id: str, success: bool = True):
        """Mark a job as completed and check if dependent jobs can be released"""
        with self.lock:
            self.completed_jobs.add(job_id)

            # Find jobs that depend on this completed job
            dependent_jobs = self.dependency_graph.get(job_id, set())

            return list(dependent_jobs)

    def can_run_job(self, job_id: str) -> Tuple[bool, str]:
        """Check if a job's dependencies are satisfied"""
        with self.lock:
            if job_id not in self.job_dependencies:
                return True, "No dependencies"

            dependency = self.job_dependencies[job_id]

            for dep_job_id in dependency.depends_on:
                if dep_job_id not in self.completed_jobs:
                    return False, f"Dependency {dep_job_id} not completed"

            return True, "All dependencies satisfied"

    def get_pending_dependencies(self, job_id: str) -> List[str]:
        """Get list of pending dependencies for a job"""
        with self.lock:
            if job_id not in self.job_dependencies:
                return []

            dependency = self.job_dependencies[job_id]
            return [dep_id for dep_id in dependency.depends_on if dep_id not in self.completed_jobs]


class BatchManager:
    """Manages job batching for improved efficiency"""

    def __init__(self):
        self.batch_groups: Dict[str, BatchGroup] = {}
        self.job_to_batch: Dict[str, str] = {}  # job_id -> batch_id
        self.lock = threading.Lock()

    def create_batch_group(self, jobs: List[QueuedJob], max_concurrent: int = 3) -> str:
        """Create a new batch group"""
        batch_id = str(uuid.uuid4())

        batch_group = BatchGroup(
            batch_id=batch_id,
            jobs=jobs,
            max_concurrent_jobs=max_concurrent
        )

        with self.lock:
            self.batch_groups[batch_id] = batch_group
            for job in jobs:
                self.job_to_batch[job.job_id] = batch_id

        return batch_id

    def can_start_batch_job(self, batch_id: str) -> Tuple[bool, int]:
        """Check if a job from a batch can be started"""
        with self.lock:
            if batch_id not in self.batch_groups:
                return False, 0

            batch = self.batch_groups[batch_id]

            # Count currently running jobs in this batch
            running_jobs = sum(1 for job in batch.jobs if job.status == JobStatus.RUNNING)

            return running_jobs < batch.max_concurrent_jobs, batch.max_concurrent_jobs - running_jobs

    def mark_batch_job_started(self, job_id: str):
        """Mark a job from a batch as started"""
        with self.lock:
            if job_id in self.job_to_batch:
                batch_id = self.job_to_batch[job_id]
                if batch_id in self.batch_groups:
                    # Update job status
                    for job in self.batch_groups[batch_id].jobs:
                        if job.job_id == job_id:
                            job.status = JobStatus.RUNNING
                            break

    def mark_batch_job_completed(self, job_id: str):
        """Mark a job from a batch as completed"""
        with self.lock:
            if job_id in self.job_to_batch:
                batch_id = self.job_to_batch[job_id]
                if batch_id in self.batch_groups:
                    # Update job status
                    for job in self.batch_groups[batch_id].jobs:
                        if job.job_id == job_id:
                            job.status = JobStatus.COMPLETED
                            break


class IntelligentScheduler:
    """Intelligent job scheduler with resource-aware algorithms"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = Redis.from_url(redis_url)
        self.logger = StructuredLogger("intelligent_scheduler")

        # Core components
        self.resource_manager = get_resource_manager()
        self.priority_queue = PriorityQueue()
        self.dependency_manager = DependencyManager()
        self.batch_manager = BatchManager()

        # Job tracking
        self.queued_jobs: Dict[str, QueuedJob] = {}
        self.running_jobs: Set[str] = set()
        self.completed_jobs: Set[str] = set()

        # Locks
        self._scheduler_lock = threading.Lock()
        self._job_lock = threading.Lock()

        # Configuration
        self.max_concurrent_jobs = 4
        self.batch_similarity_threshold = 0.7
        self.resource_check_interval = 10  # seconds

        # Background tasks
        self._scheduler_task = None
        self._resource_monitor_task = None
        self._running = False

        # Performance tracking
        self.schedule_attempts = 0
        self.successful_schedules = 0
        self.resource_conflicts = 0

    def submit_job(self, job_id: str, resource_requirements: JobResourceRequirement,
                   dependencies: Optional[JobDependency] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Submit a job for scheduling"""

        # Create queued job
        queued_job = QueuedJob(
            job_id=job_id,
            priority=resource_requirements.priority,
            queue_type=self._determine_queue_type(resource_requirements),
            resource_requirements=resource_requirements,
            submitted_at=datetime.utcnow(),
            dependencies=dependencies,
            metadata=metadata or {}
        )

        # Register with resource manager
        self.resource_manager.register_job_requirements(job_id, resource_requirements)

        # Add dependencies if specified
        if dependencies:
            self.dependency_manager.add_dependency(dependencies)

        # Add to priority queue and tracking
        with self._job_lock:
            self.queued_jobs[job_id] = queued_job
            self.priority_queue.enqueue(queued_job)

        # Update monitoring
        self._update_scheduler_metrics()

        self.logger.info("Job submitted for scheduling",
                        context={"job_id": job_id, "priority": resource_requirements.priority.name,
                               "queue_type": queued_job.queue_type.value})

        return True

    def _determine_queue_type(self, requirements: JobResourceRequirement) -> QueueType:
        """Determine the appropriate queue type for a job"""
        if requirements.priority in [Priority.CRITICAL, Priority.HIGH]:
            return QueueType.HIGH_PRIORITY
        elif requirements.memory_mb > 1024 or requirements.cpu_cores > 2:
            return QueueType.BATCH  # Large jobs might benefit from batching
        elif requirements.priority == Priority.LOW:
            return QueueType.LOW_PRIORITY
        else:
            return QueueType.NORMAL

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job"""
        with self._job_lock:
            if job_id in self.queued_jobs:
                # Remove from priority queue
                removed_job = self.priority_queue.remove_job(job_id)

                if removed_job:
                    removed_job.status = JobStatus.CANCELLED

                    # Remove from tracking
                    del self.queued_jobs[job_id]

                    self.logger.info("Job cancelled",
                                   context={"job_id": job_id})

                    return True

        return False

    def get_next_job(self) -> Optional[QueuedJob]:
        """Get the next job that should be scheduled"""
        with self._scheduler_lock:
            self.schedule_attempts += 1

            # Get next job from priority queue
            job = self.priority_queue.dequeue()

            if not job:
                return None

            # Check if job can run (dependencies satisfied)
            can_run, reason = self.dependency_manager.can_run_job(job.job_id)

            if not can_run:
                # Put job back in queue (it will be retried later)
                self.priority_queue.enqueue(job)
                return None

            # Check if resources are available
            can_allocate, reason = self.resource_manager.can_allocate_resources(job.job_id)

            if not can_allocate:
                self.resource_conflicts += 1

                # For high priority jobs, consider preemption
                if job.priority in [Priority.CRITICAL, Priority.HIGH]:
                    preempted_job_id = self._attempt_preemption(job)
                    if preempted_job_id:
                        self.logger.info("Preempted job for high priority task",
                                       context={"preempted_job": preempted_job_id, "new_job": job.job_id})

                # Put job back in queue
                self.priority_queue.enqueue(job)
                return None

            # Check batch constraints if applicable
            if job.job_id in self.batch_manager.job_to_batch:
                batch_id = self.batch_manager.job_to_batch[job.job_id]
                can_start_batch, available_slots = self.batch_manager.can_start_batch_job(batch_id)

                if not can_start_batch:
                    # Put job back in queue
                    self.priority_queue.enqueue(job)
                    return None

            self.successful_schedules += 1

            # Mark job as running
            job.status = JobStatus.RUNNING
            with self._job_lock:
                self.running_jobs.add(job.job_id)
                if job.job_id in self.queued_jobs:
                    del self.queued_jobs[job.job_id]

            # Allocate resources
            allocation_id = self.resource_manager.allocate_resources(job.job_id)

            if allocation_id:
                # Update batch manager if applicable
                if job.job_id in self.batch_manager.job_to_batch:
                    self.batch_manager.mark_batch_job_started(job.job_id)

                self.logger.info("Job scheduled successfully",
                               context={"job_id": job.job_id, "allocation_id": allocation_id})

                return job

            # Failed to allocate resources, put job back
            job.status = JobStatus.WAITING_FOR_RESOURCES
            self.priority_queue.enqueue(job)
            return None

    def _attempt_preemption(self, high_priority_job: QueuedJob) -> Optional[str]:
        """Attempt to preempt a lower priority job for a high priority one"""
        # Find running jobs that could be preempted
        for job_id in list(self.running_jobs):
            # Get job's original queued job info
            # In a real implementation, you'd want to track this better
            # For now, we'll skip preemption to avoid complexity
            pass

        return None

    def mark_job_completed(self, job_id: str, success: bool = True, error: Optional[str] = None):
        """Mark a job as completed"""
        with self._job_lock:
            if job_id in self.running_jobs:
                self.running_jobs.remove(job_id)
                self.completed_jobs.add(job_id)

        # Check if this completion enables dependent jobs
        enabled_jobs = self.dependency_manager.mark_job_completed(job_id, success)

        # Release resources
        allocation = self.resource_manager.get_allocation_info(job_id)
        if allocation:
            self.resource_manager.free_resources(allocation.allocation_id)

        # Update batch manager if applicable
        if job_id in self.batch_manager.job_to_batch:
            self.batch_manager.mark_batch_job_completed(job_id)

        # Update monitoring
        self._update_scheduler_metrics()

        self.logger.info("Job marked as completed",
                        context={"job_id": job_id, "success": success, "enabled_jobs": len(enabled_jobs)})

        return enabled_jobs

    def mark_job_failed(self, job_id: str, error: str):
        """Mark a job as failed"""
        with self._job_lock:
            if job_id in self.running_jobs:
                self.running_jobs.remove(job_id)

        # Release resources
        allocation = self.resource_manager.get_allocation_info(job_id)
        if allocation:
            self.resource_manager.free_resources(allocation.allocation_id)

        # Update monitoring
        self._update_scheduler_metrics()

        self.logger.error("Job marked as failed",
                         context={"job_id": job_id, "error": error})

    def _update_scheduler_metrics(self):
        """Update scheduler performance metrics"""
        try:
            total_jobs = len(self.queued_jobs) + len(self.running_jobs) + len(self.completed_jobs)

            self.resource_manager.monitor.metrics.set_gauge("scheduler_queued_jobs", len(self.queued_jobs))
            self.resource_manager.monitor.metrics.set_gauge("scheduler_running_jobs", len(self.running_jobs))
            self.resource_manager.monitor.metrics.set_gauge("scheduler_completed_jobs", len(self.completed_jobs))
            self.resource_manager.monitor.metrics.set_gauge("scheduler_total_jobs", total_jobs)

            # Queue lengths by priority
            queue_lengths = self.priority_queue.get_queue_lengths()
            for priority, length in queue_lengths.items():
                self.resource_manager.monitor.metrics.set_gauge(f"scheduler_queue_{priority.lower()}_length", length)

            # Performance ratios
            if self.schedule_attempts > 0:
                success_rate = self.successful_schedules / self.schedule_attempts
                conflict_rate = self.resource_conflicts / self.schedule_attempts

                self.resource_manager.monitor.metrics.set_gauge("scheduler_success_rate", success_rate)
                self.resource_manager.monitor.metrics.set_gauge("scheduler_conflict_rate", conflict_rate)

        except Exception as e:
            self.logger.error("Failed to update scheduler metrics",
                            context={"error": str(e)})

    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status"""
        queue_lengths = self.priority_queue.get_queue_lengths()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "queued_jobs": len(self.queued_jobs),
            "running_jobs": len(self.running_jobs),
            "completed_jobs": len(self.completed_jobs),
            "queue_lengths": queue_lengths,
            "total_jobs": len(self.queued_jobs) + len(self.running_jobs) + len(self.completed_jobs),
            "schedule_attempts": self.schedule_attempts,
            "successful_schedules": self.successful_schedules,
            "resource_conflicts": self.resource_conflicts,
            "performance": {
                "success_rate": self.successful_schedules / self.schedule_attempts if self.schedule_attempts > 0 else 0,
                "conflict_rate": self.resource_conflicts / self.schedule_attempts if self.schedule_attempts > 0 else 0
            }
        }

    def find_similar_jobs_for_batching(self, job: QueuedJob, threshold: float = 0.7) -> List[QueuedJob]:
        """Find similar jobs that could be batched together"""
        similar_jobs = []

        for queued_job in self.queued_jobs.values():
            if queued_job.job_id == job.job_id:
                continue

            # Calculate similarity based on resource requirements and metadata
            similarity = self._calculate_job_similarity(job, queued_job)

            if similarity >= threshold:
                similar_jobs.append(queued_job)

        return similar_jobs

    def _calculate_job_similarity(self, job1: QueuedJob, job2: QueuedJob) -> float:
        """Calculate similarity between two jobs for batching"""
        # Simple similarity based on resource requirements
        cpu_similarity = 1 - abs(job1.resource_requirements.cpu_cores - job2.resource_requirements.cpu_cores) / max(job1.resource_requirements.cpu_cores, job2.resource_requirements.cpu_cores, 1)
        memory_similarity = 1 - abs(job1.resource_requirements.memory_mb - job2.resource_requirements.memory_mb) / max(job1.resource_requirements.memory_mb, job2.resource_requirements.memory_mb, 1)

        # Consider style/type similarity if available in metadata
        style_similarity = 0.0
        if "style" in job1.metadata and "style" in job2.metadata:
            style_similarity = 1.0 if job1.metadata["style"] == job2.metadata["style"] else 0.0

        # Weighted average of similarities
        return (cpu_similarity * 0.3 + memory_similarity * 0.3 + style_similarity * 0.4)

    def create_optimal_batch(self, jobs: List[QueuedJob]) -> Optional[str]:
        """Create an optimal batch from similar jobs"""
        if len(jobs) < 2:
            return None

        # Sort jobs by priority (highest first)
        sorted_jobs = sorted(jobs, key=lambda j: j.priority.value, reverse=True)

        # Determine optimal batch size based on resource requirements
        total_cpu = sum(job.resource_requirements.cpu_cores for job in sorted_jobs)
        total_memory = sum(job.resource_requirements.memory_mb for job in sorted_jobs)

        # Estimate optimal concurrent jobs (don't overload system)
        max_cpu_per_job = max(job.resource_requirements.cpu_cores for job in sorted_jobs) or 1.0
        max_memory_per_job = max(job.resource_requirements.memory_mb for job in sorted_jobs) or 512.0

        max_concurrent = min(
            len(sorted_jobs),
            max(1, int(self.resource_manager.cpu_cores / max_cpu_per_job)),
            max(1, int(4096 / max_memory_per_job))  # Assume 4GB available for batch
        )

        return self.batch_manager.create_batch_group(sorted_jobs[:max_concurrent], max_concurrent)

    async def start_scheduler(self):
        """Start the intelligent scheduler"""
        if self._running:
            return

        self._running = True

        # Start resource monitoring
        await self.resource_manager.start_resource_monitoring()

        # Start scheduler loop
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._resource_monitor_task = asyncio.create_task(self._resource_monitoring_loop())

        self.logger.info("Intelligent scheduler started")

    async def stop_scheduler(self):
        """Stop the intelligent scheduler"""
        if not self._running:
            return

        self._running = False

        # Stop resource monitoring
        await self.resource_manager.stop_resource_monitoring()

        # Cancel tasks
        if self._scheduler_task:
            self._scheduler_task.cancel()
        if self._resource_monitor_task:
            self._resource_monitor_task.cancel()

        self.logger.info("Intelligent scheduler stopped")

    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self._running:
            try:
                # Look for jobs that can be scheduled
                job = self.get_next_job()

                if job:
                    # In a real implementation, this would trigger the actual job execution
                    # For now, we'll just mark it as completed after a delay to simulate processing
                    asyncio.create_task(self._simulate_job_execution(job))

                # Check for batching opportunities
                await self._check_batching_opportunities()

            except Exception as e:
                self.logger.error("Error in scheduler loop",
                                context={"error": str(e)})

            await asyncio.sleep(1)  # Check every second

    async def _resource_monitoring_loop(self):
        """Monitor resource usage and adjust scheduling"""
        while self._running:
            try:
                # Update scheduler metrics
                self._update_scheduler_metrics()

                # Check for resource pressure and adjust scheduling
                await self._adjust_for_resource_pressure()

            except Exception as e:
                self.logger.error("Error in resource monitoring loop",
                                context={"error": str(e)})

            await asyncio.sleep(self.resource_check_interval)

    async def _simulate_job_execution(self, job: QueuedJob):
        """Simulate job execution (for testing purposes)"""
        try:
            # Simulate processing time based on resource requirements
            processing_time = job.resource_requirements.estimated_duration * 0.1  # 10% of estimated time for simulation

            await asyncio.sleep(min(processing_time, 30))  # Max 30 seconds for simulation

            # Mark as completed
            self.mark_job_completed(job.job_id, success=True)

        except Exception as e:
            self.logger.error("Error simulating job execution",
                            context={"job_id": job.job_id, "error": str(e)})
            self.mark_job_failed(job.job_id, str(e))

    async def _check_batching_opportunities(self):
        """Look for opportunities to create efficient batches"""
        try:
            # Find jobs that could benefit from batching
            for job in list(self.queued_jobs.values()):
                if job.status == JobStatus.QUEUED:
                    similar_jobs = self.find_similar_jobs_for_batching(job)

                    if len(similar_jobs) >= 2:  # Need at least 3 jobs for batching (including current)
                        all_jobs = [job] + similar_jobs[:2]  # Current job + 2 similar ones

                        # Create batch if beneficial
                        batch_id = self.create_optimal_batch(all_jobs)
                        if batch_id:
                            self.logger.info("Created optimal batch",
                                           context={"batch_id": batch_id, "num_jobs": len(all_jobs)})

        except Exception as e:
            self.logger.error("Error checking batching opportunities",
                            context={"error": str(e)})

    async def _adjust_for_resource_pressure(self):
        """Adjust scheduling based on resource pressure"""
        try:
            resource_status = self.resource_manager.get_system_resource_status()

            cpu_usage = resource_status.get("cpu", {}).get("current_usage_percent", 0)
            memory_usage = resource_status.get("memory", {}).get("usage_percent", 0)
            disk_usage = resource_status.get("disk", {}).get("usage_percent", 0)

            # If system is under pressure, reduce concurrent jobs
            if cpu_usage > 80 or memory_usage > 85 or disk_usage > 90:
                self.max_concurrent_jobs = max(1, self.max_concurrent_jobs - 1)
                self.logger.warning("Reduced concurrent jobs due to resource pressure",
                                  context={"cpu_usage": cpu_usage, "memory_usage": memory_usage,
                                         "disk_usage": disk_usage, "new_max": self.max_concurrent_jobs})

            # If system has plenty of resources, increase concurrent jobs
            elif cpu_usage < 50 and memory_usage < 60 and disk_usage < 70:
                self.max_concurrent_jobs = min(8, self.max_concurrent_jobs + 1)
                self.logger.info("Increased concurrent jobs due to available resources",
                               context={"cpu_usage": cpu_usage, "memory_usage": memory_usage,
                                      "disk_usage": disk_usage, "new_max": self.max_concurrent_jobs})

        except Exception as e:
            self.logger.error("Error adjusting for resource pressure",
                            context={"error": str(e)})


# Global scheduler instance
_scheduler_instance = None
_scheduler_lock = threading.Lock()


def get_scheduler() -> IntelligentScheduler:
    """Get the global scheduler instance"""
    global _scheduler_instance

    if _scheduler_instance is None:
        with _scheduler_lock:
            if _scheduler_instance is None:
                _scheduler_instance = IntelligentScheduler()

    return _scheduler_instance