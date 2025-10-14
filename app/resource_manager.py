# app/resource_manager.py
import asyncio
import json
import logging
import psutil
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import redis
import gc

from .monitoring import get_monitor, StructuredLogger


class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"


class Priority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ResourceUsage:
    resource_type: ResourceType
    used: float
    total: float
    percentage: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobResourceRequirement:
    job_id: str
    priority: Priority
    cpu_cores: float = 1.0
    memory_mb: float = 512.0
    disk_mb: float = 1024.0
    estimated_duration: float = 300.0  # seconds
    network_bandwidth: float = 10.0  # Mbps
    gpu_memory_mb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceAllocation:
    job_id: str
    allocation_id: str
    resources: Dict[ResourceType, float]
    start_time: datetime
    estimated_end_time: datetime
    priority: Priority
    status: str = "active"  # active, completed, failed, cancelled


class MemoryPool:
    """Memory pool management for audio processing operations"""

    def __init__(self, total_size_mb: float = 2048.0, block_size_mb: float = 64.0):
        self.total_size = total_size_mb * 1024 * 1024  # Convert to bytes
        self.block_size = block_size_mb * 1024 * 1024
        self.available_blocks = []
        self.allocated_blocks: Dict[str, List[int]] = {}
        self.lock = threading.Lock()

        # Initialize memory pool
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize the memory pool with blocks"""
        num_blocks = int(self.total_size / self.block_size)
        self.available_blocks = list(range(num_blocks))

    def allocate(self, size_bytes: int, job_id: str) -> Optional[List[int]]:
        """Allocate memory blocks for a job"""
        with self.lock:
            required_blocks = (size_bytes + self.block_size - 1) // self.block_size

            if len(self.available_blocks) < required_blocks:
                return None

            # Allocate contiguous blocks if possible, otherwise take any available
            allocated = []
            remaining_needed = required_blocks

            # Try to allocate contiguous blocks first
            for i in range(len(self.available_blocks) - required_blocks + 1):
                if all(j in self.available_blocks for j in range(i, i + required_blocks)):
                    allocated = list(range(i, i + required_blocks))
                    for block in allocated:
                        self.available_blocks.remove(block)
                    break

            # If contiguous allocation failed, take any available blocks
            if not allocated:
                allocated = self.available_blocks[:required_blocks]
                for block in allocated:
                    self.available_blocks.remove(block)

            if allocated:
                self.allocated_blocks[job_id] = allocated
                return allocated

            return None

    def free(self, job_id: str) -> bool:
        """Free memory blocks allocated to a job"""
        with self.lock:
            if job_id in self.allocated_blocks:
                blocks = self.allocated_blocks[job_id]
                self.available_blocks.extend(blocks)
                del self.allocated_blocks[job_id]
                return True
            return False

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get memory pool usage statistics"""
        with self.lock:
            total_blocks = len(self.available_blocks) + sum(len(blocks) for blocks in self.allocated_blocks.values())
            allocated_blocks = sum(len(blocks) for blocks in self.allocated_blocks.values())

            return {
                "total_blocks": total_blocks,
                "available_blocks": len(self.available_blocks),
                "allocated_blocks": allocated_blocks,
                "allocation_ratio": allocated_blocks / total_blocks if total_blocks > 0 else 0,
                "active_allocations": len(self.allocated_blocks),
                "memory_utilization": (allocated_blocks / total_blocks) * 100 if total_blocks > 0 else 0
            }


class ResourceManager:
    """Comprehensive resource management system for concurrent job processing"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = redis.from_url(redis_url)
        self.logger = StructuredLogger("resource_manager")

        # Resource tracking
        self.resource_usage: Dict[ResourceType, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.job_requirements: Dict[str, JobResourceRequirement] = {}

        # Memory pool management
        self.memory_pool = MemoryPool()

        # CPU management
        self.cpu_cores = psutil.cpu_count(logical=True)
        self.cpu_allocations: Dict[str, Set[int]] = defaultdict(set)

        # Disk management
        self.disk_paths: Set[Path] = set()
        self.disk_cleanup_strategies: List[Dict[str, Any]] = []

        # Locks for thread safety
        self._allocation_lock = threading.Lock()
        self._resource_lock = threading.Lock()

        # Monitoring integration
        self.monitor = get_monitor()

        # Background tasks
        self._monitoring_task = None
        self._cleanup_task = None
        self._running = False

        # Configuration
        self.max_memory_per_job_mb = 1024.0
        self.max_cpu_per_job = 2.0
        self.disk_cleanup_threshold = 0.85  # 85% disk usage
        self.resource_check_interval = 30  # seconds

        self._setup_disk_cleanup_strategies()
        self._initialize_resource_tracking()

    def _setup_disk_cleanup_strategies(self):
        """Setup automatic disk cleanup strategies"""
        self.disk_cleanup_strategies = [
            {
                "name": "old_job_cleanup",
                "pattern": "*/pipeline/temp/*",
                "max_age_hours": 24,
                "priority": 1
            },
            {
                "name": "failed_job_cleanup",
                "pattern": "*/completion_status.json",
                "max_age_hours": 48,
                "priority": 2
            },
            {
                "name": "temp_file_cleanup",
                "pattern": "*.tmp",
                "max_age_hours": 12,
                "priority": 3
            }
        ]

    def _initialize_resource_tracking(self):
        """Initialize resource tracking with current system state"""
        try:
            # Get initial system metrics
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            initial_usage = {
                ResourceType.MEMORY: ResourceUsage(
                    resource_type=ResourceType.MEMORY,
                    used=memory.used,
                    total=memory.total,
                    percentage=memory.percent,
                    timestamp=datetime.utcnow()
                ),
                ResourceType.DISK: ResourceUsage(
                    resource_type=ResourceType.DISK,
                    used=disk.used,
                    total=disk.total,
                    percentage=disk.percent,
                    timestamp=datetime.utcnow()
                ),
                ResourceType.CPU: ResourceUsage(
                    resource_type=ResourceType.CPU,
                    used=psutil.cpu_percent(interval=None),
                    total=100.0,
                    percentage=psutil.cpu_percent(interval=None),
                    timestamp=datetime.utcnow()
                )
            }

            for resource_type, usage in initial_usage.items():
                self.resource_usage[resource_type].append(usage)

        except Exception as e:
            self.logger.error("Failed to initialize resource tracking",
                            context={"error": str(e), "component": "resource_manager"})

    def register_job_requirements(self, job_id: str, requirements: JobResourceRequirement) -> bool:
        """Register resource requirements for a job"""
        with self._allocation_lock:
            self.job_requirements[job_id] = requirements

            # Update monitoring
            self.monitor.metrics.set_gauge("job_resource_requirement_cpu",
                                         requirements.cpu_cores, labels={"job_id": job_id})
            self.monitor.metrics.set_gauge("job_resource_requirement_memory",
                                         requirements.memory_mb, labels={"job_id": job_id})

            self.logger.info("Job requirements registered",
                           context={"job_id": job_id, "priority": requirements.priority.name,
                                  "cpu_cores": requirements.cpu_cores, "memory_mb": requirements.memory_mb})

            return True

    def can_allocate_resources(self, job_id: str) -> Tuple[bool, str]:
        """Check if resources can be allocated for a job"""
        if job_id not in self.job_requirements:
            return False, "Job requirements not registered"

        requirements = self.job_requirements[job_id]

        # Check CPU availability
        current_cpu_usage = self._get_current_cpu_usage()
        available_cpu_cores = max(0, self.cpu_cores - current_cpu_usage)

        if requirements.cpu_cores > available_cpu_cores:
            return False, f"Insufficient CPU cores: need {requirements.cpu_cores}, available {available_cpu_cores}"

        # Check memory availability
        current_memory_usage = self._get_current_memory_usage()
        available_memory_mb = (psutil.virtual_memory().available / (1024 * 1024)) - current_memory_usage

        if requirements.memory_mb > available_memory_mb:
            return False, f"Insufficient memory: need {requirements.memory_mb}MB, available {available_memory_mb:.1f}MB"

        # Check disk space availability
        current_disk_usage = self._get_current_disk_usage()
        available_disk_mb = self._get_available_disk_space_mb()

        if requirements.disk_mb > available_disk_mb:
            return False, f"Insufficient disk space: need {requirements.disk_mb}MB, available {available_disk_mb:.1f}MB"

        return True, "Resources available"

    def allocate_resources(self, job_id: str) -> Optional[str]:
        """Allocate resources for a job"""
        if job_id not in self.job_requirements:
            self.logger.error("Cannot allocate resources: job requirements not registered",
                            context={"job_id": job_id})
            return None

        can_allocate, reason = self.can_allocate_resources(job_id)
        if not can_allocate:
            self.logger.warning("Resource allocation denied",
                              context={"job_id": job_id, "reason": reason})
            return None

        requirements = self.job_requirements[job_id]
        allocation_id = str(uuid.uuid4())

        # Allocate CPU cores
        allocated_cpu_cores = self._allocate_cpu_cores(job_id, requirements.cpu_cores)

        # Allocate memory
        memory_allocated = self._allocate_memory(job_id, requirements.memory_mb)

        if not allocated_cpu_cores or not memory_allocated:
            # Rollback allocations
            if allocated_cpu_cores:
                self._free_cpu_cores(job_id)
            if memory_allocated:
                self.memory_pool.free(job_id)
            return None

        # Create allocation record
        allocation = ResourceAllocation(
            job_id=job_id,
            allocation_id=allocation_id,
            resources={
                ResourceType.CPU: requirements.cpu_cores,
                ResourceType.MEMORY: requirements.memory_mb,
                ResourceType.DISK: requirements.disk_mb
            },
            start_time=datetime.utcnow(),
            estimated_end_time=datetime.utcnow() + timedelta(seconds=requirements.estimated_duration),
            priority=requirements.priority
        )

        with self._allocation_lock:
            self.allocations[allocation_id] = allocation

        # Update monitoring
        self.monitor.metrics.set_gauge("resource_allocation_active", 1,
                                     labels={"job_id": job_id, "allocation_id": allocation_id})

        self.logger.info("Resources allocated successfully",
                        context={"job_id": job_id, "allocation_id": allocation_id,
                               "cpu_cores": requirements.cpu_cores, "memory_mb": requirements.memory_mb})

        return allocation_id

    def _allocate_cpu_cores(self, job_id: str, requested_cores: float) -> bool:
        """Allocate CPU cores for a job"""
        available_cores = self._get_available_cpu_cores()

        if requested_cores > available_cores:
            return False

        # Simple allocation strategy - assign cores from available pool
        allocated_cores = set()
        for core_id in range(int(requested_cores)):
            if core_id not in self._get_allocated_cpu_cores():
                allocated_cores.add(core_id)

        if len(allocated_cores) >= requested_cores:
            self.cpu_allocations[job_id] = allocated_cores
            return True

        return False

    def _allocate_memory(self, job_id: str, memory_mb: float) -> bool:
        """Allocate memory for a job"""
        memory_bytes = int(memory_mb * 1024 * 1024)

        allocated_blocks = self.memory_pool.allocate(memory_bytes, job_id)
        return allocated_blocks is not None

    def _free_cpu_cores(self, job_id: str):
        """Free CPU cores allocated to a job"""
        if job_id in self.cpu_allocations:
            del self.cpu_allocations[job_id]

    def free_resources(self, allocation_id: str) -> bool:
        """Free resources for a completed job"""
        with self._allocation_lock:
            if allocation_id not in self.allocations:
                return False

            allocation = self.allocations[allocation_id]

            # Free CPU cores
            self._free_cpu_cores(allocation.job_id)

            # Free memory
            self.memory_pool.free(allocation.job_id)

            # Remove allocation record
            del self.allocations[allocation_id]

            # Update monitoring
            self.monitor.metrics.set_gauge("resource_allocation_active", 0,
                                         labels={"job_id": allocation.job_id, "allocation_id": allocation_id})

            self.logger.info("Resources freed",
                           context={"job_id": allocation.job_id, "allocation_id": allocation_id})

            return True

    def _get_current_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        return psutil.cpu_percent(interval=None)

    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        memory = psutil.virtual_memory()
        return (memory.used - memory.available) / (1024 * 1024)

    def _get_current_disk_usage(self) -> float:
        """Get current disk usage percentage"""
        disk = psutil.disk_usage('/')
        return disk.percent

    def _get_available_cpu_cores(self) -> float:
        """Get number of available CPU cores"""
        current_usage = self._get_current_cpu_usage()
        return max(0, self.cpu_cores - current_usage)

    def _get_available_disk_space_mb(self) -> float:
        """Get available disk space in MB"""
        disk = psutil.disk_usage('/')
        return (disk.free) / (1024 * 1024)

    def _get_allocated_cpu_cores(self) -> Set[int]:
        """Get set of allocated CPU core IDs"""
        allocated = set()
        for cores in self.cpu_allocations.values():
            allocated.update(cores)
        return allocated

    def get_resource_usage(self, resource_type: ResourceType) -> List[ResourceUsage]:
        """Get historical resource usage data"""
        return list(self.resource_usage[resource_type])

    def get_allocation_info(self, job_id: str) -> Optional[ResourceAllocation]:
        """Get allocation information for a job"""
        for allocation in self.allocations.values():
            if allocation.job_id == job_id:
                return allocation
        return None

    def get_system_resource_status(self) -> Dict[str, Any]:
        """Get comprehensive system resource status"""
        try:
            # CPU information
            cpu_info = {
                "total_cores": self.cpu_cores,
                "current_usage_percent": self._get_current_cpu_usage(),
                "available_cores": self._get_available_cpu_cores(),
                "allocated_cores": len(self._get_allocated_cpu_cores())
            }

            # Memory information
            memory = psutil.virtual_memory()
            memory_pool_stats = self.memory_pool.get_usage_stats()
            memory_info = {
                "total_mb": memory.total / (1024 * 1024),
                "used_mb": memory.used / (1024 * 1024),
                "available_mb": memory.available / (1024 * 1024),
                "usage_percent": memory.percent,
                "pool_utilization_percent": memory_pool_stats["memory_utilization"],
                "active_allocations": memory_pool_stats["active_allocations"]
            }

            # Disk information
            disk = psutil.disk_usage('/')
            disk_info = {
                "total_mb": disk.total / (1024 * 1024),
                "used_mb": disk.used / (1024 * 1024),
                "free_mb": disk.free / (1024 * 1024),
                "usage_percent": disk.percent,
                "available_mb": self._get_available_disk_space_mb()
            }

            # Allocation summary
            allocation_summary = {
                "total_allocations": len(self.allocations),
                "active_jobs": len(set(alloc.job_id for alloc in self.allocations.values())),
                "high_priority_jobs": len([alloc for alloc in self.allocations.values()
                                         if alloc.priority in [Priority.HIGH, Priority.CRITICAL]])
            }

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "cpu": cpu_info,
                "memory": memory_info,
                "disk": disk_info,
                "allocations": allocation_summary,
                "memory_pool": memory_pool_stats
            }

        except Exception as e:
            self.logger.error("Failed to get system resource status",
                            context={"error": str(e)})
            return {"error": str(e)}

    async def start_resource_monitoring(self):
        """Start background resource monitoring"""
        if self._running:
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._resource_monitoring_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        self.logger.info("Resource monitoring started")

    async def stop_resource_monitoring(self):
        """Stop background resource monitoring"""
        if not self._running:
            return

        self._running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()

        self.logger.info("Resource monitoring stopped")

    async def _resource_monitoring_loop(self):
        """Background loop for monitoring resource usage"""
        while self._running:
            try:
                # Update resource usage tracking
                current_time = datetime.utcnow()

                # CPU usage
                cpu_usage = ResourceUsage(
                    resource_type=ResourceType.CPU,
                    used=self._get_current_cpu_usage(),
                    total=100.0,
                    percentage=self._get_current_cpu_usage(),
                    timestamp=current_time
                )
                self.resource_usage[ResourceType.CPU].append(cpu_usage)

                # Memory usage
                memory = psutil.virtual_memory()
                memory_usage = ResourceUsage(
                    resource_type=ResourceType.MEMORY,
                    used=memory.used,
                    total=memory.total,
                    percentage=memory.percent,
                    timestamp=current_time
                )
                self.resource_usage[ResourceType.MEMORY].append(memory_usage)

                # Disk usage
                disk = psutil.disk_usage('/')
                disk_usage = ResourceUsage(
                    resource_type=ResourceType.DISK,
                    used=disk.used,
                    total=disk.total,
                    percentage=disk.percent,
                    timestamp=current_time
                )
                self.resource_usage[ResourceType.DISK].append(disk_usage)

                # Update Prometheus metrics
                self._update_prometheus_metrics()

                # Check for resource pressure and trigger cleanup if needed
                await self._check_resource_pressure()

            except Exception as e:
                self.logger.error("Error in resource monitoring loop",
                                context={"error": str(e)})

            await asyncio.sleep(self.resource_check_interval)

    async def _cleanup_loop(self):
        """Background loop for resource cleanup"""
        while self._running:
            try:
                # Run disk cleanup strategies
                await self._run_disk_cleanup()

                # Clean up expired allocations
                self._cleanup_expired_allocations()

                # Force garbage collection periodically
                if len(self.allocations) % 10 == 0:  # Every 10 allocations
                    gc.collect()

            except Exception as e:
                self.logger.error("Error in cleanup loop",
                                context={"error": str(e)})

            await asyncio.sleep(300)  # Run every 5 minutes

    def _update_prometheus_metrics(self):
        """Update Prometheus metrics for monitoring"""
        try:
            # CPU metrics
            cpu_usage = self._get_current_cpu_usage()
            self.monitor.metrics.set_gauge("resource_manager_cpu_usage_percent", cpu_usage)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.monitor.metrics.set_gauge("resource_manager_memory_usage_percent", memory.percent)
            self.monitor.metrics.set_gauge("resource_manager_memory_available_mb",
                                         memory.available / (1024 * 1024))

            # Disk metrics
            disk = psutil.disk_usage('/')
            self.monitor.metrics.set_gauge("resource_manager_disk_usage_percent", disk.percent)

            # Allocation metrics
            self.monitor.metrics.set_gauge("resource_manager_active_allocations", len(self.allocations))
            self.monitor.metrics.set_gauge("resource_manager_memory_pool_utilization",
                                         self.memory_pool.get_usage_stats()["memory_utilization"])

        except Exception as e:
            self.logger.error("Failed to update Prometheus metrics",
                            context={"error": str(e)})

    async def _check_resource_pressure(self):
        """Check for resource pressure and take corrective actions"""
        try:
            disk_usage = self._get_current_disk_usage()
            memory_usage = psutil.virtual_memory().percent

            # Check disk pressure
            if disk_usage > self.disk_cleanup_threshold * 100:
                self.logger.warning("High disk usage detected, triggering cleanup",
                                  context={"disk_usage_percent": disk_usage})
                await self._run_disk_cleanup()

            # Check memory pressure
            if memory_usage > 85:
                self.logger.warning("High memory usage detected",
                                  context={"memory_usage_percent": memory_usage})

                # Force garbage collection
                gc.collect()

                # If still high, consider freeing some resources
                if psutil.virtual_memory().percent > 90:
                    await self._emergency_memory_cleanup()

        except Exception as e:
            self.logger.error("Error checking resource pressure",
                            context={"error": str(e)})

    async def _run_disk_cleanup(self):
        """Run disk cleanup strategies"""
        try:
            cleanup_summary = {"strategies_run": 0, "files_removed": 0, "space_freed_mb": 0}

            for strategy in self.disk_cleanup_strategies:
                try:
                    files_removed, space_freed = await self._run_cleanup_strategy(strategy)
                    cleanup_summary["files_removed"] += files_removed
                    cleanup_summary["space_freed_mb"] += space_freed
                    cleanup_summary["strategies_run"] += 1

                except Exception as e:
                    self.logger.error(f"Cleanup strategy {strategy['name']} failed",
                                    context={"error": str(e)})

            if cleanup_summary["files_removed"] > 0:
                self.logger.info("Disk cleanup completed",
                               context=cleanup_summary)

        except Exception as e:
            self.logger.error("Error running disk cleanup",
                            context={"error": str(e)})

    async def _run_cleanup_strategy(self, strategy: Dict[str, Any]) -> Tuple[int, float]:
        """Run a specific cleanup strategy"""
        files_removed = 0
        space_freed = 0.0

        try:
            # This would implement the actual cleanup logic based on the strategy
            # For now, we'll implement a basic version
            cutoff_time = datetime.utcnow() - timedelta(hours=strategy["max_age_hours"])

            # Find files matching the pattern and older than cutoff
            # This is a simplified implementation - in practice, you'd want more sophisticated pattern matching
            if strategy["pattern"] == "*.tmp":
                temp_files = Path("/tmp").glob("*.tmp")
                for temp_file in temp_files:
                    if temp_file.stat().st_mtime < cutoff_time.timestamp():
                        try:
                            file_size = temp_file.stat().st_size
                            temp_file.unlink()
                            files_removed += 1
                            space_freed += file_size / (1024 * 1024)
                        except Exception as e:
                            self.logger.warning(f"Failed to remove temp file {temp_file}: {e}")

        except Exception as e:
            self.logger.error(f"Error in cleanup strategy {strategy['name']}",
                            context={"error": str(e)})

        return files_removed, space_freed

    def _cleanup_expired_allocations(self):
        """Clean up expired resource allocations"""
        current_time = datetime.utcnow()
        expired_allocations = []

        with self._allocation_lock:
            for allocation_id, allocation in self.allocations.items():
                # Consider allocation expired if it's been running much longer than estimated
                if (current_time - allocation.start_time).total_seconds() > allocation.estimated_end_time.timestamp() * 2:
                    expired_allocations.append(allocation_id)

        for allocation_id in expired_allocations:
            self.logger.warning("Freeing expired allocation",
                              context={"allocation_id": allocation_id})
            self.free_resources(allocation_id)

    async def _emergency_memory_cleanup(self):
        """Emergency memory cleanup when system is under extreme pressure"""
        try:
            # Free memory from completed/failed jobs first
            completed_jobs = []
            for allocation in self.allocations.values():
                if allocation.status in ["completed", "failed"]:
                    completed_jobs.append(allocation.allocation_id)

            for allocation_id in completed_jobs[:5]:  # Free first 5
                self.free_resources(allocation_id)

            # Force garbage collection
            gc.collect()

            self.logger.info("Emergency memory cleanup completed",
                           context={"freed_allocations": len(completed_jobs[:5])})

        except Exception as e:
            self.logger.error("Error in emergency memory cleanup",
                            context={"error": str(e)})


# Global resource manager instance
_resource_manager_instance = None
_resource_manager_lock = threading.Lock()


def get_resource_manager() -> ResourceManager:
    """Get the global resource manager instance"""
    global _resource_manager_instance

    if _resource_manager_instance is None:
        with _resource_manager_lock:
            if _resource_manager_instance is None:
                _resource_manager_instance = ResourceManager()

    return _resource_manager_instance


@asynccontextmanager
async def resource_allocation_context(job_id: str, requirements: JobResourceRequirement):
    """Context manager for automatic resource allocation and cleanup"""
    manager = get_resource_manager()

    # Register requirements
    manager.register_job_requirements(job_id, requirements)

    # Allocate resources
    allocation_id = manager.allocate_resources(job_id)

    if not allocation_id:
        raise RuntimeError(f"Failed to allocate resources for job {job_id}")

    try:
        yield allocation_id
    finally:
        # Free resources
        manager.free_resources(allocation_id)