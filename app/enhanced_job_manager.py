# app/enhanced_job_manager.py
import asyncio
import json
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

from .resource_manager import (
    get_resource_manager, ResourceManager, JobResourceRequirement,
    Priority, ResourceType, resource_allocation_context
)
from .job_scheduler import get_scheduler, IntelligentScheduler, JobDependency, QueuedJob
from .worker_scaler import get_worker_manager, get_load_balancer
from .performance_optimizer import get_performance_optimizer
from .monitoring import get_monitor, StructuredLogger


class JobMigrationReason(Enum):
    LOAD_BALANCING = "load_balancing"
    RESOURCE_PRESSURE = "resource_pressure"
    WORKER_FAILURE = "worker_failure"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"


@dataclass
class ResourceReservation:
    reservation_id: str
    job_id: str
    resources: Dict[ResourceType, float]
    reserved_at: datetime
    expires_at: datetime
    status: str = "active"  # active, used, expired, cancelled


@dataclass
class JobMigration:
    migration_id: str
    job_id: str
    from_worker: str
    to_worker: str
    reason: JobMigrationReason
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "in_progress"  # in_progress, completed, failed
    metadata: Dict[str, Any] = field(default_factory=dict)


class PredictiveScaler:
    """Predictive scaling based on historical usage patterns"""

    def __init__(self):
        self.logger = StructuredLogger("predictive_scaler")

        # Historical data for prediction
        self.usage_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.prediction_models: Dict[str, Any] = {}

        # Prediction configuration
        self.prediction_window = 300  # 5 minutes
        self.min_confidence_threshold = 0.7
        self.scale_ahead_time = 120  # Scale 2 minutes ahead of predicted demand

    def record_usage_pattern(self, metric_name: str, value: float, timestamp: datetime):
        """Record usage pattern for prediction"""
        self.usage_history[metric_name].append({
            "value": value,
            "timestamp": timestamp,
            "hour": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "minute": timestamp.minute
        })

    def predict_usage(self, metric_name: str, future_seconds: int = 300) -> Tuple[Optional[float], float]:
        """Predict future usage based on historical patterns"""
        try:
            if metric_name not in self.usage_history or len(self.usage_history[metric_name]) < 10:
                return None, 0.0

            history = list(self.usage_history[metric_name])

            # Simple trend-based prediction
            if len(history) >= 5:
                recent_values = [h["value"] for h in history[-5:]]
                trend = self._calculate_trend(recent_values)

                # Predict based on trend
                predicted_value = recent_values[-1] + (trend * future_seconds / 60.0)  # Scale by minutes

                # Calculate confidence based on trend consistency
                confidence = self._calculate_prediction_confidence(recent_values)

                return max(0, predicted_value), confidence

            return None, 0.0

        except Exception as e:
            self.logger.error("Error predicting usage",
                            context={"error": str(e), "metric": metric_name})
            return None, 0.0

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend from recent values"""
        if len(values) < 2:
            return 0.0

        # Simple linear trend
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n

        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, values))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _calculate_prediction_confidence(self, values: List[float]) -> float:
        """Calculate confidence in prediction based on value consistency"""
        if len(values) < 3:
            return 0.0

        try:
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            std_dev = variance ** 0.5

            # Confidence decreases with higher variance
            if mean_val > 0:
                coefficient_of_variation = std_dev / mean_val
                confidence = max(0.0, min(1.0, 1.0 - coefficient_of_variation))
                return confidence
            else:
                return 0.5  # Neutral confidence for zero values

        except Exception:
            return 0.0

    def should_scale_up_predictively(self) -> Tuple[bool, str, float]:
        """Check if we should scale up based on predicted demand"""
        try:
            # Predict CPU and memory usage
            cpu_prediction, cpu_confidence = self.predict_usage("cpu_usage", self.scale_ahead_time)
            memory_prediction, memory_confidence = self.predict_usage("memory_usage", self.scale_ahead_time)

            # Check if predictions indicate high future demand
            scale_up_reasons = []

            if cpu_prediction and cpu_confidence > self.min_confidence_threshold:
                if cpu_prediction > 70:  # Predicted > 70% CPU usage
                    scale_up_reasons.append(f"Predicted CPU usage: {cpu_prediction:.1f}%")

            if memory_prediction and memory_confidence > self.min_confidence_threshold:
                if memory_prediction > 75:  # Predicted > 75% memory usage
                    scale_up_reasons.append(f"Predicted memory usage: {memory_prediction:.1f}%")

            if scale_up_reasons:
                reason = "; ".join(scale_up_reasons)
                avg_confidence = (cpu_confidence + memory_confidence) / 2 if cpu_prediction and memory_prediction else max(cpu_confidence, memory_confidence)
                return True, reason, avg_confidence

            return False, "No high demand predicted", 0.0

        except Exception as e:
            self.logger.error("Error in predictive scaling check",
                            context={"error": str(e)})
            return False, f"Error: {str(e)}", 0.0


class EnhancedJobManager:
    """Enhanced job manager with advanced features"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = Redis.from_url(redis_url)
        self.logger = StructuredLogger("enhanced_job_manager")

        # Core components
        self.resource_manager = get_resource_manager()
        self.scheduler = get_scheduler()
        self.worker_manager = get_worker_manager()
        self.load_balancer = get_load_balancer()
        self.performance_optimizer = get_performance_optimizer()

        # Advanced features
        self.predictive_scaler = PredictiveScaler()

        # Resource reservations
        self.resource_reservations: Dict[str, ResourceReservation] = {}
        self.reservation_lock = threading.Lock()

        # Job migrations
        self.job_migrations: Dict[str, JobMigration] = {}
        self.migration_lock = threading.Lock()

        # Background tasks
        self._migration_task = None
        self._reservation_cleanup_task = None
        self._predictive_scaling_task = None
        self._running = False

        # Configuration
        self.migration_check_interval = 60  # seconds
        self.reservation_cleanup_interval = 300  # 5 minutes
        self.predictive_scaling_interval = 120  # 2 minutes

    def submit_enhanced_job(self, job_id: str, resource_requirements: JobResourceRequirement,
                           dependencies: Optional[JobDependency] = None,
                           reserve_resources: bool = False,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Submit a job with enhanced features"""

        try:
            # Reserve resources if requested
            reservation_id = None
            if reserve_resources:
                reservation_id = self._reserve_resources_for_job(job_id, resource_requirements)

            # Submit to scheduler
            success = self.scheduler.submit_job(job_id, resource_requirements, dependencies, metadata)

            if not success:
                # Clean up reservation if job submission failed
                if reservation_id:
                    self._cancel_resource_reservation(reservation_id)
                return {"success": False, "error": "Failed to submit job to scheduler"}

            # Record usage patterns for predictive scaling
            current_time = datetime.utcnow()
            self.predictive_scaler.record_usage_pattern("job_submissions", 1.0, current_time)

            response = {
                "success": True,
                "job_id": job_id,
                "reservation_id": reservation_id,
                "submitted_at": current_time.isoformat(),
                "estimated_start_time": self._estimate_job_start_time(job_id)
            }

            self.logger.info("Enhanced job submitted",
                           context={"job_id": job_id, "reservation_id": reservation_id})

            return response

        except Exception as e:
            self.logger.error("Error submitting enhanced job",
                            context={"job_id": job_id, "error": str(e)})
            return {"success": False, "error": str(e)}

    def _reserve_resources_for_job(self, job_id: str, requirements: JobResourceRequirement) -> Optional[str]:
        """Reserve resources for a large job"""
        try:
            reservation_id = str(uuid.uuid4())

            # Create reservation
            reservation = ResourceReservation(
                reservation_id=reservation_id,
                job_id=job_id,
                resources={
                    ResourceType.CPU: requirements.cpu_cores,
                    ResourceType.MEMORY: requirements.memory_mb,
                    ResourceType.DISK: requirements.disk_mb
                },
                reserved_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=1)  # 1 hour expiry
            )

            with self.reservation_lock:
                self.resource_reservations[reservation_id] = reservation

            # Update monitoring
            self.resource_manager.monitor.metrics.set_gauge("resource_reservations_active", 1,
                                                          labels={"job_id": job_id, "reservation_id": reservation_id})

            self.logger.info("Resources reserved for job",
                           context={"job_id": job_id, "reservation_id": reservation_id})

            return reservation_id

        except Exception as e:
            self.logger.error("Error reserving resources",
                           context={"job_id": job_id, "error": str(e)})
            return None

    def _cancel_resource_reservation(self, reservation_id: str) -> bool:
        """Cancel a resource reservation"""
        with self.reservation_lock:
            if reservation_id in self.resource_reservations:
                reservation = self.resource_reservations[reservation_id]
                reservation.status = "cancelled"

                # Update monitoring
                self.resource_manager.monitor.metrics.set_gauge("resource_reservations_active", 0,
                                                              labels={"job_id": reservation.job_id, "reservation_id": reservation_id})

                self.logger.info("Resource reservation cancelled",
                               context={"reservation_id": reservation_id})

                return True

        return False

    def preempt_job(self, job_id: str, reason: str = "High priority job waiting") -> Dict[str, Any]:
        """Preempt a running job for a higher priority one"""
        try:
            # Find the job in running jobs
            if job_id not in self.scheduler.running_jobs:
                return {"success": False, "error": "Job is not currently running"}

            # Get job information (in a real implementation, you'd track this better)
            # For now, we'll simulate preemption by marking the job as failed
            # In practice, you'd want to pause and resume jobs

            # Mark job as preempted
            self.scheduler.mark_job_failed(job_id, f"Preempted: {reason}")

            # Release resources
            allocation = self.resource_manager.get_allocation_info(job_id)
            if allocation:
                self.resource_manager.free_resources(allocation.allocation_id)

            response = {
                "success": True,
                "job_id": job_id,
                "preempted_at": datetime.utcnow().isoformat(),
                "reason": reason
            }

            self.logger.info("Job preempted",
                           context={"job_id": job_id, "reason": reason})

            return response

        except Exception as e:
            self.logger.error("Error preempting job",
                            context={"job_id": job_id, "error": str(e)})
            return {"success": False, "error": str(e)}

    def migrate_job(self, job_id: str, to_worker: str, reason: JobMigrationReason) -> Dict[str, Any]:
        """Migrate a job from one worker to another"""
        try:
            migration_id = str(uuid.uuid4())

            # In a real implementation, this would involve:
            # 1. Pausing the job on the current worker
            # 2. Transferring job state and data
            # 3. Resuming the job on the target worker

            # For now, we'll simulate migration
            migration = JobMigration(
                migration_id=migration_id,
                job_id=job_id,
                from_worker="current_worker",  # Would be determined dynamically
                to_worker=to_worker,
                reason=reason,
                started_at=datetime.utcnow(),
                status="completed"  # Simulate immediate completion for this example
            )

            with self.migration_lock:
                self.job_migrations[migration_id] = migration

            response = {
                "success": True,
                "migration_id": migration_id,
                "job_id": job_id,
                "to_worker": to_worker,
                "reason": reason.value,
                "started_at": migration.started_at.isoformat()
            }

            self.logger.info("Job migration initiated",
                           context={"migration_id": migration_id, "job_id": job_id, "reason": reason.value})

            return response

        except Exception as e:
            self.logger.error("Error migrating job",
                            context={"job_id": job_id, "error": str(e)})
            return {"success": False, "error": str(e)}

    def _estimate_job_start_time(self, job_id: str) -> Optional[str]:
        """Estimate when a job will start based on queue position and resources"""
        try:
            # Get queue position
            scheduler_status = self.scheduler.get_scheduler_status()
            queued_jobs = scheduler_status.get("queued_jobs", 0)

            # Estimate based on average processing time and available workers
            worker_status = self.worker_manager.get_worker_status()
            active_workers = worker_status.get("active_workers", 1)

            if active_workers > 0:
                # Rough estimate: assume average job takes 5 minutes
                estimated_wait_minutes = (queued_jobs / active_workers) * 5
                estimated_start = datetime.utcnow() + timedelta(minutes=estimated_wait_minutes)
                return estimated_start.isoformat()

            return None

        except Exception as e:
            self.logger.error("Error estimating job start time",
                            context={"job_id": job_id, "error": str(e)})
            return None

    def get_job_insights(self, job_id: str) -> Dict[str, Any]:
        """Get comprehensive insights for a job"""
        try:
            # Get job information from various systems
            scheduler_status = self.scheduler.get_scheduler_status()
            resource_status = self.resource_manager.get_system_resource_status()
            worker_status = self.worker_manager.get_worker_status()

            # Get resource allocation if exists
            allocation = self.resource_manager.get_allocation_info(job_id)

            # Check for reservations
            reservation = None
            for res in self.resource_reservations.values():
                if res.job_id == job_id:
                    reservation = res
                    break

            # Get performance insights
            performance_report = self.performance_optimizer.get_performance_report()

            insights = {
                "job_id": job_id,
                "timestamp": datetime.utcnow().isoformat(),
                "queue_position": self._get_job_queue_position(job_id),
                "estimated_wait_time": self._estimate_job_wait_time(job_id),
                "resource_allocation": {
                    "allocated": allocation is not None,
                    "allocation_id": allocation.allocation_id if allocation else None,
                    "resources": allocation.resources if allocation else {}
                } if allocation else None,
                "resource_reservation": {
                    "reservation_id": reservation.reservation_id,
                    "expires_at": reservation.expires_at.isoformat(),
                    "status": reservation.status
                } if reservation else None,
                "system_health": {
                    "cpu_usage": resource_status.get("cpu", {}).get("current_usage_percent", 0),
                    "memory_usage": resource_status.get("memory", {}).get("usage_percent", 0),
                    "active_workers": worker_status.get("active_workers", 0),
                    "queued_jobs": scheduler_status.get("queued_jobs", 0)
                },
                "recommendations": self._get_job_specific_recommendations(job_id, performance_report)
            }

            return insights

        except Exception as e:
            self.logger.error("Error getting job insights",
                            context={"job_id": job_id, "error": str(e)})
            return {"error": str(e)}

    def _get_job_queue_position(self, job_id: str) -> Optional[int]:
        """Get the position of a job in the queue"""
        # This would need to be implemented based on the actual queue structure
        # For now, return a placeholder
        return None

    def _estimate_job_wait_time(self, job_id: str) -> Optional[str]:
        """Estimate how long a job will wait in queue"""
        return self._estimate_job_start_time(job_id)

    def _get_job_specific_recommendations(self, job_id: str, performance_report: Dict[str, Any]) -> List[str]:
        """Get job-specific recommendations"""
        recommendations = []

        # Check for bottlenecks that might affect this job
        bottlenecks = performance_report.get("current_bottlenecks", [])

        for bottleneck in bottlenecks:
            if bottleneck.get("severity", 0) > 0.7:  # High severity bottlenecks
                recommendations.append(f"System bottleneck detected: {bottleneck.get('description', '')}")

        # Check for resource pressure
        system_health = performance_report.get("summary", {})
        if system_health.get("total_bottlenecks", 0) > 0:
            recommendations.append("System is experiencing performance bottlenecks that may delay job processing")

        return recommendations

    async def start_enhanced_management(self):
        """Start the enhanced job management system"""
        if self._running:
            return

        self._running = True

        # Start core systems
        await self.resource_manager.start_resource_monitoring()
        await self.scheduler.start_scheduler()
        await self.worker_manager.start_worker_management()
        await self.performance_optimizer.start_optimization()

        # Start advanced features
        self._migration_task = asyncio.create_task(self._migration_monitoring_loop())
        self._reservation_cleanup_task = asyncio.create_task(self._reservation_cleanup_loop())
        self._predictive_scaling_task = asyncio.create_task(self._predictive_scaling_loop())

        self.logger.info("Enhanced job management system started")

    async def stop_enhanced_management(self):
        """Stop the enhanced job management system"""
        if not self._running:
            return

        self._running = False

        # Stop advanced features
        if self._migration_task:
            self._migration_task.cancel()
        if self._reservation_cleanup_task:
            self._reservation_cleanup_task.cancel()
        if self._predictive_scaling_task:
            self._predictive_scaling_task.cancel()

        # Stop core systems
        await self.performance_optimizer.stop_optimization()
        await self.worker_manager.stop_worker_management()
        await self.scheduler.stop_scheduler()
        await self.resource_manager.stop_resource_monitoring()

        self.logger.info("Enhanced job management system stopped")

    async def _migration_monitoring_loop(self):
        """Monitor and execute job migrations for load balancing"""
        while self._running:
            try:
                # Check for migration opportunities
                await self._check_migration_opportunities()

            except Exception as e:
                self.logger.error("Error in migration monitoring loop",
                                context={"error": str(e)})

            await asyncio.sleep(self.migration_check_interval)

    async def _reservation_cleanup_loop(self):
        """Clean up expired resource reservations"""
        while self._running:
            try:
                # Clean up expired reservations
                self._cleanup_expired_reservations()

            except Exception as e:
                self.logger.error("Error in reservation cleanup loop",
                                context={"error": str(e)})

            await asyncio.sleep(self.reservation_cleanup_interval)

    async def _predictive_scaling_loop(self):
        """Monitor and execute predictive scaling"""
        while self._running:
            try:
                # Check if predictive scaling is needed
                should_scale, reason, confidence = self.predictive_scaler.should_scale_up_predictively()

                if should_scale and confidence > self.predictive_scaler.min_confidence_threshold:
                    # Get current worker count
                    worker_status = self.worker_manager.get_worker_status()
                    current_workers = worker_status.get("active_workers", 0)

                    # Scale up by 1 worker
                    if current_workers < self.worker_manager.max_workers:
                        worker_id = self.worker_manager.start_worker()
                        if worker_id:
                            self.logger.info("Predictive scaling: started worker",
                                           context={"worker_id": worker_id, "reason": reason, "confidence": confidence})

            except Exception as e:
                self.logger.error("Error in predictive scaling loop",
                                context={"error": str(e)})

            await asyncio.sleep(self.predictive_scaling_interval)

    async def _check_migration_opportunities(self):
        """Check for opportunities to migrate jobs for better load balancing"""
        try:
            # Get worker load information
            worker_status = self.worker_manager.get_worker_status()

            if worker_status.get("active_workers", 0) < 2:
                return  # Need at least 2 workers for migration

            # Check for overloaded workers
            for worker_id, worker in self.worker_manager.workers.items():
                if worker.status == WorkerStatus.RUNNING:
                    # Check if worker is overloaded (high resource usage)
                    if worker.resource_usage.get("cpu_percent", 0) > 80:
                        # Find a less loaded worker to migrate to
                        target_worker = self._find_least_loaded_worker(exclude_worker=worker_id)

                        if target_worker:
                            # In a real implementation, you'd select a job to migrate
                            # For now, we'll just log the opportunity
                            self.logger.info("Migration opportunity detected",
                                           context={
                                               "from_worker": worker_id,
                                               "to_worker": target_worker,
                                               "reason": "High CPU usage on source worker"
                                           })

        except Exception as e:
            self.logger.error("Error checking migration opportunities",
                            context={"error": str(e)})

    def _find_least_loaded_worker(self, exclude_worker: str) -> Optional[str]:
        """Find the least loaded worker, excluding the specified worker"""
        least_loaded = None
        min_load = float('inf')

        for worker_id, worker in self.worker_manager.workers.items():
            if worker_id == exclude_worker or worker.status != WorkerStatus.RUNNING:
                continue

            # Calculate load score
            cpu_usage = worker.resource_usage.get("cpu_percent", 50)
            memory_usage = worker.resource_usage.get("memory_percent", 50)
            load_score = (cpu_usage + memory_usage) / 2

            if load_score < min_load:
                min_load = load_score
                least_loaded = worker_id

        return least_loaded

    def _cleanup_expired_reservations(self):
        """Clean up expired resource reservations"""
        current_time = datetime.utcnow()
        expired_reservations = []

        with self.reservation_lock:
            for reservation_id, reservation in self.resource_reservations.items():
                if reservation.expires_at < current_time and reservation.status == "active":
                    expired_reservations.append(reservation_id)

        for reservation_id in expired_reservations:
            with self.reservation_lock:
                if reservation_id in self.resource_reservations:
                    reservation = self.resource_reservations[reservation_id]
                    reservation.status = "expired"

                    # Update monitoring
                    self.resource_manager.monitor.metrics.set_gauge("resource_reservations_active", 0,
                                                                  labels={"job_id": reservation.job_id, "reservation_id": reservation_id})

            self.logger.info("Expired resource reservation cleaned up",
                           context={"reservation_id": reservation_id})

    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get comprehensive enhanced job management status"""
        try:
            # Get status from all components
            scheduler_status = self.scheduler.get_scheduler_status()
            worker_status = self.worker_manager.get_worker_status()
            resource_status = self.resource_manager.get_system_resource_status()
            performance_report = self.performance_optimizer.get_performance_report()

            # Get advanced features status
            active_reservations = len([r for r in self.resource_reservations.values() if r.status == "active"])
            active_migrations = len([m for m in self.job_migrations.values() if m.status == "in_progress"])

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "system_status": "running" if self._running else "stopped",
                "components": {
                    "scheduler": scheduler_status,
                    "worker_manager": worker_status,
                    "resource_manager": resource_status,
                    "performance_optimizer": performance_report.get("summary", {})
                },
                "advanced_features": {
                    "active_reservations": active_reservations,
                    "active_migrations": active_migrations,
                    "predictive_scaling_enabled": True,
                    "job_preemption_enabled": True,
                    "load_balancing_enabled": True
                },
                "overall_health": self._calculate_overall_health()
            }

        except Exception as e:
            self.logger.error("Error getting enhanced status",
                            context={"error": str(e)})
            return {"error": str(e)}

    def _calculate_overall_health(self) -> str:
        """Calculate overall system health"""
        try:
            # Simple health calculation based on various metrics
            health_score = 100

            # Deduct points for bottlenecks
            performance_report = self.performance_optimizer.get_performance_report()
            bottlenecks = performance_report.get("current_bottlenecks", [])
            health_score -= len(bottlenecks) * 10

            # Deduct points for high resource usage
            resource_status = self.resource_manager.get_system_resource_status()
            cpu_usage = resource_status.get("cpu", {}).get("current_usage_percent", 0)
            memory_usage = resource_status.get("memory", {}).get("usage_percent", 0)

            if cpu_usage > 80:
                health_score -= (cpu_usage - 80)
            if memory_usage > 80:
                health_score -= (memory_usage - 80)

            # Deduct points for queue congestion
            scheduler_status = self.scheduler.get_scheduler_status()
            queued_jobs = scheduler_status.get("queued_jobs", 0)
            if queued_jobs > 20:
                health_score -= min(50, queued_jobs - 20)

            # Determine health status
            if health_score >= 80:
                return "healthy"
            elif health_score >= 60:
                return "degraded"
            else:
                return "unhealthy"

        except Exception:
            return "unknown"


# Global enhanced job manager instance
_enhanced_manager_instance = None
_enhanced_manager_lock = threading.Lock()


def get_enhanced_job_manager() -> EnhancedJobManager:
    """Get the global enhanced job manager instance"""
    global _enhanced_manager_instance

    if _enhanced_manager_instance is None:
        with _enhanced_manager_lock:
            if _enhanced_manager_instance is None:
                _enhanced_manager_instance = EnhancedJobManager()

    return _enhanced_manager_instance