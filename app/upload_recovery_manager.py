# app/upload_recovery_manager.py
import asyncio
import json
import os
import shutil
import subprocess
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple, Set
import threading
import redis
import logging

from .monitoring import StructuredLogger, MetricsCollector
from .error_recovery import ErrorRecoveryManager, ErrorContext, ErrorCategory, ErrorSeverity, RecoveryStrategy
from .upload_health_checker import UploadHealthChecker, UploadComponentType, UploadComponentStatus
from .upload_alert_manager import UploadAlertManager, AlertSeverity


class RecoveryAction(Enum):
    RESTART_SERVICE = "restart_service"
    CLEAR_CACHE = "clear_cache"
    FREE_DISK_SPACE = "free_disk_space"
    RESTART_WORKER = "restart_worker"
    REINSTALL_PACKAGE = "reinstall_package"
    CHECK_PERMISSIONS = "check_permissions"
    VALIDATE_CONFIG = "validate_config"
    NETWORK_RESET = "network_reset"
    MEMORY_CLEANUP = "memory_cleanup"
    REDIS_FLUSH = "redis_flush"


class RemediationStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RemediationTask:
    """Automated remediation task"""
    task_id: str
    component_type: UploadComponentType
    action: RecoveryAction
    description: str
    status: RemediationStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class SelfHealingConfig:
    """Configuration for self-healing capabilities"""
    enabled: bool = True
    max_concurrent_remediations: int = 3
    remediation_timeout: int = 300  # 5 minutes
    cooldown_between_attempts: int = 600  # 10 minutes
    enable_graceful_degradation: bool = True
    preserve_user_data: bool = True
    backup_before_remediation: bool = True


class UploadRecoveryManager:
    """Advanced recovery manager for upload operations with automated remediation"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = redis.from_url(redis_url)
        self.logger = StructuredLogger("upload_recovery_manager")

        # Core components
        self.health_checker = None
        self.alert_manager = None
        self.error_recovery = None

        # Initialize components
        try:
            self.health_checker = UploadHealthChecker(redis_url)
            self.alert_manager = UploadAlertManager(redis_url)
            self.error_recovery = ErrorRecoveryManager(redis_url)
        except Exception as e:
            self.logger.error("Failed to initialize recovery components",
                            context={"error": str(e)})

        # Remediation tracking
        self.active_remediations: Dict[str, RemediationTask] = {}
        self.remediation_history: Dict[str, RemediationTask] = {}
        self.component_recovery_history: Dict[UploadComponentType, List[datetime]] = defaultdict(list)

        # Configuration
        self.self_healing_config = SelfHealingConfig()
        self.remediation_procedures: Dict[Tuple[UploadComponentType, UploadComponentStatus], List[RecoveryAction]] = {}
        self.last_remediation_attempt: Dict[UploadComponentType, datetime] = {}

        # Locks for thread safety
        self._remediation_lock = threading.Lock()
        self._history_lock = threading.Lock()

        # Background tasks
        self._remediation_task = None
        self._monitoring_task = None
        self._cleanup_task = None
        self._running = False

        # Setup remediation procedures
        self._setup_remediation_procedures()

    def _setup_remediation_procedures(self):
        """Setup automated remediation procedures for different failure scenarios"""

        # Disk storage remediation
        self.remediation_procedures[(UploadComponentType.DISK_STORAGE, UploadComponentStatus.UNHEALTHY)] = [
            RecoveryAction.FREE_DISK_SPACE,
            RecoveryAction.CHECK_PERMISSIONS,
            RecoveryAction.VALIDATE_CONFIG
        ]

        self.remediation_procedures[(UploadComponentType.DISK_STORAGE, UploadComponentStatus.DEGRADED)] = [
            RecoveryAction.CLEAR_CACHE,
            RecoveryAction.FREE_DISK_SPACE
        ]

        # Network remediation
        self.remediation_procedures[(UploadComponentType.NETWORK, UploadComponentStatus.UNHEALTHY)] = [
            RecoveryAction.NETWORK_RESET,
            RecoveryAction.VALIDATE_CONFIG
        ]

        self.remediation_procedures[(UploadComponentType.NETWORK, UploadComponentStatus.DEGRADED)] = [
            RecoveryAction.NETWORK_RESET
        ]

        # FFmpeg remediation
        self.remediation_procedures[(UploadComponentType.FFMPEG, UploadComponentStatus.UNHEALTHY)] = [
            RecoveryAction.REINSTALL_PACKAGE,
            RecoveryAction.VALIDATE_CONFIG
        ]

        self.remediation_procedures[(UploadComponentType.FFMPEG, UploadComponentStatus.DEGRADED)] = [
            RecoveryAction.RESTART_SERVICE
        ]

        # Audio libraries remediation
        self.remediation_procedures[(UploadComponentType.AUDIO_LIBRARIES, UploadComponentStatus.UNHEALTHY)] = [
            RecoveryAction.REINSTALL_PACKAGE,
            RecoveryAction.VALIDATE_CONFIG
        ]

        # Redis remediation
        self.remediation_procedures[(UploadComponentType.REDIS, UploadComponentStatus.UNHEALTHY)] = [
            RecoveryAction.REDIS_FLUSH,
            RecoveryAction.RESTART_SERVICE
        ]

        self.remediation_procedures[(UploadComponentType.REDIS, UploadComponentStatus.DEGRADED)] = [
            RecoveryAction.CLEAR_CACHE
        ]

        # B2 Storage remediation
        self.remediation_procedures[(UploadComponentType.B2_STORAGE, UploadComponentStatus.UNHEALTHY)] = [
            RecoveryAction.VALIDATE_CONFIG,
            RecoveryAction.NETWORK_RESET
        ]

        # Memory remediation
        self.remediation_procedures[(UploadComponentType.MEMORY, UploadComponentStatus.UNHEALTHY)] = [
            RecoveryAction.MEMORY_CLEANUP,
            RecoveryAction.RESTART_WORKER
        ]

        self.remediation_procedures[(UploadComponentType.MEMORY, UploadComponentStatus.DEGRADED)] = [
            RecoveryAction.MEMORY_CLEANUP
        ]

        # CPU remediation
        self.remediation_procedures[(UploadComponentType.CPU, UploadComponentStatus.DEGRADED)] = [
            RecoveryAction.RESTART_WORKER
        ]

    async def start_recovery_management(self):
        """Start the recovery management system"""
        if self._running:
            return

        self._running = True
        self.logger.info("Starting upload recovery management")

        # Start core components
        if self.health_checker:
            await self.health_checker.start_health_monitoring()
        if self.alert_manager:
            await self.alert_manager.start_alert_management()

        # Start background tasks
        self._remediation_task = asyncio.create_task(self._remediation_loop())
        self._monitoring_task = asyncio.create_task(self._recovery_monitoring_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_recovery_management(self):
        """Stop the recovery management system"""
        if not self._running:
            return

        self._running = False
        self.logger.info("Stopping upload recovery management")

        # Stop core components
        if self.health_checker:
            await self.health_checker.stop_health_monitoring()
        if self.alert_manager:
            await self.alert_manager.stop_alert_management()

        # Stop background tasks
        tasks = [self._remediation_task, self._monitoring_task, self._cleanup_task]
        for task in tasks:
            if task:
                task.cancel()

    async def _remediation_loop(self):
        """Main remediation loop"""
        while self._running:
            try:
                # Check for components that need remediation
                await self._check_and_trigger_remediation()

                # Process active remediations
                await self._process_active_remediations()

            except Exception as e:
                self.logger.error("Error in remediation loop",
                                context={"error": str(e)})

            await asyncio.sleep(60)  # Check every minute

    async def _check_and_trigger_remediation(self):
        """Check component health and trigger remediation if needed"""
        if not self.health_checker:
            return

        try:
            # Get current health status for all components
            health_status = self.health_checker.get_component_health()

            for component_type in UploadComponentType:
                component_key = component_type.value
                if component_key not in health_status:
                    continue

                component_info = health_status[component_key]
                current_status = UploadComponentStatus(component_info["status"])

                # Check if remediation is needed
                remediation_key = (component_type, current_status)
                if remediation_key in self.remediation_procedures:
                    if self._should_attempt_remediation(component_type, current_status):
                        # Trigger remediation
                        await self._trigger_component_remediation(component_type, current_status)

        except Exception as e:
            self.logger.error("Error checking for remediation needs",
                            context={"error": str(e)})

    def _should_attempt_remediation(self, component_type: UploadComponentType,
                                   status: UploadComponentStatus) -> bool:
        """Check if remediation should be attempted for a component"""
        current_time = datetime.utcnow()

        # Check cooldown period
        if component_type in self.last_remediation_attempt:
            time_since_last = current_time - self.last_remediation_attempt[component_type]
            if time_since_last < timedelta(seconds=self.self_healing_config.cooldown_between_attempts):
                return False

        # Check if already have active remediation for this component
        active_for_component = [
            task for task in self.active_remediations.values()
            if task.component_type == component_type and task.status == RemediationStatus.IN_PROGRESS
        ]

        if len(active_for_component) > 0:
            return False

        # Check recent failure history
        if component_type in self.component_recovery_history:
            recent_failures = [
                failure_time for failure_time in self.component_recovery_history[component_type]
                if failure_time > current_time - timedelta(hours=1)
            ]

            # If too many recent failures, skip remediation
            if len(recent_failures) >= 5:
                self.logger.warning("Skipping remediation due to recent failures",
                                  context={
                                      "component": component_type.value,
                                      "recent_failures": len(recent_failures)
                                  })
                return False

        return True

    async def _trigger_component_remediation(self, component_type: UploadComponentType,
                                           status: UploadComponentStatus):
        """Trigger remediation for a component"""
        remediation_key = (component_type, status)
        actions = self.remediation_procedures.get(remediation_key, [])

        if not actions:
            return

        for action in actions:
            task_id = str(uuid.uuid4())
            task = RemediationTask(
                task_id=task_id,
                component_type=component_type,
                action=action,
                description=f"Automated remediation for {component_type.value}: {action.value}",
                status=RemediationStatus.PENDING,
                created_at=datetime.utcnow(),
                metadata={
                    "triggered_by": "health_check",
                    "component_status": status.value,
                    "auto_remediation": True
                }
            )

            with self._remediation_lock:
                self.active_remediations[task_id] = task

            self.logger.info("Remediation task created",
                           context={
                               "task_id": task_id,
                               "component": component_type.value,
                               "action": action.value,
                               "status": status.value
                           })

            # Update last remediation attempt time
            self.last_remediation_attempt[component_type] = datetime.utcnow()

    async def _process_active_remediations(self):
        """Process active remediation tasks"""
        with self._remediation_lock:
            tasks_to_process = [
                task for task in self.active_remediations.values()
                if task.status == RemediationStatus.PENDING
            ]

        # Limit concurrent remediations
        max_concurrent = self.self_healing_config.max_concurrent_remediations
        tasks_to_process = tasks_to_process[:max_concurrent]

        for task in tasks_to_process:
            try:
                # Start the remediation task
                task.status = RemediationStatus.IN_PROGRESS
                task.started_at = datetime.utcnow()

                self.logger.info("Starting remediation task",
                               context={
                                   "task_id": task.task_id,
                                   "component": task.component_type.value,
                                   "action": task.action.value
                               })

                # Execute the remediation action
                success = await self._execute_remediation_action(task)

                # Update task status
                task.completed_at = datetime.utcnow()
                if success:
                    task.status = RemediationStatus.SUCCESS
                    self.logger.info("Remediation task completed successfully",
                                   context={"task_id": task.task_id})
                else:
                    task.status = RemediationStatus.FAILED
                    task.retry_count += 1

                    # Check if we should retry
                    if task.retry_count < task.max_retries:
                        task.status = RemediationStatus.PENDING
                        self.logger.warning("Remediation task failed, will retry",
                                          context={
                                              "task_id": task.task_id,
                                              "retry_count": task.retry_count
                                          })
                    else:
                        self.logger.error("Remediation task failed after max retries",
                                        context={
                                            "task_id": task.task_id,
                                            "retry_count": task.retry_count
                                        })

            except Exception as e:
                task.status = RemediationStatus.FAILED
                task.error_message = str(e)
                task.completed_at = datetime.utcnow()

                self.logger.error("Error executing remediation task",
                                context={
                                    "task_id": task.task_id,
                                    "error": str(e)
                                })

    async def _execute_remediation_action(self, task: RemediationTask) -> bool:
        """Execute a specific remediation action"""
        try:
            if task.action == RecoveryAction.FREE_DISK_SPACE:
                return await self._free_disk_space(task)
            elif task.action == RecoveryAction.CLEAR_CACHE:
                return await self._clear_cache(task)
            elif task.action == RecoveryAction.RESTART_SERVICE:
                return await self._restart_service(task)
            elif task.action == RecoveryAction.RESTART_WORKER:
                return await self._restart_worker(task)
            elif task.action == RecoveryAction.REINSTALL_PACKAGE:
                return await self._reinstall_package(task)
            elif task.action == RecoveryAction.CHECK_PERMISSIONS:
                return await self._check_permissions(task)
            elif task.action == RecoveryAction.VALIDATE_CONFIG:
                return await self._validate_config(task)
            elif task.action == RecoveryAction.NETWORK_RESET:
                return await self._network_reset(task)
            elif task.action == RecoveryAction.MEMORY_CLEANUP:
                return await self._memory_cleanup(task)
            elif task.action == RecoveryAction.REDIS_FLUSH:
                return await self._redis_flush(task)
            else:
                self.logger.error("Unknown remediation action",
                                context={"action": task.action.value})
                return False

        except Exception as e:
            task.error_message = str(e)
            self.logger.error("Remediation action failed",
                           context={
                               "task_id": task.task_id,
                               "action": task.action.value,
                               "error": str(e)
                           })
            return False

    async def _free_disk_space(self, task: RemediationTask) -> bool:
        """Free up disk space by cleaning temporary files"""
        try:
            # Clean up temporary directories
            temp_dirs = [
                Path("./tmp"),
                Path("./pipeline/temp"),
                Path("./uploads/temp")
            ]

            freed_bytes = 0

            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    # Remove files older than 1 hour
                    cutoff_time = time.time() - 3600

                    for file_path in temp_dir.rglob("*"):
                        if file_path.is_file():
                            try:
                                # Check if file is old enough to delete
                                if file_path.stat().st_mtime < cutoff_time:
                                    file_size = file_path.stat().st_size
                                    file_path.unlink()
                                    freed_bytes += file_size

                                    task.metadata[f"deleted_file_{file_path.name}"] = {
                                        "size_bytes": file_size,
                                        "path": str(file_path)
                                    }

                            except Exception as e:
                                self.logger.warning("Failed to delete temp file",
                                                  context={"file": str(file_path), "error": str(e)})

            # Clean up empty directories
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    try:
                        # Remove empty subdirectories
                        for dir_path in list(temp_dir.rglob("*")):
                            if dir_path.is_dir() and not any(dir_path.iterdir()):
                                dir_path.rmdir()
                    except Exception as e:
                        self.logger.warning("Failed to clean empty directories",
                                          context={"directory": str(temp_dir), "error": str(e)})

            freed_mb = freed_bytes / (1024 * 1024)
            task.metadata["total_freed_mb"] = round(freed_mb, 2)

            self.logger.info("Disk space cleanup completed",
                           context={"freed_mb": freed_mb})

            return freed_bytes > 0

        except Exception as e:
            self.logger.error("Disk space cleanup failed",
                            context={"error": str(e)})
            return False

    async def _clear_cache(self, task: RemediationTask) -> bool:
        """Clear various caches"""
        try:
            success = True

            # Clear Redis cache if available
            try:
                # Clear upload-related cache keys
                cache_patterns = [
                    "upload_cache:*",
                    "audio_cache:*",
                    "temp_upload:*"
                ]

                for pattern in cache_patterns:
                    keys = self.redis.keys(pattern)
                    if keys:
                        self.redis.delete(*keys)

                task.metadata["redis_cache_cleared"] = True

            except Exception as e:
                self.logger.warning("Failed to clear Redis cache",
                                  context={"error": str(e)})
                success = False

            # Clear local cache directories
            cache_dirs = [
                Path("./pipeline/cache"),
                Path("./.cache")
            ]

            for cache_dir in cache_dirs:
                if cache_dir.exists():
                    try:
                        # Remove files older than 30 minutes
                        cutoff_time = time.time() - 1800

                        for file_path in cache_dir.rglob("*"):
                            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                                file_path.unlink()

                        task.metadata[f"cleared_cache_{cache_dir.name}"] = True

                    except Exception as e:
                        self.logger.warning("Failed to clear local cache",
                                          context={"directory": str(cache_dir), "error": str(e)})
                        success = False

            return success

        except Exception as e:
            self.logger.error("Cache clearing failed",
                            context={"error": str(e)})
            return False

    async def _restart_service(self, task: RemediationTask) -> bool:
        """Restart system services"""
        try:
            # This would restart relevant services
            # For now, simulate service restart

            task.metadata["service_restart_simulated"] = True

            # In a real implementation, you would:
            # - Restart FFmpeg-related processes
            # - Restart Redis if it's a local instance
            # - Restart web server processes

            self.logger.info("Service restart simulated",
                           context={"component": task.component_type.value})

            return True

        except Exception as e:
            self.logger.error("Service restart failed",
                            context={"error": str(e)})
            return False

    async def _restart_worker(self, task: RemediationTask) -> bool:
        """Restart worker processes"""
        try:
            # This would restart RQ workers or similar
            # For now, simulate worker restart

            task.metadata["worker_restart_simulated"] = True

            self.logger.info("Worker restart simulated",
                           context={"component": task.component_type.value})

            return True

        except Exception as e:
            self.logger.error("Worker restart failed",
                            context={"error": str(e)})
            return False

    async def _reinstall_package(self, task: RemediationTask) -> bool:
        """Reinstall problematic packages"""
        try:
            # This would reinstall packages like ffmpeg, audio libraries, etc.
            # For now, simulate package reinstallation

            task.metadata["package_reinstall_simulated"] = True

            self.logger.info("Package reinstallation simulated",
                           context={"component": task.component_type.value})

            return True

        except Exception as e:
            self.logger.error("Package reinstallation failed",
                            context={"error": str(e)})
            return False

    async def _check_permissions(self, task: RemediationTask) -> bool:
        """Check and fix file permissions"""
        try:
            # Check and fix permissions for upload directories
            upload_dirs = [
                Path("./uploads"),
                Path("./pipeline"),
                Path("./tmp")
            ]

            fixed_permissions = []

            for upload_dir in upload_dirs:
                if upload_dir.exists():
                    try:
                        # Ensure directory is writable
                        test_file = upload_dir / f".perm_check_{uuid.uuid4()}.tmp"
                        test_file.write_text("test")
                        test_file.unlink()

                        task.metadata[f"permissions_ok_{upload_dir.name}"] = True

                    except Exception as e:
                        # Try to fix permissions (on Unix systems)
                        try:
                            upload_dir.chmod(0o755)
                            fixed_permissions.append(str(upload_dir))

                            task.metadata[f"permissions_fixed_{upload_dir.name}"] = True

                        except Exception as fix_error:
                            self.logger.error("Failed to fix permissions",
                                            context={"directory": str(upload_dir), "error": str(fix_error)})
                            task.metadata[f"permissions_error_{upload_dir.name}"] = str(fix_error)

            task.metadata["fixed_permissions_count"] = len(fixed_permissions)

            return len(fixed_permissions) == 0  # Success if no fixes were needed

        except Exception as e:
            self.logger.error("Permission check failed",
                            context={"error": str(e)})
            return False

    async def _validate_config(self, task: RemediationTask) -> bool:
        """Validate system configuration"""
        try:
            # Validate environment variables and configuration files
            config_issues = []

            # Check required environment variables for upload operations
            required_env_vars = [
                "B2_KEY_ID",
                "B2_KEY"
            ]

            for env_var in required_env_vars:
                if not os.getenv(env_var):
                    config_issues.append(f"Missing environment variable: {env_var}")

            # Check configuration files
            config_files = [
                Path("./.env"),
                Path("./pipeline/config.py")
            ]

            for config_file in config_files:
                if config_file.exists():
                    try:
                        # Basic validation - check if file is readable and not empty
                        if config_file.stat().st_size == 0:
                            config_issues.append(f"Empty config file: {config_file}")

                        # Try to parse as JSON if it's a .env file
                        if config_file.suffix == ".env":
                            # Basic .env validation
                            content = config_file.read_text()
                            if not content.strip():
                                config_issues.append(f"Empty .env file: {config_file}")

                    except Exception as e:
                        config_issues.append(f"Config file error {config_file}: {str(e)}")

            task.metadata["config_issues"] = config_issues
            task.metadata["config_issues_count"] = len(config_issues)

            return len(config_issues) == 0

        except Exception as e:
            self.logger.error("Config validation failed",
                            context={"error": str(e)})
            return False

    async def _network_reset(self, task: RemediationTask) -> bool:
        """Reset network connections"""
        try:
            # This would reset network interfaces, restart networking services, etc.
            # For now, simulate network reset

            task.metadata["network_reset_simulated"] = True

            self.logger.info("Network reset simulated",
                           context={"component": task.component_type.value})

            return True

        except Exception as e:
            self.logger.error("Network reset failed",
                            context={"error": str(e)})
            return False

    async def _memory_cleanup(self, task: RemediationTask) -> bool:
        """Clean up memory usage"""
        try:
            # Force garbage collection
            import gc
            gc.collect()

            # Clear any large caches or temporary data structures
            # This would depend on the specific application

            task.metadata["gc_collected"] = True

            self.logger.info("Memory cleanup completed",
                           context={"component": task.component_type.value})

            return True

        except Exception as e:
            self.logger.error("Memory cleanup failed",
                            context={"error": str(e)})
            return False

    async def _redis_flush(self, task: RemediationTask) -> bool:
        """Flush Redis database"""
        try:
            # Flush all Redis data (dangerous - use with caution)
            # In practice, you might want to flush only specific patterns

            confirmation = task.metadata.get("confirm_dangerous_operation", False)
            if not confirmation:
                self.logger.warning("Redis flush skipped - requires explicit confirmation",
                                  context={"task_id": task.task_id})
                return False

            # Flush Redis
            self.redis.flushdb()

            task.metadata["redis_flushed"] = True

            self.logger.warning("Redis database flushed",
                              context={"task_id": task.task_id})

            return True

        except Exception as e:
            self.logger.error("Redis flush failed",
                            context={"error": str(e)})
            return False

    async def _recovery_monitoring_loop(self):
        """Monitor recovery effectiveness and system health"""
        while self._running:
            try:
                # Monitor component health and trigger alerts if needed
                await self._monitor_recovery_effectiveness()

                # Update recovery statistics
                self._update_recovery_statistics()

            except Exception as e:
                self.logger.error("Error in recovery monitoring loop",
                                context={"error": str(e)})

            await asyncio.sleep(120)  # Monitor every 2 minutes

    async def _monitor_recovery_effectiveness(self):
        """Monitor how effective recovery actions are"""
        if not self.health_checker:
            return

        try:
            # Get current health status
            health_status = self.health_checker.get_component_health()

            for component_type in UploadComponentType:
                component_key = component_type.value
                if component_key not in health_status:
                    continue

                component_info = health_status[component_key]
                current_status = UploadComponentStatus(component_info["status"])

                # Check if component is still unhealthy after remediation attempts
                if current_status == UploadComponentStatus.UNHEALTHY:
                    recent_remediations = [
                        task for task in self.active_remediations.values()
                        if (task.component_type == component_type and
                            task.created_at > datetime.utcnow() - timedelta(hours=1))
                    ]

                    if recent_remediations:
                        # Alert about ineffective remediation
                        if self.alert_manager:
                            self.alert_manager.create_alert(
                                rule_id="remediation_ineffective",
                                title=f"Remediation Ineffective: {component_type.value}",
                                message=f"Component {component_type.value} remains unhealthy after remediation attempts",
                                severity=AlertSeverity.WARNING,
                                component_type=component_type,
                                metadata={
                                    "remediation_attempts": len(recent_remediations),
                                    "component_status": current_status.value
                                }
                            )

        except Exception as e:
            self.logger.error("Error monitoring recovery effectiveness",
                            context={"error": str(e)})

    def _update_recovery_statistics(self):
        """Update recovery statistics for monitoring"""
        with self._remediation_lock:
            total_tasks = len(self.active_remediations) + len(self.remediation_history)
            successful_tasks = len([
                task for task in list(self.active_remediations.values()) + list(self.remediation_history.values())
                if task.status == RemediationStatus.SUCCESS
            ])

            success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0

            # Update statistics in task metadata
            for task in self.active_remediations.values():
                task.metadata["overall_success_rate"] = success_rate

    async def _cleanup_loop(self):
        """Clean up old remediation history"""
        while self._running:
            try:
                current_time = datetime.utcnow()
                retention_cutoff = current_time - timedelta(hours=24)  # 24 hours retention

                with self._history_lock:
                    # Move completed tasks to history
                    completed_tasks = [
                        task_id for task_id, task in self.active_remediations.items()
                        if (task.completed_at and task.completed_at < retention_cutoff)
                    ]

                    for task_id in completed_tasks:
                        task = self.active_remediations.pop(task_id)
                        self.remediation_history[task_id] = task

                        # Track recovery history for components
                        if task.status == RemediationStatus.FAILED:
                            self.component_recovery_history[task.component_type].append(task.completed_at)

                    # Limit history size
                    if len(self.remediation_history) > 1000:
                        oldest_tasks = sorted(
                            self.remediation_history.values(),
                            key=lambda t: t.completed_at or t.created_at
                        )[:100]  # Remove oldest 100

                        for task in oldest_tasks:
                            del self.remediation_history[task.task_id]

            except Exception as e:
                self.logger.error("Error in cleanup loop",
                                context={"error": str(e)})

            await asyncio.sleep(3600)  # Clean up every hour

    def trigger_manual_remediation(self, component_type: UploadComponentType,
                                 action: RecoveryAction, description: str,
                                 requested_by: str) -> str:
        """Trigger manual remediation"""
        task_id = str(uuid.uuid4())
        task = RemediationTask(
            task_id=task_id,
            component_type=component_type,
            action=action,
            description=f"Manual remediation: {description}",
            status=RemediationStatus.PENDING,
            created_at=datetime.utcnow(),
            metadata={
                "triggered_by": "manual",
                "requested_by": requested_by,
                "manual_remediation": True
            }
        )

        with self._remediation_lock:
            self.active_remediations[task_id] = task

        self.logger.info("Manual remediation triggered",
                       context={
                           "task_id": task_id,
                           "component": component_type.value,
                           "action": action.value,
                           "requested_by": requested_by
                       })

        return task_id

    def get_remediation_status(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Get remediation status"""
        if task_id:
            # Get specific task
            task = self.active_remediations.get(task_id) or self.remediation_history.get(task_id)
            if task:
                return {
                    "task_id": task.task_id,
                    "component_type": task.component_type.value,
                    "action": task.action.value,
                    "status": task.status.value,
                    "description": task.description,
                    "created_at": task.created_at.isoformat(),
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "error_message": task.error_message,
                    "retry_count": task.retry_count,
                    "metadata": task.metadata
                }
            return {"error": "Task not found"}

        # Get all active remediations
        with self._remediation_lock:
            active_tasks = [
                {
                    "task_id": task.task_id,
                    "component_type": task.component_type.value,
                    "action": task.action.value,
                    "status": task.status.value,
                    "created_at": task.created_at.isoformat(),
                    "description": task.description
                }
                for task in self.active_remediations.values()
            ]

        return {
            "active_remediations": active_tasks,
            "total_active": len(active_tasks),
            "total_history": len(self.remediation_history)
        }

    def get_recovery_summary(self) -> Dict[str, Any]:
        """Get recovery system summary"""
        with self._remediation_lock:
            # Calculate statistics
            all_tasks = list(self.active_remediations.values()) + list(self.remediation_history.values())

            if not all_tasks:
                return {"message": "No recovery tasks found"}

            successful_tasks = [t for t in all_tasks if t.status == RemediationStatus.SUCCESS]
            failed_tasks = [t for t in all_tasks if t.status == RemediationStatus.FAILED]

            # Group by component
            component_stats = defaultdict(lambda: {"total": 0, "successful": 0, "failed": 0})
            for task in all_tasks:
                comp = task.component_type.value
                component_stats[comp]["total"] += 1
                if task.status == RemediationStatus.SUCCESS:
                    component_stats[comp]["successful"] += 1
                elif task.status == RemediationStatus.FAILED:
                    component_stats[comp]["failed"] += 1

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "recovery_system_active": self._running,
                "total_recovery_tasks": len(all_tasks),
                "successful_tasks": len(successful_tasks),
                "failed_tasks": len(failed_tasks),
                "success_rate": len(successful_tasks) / len(all_tasks) if all_tasks else 0,
                "active_remediations": len(self.active_remediations),
                "component_statistics": dict(component_stats),
                "self_healing_enabled": self.self_healing_config.enabled,
                "max_concurrent_remediations": self.self_healing_config.max_concurrent_remediations
            }

    def graceful_degradation_check(self, component_type: UploadComponentType) -> Dict[str, Any]:
        """Check if graceful degradation should be activated for a component"""
        if not self.self_healing_config.enable_graceful_degradation:
            return {"degradation_recommended": False, "reason": "Graceful degradation disabled"}

        # Check if component has been failing consistently
        if component_type in self.component_recovery_history:
            recent_failures = [
                failure_time for failure_time in self.component_recovery_history[component_type]
                if failure_time > datetime.utcnow() - timedelta(hours=1)
            ]

            if len(recent_failures) >= 3:
                return {
                    "degradation_recommended": True,
                    "reason": f"Component has {len(recent_failures)} failures in the last hour",
                    "failure_count": len(recent_failures),
                    "recommendations": self._get_degradation_recommendations(component_type)
                }

        return {"degradation_recommended": False, "reason": "Component is functioning normally"}

    def _get_degradation_recommendations(self, component_type: UploadComponentType) -> List[str]:
        """Get recommendations for graceful degradation"""
        recommendations = []

        if component_type == UploadComponentType.FFMPEG:
            recommendations.extend([
                "Use alternative video processing methods",
                "Reduce video quality settings",
                "Skip video effects processing",
                "Use pre-processed video templates"
            ])
        elif component_type == UploadComponentType.B2_STORAGE:
            recommendations.extend([
                "Use local storage as fallback",
                "Implement upload retry with exponential backoff",
                "Queue uploads for later processing",
                "Compress files before upload"
            ])
        elif component_type == UploadComponentType.NETWORK:
            recommendations.extend([
                "Implement offline processing mode",
                "Use cached results when possible",
                "Reduce external API calls",
                "Implement request batching"
            ])
        elif component_type == UploadComponentType.DISK_STORAGE:
            recommendations.extend([
                "Process files in smaller batches",
                "Use temporary cloud storage",
                "Implement file cleanup routines",
                "Compress temporary files"
            ])

        return recommendations

    def enable_graceful_degradation(self, component_type: UploadComponentType,
                                  duration_hours: int = 1) -> bool:
        """Enable graceful degradation for a component"""
        try:
            # This would set system-wide flags for graceful degradation
            # For now, simulate the action

            self.logger.warning("Graceful degradation enabled",
                              context={
                                  "component": component_type.value,
                                  "duration_hours": duration_hours
                              })

            return True

        except Exception as e:
            self.logger.error("Failed to enable graceful degradation",
                            context={"component": component_type.value, "error": str(e)})
            return False

    def disable_graceful_degradation(self, component_type: UploadComponentType) -> bool:
        """Disable graceful degradation for a component"""
        try:
            # This would clear system-wide graceful degradation flags
            # For now, simulate the action

            self.logger.info("Graceful degradation disabled",
                           context={"component": component_type.value})

            return True

        except Exception as e:
            self.logger.error("Failed to disable graceful degradation",
                            context={"component": component_type.value, "error": str(e)})
            return False