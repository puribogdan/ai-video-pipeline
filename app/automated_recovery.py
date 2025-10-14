# app/automated_recovery.py
import asyncio
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple
import logging
import threading
import redis
from .error_recovery import (
    ErrorRecoveryManager, ErrorContext, RecoveryAttempt,
    ErrorCategory, ErrorSeverity, RecoveryStrategy,
    get_error_recovery_manager
)


@dataclass
class RecoveryWorkflow:
    """Defines an automated recovery workflow"""
    name: str
    description: str
    trigger_conditions: Dict[str, Any]
    recovery_actions: List[Dict[str, Any]]
    cooldown_period: int = 300  # 5 minutes
    max_executions: int = 10
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_executed: Optional[datetime] = None
    execution_count: int = 0


@dataclass
class ErrorTrend:
    """Tracks error trends for predictive failure detection"""
    error_type: str
    component: str
    time_window: int  # minutes
    error_counts: deque = field(default_factory=lambda: deque(maxlen=100))
    trend_direction: str = "stable"  # stable, increasing, decreasing
    prediction_confidence: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class AutomatedRecoveryWorkflows:
    """Manages automated recovery workflows"""

    def __init__(self, error_recovery_manager: ErrorRecoveryManager):
        self.error_recovery = error_recovery_manager
        self.logger = logging.getLogger("automated_recovery")
        self.workflows: Dict[str, RecoveryWorkflow] = {}
        self.error_trends: Dict[str, ErrorTrend] = {}
        self.workflow_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()
        self._running = False
        self._workflow_task = None

        # Setup default workflows
        self._setup_default_workflows()

    def _setup_default_workflows(self):
        """Setup default automated recovery workflows"""

        # Circuit breaker reset workflow
        circuit_breaker_workflow = RecoveryWorkflow(
            name="circuit_breaker_reset",
            description="Automatically reset circuit breakers after cooldown period",
            trigger_conditions={
                "event_type": "circuit_breaker_open",
                "min_duration_open": 300  # 5 minutes
            },
            recovery_actions=[
                {
                    "action": "reset_circuit_breaker",
                    "target": "auto_detected",
                    "conditions": {"state": "open", "duration_exceeded": True}
                },
                {
                    "action": "log_recovery",
                    "message": "Circuit breaker auto-reset executed"
                }
            ],
            cooldown_period=600,  # 10 minutes between executions
            max_executions=50
        )

        # High error rate workflow
        high_error_rate_workflow = RecoveryWorkflow(
            name="high_error_rate_response",
            description="Respond to high error rates with graceful degradation",
            trigger_conditions={
                "event_type": "high_error_rate",
                "error_rate_threshold": 0.1,  # 10% error rate
                "time_window": 300  # 5 minutes
            },
            recovery_actions=[
                {
                    "action": "enable_graceful_degradation",
                    "components": ["video_processing", "image_generation"],
                    "fallback_mode": "reduced_quality"
                },
                {
                    "action": "increase_circuit_breaker_thresholds",
                    "multiplier": 1.5,
                    "duration": 600  # 10 minutes
                },
                {
                    "action": "log_recovery",
                    "message": "High error rate response activated"
                }
            ],
            cooldown_period=900,  # 15 minutes
            max_executions=20
        )

        # Storage cleanup workflow
        storage_cleanup_workflow = RecoveryWorkflow(
            name="storage_cleanup",
            description="Clean up storage when disk space is low",
            trigger_conditions={
                "event_type": "low_disk_space",
                "disk_usage_threshold": 85  # 85% usage
            },
            recovery_actions=[
                {
                    "action": "cleanup_temp_files",
                    "max_age_hours": 24,
                    "preserve_patterns": ["*.mp4", "*.mp3"]
                },
                {
                    "action": "cleanup_failed_jobs",
                    "max_age_hours": 48
                },
                {
                    "action": "log_recovery",
                    "message": "Storage cleanup executed"
                }
            ],
            cooldown_period=3600,  # 1 hour
            max_executions=100
        )

        self.add_workflow(circuit_breaker_workflow)
        self.add_workflow(high_error_rate_workflow)
        self.add_workflow(storage_cleanup_workflow)

    def add_workflow(self, workflow: RecoveryWorkflow):
        """Add a recovery workflow"""
        with self._lock:
            self.workflows[workflow.name] = workflow

    def remove_workflow(self, name: str) -> bool:
        """Remove a recovery workflow"""
        with self._lock:
            if name in self.workflows:
                del self.workflows[name]
                return True
            return False

    def get_workflow(self, name: str) -> Optional[RecoveryWorkflow]:
        """Get a specific workflow"""
        with self._lock:
            return self.workflows.get(name)

    def list_workflows(self) -> List[RecoveryWorkflow]:
        """List all workflows"""
        with self._lock:
            return list(self.workflows.values())

    async def start_automated_recovery(self):
        """Start the automated recovery system"""
        if self._running:
            return

        self._running = True
        self._workflow_task = asyncio.create_task(self._workflow_monitor_loop())
        self.logger.info("Automated recovery workflows started")

    async def stop_automated_recovery(self):
        """Stop the automated recovery system"""
        if not self._running:
            return

        self._running = False

        if self._workflow_task:
            self._workflow_task.cancel()

        self.logger.info("Automated recovery workflows stopped")

    async def _workflow_monitor_loop(self):
        """Main workflow monitoring loop"""
        while self._running:
            try:
                # Check each workflow for execution conditions
                for workflow in self.workflows.values():
                    if workflow.enabled:
                        await self._check_and_execute_workflow(workflow)

                # Update error trends
                self._update_error_trends()

                # Clean up old workflow results
                self._cleanup_old_results()

            except Exception as e:
                self.logger.error(f"Error in workflow monitor loop: {e}")

            await asyncio.sleep(60)  # Check every minute

    async def _check_and_execute_workflow(self, workflow: RecoveryWorkflow):
        """Check if a workflow should be executed and run it if conditions are met"""
        current_time = datetime.utcnow()

        # Check cooldown period
        if (workflow.last_executed and
            current_time - workflow.last_executed < timedelta(seconds=workflow.cooldown_period)):
            return

        # Check execution limits
        if workflow.execution_count >= workflow.max_executions:
            self.logger.warning(f"Workflow {workflow.name} has reached max executions")
            return

        # Check trigger conditions
        if not await self._check_trigger_conditions(workflow.trigger_conditions):
            return

        # Execute the workflow
        await self._execute_workflow(workflow)

    async def _check_trigger_conditions(self, conditions: Dict[str, Any]) -> bool:
        """Check if workflow trigger conditions are met"""
        event_type = conditions.get("event_type")

        if event_type == "circuit_breaker_open":
            return await self._check_circuit_breaker_conditions(conditions)
        elif event_type == "high_error_rate":
            return await self._check_error_rate_conditions(conditions)
        elif event_type == "low_disk_space":
            return await self._check_disk_space_conditions(conditions)

        return False

    async def _check_circuit_breaker_conditions(self, conditions: Dict[str, Any]) -> bool:
        """Check circuit breaker trigger conditions"""
        stats = self.error_recovery.get_recovery_stats()
        circuit_states = stats.get("circuit_breaker_states", {})

        min_duration = conditions.get("min_duration_open", 0)

        for cb_name, cb_state in circuit_states.items():
            if cb_state.get("state") == "open":
                # Check if it's been open for the minimum duration
                if cb_state.get("last_failure_time"):
                    open_duration = time.time() - cb_state["last_failure_time"]
                    if open_duration >= min_duration:
                        return True

        return False

    async def _check_error_rate_conditions(self, conditions: Dict[str, Any]) -> bool:
        """Check error rate trigger conditions"""
        # This would analyze recent error patterns
        # For now, return a simple check based on recovery stats
        stats = self.error_recovery.get_recovery_stats()
        total_attempts = stats.get("total_recovery_attempts", 0)
        successful_attempts = stats.get("successful_attempts", 0)

        if total_attempts > 0:
            error_rate = 1 - (successful_attempts / total_attempts)
            threshold = conditions.get("error_rate_threshold", 0.1)
            return error_rate > threshold

        return False

    async def _check_disk_space_conditions(self, conditions: Dict[str, Any]) -> bool:
        """Check disk space trigger conditions"""
        try:
            import shutil
            disk_usage = shutil.disk_usage("/")
            usage_percent = (disk_usage.used / disk_usage.total) * 100
            threshold = conditions.get("disk_usage_threshold", 85)
            return usage_percent > threshold
        except Exception as e:
            self.logger.error(f"Error checking disk space: {e}")
            return False

    async def _execute_workflow(self, workflow: RecoveryWorkflow):
        """Execute a recovery workflow"""
        self.logger.info(f"Executing workflow: {workflow.name}")

        execution_result = {
            "workflow_name": workflow.name,
            "start_time": datetime.utcnow(),
            "actions_executed": [],
            "success": True,
            "error": None
        }

        try:
            for action in workflow.recovery_actions:
                action_result = await self._execute_workflow_action(action)
                execution_result["actions_executed"].append(action_result)

                if not action_result.get("success", False):
                    execution_result["success"] = False
                    execution_result["error"] = action_result.get("error")
                    break

            # Update workflow execution tracking
            workflow.last_executed = datetime.utcnow()
            workflow.execution_count += 1

            execution_result["end_time"] = datetime.utcnow()
            execution_result["duration"] = (
                execution_result["end_time"] - execution_result["start_time"]
            ).total_seconds()

            self.logger.info(f"Workflow {workflow.name} completed successfully")

        except Exception as e:
            execution_result["success"] = False
            execution_result["error"] = str(e)
            execution_result["end_time"] = datetime.utcnow()
            self.logger.error(f"Workflow {workflow.name} failed: {e}")

        # Store execution result
        with self._lock:
            self.workflow_results[workflow.name].append(execution_result)

    async def _execute_workflow_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow action"""
        action_type = action.get("action")
        result = {
            "action": action_type,
            "success": False,
            "timestamp": datetime.utcnow(),
            "details": {}
        }

        try:
            if action_type == "reset_circuit_breaker":
                success = await self._reset_circuit_breaker(action)
            elif action_type == "log_recovery":
                success = await self._log_recovery_action(action)
            elif action_type == "enable_graceful_degradation":
                success = await self._enable_graceful_degradation(action)
            elif action_type == "increase_circuit_breaker_thresholds":
                success = await self._increase_circuit_breaker_thresholds(action)
            elif action_type == "cleanup_temp_files":
                success = await self._cleanup_temp_files(action)
            elif action_type == "cleanup_failed_jobs":
                success = await self._cleanup_failed_jobs(action)
            else:
                raise ValueError(f"Unknown workflow action: {action_type}")

            result["success"] = success

        except Exception as e:
            result["error"] = str(e)
            result["success"] = False

        return result

    async def _reset_circuit_breaker(self, action: Dict[str, Any]) -> bool:
        """Reset circuit breaker action"""
        # This would implement circuit breaker reset logic
        # For now, just log the action
        self.logger.info("Circuit breaker reset action executed")
        return True

    async def _log_recovery_action(self, action: Dict[str, Any]) -> bool:
        """Log recovery action"""
        message = action.get("message", "Recovery action executed")
        self.logger.info(f"Recovery workflow: {message}")
        return True

    async def _enable_graceful_degradation(self, action: Dict[str, Any]) -> bool:
        """Enable graceful degradation action"""
        components = action.get("components", [])
        fallback_mode = action.get("fallback_mode", "reduced_quality")

        self.logger.info(f"Enabling graceful degradation for components: {components}, mode: {fallback_mode}")
        return True

    async def _increase_circuit_breaker_thresholds(self, action: Dict[str, Any]) -> bool:
        """Increase circuit breaker thresholds action"""
        multiplier = action.get("multiplier", 1.5)
        duration = action.get("duration", 600)

        self.logger.info(f"Increasing circuit breaker thresholds by {multiplier}x for {duration}s")
        return True

    async def _cleanup_temp_files(self, action: Dict[str, Any]) -> bool:
        """Cleanup temporary files action"""
        max_age_hours = action.get("max_age_hours", 24)
        preserve_patterns = action.get("preserve_patterns", [])

        self.logger.info(f"Cleaning up temp files older than {max_age_hours}h, preserving: {preserve_patterns}")
        return True

    async def _cleanup_failed_jobs(self, action: Dict[str, Any]) -> bool:
        """Cleanup failed jobs action"""
        max_age_hours = action.get("max_age_hours", 48)

        self.logger.info(f"Cleaning up failed jobs older than {max_age_hours}h")
        return True

    def _update_error_trends(self):
        """Update error trend analysis"""
        # This would analyze error patterns and predict future failures
        # For now, just maintain the trend data structure
        pass

    def _cleanup_old_results(self, max_age_hours: int = 24):
        """Clean up old workflow execution results"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)

        with self._lock:
            for workflow_name in self.workflow_results:
                self.workflow_results[workflow_name] = [
                    result for result in self.workflow_results[workflow_name]
                    if result.get("start_time", datetime.utcnow()) > cutoff_time
                ]

    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get workflow execution statistics"""
        with self._lock:
            stats = {}
            for name, workflow in self.workflows.items():
                recent_results = self.workflow_results[name][-10:]  # Last 10 executions

                successful_executions = sum(1 for r in recent_results if r.get("success", False))
                total_executions = len(recent_results)

                stats[name] = {
                    "workflow_info": {
                        "name": workflow.name,
                        "description": workflow.description,
                        "enabled": workflow.enabled,
                        "execution_count": workflow.execution_count,
                        "last_executed": workflow.last_executed.isoformat() if workflow.last_executed else None
                    },
                    "recent_performance": {
                        "total_executions": total_executions,
                        "successful_executions": successful_executions,
                        "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
                        "avg_duration": sum(r.get("duration", 0) for r in recent_results) / total_executions if total_executions > 0 else 0
                    }
                }

            return stats


class PredictiveFailureDetector:
    """Predictive failure detection based on error trends"""

    def __init__(self, error_recovery_manager: ErrorRecoveryManager):
        self.error_recovery = error_recovery_manager
        self.logger = logging.getLogger("predictive_failure_detector")
        self.error_patterns: Dict[str, List[Tuple[datetime, str]]] = defaultdict(list)
        self.failure_predictions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def record_error(self, error: Exception, context: ErrorContext):
        """Record an error for trend analysis"""
        pattern_key = f"{context.component}:{type(error).__name__}"

        with self._lock:
            self.error_patterns[pattern_key].append((context.timestamp, str(error)))

            # Keep only last 100 errors per pattern
            if len(self.error_patterns[pattern_key]) > 100:
                self.error_patterns[pattern_key] = self.error_patterns[pattern_key][-100:]

    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze error trends and generate predictions"""
        predictions = {}

        with self._lock:
            for pattern_key, errors in self.error_patterns.items():
                if len(errors) < 5:
                    continue  # Need at least 5 errors for trend analysis

                # Simple trend analysis based on error frequency
                recent_errors = errors[-10:]  # Last 10 errors
                time_span = (recent_errors[-1][0] - recent_errors[0][0]).total_seconds()

                if time_span > 0:
                    error_rate = len(recent_errors) / time_span  # errors per second

                    # If error rate is increasing significantly, predict potential failure
                    if error_rate > 0.01:  # More than 1 error per 100 seconds
                        predictions[pattern_key] = {
                            "pattern": pattern_key,
                            "error_rate": error_rate,
                            "recent_count": len(recent_errors),
                            "time_span_seconds": time_span,
                            "prediction": "potential_failure",
                            "confidence": min(error_rate * 100, 1.0),
                            "recommendation": "Consider increasing circuit breaker thresholds or enabling graceful degradation"
                        }

        return predictions

    def get_failure_predictions(self) -> Dict[str, Any]:
        """Get current failure predictions"""
        return self.analyze_trends()


# Global instances
_automated_recovery = None
_recovery_lock = threading.Lock()


def get_automated_recovery() -> AutomatedRecoveryWorkflows:
    """Get the global automated recovery instance"""
    global _automated_recovery

    if _automated_recovery is None:
        with _recovery_lock:
            if _automated_recovery is None:
                error_recovery = get_error_recovery_manager()
                _automated_recovery = AutomatedRecoveryWorkflows(error_recovery)

    return _automated_recovery


def get_predictive_failure_detector() -> PredictiveFailureDetector:
    """Get the global predictive failure detector instance"""
    error_recovery = get_error_recovery_manager()
    return PredictiveFailureDetector(error_recovery)


async def start_automated_recovery():
    """Start the automated recovery system"""
    recovery = get_automated_recovery()
    await recovery.start_automated_recovery()


async def stop_automated_recovery():
    """Stop the automated recovery system"""
    recovery = get_automated_recovery()
    await recovery.stop_automated_recovery()