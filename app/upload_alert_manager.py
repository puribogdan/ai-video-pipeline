# app/upload_alert_manager.py
import asyncio
import json
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple, Set
import threading
import redis
import logging

from .monitoring import StructuredLogger, MetricsCollector, Alert
from .error_recovery import ErrorRecoveryManager, ErrorContext, ErrorCategory, ErrorSeverity
from .upload_health_checker import UploadHealthChecker, UploadComponentType, UploadComponentStatus


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AlertChannel(Enum):
    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"


@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    description: str
    severity: AlertSeverity
    condition_func: Callable
    channels: List[AlertChannel]
    cooldown_seconds: int = 300  # 5 minutes default cooldown
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UploadAlert:
    """Upload-specific alert"""
    alert_id: str
    rule_id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    component_type: Optional[UploadComponentType] = None
    error_context: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalation_level: int = 0
    channels_notified: List[AlertChannel] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertCorrelation:
    """Correlated alerts group"""
    correlation_id: str
    alert_ids: Set[str]
    root_cause_alert_id: Optional[str]
    pattern: str
    first_seen: datetime
    last_seen: datetime
    count: int
    status: str = "active"


class AlertEscalationPolicy:
    """Escalation policy for alerts"""

    def __init__(self, name: str):
        self.name = name
        self.escalation_levels = [
            {"delay_seconds": 0, "channels": [AlertChannel.LOG, AlertChannel.DASHBOARD]},
            {"delay_seconds": 300, "channels": [AlertChannel.EMAIL]},  # 5 minutes
            {"delay_seconds": 900, "channels": [AlertChannel.WEBHOOK]},  # 15 minutes
            {"delay_seconds": 1800, "channels": [AlertChannel.EMAIL, AlertChannel.WEBHOOK]}  # 30 minutes
        ]

    def get_escalation_for_level(self, level: int) -> Dict[str, Any]:
        """Get escalation configuration for a specific level"""
        if 0 <= level < len(self.escalation_levels):
            return self.escalation_levels[level]
        return self.escalation_levels[-1]  # Return highest level if out of bounds


class UploadAlertManager:
    """Advanced alert management system for upload operations"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = redis.from_url(redis_url)
        self.logger = StructuredLogger("upload_alert_manager")

        # Alert storage
        self.active_alerts: Dict[str, UploadAlert] = {}
        self.alert_history: Dict[str, UploadAlert] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.correlations: Dict[str, AlertCorrelation] = {}

        # Alert tracking
        self.alert_counts: Dict[str, int] = defaultdict(int)
        self.last_alert_times: Dict[str, datetime] = {}

        # Configuration
        self.max_alerts_per_component = 100
        self.alert_history_retention = 1000
        self.correlation_window = 300  # 5 minutes
        self.cooldown_periods: Dict[str, datetime] = {}

        # Locks for thread safety
        self._alert_lock = threading.Lock()
        self._rule_lock = threading.Lock()
        self._correlation_lock = threading.Lock()

        # Background tasks
        self._alert_processing_task = None
        self._escalation_task = None
        self._correlation_task = None
        self._cleanup_task = None
        self._running = False

        # Escalation policies
        self.escalation_policies = {
            AlertSeverity.INFO: AlertEscalationPolicy("info_escalation"),
            AlertSeverity.WARNING: AlertEscalationPolicy("warning_escalation"),
            AlertSeverity.ERROR: AlertEscalationPolicy("error_escalation"),
            AlertSeverity.CRITICAL: AlertEscalationPolicy("critical_escalation")
        }

        # Error recovery integration
        self.error_recovery = None
        try:
            self.error_recovery = ErrorRecoveryManager(redis_url)
        except Exception as e:
            self.logger.warning("Error recovery manager not available", context={"error": str(e)})

        # Setup default alert rules
        self._setup_default_alert_rules()

    def _setup_default_alert_rules(self):
        """Setup default alert rules for upload operations"""

        # Component health degradation rule
        def component_health_check(metrics_collector: MetricsCollector) -> Tuple[bool, str, Dict[str, Any]]:
            """Check for component health degradation"""
            # This would integrate with the health checker
            # For now, return a placeholder
            return False, "", {}

        # Upload failure rate rule
        def upload_failure_rate_check(metrics_collector: MetricsCollector) -> Tuple[bool, str, Dict[str, Any]]:
            """Check for high upload failure rate"""
            recent_failures = metrics_collector.get_metrics("upload_failures_total", limit=10)
            if len(recent_failures) >= 5:
                failure_rate = len(recent_failures) / 10.0
                return True, f"High upload failure rate: {failure_rate:.1%}", {"failure_rate": failure_rate}
            return False, "", {}

        # Performance degradation rule
        def performance_degradation_check(metrics_collector: MetricsCollector) -> Tuple[bool, str, Dict[str, Any]]:
            """Check for upload performance degradation"""
            recent_times = metrics_collector.get_metrics("upload_processing_time", limit=10)
            if recent_times:
                avg_time = sum(m.value for m in recent_times) / len(recent_times)
                if avg_time > 300:  # 5 minutes average
                    return True, f"Slow upload processing: {avg_time:.1f}s average", {"avg_time": avg_time}
            return False, "", {}

        # Disk space critical rule
        def disk_space_critical_check(metrics_collector: MetricsCollector) -> Tuple[bool, str, Dict[str, Any]]:
            """Check for critical disk space"""
            recent_disk = metrics_collector.get_metrics("disk_free_gb", limit=5)
            if recent_disk:
                min_free = min(m.value for m in recent_disk)
                if min_free < 1.0:  # Less than 1GB free
                    return True, f"Critical disk space: {min_free:.1f}GB free", {"free_gb": min_free}
            return False, "", {}

        # Memory pressure rule
        def memory_pressure_check(metrics_collector: MetricsCollector) -> Tuple[bool, str, Dict[str, Any]]:
            """Check for memory pressure"""
            recent_memory = metrics_collector.get_metrics("memory_usage_percent", limit=5)
            if recent_memory:
                max_memory = max(m.value for m in recent_memory)
                if max_memory > 90:  # Over 90% memory usage
                    return True, f"High memory usage: {max_memory:.1f}%", {"memory_percent": max_memory}
            return False, "", {}

        # Network issues rule
        def network_issues_check(metrics_collector: MetricsCollector) -> Tuple[bool, str, Dict[str, Any]]:
            """Check for network connectivity issues"""
            recent_connectivity = metrics_collector.get_metrics("network_connectivity_failures", limit=5)
            if len(recent_connectivity) >= 3:
                return True, "Multiple network connectivity failures", {"failure_count": len(recent_connectivity)}
            return False, "", {}

        # Add default rules
        default_rules = [
            AlertRule(
                rule_id="component_health_degraded",
                name="Component Health Degraded",
                description="Upload component health has degraded",
                severity=AlertSeverity.WARNING,
                condition_func=component_health_check,
                channels=[AlertChannel.LOG, AlertChannel.DASHBOARD],
                cooldown_seconds=300
            ),
            AlertRule(
                rule_id="high_upload_failure_rate",
                name="High Upload Failure Rate",
                description="Upload failure rate exceeds threshold",
                severity=AlertSeverity.ERROR,
                condition_func=upload_failure_rate_check,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.DASHBOARD],
                cooldown_seconds=600
            ),
            AlertRule(
                rule_id="slow_upload_processing",
                name="Slow Upload Processing",
                description="Upload processing time exceeds normal range",
                severity=AlertSeverity.WARNING,
                condition_func=performance_degradation_check,
                channels=[AlertChannel.LOG, AlertChannel.DASHBOARD],
                cooldown_seconds=900
            ),
            AlertRule(
                rule_id="critical_disk_space",
                name="Critical Disk Space",
                description="Available disk space is critically low",
                severity=AlertSeverity.CRITICAL,
                condition_func=disk_space_critical_check,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.WEBHOOK],
                cooldown_seconds=1800
            ),
            AlertRule(
                rule_id="memory_pressure",
                name="Memory Pressure",
                description="System memory usage is high",
                severity=AlertSeverity.WARNING,
                condition_func=memory_pressure_check,
                channels=[AlertChannel.LOG, AlertChannel.DASHBOARD],
                cooldown_seconds=300
            ),
            AlertRule(
                rule_id="network_issues",
                name="Network Issues",
                description="Network connectivity problems detected",
                severity=AlertSeverity.ERROR,
                condition_func=network_issues_check,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.DASHBOARD],
                cooldown_seconds=600
            )
        ]

        for rule in default_rules:
            self.add_alert_rule(rule)

    async def start_alert_management(self):
        """Start the alert management system"""
        if self._running:
            return

        self._running = True
        self.logger.info("Starting upload alert management")

        # Start background tasks
        self._alert_processing_task = asyncio.create_task(self._alert_processing_loop())
        self._escalation_task = asyncio.create_task(self._escalation_loop())
        self._correlation_task = asyncio.create_task(self._correlation_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_alert_management(self):
        """Stop the alert management system"""
        if not self._running:
            return

        self._running = False
        self.logger.info("Stopping upload alert management")

        # Stop background tasks
        tasks = [self._alert_processing_task, self._escalation_task,
                self._correlation_task, self._cleanup_task]

        for task in tasks:
            if task:
                task.cancel()

    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        with self._rule_lock:
            self.alert_rules[rule.rule_id] = rule
            self.logger.info("Alert rule added",
                           context={"rule_id": rule.rule_id, "name": rule.name})

    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule"""
        with self._rule_lock:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                self.logger.info("Alert rule removed", context={"rule_id": rule_id})
                return True
            return False

    def create_alert(self, rule_id: str, title: str, message: str,
                    severity: AlertSeverity, component_type: Optional[UploadComponentType] = None,
                    error_context: Optional[Dict[str, Any]] = None,
                    correlation_id: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Optional[UploadAlert]:
        """Create a new alert"""

        # Check cooldown period
        if self._is_in_cooldown(rule_id):
            self.logger.debug("Alert suppressed due to cooldown",
                            context={"rule_id": rule_id, "title": title})
            return None

        # Check if we should suppress due to correlation
        if self._should_suppress_alert(rule_id, component_type, error_context):
            self.logger.debug("Alert suppressed due to correlation",
                            context={"rule_id": rule_id, "title": title})
            return None

        # Create alert
        alert_id = str(uuid.uuid4())
        alert = UploadAlert(
            alert_id=alert_id,
            rule_id=rule_id,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.utcnow(),
            component_type=component_type,
            error_context=error_context,
            correlation_id=correlation_id,
            metadata=metadata or {}
        )

        # Store alert
        with self._alert_lock:
            self.active_alerts[alert_id] = alert
            self.alert_counts[rule_id] += 1

            # Limit active alerts per component
            if component_type:
                component_alerts = [a for a in self.active_alerts.values()
                                  if a.component_type == component_type]
                if len(component_alerts) > self.max_alerts_per_component:
                    # Remove oldest alert for this component
                    oldest_alert = min(component_alerts, key=lambda a: a.timestamp)
                    del self.active_alerts[oldest_alert.alert_id]

        # Update cooldown
        self.cooldown_periods[rule_id] = datetime.utcnow() + timedelta(seconds=self.alert_rules[rule_id].cooldown_seconds)

        self.logger.info("Alert created",
                        context={
                            "alert_id": alert_id,
                            "rule_id": rule_id,
                            "severity": severity.value,
                            "title": title
                        })

        return alert

    def _is_in_cooldown(self, rule_id: str) -> bool:
        """Check if an alert rule is in cooldown period"""
        if rule_id in self.cooldown_periods:
            return datetime.utcnow() < self.cooldown_periods[rule_id]
        return False

    def _should_suppress_alert(self, rule_id: str, component_type: Optional[UploadComponentType],
                              error_context: Optional[Dict[str, Any]]) -> bool:
        """Check if alert should be suppressed due to correlation or other factors"""
        # Check for similar recent alerts
        current_time = datetime.utcnow()
        threshold_time = current_time - timedelta(seconds=self.correlation_window)

        with self._alert_lock:
            similar_alerts = [
                alert for alert in self.active_alerts.values()
                if (alert.rule_id == rule_id and
                    alert.component_type == component_type and
                    alert.timestamp > threshold_time)
            ]

            # If we have too many similar alerts recently, suppress new ones
            if len(similar_alerts) >= 5:
                return True

        return False

    async def _alert_processing_loop(self):
        """Main alert processing loop"""
        while self._running:
            try:
                # Process pending alerts
                await self._process_pending_alerts()

                # Check alert rules
                await self._check_alert_rules()

            except Exception as e:
                self.logger.error("Error in alert processing loop",
                                context={"error": str(e)})

            await asyncio.sleep(30)  # Check every 30 seconds

    async def _process_pending_alerts(self):
        """Process pending alerts (send notifications, etc.)"""
        with self._alert_lock:
            alerts_to_process = list(self.active_alerts.values())

        for alert in alerts_to_process:
            try:
                # Send notifications based on escalation level
                await self._send_alert_notifications(alert)

                # Update alert metadata
                alert.metadata["processed_at"] = datetime.utcnow().isoformat()

            except Exception as e:
                self.logger.error("Error processing alert",
                                context={"alert_id": alert.alert_id, "error": str(e)})

    async def _send_alert_notifications(self, alert: UploadAlert):
        """Send alert notifications via configured channels"""
        rule = self.alert_rules.get(alert.rule_id)
        if not rule:
            return

        # Get escalation configuration
        escalation_config = self.escalation_policies[alert.severity].get_escalation_for_level(alert.escalation_level)

        for channel in escalation_config["channels"]:
            if channel not in alert.channels_notified:
                try:
                    await self._send_notification(alert, channel)
                    alert.channels_notified.append(channel)

                    self.logger.debug("Alert notification sent",
                                    context={
                                        "alert_id": alert.alert_id,
                                        "channel": channel.value,
                                        "escalation_level": alert.escalation_level
                                    })

                except Exception as e:
                    self.logger.error("Failed to send alert notification",
                                    context={
                                        "alert_id": alert.alert_id,
                                        "channel": channel.value,
                                        "error": str(e)
                                    })

    async def _send_notification(self, alert: UploadAlert, channel: AlertChannel):
        """Send notification via specific channel"""
        if channel == AlertChannel.LOG:
            await self._send_log_notification(alert)
        elif channel == AlertChannel.EMAIL:
            await self._send_email_notification(alert)
        elif channel == AlertChannel.WEBHOOK:
            await self._send_webhook_notification(alert)
        elif channel == AlertChannel.DASHBOARD:
            await self._send_dashboard_notification(alert)

    async def _send_log_notification(self, alert: UploadAlert):
        """Send alert to logging system"""
        self.logger.log(
            level=self._severity_to_log_level(alert.severity),
            f"Upload Alert: {alert.title}",
            context={
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "component": alert.component_type.value if alert.component_type else "unknown",
                "message": alert.message
            }
        )

    async def _send_email_notification(self, alert: UploadAlert):
        """Send alert via email"""
        # This would integrate with an email service
        # For now, just log the intent
        self.logger.info("Email notification would be sent",
                        context={
                            "alert_id": alert.alert_id,
                            "title": alert.title,
                            "recipient": "admin@example.com"  # Would be configurable
                        })

    async def _send_webhook_notification(self, alert: UploadAlert):
        """Send alert via webhook"""
        # This would send HTTP POST to configured webhook URLs
        # For now, just log the intent
        self.logger.info("Webhook notification would be sent",
                        context={
                            "alert_id": alert.alert_id,
                            "title": alert.title,
                            "webhook_url": "https://example.com/webhook"  # Would be configurable
                        })

    async def _send_dashboard_notification(self, alert: UploadAlert):
        """Send alert to dashboard WebSocket clients"""
        # This would integrate with the WebSocket manager
        # For now, just log the intent
        self.logger.info("Dashboard notification would be sent",
                        context={"alert_id": alert.alert_id, "title": alert.title})

    def _severity_to_log_level(self, severity: AlertSeverity) -> int:
        """Convert alert severity to logging level"""
        mapping = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }
        return mapping.get(severity, logging.INFO)

    async def _check_alert_rules(self):
        """Check all alert rules against current system state"""
        # This would integrate with metrics collector and health checker
        # For now, create a placeholder implementation
        pass

    async def _escalation_loop(self):
        """Handle alert escalation"""
        while self._running:
            try:
                current_time = datetime.utcnow()

                with self._alert_lock:
                    alerts_to_escalate = [
                        alert for alert in self.active_alerts.values()
                        if (alert.status == AlertStatus.ACTIVE and
                            current_time - alert.timestamp >
                            timedelta(seconds=self._get_escalation_delay(alert.severity, alert.escalation_level)))
                    ]

                for alert in alerts_to_escalate:
                    # Escalate the alert
                    alert.escalation_level += 1

                    self.logger.info("Alert escalated",
                                   context={
                                       "alert_id": alert.alert_id,
                                       "new_level": alert.escalation_level,
                                       "severity": alert.severity.value
                                   })

                    # Re-process with new escalation level
                    await self._process_pending_alerts()

            except Exception as e:
                self.logger.error("Error in escalation loop",
                                context={"error": str(e)})

            await asyncio.sleep(60)  # Check every minute

    def _get_escalation_delay(self, severity: AlertSeverity, level: int) -> int:
        """Get delay before escalation for a given level"""
        escalation_config = self.escalation_policies[severity].get_escalation_for_level(level)
        return escalation_config.get("delay_seconds", 300)

    async def _correlation_loop(self):
        """Correlate related alerts to reduce noise"""
        while self._running:
            try:
                await self._correlate_alerts()
            except Exception as e:
                self.logger.error("Error in correlation loop",
                                context={"error": str(e)})

            await asyncio.sleep(120)  # Correlate every 2 minutes

    async def _correlate_alerts(self):
        """Find and correlate related alerts"""
        current_time = datetime.utcnow()
        window_start = current_time - timedelta(seconds=self.correlation_window)

        with self._alert_lock:
            # Get recent active alerts
            recent_alerts = [
                alert for alert in self.active_alerts.values()
                if alert.timestamp > window_start
            ]

        if len(recent_alerts) < 3:
            return  # Need at least 3 alerts to correlate

        # Group alerts by component and error patterns
        correlation_groups = self._group_alerts_for_correlation(recent_alerts)

        for group_key, alerts in correlation_groups.items():
            if len(alerts) >= 3:  # Minimum group size for correlation
                # Create correlation
                correlation_id = str(uuid.uuid4())
                correlation = AlertCorrelation(
                    correlation_id=correlation_id,
                    alert_ids={alert.alert_id for alert in alerts},
                    root_cause_alert_id=self._identify_root_cause(alerts),
                    pattern=group_key,
                    first_seen=min(alert.timestamp for alert in alerts),
                    last_seen=max(alert.timestamp for alert in alerts),
                    count=len(alerts)
                )

                with self._correlation_lock:
                    self.correlations[correlation_id] = correlation

                # Suppress non-root-cause alerts in the correlation group
                root_cause_id = correlation.root_cause_alert_id
                for alert in alerts:
                    if alert.alert_id != root_cause_id and alert.status == AlertStatus.ACTIVE:
                        alert.status = AlertStatus.SUPPRESSED
                        alert.metadata["suppressed_by_correlation"] = correlation_id

                self.logger.info("Alert correlation created",
                               context={
                                   "correlation_id": correlation_id,
                                   "alert_count": len(alerts),
                                   "pattern": group_key
                               })

    def _group_alerts_for_correlation(self, alerts: List[UploadAlert]) -> Dict[str, List[UploadAlert]]:
        """Group alerts for correlation analysis"""
        groups = defaultdict(list)

        for alert in alerts:
            # Group by component type
            if alert.component_type:
                key = f"component:{alert.component_type.value}"
                groups[key].append(alert)

            # Group by error context similarity
            if alert.error_context:
                error_type = alert.error_context.get("error_type", "unknown")
                key = f"error_type:{error_type}"
                groups[key].append(alert)

        return groups

    def _identify_root_cause(self, alerts: List[UploadAlert]) -> Optional[str]:
        """Identify the root cause alert in a correlation group"""
        if not alerts:
            return None

        # Simple heuristic: return the alert with highest severity first,
        # then earliest timestamp
        highest_severity = max(alerts, key=lambda a: self._severity_priority(a.severity))
        return highest_severity.alert_id

    def _severity_priority(self, severity: AlertSeverity) -> int:
        """Get priority value for alert severity (higher = more severe)"""
        priorities = {
            AlertSeverity.INFO: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.ERROR: 3,
            AlertSeverity.CRITICAL: 4
        }
        return priorities.get(severity, 0)

    async def _cleanup_loop(self):
        """Clean up old alerts and correlations"""
        while self._running:
            try:
                current_time = datetime.utcnow()
                retention_cutoff = current_time - timedelta(hours=24)  # 24 hours retention

                # Clean up old alerts
                with self._alert_lock:
                    old_alerts = [
                        alert_id for alert_id, alert in self.active_alerts.items()
                        if alert.timestamp < retention_cutoff and alert.status != AlertStatus.ACTIVE
                    ]

                    for alert_id in old_alerts:
                        alert = self.active_alerts.pop(alert_id)
                        self.alert_history[alert_id] = alert

                        # Limit history size
                        if len(self.alert_history) > self.alert_history_retention:
                            oldest_alert = min(self.alert_history.values(), key=lambda a: a.timestamp)
                            del self.alert_history[oldest_alert.alert_id]

                # Clean up old correlations
                with self._correlation_lock:
                    old_correlations = [
                        corr_id for corr_id, correlation in self.correlations.items()
                        if correlation.last_seen < retention_cutoff
                    ]

                    for corr_id in old_correlations:
                        del self.correlations[corr_id]

            except Exception as e:
                self.logger.error("Error in cleanup loop",
                                context={"error": str(e)})

            await asyncio.sleep(3600)  # Clean up every hour

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        with self._alert_lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.utcnow()

                self.logger.info("Alert acknowledged",
                               context={
                                   "alert_id": alert_id,
                                   "acknowledged_by": acknowledged_by
                               })
                return True
        return False

    def resolve_alert(self, alert_id: str, resolved_by: Optional[str] = None) -> bool:
        """Resolve an alert"""
        with self._alert_lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.utcnow()

                # Update metadata
                if resolved_by:
                    alert.metadata["resolved_by"] = resolved_by

                self.logger.info("Alert resolved",
                               context={"alert_id": alert_id, "resolved_by": resolved_by})
                return True
        return False

    def get_active_alerts(self, component_type: Optional[UploadComponentType] = None,
                         severity: Optional[AlertSeverity] = None) -> List[UploadAlert]:
        """Get active alerts with optional filtering"""
        with self._alert_lock:
            alerts = [alert for alert in self.active_alerts.values()
                     if alert.status == AlertStatus.ACTIVE]

            if component_type:
                alerts = [alert for alert in alerts if alert.component_type == component_type]

            if severity:
                alerts = [alert for alert in alerts if alert.severity == severity]

            return sorted(alerts, key=lambda a: (self._severity_priority(a.severity), a.timestamp), reverse=True)

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status"""
        with self._alert_lock:
            active_alerts = self.get_active_alerts()

            # Count by severity
            severity_counts = defaultdict(int)
            for alert in active_alerts:
                severity_counts[alert.severity.value] += 1

            # Count by component
            component_counts = defaultdict(int)
            for alert in active_alerts:
                if alert.component_type:
                    component_counts[alert.component_type.value] += 1

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "total_active_alerts": len(active_alerts),
                "alerts_by_severity": dict(severity_counts),
                "alerts_by_component": dict(component_counts),
                "escalated_alerts": len([a for a in active_alerts if a.escalation_level > 0]),
                "acknowledged_alerts": len([a for a in active_alerts if a.status == AlertStatus.ACKNOWLEDGED]),
                "suppressed_alerts": len([a for a in active_alerts if a.status == AlertStatus.SUPPRESSED]),
                "correlation_groups": len(self.correlations)
            }

    def get_alert_history(self, limit: int = 100) -> List[UploadAlert]:
        """Get recent alert history"""
        with self._alert_lock:
            alerts = list(self.alert_history.values())
            return sorted(alerts, key=lambda a: a.timestamp, reverse=True)[:limit]

    def trigger_upload_failure_alert(self, job_id: str, error: Exception,
                                   component_type: UploadComponentType,
                                   metadata: Optional[Dict[str, Any]] = None):
        """Trigger an alert for upload failure"""
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "job_id": job_id,
            "component": component_type.value
        }

        if metadata:
            error_context.update(metadata)

        return self.create_alert(
            rule_id="upload_failure",
            title=f"Upload Failed: {component_type.value}",
            message=f"Upload operation failed for job {job_id}: {str(error)}",
            severity=AlertSeverity.ERROR,
            component_type=component_type,
            error_context=error_context,
            metadata={"job_id": job_id, "operation": "upload"}
        )

    def trigger_performance_alert(self, component_type: UploadComponentType,
                                metric_name: str, current_value: float,
                                threshold_value: float):
        """Trigger an alert for performance issues"""
        return self.create_alert(
            rule_id="performance_degradation",
            title=f"Performance Issue: {component_type.value}",
            message=f"{metric_name} is {current_value:.2f}, exceeds threshold {threshold_value:.2f}",
            severity=AlertSeverity.WARNING,
            component_type=component_type,
            metadata={
                "metric_name": metric_name,
                "current_value": current_value,
                "threshold_value": threshold_value
            }
        )

    def trigger_health_check_alert(self, component_type: UploadComponentType,
                                 status: UploadComponentStatus, message: str):
        """Trigger an alert for health check failures"""
        severity = {
            UploadComponentStatus.HEALTHY: AlertSeverity.INFO,
            UploadComponentStatus.DEGRADED: AlertSeverity.WARNING,
            UploadComponentStatus.UNHEALTHY: AlertSeverity.ERROR,
            UploadComponentStatus.UNKNOWN: AlertSeverity.WARNING
        }.get(status, AlertSeverity.WARNING)

        return self.create_alert(
            rule_id="component_health_degraded",
            title=f"Health Check: {component_type.value}",
            message=f"Component {component_type.value} is {status.value}: {message}",
            severity=severity,
            component_type=component_type,
            metadata={"health_status": status.value, "health_message": message}
        )