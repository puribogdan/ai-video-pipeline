# app/upload_health_monitoring.py
import asyncio
import json
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import threading
import redis
import logging

from .monitoring import StructuredLogger, MetricsCollector
from .upload_health_checker import UploadHealthChecker, UploadComponentType, UploadComponentStatus
from .upload_alert_manager import UploadAlertManager, AlertSeverity, AlertStatus
from .upload_recovery_manager import UploadRecoveryManager, RecoveryAction, RemediationStatus
from .upload_analytics_manager import UploadAnalyticsManager, MetricType


class UploadHealthMonitoringSystem:
    """Comprehensive upload health monitoring and management system"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = redis.from_url(redis_url)
        self.logger = StructuredLogger("upload_health_monitoring")

        # Core components
        self.health_checker = None
        self.alert_manager = None
        self.recovery_manager = None
        self.analytics_manager = None

        # Initialize all components
        try:
            self.health_checker = UploadHealthChecker(redis_url)
            self.alert_manager = UploadAlertManager(redis_url)
            self.recovery_manager = UploadRecoveryManager(redis_url)
            self.analytics_manager = UploadAnalyticsManager(redis_url)
        except Exception as e:
            self.logger.error("Failed to initialize monitoring components",
                            context={"error": str(e)})

        # System state
        self._running = False
        self._start_time = None
        self._system_stats = {}

        # Integration tracking
        self._integration_lock = threading.Lock()

        # Prometheus metrics integration
        self._prometheus_metrics = None

    async def start_system(self):
        """Start the complete upload health monitoring system"""
        if self._running:
            return {"success": False, "error": "System already running"}

        self._running = True
        self._start_time = datetime.utcnow()

        self.logger.info("Starting upload health monitoring system")

        try:
            # Start all components
            if self.health_checker:
                await self.health_checker.start_health_monitoring()
            if self.alert_manager:
                await self.alert_manager.start_alert_management()
            if self.recovery_manager:
                await self.recovery_manager.start_recovery_management()
            if self.analytics_manager:
                await self.analytics_manager.start_analytics()

            # Start integration monitoring
            asyncio.create_task(self._integration_monitoring_loop())

            self.logger.info("Upload health monitoring system started successfully")
            return {"success": True, "started_at": self._start_time.isoformat()}

        except Exception as e:
            self.logger.error("Failed to start upload health monitoring system",
                            context={"error": str(e)})
            return {"success": False, "error": str(e)}

    async def stop_system(self):
        """Stop the complete upload health monitoring system"""
        if not self._running:
            return {"success": False, "error": "System not running"}

        self._running = False

        self.logger.info("Stopping upload health monitoring system")

        try:
            # Stop all components
            if self.health_checker:
                await self.health_checker.stop_health_monitoring()
            if self.alert_manager:
                await self.alert_manager.stop_alert_management()
            if self.recovery_manager:
                await self.recovery_manager.stop_recovery_management()
            if self.analytics_manager:
                await self.analytics_manager.stop_analytics()

            self.logger.info("Upload health monitoring system stopped successfully")
            return {"success": True}

        except Exception as e:
            self.logger.error("Error stopping upload health monitoring system",
                            context={"error": str(e)})
            return {"success": False, "error": str(e)}

    async def _integration_monitoring_loop(self):
        """Monitor integration between components"""
        while self._running:
            try:
                # Update system statistics
                await self._update_system_statistics()

                # Check component integration health
                await self._check_component_integration()

                # Update Prometheus metrics
                await self._update_prometheus_metrics()

            except Exception as e:
                self.logger.error("Error in integration monitoring loop",
                                context={"error": str(e)})

            await asyncio.sleep(60)  # Monitor every minute

    async def _update_system_statistics(self):
        """Update overall system statistics"""
        try:
            stats = {
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds() if self._start_time else 0,
                "system_running": self._running
            }

            # Get component statistics
            if self.health_checker:
                health_summary = self.health_checker.get_system_health_summary()
                stats["health"] = health_summary

            if self.alert_manager:
                alert_summary = self.alert_manager.get_alert_summary()
                stats["alerts"] = alert_summary

            if self.recovery_manager:
                recovery_summary = self.recovery_manager.get_recovery_summary()
                stats["recovery"] = recovery_summary

            if self.analytics_manager:
                analytics_summary = self.analytics_manager.get_analytics_summary()
                stats["analytics"] = analytics_summary

            self._system_stats = stats

            # Store in Redis for API access
            self.redis.setex("upload_health_system_stats", 300, json.dumps(stats))

        except Exception as e:
            self.logger.error("Error updating system statistics",
                            context={"error": str(e)})

    async def _check_component_integration(self):
        """Check health of component integration"""
        try:
            integration_issues = []

            # Check if all components are running
            components = {
                "health_checker": self.health_checker,
                "alert_manager": self.alert_manager,
                "recovery_manager": self.recovery_manager,
                "analytics_manager": self.analytics_manager
            }

            for name, component in components.items():
                if component is None:
                    integration_issues.append(f"{name} not initialized")
                elif not hasattr(component, '_running') or not component._running:
                    integration_issues.append(f"{name} not running")

            if integration_issues:
                self.logger.warning("Component integration issues detected",
                                  context={"issues": integration_issues})

                # Create alert for integration issues
                if self.alert_manager:
                    self.alert_manager.create_alert(
                        rule_id="integration_issues",
                        title="Component Integration Issues",
                        message=f"Integration issues: {', '.join(integration_issues)}",
                        severity=AlertSeverity.WARNING,
                        metadata={"issues": integration_issues}
                    )

        except Exception as e:
            self.logger.error("Error checking component integration",
                            context={"error": str(e)})

    async def _update_prometheus_metrics(self):
        """Update Prometheus metrics for upload health"""
        try:
            if not self._prometheus_metrics:
                # Initialize Prometheus metrics if available
                try:
                    from prometheus_client import Gauge, Counter, Histogram
                    self._prometheus_metrics = {
                        "upload_health_score": Gauge("upload_system_health_score", "Overall upload system health score", ["component"]),
                        "upload_alerts_active": Gauge("upload_alerts_active_total", "Total active upload alerts", ["severity"]),
                        "upload_recovery_success_rate": Gauge("upload_recovery_success_rate", "Upload recovery success rate"),
                        "upload_analytics_quality_score": Gauge("upload_analytics_quality_score", "Upload quality score", ["dimension"]),
                        "upload_component_status": Gauge("upload_component_status", "Upload component status", ["component", "status"]),
                        "upload_remediation_tasks": Gauge("upload_remediation_tasks_total", "Total upload remediation tasks", ["status"])
                    }
                except ImportError:
                    self.logger.debug("Prometheus client not available")
                    return

            # Update health metrics
            if self.health_checker:
                health_status = self.health_checker.get_component_health()
                for component, info in health_status.items():
                    status_value = {"healthy": 1.0, "degraded": 0.5, "unhealthy": 0.0}.get(info["status"], 0.0)
                    self._prometheus_metrics["upload_component_status"].labels(
                        component=component, status=info["status"]
                    ).set(status_value)

            # Update alert metrics
            if self.alert_manager:
                alert_summary = self.alert_manager.get_alert_summary()
                for severity, count in alert_summary.get("alerts_by_severity", {}).items():
                    self._prometheus_metrics["upload_alerts_active"].labels(severity=severity).set(count)

            # Update recovery metrics
            if self.recovery_manager:
                recovery_summary = self.recovery_manager.get_recovery_summary()
                success_rate = recovery_summary.get("success_rate", 0)
                self._prometheus_metrics["upload_recovery_success_rate"].set(success_rate)

            # Update analytics metrics
            if self.analytics_manager:
                quality_metrics = self.analytics_manager.get_quality_metrics()
                for metric in quality_metrics.get("metrics", []):
                    self._prometheus_metrics["upload_analytics_quality_score"].labels(
                        dimension=metric["quality_dimension"]
                    ).set(metric["score"])

        except Exception as e:
            self.logger.error("Error updating Prometheus metrics",
                            context={"error": str(e)})

    # API Methods for Health Status

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system_running": self._running,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "components": {
                "health_checker": self.health_checker is not None,
                "alert_manager": self.alert_manager is not None,
                "recovery_manager": self.recovery_manager is not None,
                "analytics_manager": self.analytics_manager is not None
            },
            "statistics": self._system_stats
        }

    def get_component_health(self, component_type: Optional[str] = None) -> Dict[str, Any]:
        """Get component health status"""
        if not self.health_checker:
            return {"error": "Health checker not available"}

        if component_type:
            try:
                comp_type = UploadComponentType(component_type)
                return self.health_checker.get_component_health(comp_type)
            except ValueError:
                return {"error": f"Invalid component type: {component_type}"}

        return self.health_checker.get_component_health()

    def get_alerts(self, component_type: Optional[str] = None,
                   severity: Optional[str] = None, active_only: bool = True) -> Dict[str, Any]:
        """Get alerts with optional filtering"""
        if not self.alert_manager:
            return {"error": "Alert manager not available"}

        try:
            comp_type = UploadComponentType(component_type) if component_type else None
            sev = AlertSeverity(severity) if severity else None

            alerts = self.alert_manager.get_active_alerts(comp_type, sev) if active_only else []

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "total_alerts": len(alerts),
                "alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "severity": alert.severity.value,
                        "title": alert.title,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "status": alert.status.value,
                        "component_type": alert.component_type.value if alert.component_type else None,
                        "acknowledged": alert.acknowledged_at is not None,
                        "escalation_level": alert.escalation_level
                    }
                    for alert in alerts
                ]
            }

        except ValueError as e:
            return {"error": str(e)}

    def get_recovery_status(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Get recovery/remediation status"""
        if not self.recovery_manager:
            return {"error": "Recovery manager not available"}

        return self.recovery_manager.get_remediation_status(task_id)

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary"""
        if not self.analytics_manager:
            return {"error": "Analytics manager not available"}

        return self.analytics_manager.get_analytics_summary()

    def get_upload_trends(self, component: Optional[str] = None,
                         time_period: Optional[str] = None) -> Dict[str, Any]:
        """Get upload operation trends"""
        if not self.analytics_manager:
            return {"error": "Analytics manager not available"}

        return self.analytics_manager.get_upload_trends(component, time_period)

    def get_bottleneck_analysis(self, active_only: bool = True) -> Dict[str, Any]:
        """Get bottleneck analysis"""
        if not self.analytics_manager:
            return {"error": "Analytics manager not available"}

        return self.analytics_manager.get_bottleneck_analysis(active_only)

    def get_quality_metrics(self, dimension: Optional[str] = None) -> Dict[str, Any]:
        """Get quality metrics"""
        if not self.analytics_manager:
            return {"error": "Analytics manager not available"}

        return self.analytics_manager.get_quality_metrics(dimension)

    # Control Methods

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> Dict[str, Any]:
        """Acknowledge an alert"""
        if not self.alert_manager:
            return {"success": False, "error": "Alert manager not available"}

        success = self.alert_manager.acknowledge_alert(alert_id, acknowledged_by)
        return {"success": success, "alert_id": alert_id, "acknowledged_by": acknowledged_by}

    def resolve_alert(self, alert_id: str, resolved_by: Optional[str] = None) -> Dict[str, Any]:
        """Resolve an alert"""
        if not self.alert_manager:
            return {"success": False, "error": "Alert manager not available"}

        success = self.alert_manager.resolve_alert(alert_id, resolved_by)
        return {"success": success, "alert_id": alert_id, "resolved_by": resolved_by}

    def trigger_manual_remediation(self, component_type: str, action: str,
                                 description: str, requested_by: str) -> Dict[str, Any]:
        """Trigger manual remediation"""
        if not self.recovery_manager:
            return {"success": False, "error": "Recovery manager not available"}

        try:
            comp_type = UploadComponentType(component_type)
            rec_action = RecoveryAction(action)

            task_id = self.recovery_manager.trigger_manual_remediation(
                comp_type, rec_action, description, requested_by
            )

            return {
                "success": True,
                "task_id": task_id,
                "component_type": component_type,
                "action": action,
                "requested_by": requested_by
            }

        except ValueError as e:
            return {"success": False, "error": str(e)}

    def enable_graceful_degradation(self, component_type: str, duration_hours: int = 1) -> Dict[str, Any]:
        """Enable graceful degradation for a component"""
        if not self.recovery_manager:
            return {"success": False, "error": "Recovery manager not available"}

        try:
            comp_type = UploadComponentType(component_type)
            success = self.recovery_manager.enable_graceful_degradation(comp_type, duration_hours)

            return {
                "success": success,
                "component_type": component_type,
                "duration_hours": duration_hours,
                "graceful_degradation_enabled": success
            }

        except ValueError as e:
            return {"success": False, "error": str(e)}

    def disable_graceful_degradation(self, component_type: str) -> Dict[str, Any]:
        """Disable graceful degradation for a component"""
        if not self.recovery_manager:
            return {"success": False, "error": "Recovery manager not available"}

        try:
            comp_type = UploadComponentType(component_type)
            success = self.recovery_manager.disable_graceful_degradation(comp_type)

            return {
                "success": success,
                "component_type": component_type,
                "graceful_degradation_disabled": success
            }

        except ValueError as e:
            return {"success": False, "error": str(e)}

    # Integration Methods

    def record_upload_metric(self, job_id: str, operation: str, component: str,
                           metric_type: MetricType, value: float, unit: str,
                           metadata: Optional[Dict[str, Any]] = None,
                           tags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Record an upload metric for analytics"""
        if not self.analytics_manager:
            return {"success": False, "error": "Analytics manager not available"}

        try:
            self.analytics_manager.record_upload_metric(
                job_id, operation, component, metric_type, value, unit, metadata, tags
            )

            return {
                "success": True,
                "job_id": job_id,
                "operation": operation,
                "component": component,
                "value": value,
                "unit": unit
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def trigger_upload_failure_alert(self, job_id: str, error: Exception,
                                   component_type: str) -> Dict[str, Any]:
        """Trigger an alert for upload failure"""
        if not self.alert_manager:
            return {"success": False, "error": "Alert manager not available"}

        try:
            comp_type = UploadComponentType(component_type)
            alert = self.alert_manager.trigger_upload_failure_alert(job_id, error, comp_type)

            return {
                "success": alert is not None,
                "alert_id": alert.alert_id if alert else None,
                "job_id": job_id,
                "component_type": component_type,
                "error": str(error)
            }

        except ValueError as e:
            return {"success": False, "error": str(e)}

    def trigger_performance_alert(self, component_type: str, metric_name: str,
                                current_value: float, threshold_value: float) -> Dict[str, Any]:
        """Trigger an alert for performance issues"""
        if not self.alert_manager:
            return {"success": False, "error": "Alert manager not available"}

        try:
            comp_type = UploadComponentType(component_type)
            alert = self.alert_manager.trigger_performance_alert(
                comp_type, metric_name, current_value, threshold_value
            )

            return {
                "success": alert is not None,
                "alert_id": alert.alert_id if alert else None,
                "component_type": component_type,
                "metric_name": metric_name,
                "current_value": current_value,
                "threshold_value": threshold_value
            }

        except ValueError as e:
            return {"success": False, "error": str(e)}

    def trigger_health_check_alert(self, component_type: str, status: str, message: str) -> Dict[str, Any]:
        """Trigger an alert for health check failures"""
        if not self.alert_manager:
            return {"success": False, "error": "Alert manager not available"}

        try:
            comp_type = UploadComponentType(component_type)
            status_enum = UploadComponentStatus(status)
            alert = self.alert_manager.trigger_health_check_alert(comp_type, status_enum, message)

            return {
                "success": alert is not None,
                "alert_id": alert.alert_id if alert else None,
                "component_type": component_type,
                "status": status,
                "message": message
            }

        except ValueError as e:
            return {"success": False, "error": str(e)}

    # Dashboard Integration Methods

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive data for monitoring dashboard"""
        try:
            dashboard_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "system_status": self.get_system_status(),
                "component_health": self.get_component_health(),
                "active_alerts": self.get_alerts(active_only=True),
                "recovery_status": self.get_recovery_status(),
                "analytics_summary": self.get_analytics_summary(),
                "recent_trends": self.get_upload_trends(time_period="short_term"),
                "active_bottlenecks": self.get_bottleneck_analysis(active_only=True),
                "quality_overview": self.get_quality_metrics()
            }

            return dashboard_data

        except Exception as e:
            self.logger.error("Error generating dashboard data",
                            context={"error": str(e)})
            return {"error": str(e)}

    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        try:
            if self._prometheus_metrics:
                # Force update of Prometheus metrics
                asyncio.create_task(self._update_prometheus_metrics())

                # Return formatted metrics (this would use prometheus_client.generate_latest)
                return "# Upload Health Monitoring Metrics\n# Generated at " + datetime.utcnow().isoformat()
            else:
                return "# Prometheus metrics not available"

        except Exception as e:
            self.logger.error("Error generating Prometheus metrics",
                            context={"error": str(e)})
            return f"# Error generating metrics: {str(e)}"

    # Utility Methods

    def get_system_health_score(self) -> Dict[str, Any]:
        """Get overall system health score"""
        try:
            if not self.health_checker:
                return {"error": "Health checker not available"}

            health_summary = self.health_checker.get_system_health_summary()

            # Calculate weighted health score
            components = health_summary.get("components", {})
            total_components = components.get("total", 0)

            if total_components == 0:
                return {"health_score": 0.0, "status": "unknown"}

            # Weight calculation: healthy=1.0, degraded=0.5, unhealthy=0.0
            healthy_weight = components.get("healthy", 0) * 1.0
            degraded_weight = components.get("degraded", 0) * 0.5
            unhealthy_weight = components.get("unhealthy", 0) * 0.0

            weighted_score = (healthy_weight + degraded_weight + unhealthy_weight) / total_components
            health_score = weighted_score * 100

            return {
                "health_score": round(health_score, 2),
                "status": health_summary.get("overall_status", "unknown"),
                "components": components,
                "last_check": health_summary.get("last_check")
            }

        except Exception as e:
            self.logger.error("Error calculating system health score",
                            context={"error": str(e)})
            return {"error": str(e)}

    def get_upload_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive upload health report"""
        try:
            report = {
                "report_id": str(uuid.uuid4()),
                "generated_at": datetime.utcnow().isoformat(),
                "system_overview": self.get_system_status(),
                "health_score": self.get_system_health_score(),
                "component_health": self.get_component_health(),
                "alert_summary": self.get_alerts(active_only=True),
                "recovery_summary": self.get_recovery_status(),
                "analytics_overview": self.get_analytics_summary(),
                "quality_assessment": self.get_quality_metrics(),
                "recommendations": self._generate_system_recommendations()
            }

            return report

        except Exception as e:
            self.logger.error("Error generating upload health report",
                            context={"error": str(e)})
            return {"error": str(e)}

    def _generate_system_recommendations(self) -> List[str]:
        """Generate system-wide recommendations"""
        recommendations = []

        try:
            # Health-based recommendations
            if self.health_checker:
                health_summary = self.health_checker.get_system_health_summary()
                if health_summary.get("overall_status") == "unhealthy":
                    recommendations.append("Immediate attention required: System health is unhealthy")
                elif health_summary.get("overall_status") == "degraded":
                    recommendations.append("System health is degraded - investigate component issues")

            # Alert-based recommendations
            if self.alert_manager:
                alert_summary = self.alert_manager.get_alert_summary()
                critical_alerts = alert_summary.get("alerts_by_severity", {}).get("critical", 0)
                if critical_alerts > 0:
                    recommendations.append(f"Address {critical_alerts} critical alerts immediately")

            # Recovery-based recommendations
            if self.recovery_manager:
                recovery_summary = self.recovery_manager.get_recovery_summary()
                success_rate = recovery_summary.get("success_rate", 1.0)
                if success_rate < 0.7:
                    recommendations.append("Recovery success rate is low - review remediation procedures")

            # Analytics-based recommendations
            if self.analytics_manager:
                bottlenecks = self.analytics_manager.get_bottleneck_analysis(active_only=True)
                if bottlenecks.get("active_bottlenecks", 0) > 0:
                    recommendations.append(f"Address {bottlenecks['active_bottlenecks']} active bottlenecks")

                quality_metrics = self.analytics_manager.get_quality_metrics()
                metrics = quality_metrics.get("metrics", [])
                if metrics:
                    avg_quality = sum(m["score"] for m in metrics[:5]) / min(5, len(metrics))
                    if avg_quality < 0.7:
                        recommendations.append("Overall upload quality is below acceptable threshold")

            if not recommendations:
                recommendations.append("System is operating within normal parameters")

            return recommendations

        except Exception as e:
            self.logger.error("Error generating system recommendations",
                            context={"error": str(e)})
            return [f"Error generating recommendations: {str(e)}"]

    def export_system_logs(self, hours: int = 24) -> Dict[str, Any]:
        """Export system logs for analysis"""
        try:
            # This would integrate with the logging system to export logs
            # For now, return a placeholder

            cutoff_time = datetime.utcnow() - timedelta(hours=hours)

            return {
                "export_id": str(uuid.uuid4()),
                "exported_at": datetime.utcnow().isoformat(),
                "time_range": {
                    "start": cutoff_time.isoformat(),
                    "end": datetime.utcnow().isoformat(),
                    "hours": hours
                },
                "components": {
                    "health_checker": "Logs available" if self.health_checker else "Not available",
                    "alert_manager": "Logs available" if self.alert_manager else "Not available",
                    "recovery_manager": "Logs available" if self.recovery_manager else "Not available",
                    "analytics_manager": "Logs available" if self.analytics_manager else "Not available"
                },
                "note": "Log export would be implemented based on specific logging backend"
            }

        except Exception as e:
            self.logger.error("Error exporting system logs",
                            context={"error": str(e)})
            return {"error": str(e)}


# Global upload health monitoring system instance
_upload_health_monitoring_system = None
_monitoring_lock = threading.Lock()


def get_upload_health_monitoring_system() -> UploadHealthMonitoringSystem:
    """Get the global upload health monitoring system instance"""
    global _upload_health_monitoring_system

    if _upload_health_monitoring_system is None:
        with _monitoring_lock:
            if _upload_health_monitoring_system is None:
                _upload_health_monitoring_system = UploadHealthMonitoringSystem()

    return _upload_health_monitoring_system


# Convenience functions for easy integration

async def start_upload_health_monitoring() -> Dict[str, Any]:
    """Start the upload health monitoring system"""
    system = get_upload_health_monitoring_system()
    return await system.start_system()


async def stop_upload_health_monitoring() -> Dict[str, Any]:
    """Stop the upload health monitoring system"""
    system = get_upload_health_monitoring_system()
    return await system.stop_system()


def get_upload_system_status() -> Dict[str, Any]:
    """Get current upload system status"""
    system = get_upload_health_monitoring_system()
    return system.get_system_status()


def record_upload_operation_metric(job_id: str, operation: str, component: str,
                                 value: float, unit: str = "count",
                                 metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Record an upload operation metric"""
    system = get_upload_health_monitoring_system()
    return system.record_upload_metric(
        job_id, operation, component, MetricType.GAUGE, value, unit, metadata
    )


def trigger_upload_health_alert(job_id: str, error: Exception, component: str) -> Dict[str, Any]:
    """Trigger an alert for upload health issues"""
    system = get_upload_health_monitoring_system()
    return system.trigger_upload_failure_alert(job_id, error, component)