# app/monitoring.py
import asyncio
import json
import logging
import psutil
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from fastapi import WebSocket
import redis
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create dummy classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs): pass
    class Histogram:
        def __init__(self, *args, **kwargs): pass
    class Gauge:
        def __init__(self, *args, **kwargs): pass
    class CollectorRegistry:
        def __init__(self, *args, **kwargs): pass
    def generate_latest(*args, **kwargs): return b""
import threading


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class MetricPoint:
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class HealthCheck:
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    id: str
    severity: str  # "info", "warning", "error", "critical"
    title: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    labels: Dict[str, str] = field(default_factory=dict)


class StructuredLogger:
    """Enhanced logger with correlation IDs and structured context"""

    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = JSONFormatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _log_with_context(self, level: int, message: str, correlation_id: Optional[str] = None,
                         context: Optional[Dict[str, Any]] = None, **kwargs):
        extra = {
            'correlation_id': correlation_id or str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'context': context or {},
            **kwargs
        }

        self.logger.log(level, message, extra=extra)

    def info(self, message: str, correlation_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None, **kwargs):
        self._log_with_context(logging.INFO, message, correlation_id, context, **kwargs)

    def warning(self, message: str, correlation_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None, **kwargs):
        self._log_with_context(logging.WARNING, message, correlation_id, context, **kwargs)

    def error(self, message: str, correlation_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None, **kwargs):
        self._log_with_context(logging.ERROR, message, correlation_id, context, **kwargs)

    def debug(self, message: str, correlation_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None, **kwargs):
        self._log_with_context(logging.DEBUG, message, correlation_id, context, **kwargs)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }

        # Add custom attributes if they exist
        if hasattr(record, 'correlation_id') and record.correlation_id:
            log_entry['correlation_id'] = record.correlation_id
        if hasattr(record, 'context') and record.context:
            log_entry['context'] = record.context

        return json.dumps(log_entry)


class MetricsCollector:
    """Thread-safe metrics collection and storage"""

    def __init__(self):
        self._metrics = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.Lock()
        self._registry = CollectorRegistry()

        # Prometheus metrics
        self.request_count = Counter('pipeline_requests_total', 'Total requests', ['endpoint', 'method', 'status'], registry=self._registry)
        self.request_duration = Histogram('pipeline_request_duration_seconds', 'Request duration', ['endpoint'], registry=self._registry)
        self.active_jobs = Gauge('pipeline_active_jobs', 'Number of active jobs', registry=self._registry)
        
        self.error_count = Counter('pipeline_errors_total', 'Total errors', ['type', 'component'], registry=self._registry)
        self.cache_hit_rate = Gauge('pipeline_cache_hit_rate', 'Cache hit rate', registry=self._registry)
        self.system_cpu_usage = Gauge('pipeline_system_cpu_percent', 'System CPU usage', registry=self._registry)
        self.system_memory_usage = Gauge('pipeline_system_memory_percent', 'System memory usage', registry=self._registry)

    def record_metric(self, metric: MetricPoint):
        """Record a custom metric point"""
        with self._lock:
            self._metrics[metric.name].append(metric)

    def get_metrics(self, name: str, limit: int = 100) -> List[MetricPoint]:
        """Get recent metrics for a specific metric name"""
        with self._lock:
            return list(self._metrics[name])[-limit:]

    def increment_counter(self, name: str, value: float = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        metric = MetricPoint(name, value, datetime.utcnow(), labels or {})
        self.record_metric(metric)

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        metric = MetricPoint(name, value, datetime.utcnow(), labels or {}, MetricType.GAUGE)
        self.record_metric(metric)

    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric"""
        metric = MetricPoint(name, value, datetime.utcnow(), labels or {}, MetricType.HISTOGRAM)
        self.record_metric(metric)

    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return generate_latest(self._registry).decode('utf-8')


class HealthChecker:
    """Comprehensive health checking for pipeline components"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.check_configs: Dict[str, Dict[str, Any]] = {}
        self.check_results: Dict[str, HealthCheck] = {}
        self._lock = threading.Lock()

    def add_check(self, name: str, check_func: Callable[[], tuple], interval_seconds: int = 60):
        """Add a health check function"""
        self.check_configs[name] = {
            'function': check_func,
            'interval': interval_seconds,
            'last_run': 0
        }

    def run_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all health checks and return results"""
        results = {}
        current_time = time.time()

        for name, check_config in self.check_configs.items():
            # Check if enough time has passed since last run
            if current_time - check_config['last_run'] >= check_config['interval']:
                try:
                    status, message, details = check_config['function']()
                    health_check = HealthCheck(
                        name=name,
                        status=status,
                        message=message,
                        timestamp=datetime.utcnow(),
                        details=details
                    )
                    results[name] = health_check
                    check_config['last_run'] = current_time

                except Exception as e:
                    results[name] = HealthCheck(
                        name=name,
                        status="unhealthy",
                        message=f"Health check failed: {str(e)}",
                        timestamp=datetime.utcnow()
                    )

        with self._lock:
            self.check_results.update(results)

        return results


class AlertManager:
    """Alert management and notification system"""

    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict] = []
        self._lock = threading.Lock()

    def add_alert_rule(self, name: str, condition_func, severity: str, message_template: str):
        """Add an alert rule"""
        self.alert_rules.append({
            'name': name,
            'condition': condition_func,
            'severity': severity,
            'message_template': message_template
        })

    def check_alerts(self, metrics_collector: MetricsCollector) -> List[Alert]:
        """Check all alert rules and return new alerts"""
        new_alerts = []

        for rule in self.alert_rules:
            try:
                if rule['condition'](metrics_collector):
                    alert_id = str(uuid.uuid4())
                    message = rule['message_template'].format(
                        name=rule['name'],
                        severity=rule['severity']
                    )

                    alert = Alert(
                        id=alert_id,
                        severity=rule['severity'],
                        title=f"Alert: {rule['name']}",
                        message=message,
                        timestamp=datetime.utcnow()
                    )

                    with self._lock:
                        self.alerts[alert_id] = alert

                    new_alerts.append(alert)

            except Exception as e:
                print(f"Error checking alert rule {rule['name']}: {e}")

        return new_alerts

    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        with self._lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved = True
                self.alerts[alert_id].resolved_at = datetime.utcnow()

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        with self._lock:
            return [alert for alert in self.alerts.values() if not alert.resolved]


class WebSocketManager:
    """WebSocket connection manager for real-time updates"""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = defaultdict(list)
        self._lock = threading.Lock()

    async def connect(self, client_id: str, websocket: WebSocket):
        """Add a WebSocket connection"""
        await websocket.accept()
        with self._lock:
            self.active_connections[client_id].append(websocket)

    def disconnect(self, client_id: str, websocket: WebSocket):
        """Remove a WebSocket connection"""
        with self._lock:
            if client_id in self.active_connections:
                self.active_connections[client_id].remove(websocket)
                if not self.active_connections[client_id]:
                    del self.active_connections[client_id]

    async def broadcast_to_client(self, client_id: str, message: Dict[str, Any]):
        """Broadcast message to specific client"""
        if client_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[client_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)

            # Remove disconnected connections
            for conn in disconnected:
                self.active_connections[client_id].remove(conn)

    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        disconnected_clients = []

        for client_id, connections in self.active_connections.items():
            disconnected = []
            for connection in connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)

            # Remove disconnected connections
            for conn in disconnected:
                connections.remove(conn)

            if not connections:
                disconnected_clients.append(client_id)

        # Remove clients with no connections
        for client_id in disconnected_clients:
            del self.active_connections[client_id]


class AudioPipelineMonitor:
    """Main monitoring class for the audio processing pipeline"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = redis.from_url(redis_url)
        self.logger = StructuredLogger("audio_pipeline_monitor")
        self.metrics = MetricsCollector()
        self.health_checker = HealthChecker(self.redis)
        self.alert_manager = AlertManager()
        self.websocket_manager = WebSocketManager()

        # Error recovery integration
        try:
            from .error_recovery import get_error_recovery_manager
            self.error_recovery = get_error_recovery_manager()
        except ImportError:
            self.error_recovery = None

        # Monitoring state
        self._monitoring = False
        self._monitor_task = None
        self._system_monitor_task = None

        # Performance tracking
        self.job_start_times: Dict[str, datetime] = {}
        self.job_metrics: Dict[str, Dict[str, Any]] = {}
        self._performance_lock = threading.Lock()

        self._setup_default_health_checks()
        self._setup_default_alert_rules()
        self._setup_error_recovery_metrics()

    def _setup_default_health_checks(self):
        """Setup default health checks"""

        # Redis health check
        def check_redis():
            try:
                self.redis.ping()
                return "healthy", "Redis connection successful", {"response_time": "fast"}
            except Exception as e:
                return "unhealthy", f"Redis connection failed: {str(e)}", {}

        # Disk space check
        def check_disk_space():
            try:
                disk = psutil.disk_usage('/')
                free_gb = disk.free / (1024**3)
                if free_gb < 1:
                    return "unhealthy", f"Low disk space: {free_gb:.1f}GB free", {"free_gb": free_gb}
                return "healthy", f"Disk space OK: {free_gb:.1f}GB free", {"free_gb": free_gb}
            except Exception as e:
                return "unhealthy", f"Disk check failed: {str(e)}", {}

        # Memory usage check
        def check_memory():
            try:
                memory = psutil.virtual_memory()
                usage_percent = memory.percent
                if usage_percent > 90:
                    return "unhealthy", f"High memory usage: {usage_percent:.1f}%", {"usage_percent": usage_percent}
                elif usage_percent > 75:
                    return "degraded", f"Elevated memory usage: {usage_percent:.1f}%", {"usage_percent": usage_percent}
                return "healthy", f"Memory usage OK: {usage_percent:.1f}%", {"usage_percent": usage_percent}
            except Exception as e:
                return "unhealthy", f"Memory check failed: {str(e)}", {}

        self.health_checker.add_check("redis", check_redis, 30)
        self.health_checker.add_check("disk_space", check_disk_space, 60)
        self.health_checker.add_check("memory", check_memory, 30)

    def _setup_default_alert_rules(self):
        """Setup default alert rules"""

        # High error rate alert
        def high_error_rate(metrics_collector: MetricsCollector) -> bool:
            recent_errors = metrics_collector.get_metrics("error", limit=10)
            if len(recent_errors) >= 5:
                return True
            return False

        # High memory usage alert
        def high_memory_usage(metrics_collector: MetricsCollector) -> bool:
            recent_memory = metrics_collector.get_metrics("system_memory_percent", limit=5)
            if recent_memory and any(m.value > 85 for m in recent_memory):
                return True
            return False

        # Job processing time alert
        def slow_job_processing(metrics_collector: MetricsCollector) -> bool:
            recent_jobs = metrics_collector.get_metrics("job_processing_time", limit=10)
            if recent_jobs and any(m.value > 1800 for m in recent_jobs):  # 30 minutes
                return True
            return False

        # Error recovery failure alert
        def high_recovery_failure_rate(metrics_collector: MetricsCollector) -> bool:
            if self.error_recovery:
                stats = self.error_recovery.get_recovery_stats()
                success_rate = stats.get("success_rate", 1.0)
                return success_rate < 0.5  # Less than 50% success rate
            return False

        # Circuit breaker open alert
        def circuit_breaker_open(metrics_collector: MetricsCollector) -> bool:
            if self.error_recovery:
                stats = self.error_recovery.get_recovery_stats()
                circuit_states = stats.get("circuit_breaker_states", {})
                return any(state.get("state") == "open" for state in circuit_states.values())
            return False

        self.alert_manager.add_alert_rule(
            "high_error_rate",
            high_error_rate,
            "warning",
            "High error rate detected in the last 10 metrics"
        )

        self.alert_manager.add_alert_rule(
            "high_memory_usage",
            high_memory_usage,
            "warning",
            "Memory usage exceeded 85% threshold"
        )

        self.alert_manager.add_alert_rule(
            "slow_job_processing",
            slow_job_processing,
            "warning",
            "Job processing time exceeded 30 minutes"
        )

        self.alert_manager.add_alert_rule(
            "high_recovery_failure_rate",
            high_recovery_failure_rate,
            "error",
            "Error recovery success rate below 50%"
        )

        self.alert_manager.add_alert_rule(
            "circuit_breaker_open",
            circuit_breaker_open,
            "critical",
            "One or more circuit breakers are in OPEN state"
        )

    def _setup_error_recovery_metrics(self):
        """Setup error recovery specific metrics"""
        if not self.error_recovery:
            return

        # Add Prometheus metrics for error recovery
        self.metrics.error_recovery_attempts = Counter(
            'pipeline_error_recovery_attempts_total',
            'Total error recovery attempts',
            ['strategy', 'success'],
            registry=self.metrics._registry
        )

        self.metrics.error_recovery_duration = Histogram(
            'pipeline_error_recovery_duration_seconds',
            'Error recovery duration',
            ['strategy'],
            registry=self.metrics._registry
        )

        self.metrics.circuit_breaker_state = Gauge(
            'pipeline_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 0.5=half_open)',
            ['name'],
            registry=self.metrics._registry
        )

    async def start_monitoring(self):
        """Start the monitoring system"""
        if self._monitoring:
            return

        self._monitoring = True
        self.logger.info("Starting AudioPipelineMonitor", context={"component": "monitor"})

        # Start monitoring tasks
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        self._system_monitor_task = asyncio.create_task(self._system_monitoring_loop())

        # Start health checks
        await self._start_health_checks()

    async def stop_monitoring(self):
        """Stop the monitoring system"""
        if not self._monitoring:
            return

        self._monitoring = False
        self.logger.info("Stopping AudioPipelineMonitor", context={"component": "monitor"})

        # Stop monitoring tasks
        if self._monitor_task:
            self._monitor_task.cancel()
        if self._system_monitor_task:
            self._system_monitor_task.cancel()

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                # Run health checks
                health_results = self.health_checker.run_health_checks()

                # Check for alerts
                new_alerts = self.alert_manager.check_alerts(self.metrics)

                # Update error recovery metrics
                self._update_error_recovery_metrics()

                # Broadcast updates to WebSocket clients
                if new_alerts or health_results:
                    update_message = {
                        "type": "monitoring_update",
                        "timestamp": datetime.utcnow().isoformat(),
                        "health_checks": {
                            name: {
                                "status": check.status,
                                "message": check.message,
                                "timestamp": check.timestamp.isoformat()
                            }
                            for name, check in health_results.items()
                        },
                        "alerts": [
                            {
                                "id": alert.id,
                                "severity": alert.severity,
                                "title": alert.title,
                                "message": alert.message,
                                "timestamp": alert.timestamp.isoformat()
                            }
                            for alert in new_alerts
                        ]
                    }

                    await self.websocket_manager.broadcast_to_all(update_message)

                # Update system metrics
                self._update_system_metrics()

            except Exception as e:
                self.logger.error("Error in monitoring loop",
                                context={"error": str(e), "component": "monitor"})

            await asyncio.sleep(30)  # Run every 30 seconds

    async def _system_monitoring_loop(self):
        """System resource monitoring loop"""
        while self._monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics.set_gauge("system_cpu_percent", cpu_percent)

                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics.set_gauge("system_memory_percent", memory.percent)
                self.metrics.set_gauge("system_memory_used_gb", memory.used / (1024**3))
                self.metrics.set_gauge("system_memory_available_gb", memory.available / (1024**3))

                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.metrics.set_gauge("system_disk_read_mb", disk_io.read_bytes / (1024**2))
                    self.metrics.set_gauge("system_disk_write_mb", disk_io.write_bytes / (1024**2))

                # Network I/O
                network_io = psutil.net_io_counters()
                if network_io:
                    self.metrics.set_gauge("system_network_bytes_sent_mb", network_io.bytes_sent / (1024**2))
                    self.metrics.set_gauge("system_network_bytes_recv_mb", network_io.bytes_recv / (1024**2))

            except Exception as e:
                self.logger.error("Error in system monitoring loop",
                                context={"error": str(e), "component": "system_monitor"})

            await asyncio.sleep(60)  # Run every minute

    async def _start_health_checks(self):
        """Start the health check system"""
        # This runs in the background as part of the monitoring loop
        pass

    def _update_system_metrics(self):
        """Update system-level metrics"""
        try:
            # Active connections
            active_connections = len(self.websocket_manager.active_connections)
            self.metrics.set_gauge("active_websocket_connections", active_connections)

            # Queue status (if RQ is available)
            try:
                from rq import Queue, Connection
                with Connection(self.redis):
                    queue = Queue()
                    queue_length = len(queue)
                    self.metrics.set_gauge("redis_queue_length", queue_length)
            except Exception:
                pass  # RQ not available or not configured

        except Exception as e:
            self.logger.error("Error updating system metrics",
                            context={"error": str(e), "component": "metrics"})

    def _update_error_recovery_metrics(self):
        """Update error recovery specific metrics"""
        if not self.error_recovery:
            return

        try:
            # Get recovery statistics
            stats = self.error_recovery.get_recovery_stats()

            # Update Prometheus metrics
            total_attempts = stats.get("total_recovery_attempts", 0)
            successful_attempts = stats.get("successful_attempts", 0)

            if total_attempts > 0:
                success_rate = successful_attempts / total_attempts
                self.metrics.set_gauge("error_recovery_success_rate", success_rate)

            # Update circuit breaker states
            circuit_states = stats.get("circuit_breaker_states", {})
            for cb_name, cb_state in circuit_states.items():
                state_value = {"closed": 0, "half_open": 0.5, "open": 1}.get(cb_state.get("state", "closed"), 0)
                self.metrics.set_gauge("circuit_breaker_state", state_value, labels={"name": cb_name})

            # Record error recovery metrics in our custom collector
            self.metrics.set_gauge("error_recovery_total_attempts", total_attempts)
            self.metrics.set_gauge("error_recovery_successful_attempts", successful_attempts)

        except Exception as e:
            self.logger.error("Error updating error recovery metrics",
                            context={"error": str(e), "component": "error_recovery_metrics"})

    def start_job_tracking(self, job_id: str, correlation_id: Optional[str] = None) -> str:
        """Start tracking a job's performance"""
        actual_correlation_id = correlation_id or str(uuid.uuid4())

        with self._performance_lock:
            self.job_start_times[job_id] = datetime.utcnow()
            self.job_metrics[job_id] = {
                'correlation_id': actual_correlation_id,
                'start_time': datetime.utcnow(),
                'stages': {},
                'metrics': {}
            }

        self.logger.info("Job tracking started",
                        correlation_id=actual_correlation_id,
                        context={"job_id": job_id, "component": "job_tracker"})

        return actual_correlation_id

    def record_job_stage(self, job_id: str, stage: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
        """Record a job stage completion"""
        with self._performance_lock:
            if job_id in self.job_metrics:
                self.job_metrics[job_id]['stages'][stage] = {
                    'duration': duration,
                    'timestamp': datetime.utcnow(),
                    'metadata': metadata or {}
                }

                # Update Prometheus metrics
                self.metrics.record_histogram("job_stage_duration", duration,
                                            labels={"stage": stage})

        correlation_id = self.job_metrics.get(job_id, {}).get('correlation_id')
        self.logger.info("Job stage completed",
                        correlation_id=correlation_id,
                        context={"job_id": job_id, "stage": stage, "duration": duration})

    def complete_job_tracking(self, job_id: str, status: str, error: Optional[str] = None):
        """Complete job tracking and record final metrics"""
        with self._performance_lock:
            if job_id in self.job_start_times:
                start_time = self.job_start_times[job_id]
                end_time = datetime.utcnow()
                total_duration = (end_time - start_time).total_seconds()

                job_info = self.job_metrics.get(job_id, {})
                correlation_id = job_info.get('correlation_id')

                # Record final metrics
                self.metrics.record_histogram("job_processing_time", total_duration,
                                            labels={"status": status})

                if status == "success":
                    self.metrics.increment_counter("jobs_completed_total")
                else:
                    self.metrics.increment_counter("jobs_failed_total")
                    if error:
                        self.metrics.increment_counter("job_errors", labels={"error_type": "processing"})

                self.logger.info("Job tracking completed",
                               correlation_id=correlation_id,
                               context={
                                   "job_id": job_id,
                                   "status": status,
                                   "duration": total_duration,
                                   "error": error
                               })

                # Clean up tracking data
                del self.job_start_times[job_id]
                if job_id in self.job_metrics:
                    del self.job_metrics[job_id]

    def record_video_processing_metrics(self, job_id: str, file_size: int, processing_time: float,
                                      success: bool, format_info: Optional[Dict[str, Any]] = None):
        """Record video processing specific metrics"""
        labels = {"success": str(success)}

        self.metrics.record_histogram("video_processing_time", processing_time, labels)
        self.metrics.record_histogram("video_file_size", file_size, labels)

        if success:
            self.metrics.increment_counter("video_processing_success_total")
        else:
            self.metrics.increment_counter("video_processing_failure_total")

    def record_cache_metrics(self, operation: str, hit: bool, response_time: float):
        """Record cache operation metrics"""
        labels = {"operation": operation, "result": "hit" if hit else "miss"}

        self.metrics.record_histogram("cache_operation_time", response_time, labels)

        if hit:
            self.metrics.increment_counter("cache_hits_total", labels={"operation": operation})
        else:
            self.metrics.increment_counter("cache_misses_total", labels={"operation": operation})

    def record_upload_metrics(self, file_size: int, upload_time: float, success: bool):
        """Record file upload metrics"""
        labels = {"success": str(success)}

        self.metrics.record_histogram("file_upload_time", upload_time, labels)
        self.metrics.record_histogram("file_upload_size", file_size, labels)

        if success:
            self.metrics.increment_counter("uploads_success_total")
        else:
            self.metrics.increment_counter("uploads_failed_total")

    def get_job_metrics(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific job"""
        with self._performance_lock:
            return self.job_metrics.get(job_id)

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        health_results = self.health_checker.run_health_checks()
        active_alerts = self.alert_manager.get_active_alerts()

        # Get error recovery status
        error_recovery_status = None
        if self.error_recovery:
            recovery_stats = self.error_recovery.get_recovery_stats()
            error_recovery_status = {
                "enabled": True,
                "total_recovery_attempts": recovery_stats.get("total_recovery_attempts", 0),
                "successful_attempts": recovery_stats.get("successful_attempts", 0),
                "success_rate": recovery_stats.get("success_rate", 0),
                "circuit_breaker_states": recovery_stats.get("circuit_breaker_states", {}),
                "error_contexts": recovery_stats.get("error_contexts", 0)
            }

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "monitoring_active": self._monitoring,
            "health_checks": {
                name: {
                    "status": check.status,
                    "message": check.message,
                    "timestamp": check.timestamp.isoformat()
                }
                for name, check in health_results.items()
            },
            "active_alerts": [
                {
                    "id": alert.id,
                    "severity": alert.severity,
                    "title": alert.title,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in active_alerts
            ],
            "system_metrics": {
                "active_jobs": len(self.job_start_times),
                "tracked_jobs": len(self.job_metrics),
                "websocket_connections": len(self.websocket_manager.active_connections)
            },
            "error_recovery": error_recovery_status
        }


# Global monitor instance
_monitor_instance = None
_monitor_lock = threading.Lock()


def get_monitor() -> AudioPipelineMonitor:
    """Get the global monitor instance"""
    global _monitor_instance

    if _monitor_instance is None:
        with _monitor_lock:
            if _monitor_instance is None:
                _monitor_instance = AudioPipelineMonitor()

    return _monitor_instance


@asynccontextmanager
async def monitoring_context(correlation_id: Optional[str] = None):
    """Context manager for automatic job tracking"""
    monitor = get_monitor()
    job_id = str(uuid.uuid4())
    actual_correlation_id = monitor.start_job_tracking(job_id, correlation_id)

    try:
        yield job_id, actual_correlation_id
    finally:
        # Job completion will be handled by calling complete_job_tracking
        pass