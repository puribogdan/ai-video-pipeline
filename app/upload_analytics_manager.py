# app/upload_analytics_manager.py
import asyncio
import json
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
import threading
import redis
import logging
import statistics
import numpy as np

from .monitoring import StructuredLogger, MetricsCollector
from .upload_health_checker import UploadHealthChecker, UploadComponentType, UploadComponentStatus
from .upload_alert_manager import UploadAlertManager, AlertSeverity


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    RATE = "rate"


class TrendDirection(Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    VOLATILE = "volatile"


@dataclass
class UploadMetric:
    """Upload operation metric"""
    metric_id: str
    job_id: str
    operation: str  # upload, processing, storage, etc.
    component: str
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class UploadTrend:
    """Trend analysis for upload metrics"""
    metric_name: str
    component: str
    time_period: str  # "1h", "24h", "7d", etc.
    trend_direction: TrendDirection
    trend_strength: float  # 0.0 to 1.0
    current_value: float
    baseline_value: float
    change_percent: float
    confidence: float  # 0.0 to 1.0
    data_points: int
    analysis_timestamp: datetime


@dataclass
class BottleneckAnalysis:
    """Analysis of upload bottlenecks"""
    bottleneck_id: str
    component: str
    bottleneck_type: str  # cpu, memory, disk, network, etc.
    severity: str  # low, medium, high, critical
    impact_score: float  # 0.0 to 1.0
    affected_operations: List[str]
    root_cause: str
    recommendations: List[str]
    detected_at: datetime
    resolved_at: Optional[datetime] = None


@dataclass
class QualityMetric:
    """Upload quality assessment metric"""
    metric_id: str
    job_id: str
    quality_dimension: str  # completeness, accuracy, performance, reliability
    score: float  # 0.0 to 1.0
    max_score: float
    weight: float  # importance weight for overall quality
    details: Dict[str, Any]
    assessed_at: datetime


class UploadAnalyticsManager:
    """Comprehensive analytics and insights for upload operations"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = redis.from_url(redis_url)
        self.logger = StructuredLogger("upload_analytics_manager")

        # Core components
        self.health_checker = None
        self.alert_manager = None

        # Initialize components
        try:
            self.health_checker = UploadHealthChecker(redis_url)
            self.alert_manager = UploadAlertManager(redis_url)
        except Exception as e:
            self.logger.warning("Analytics components not available",
                             context={"error": str(e)})

        # Metrics storage
        self.upload_metrics: Dict[str, List[UploadMetric]] = defaultdict(list)
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.trend_cache: Dict[str, UploadTrend] = {}

        # Analytics configuration
        self.analysis_intervals = {
            "realtime": 60,      # 1 minute
            "short_term": 300,   # 5 minutes
            "medium_term": 1800, # 30 minutes
            "long_term": 3600    # 1 hour
        }

        self.trend_analysis_window = 24 * 3600  # 24 hours
        self.min_data_points_for_trend = 10
        self.min_confidence_threshold = 0.7

        # Bottleneck detection
        self.bottleneck_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_io_wait": 60.0,
            "network_latency": 1000.0,  # ms
            "queue_depth": 50.0
        }

        self.bottlenecks: Dict[str, BottleneckAnalysis] = {}
        self.bottleneck_history: List[BottleneckAnalysis] = []

        # Quality metrics
        self.quality_metrics: Dict[str, List[QualityMetric]] = defaultdict(list)
        self.quality_baselines: Dict[str, float] = {}

        # Locks for thread safety
        self._metrics_lock = threading.Lock()
        self._trend_lock = threading.Lock()
        self._bottleneck_lock = threading.Lock()
        self._quality_lock = threading.Lock()

        # Background tasks
        self._analytics_task = None
        self._trend_task = None
        self._bottleneck_task = None
        self._quality_task = None
        self._running = False

        # Setup quality baselines
        self._setup_quality_baselines()

    def _setup_quality_baselines(self):
        """Setup quality metric baselines"""
        self.quality_baselines = {
            "upload_success_rate": 0.95,      # 95% success rate
            "processing_accuracy": 0.90,      # 90% accuracy
            "performance_efficiency": 0.85,   # 85% efficiency
            "reliability_score": 0.92,        # 92% reliability
            "user_satisfaction": 0.88         # 88% satisfaction
        }

    async def start_analytics(self):
        """Start the analytics system"""
        if self._running:
            return

        self._running = True
        self.logger.info("Starting upload analytics system")

        # Start background tasks
        self._analytics_task = asyncio.create_task(self._analytics_loop())
        self._trend_task = asyncio.create_task(self._trend_analysis_loop())
        self._bottleneck_task = asyncio.create_task(self._bottleneck_detection_loop())
        self._quality_task = asyncio.create_task(self._quality_assessment_loop())

    async def stop_analytics(self):
        """Stop the analytics system"""
        if not self._running:
            return

        self._running = False
        self.logger.info("Stopping upload analytics system")

        # Stop background tasks
        tasks = [self._analytics_task, self._trend_task, self._bottleneck_task, self._quality_task]
        for task in tasks:
            if task:
                task.cancel()

    def record_upload_metric(self, job_id: str, operation: str, component: str,
                           metric_type: MetricType, value: float, unit: str,
                           metadata: Optional[Dict[str, Any]] = None,
                           tags: Optional[Dict[str, str]] = None):
        """Record an upload operation metric"""
        metric_id = str(uuid.uuid4())
        metric = UploadMetric(
            metric_id=metric_id,
            job_id=job_id,
            operation=operation,
            component=component,
            metric_type=metric_type,
            value=value,
            unit=unit,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
            tags=tags or {}
        )

        # Store metric
        with self._metrics_lock:
            metric_key = f"{component}:{operation}"
            self.upload_metrics[metric_key].append(metric)
            self.metric_history[metric_key].append({
                "value": value,
                "timestamp": metric.timestamp,
                "metadata": metadata or {}
            })

            # Limit stored metrics per component/operation
            if len(self.upload_metrics[metric_key]) > 1000:
                self.upload_metrics[metric_key] = self.upload_metrics[metric_key][-500:]

        self.logger.debug("Upload metric recorded",
                        context={
                            "metric_id": metric_id,
                            "job_id": job_id,
                            "operation": operation,
                            "component": component,
                            "value": value,
                            "unit": unit
                        })

    async def _analytics_loop(self):
        """Main analytics processing loop"""
        while self._running:
            try:
                # Process recent metrics
                await self._process_recent_metrics()

                # Update success rate tracking
                await self._update_success_rate_tracking()

                # Generate analytics summary
                await self._generate_analytics_summary()

            except Exception as e:
                self.logger.error("Error in analytics loop",
                                context={"error": str(e)})

            await asyncio.sleep(60)  # Process every minute

    async def _process_recent_metrics(self):
        """Process and analyze recent metrics"""
        try:
            current_time = datetime.utcnow()
            cutoff_time = current_time - timedelta(minutes=5)

            with self._metrics_lock:
                recent_metrics = []
                for metric_list in self.upload_metrics.values():
                    recent_metrics.extend([
                        metric for metric in metric_list
                        if metric.timestamp > cutoff_time
                    ])

            if not recent_metrics:
                return

            # Analyze metrics by component and operation
            component_metrics = defaultdict(list)
            for metric in recent_metrics:
                key = f"{metric.component}:{metric.operation}"
                component_metrics[key].append(metric)

            # Detect anomalies and patterns
            for key, metrics in component_metrics.items():
                if len(metrics) >= 5:  # Need minimum data points
                    await self._analyze_metric_pattern(key, metrics)

        except Exception as e:
            self.logger.error("Error processing recent metrics",
                            context={"error": str(e)})

    async def _analyze_metric_pattern(self, key: str, metrics: List[UploadMetric]):
        """Analyze patterns in metrics for a specific component/operation"""
        try:
            # Extract values for analysis
            values = [m.value for m in metrics]
            timestamps = [m.timestamp for m in metrics]

            # Calculate basic statistics
            mean_val = statistics.mean(values)
            median_val = statistics.median(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 0

            # Detect anomalies (values outside 2 standard deviations)
            if std_dev > 0:
                threshold_high = mean_val + (2 * std_dev)
                threshold_low = mean_val - (2 * std_dev)

                anomalies = [
                    (i, m) for i, m in enumerate(metrics)
                    if m.value > threshold_high or m.value < threshold_low
                ]

                if anomalies:
                    self.logger.warning("Metric anomalies detected",
                                      context={
                                          "component_operation": key,
                                          "anomaly_count": len(anomalies),
                                          "mean": mean_val,
                                          "std_dev": std_dev,
                                          "threshold_high": threshold_high,
                                          "threshold_low": threshold_low
                                      })

                    # Create alert for significant anomalies
                    if len(anomalies) >= 3:  # Multiple anomalies
                        if self.alert_manager:
                            self.alert_manager.create_alert(
                                rule_id="metric_anomalies",
                                title=f"Metric Anomalies: {key}",
                                message=f"Detected {len(anomalies)} anomalies in {key} metrics",
                                severity=AlertSeverity.WARNING,
                                metadata={
                                    "component_operation": key,
                                    "anomaly_count": len(anomalies),
                                    "mean_value": mean_val,
                                    "std_dev": std_dev
                                }
                            )

        except Exception as e:
            self.logger.error("Error analyzing metric pattern",
                            context={"key": key, "error": str(e)})

    async def _update_success_rate_tracking(self):
        """Update upload success rate tracking and analysis"""
        try:
            # Analyze success rates by component and operation
            with self._metrics_lock:
                success_metrics = []
                for metric_list in self.upload_metrics.values():
                    success_metrics.extend([
                        metric for metric in metric_list
                        if "success" in metric.operation.lower() or "success_rate" in metric.tags
                    ])

            if not success_metrics:
                return

            # Group by component
            component_success = defaultdict(list)
            for metric in success_metrics:
                component_success[metric.component].append(metric.value)

            # Calculate success rates
            for component, values in component_success.items():
                if values:
                    success_rate = sum(values) / len(values)

                    # Compare against baseline
                    baseline_key = f"{component}_success_rate"
                    baseline = self.quality_baselines.get(baseline_key, 0.95)

                    if success_rate < baseline * 0.8:  # 20% below baseline
                        self.logger.warning("Low success rate detected",
                                          context={
                                              "component": component,
                                              "success_rate": success_rate,
                                              "baseline": baseline
                                          })

                        # Create alert for low success rate
                        if self.alert_manager:
                            self.alert_manager.create_alert(
                                rule_id="low_success_rate",
                                title=f"Low Success Rate: {component}",
                                message=f"Success rate {success_rate:.1%} is below baseline {baseline:.1%}",
                                severity=AlertSeverity.ERROR,
                                metadata={
                                    "component": component,
                                    "success_rate": success_rate,
                                    "baseline": baseline
                                }
                            )

        except Exception as e:
            self.logger.error("Error updating success rate tracking",
                            context={"error": str(e)})

    async def _generate_analytics_summary(self):
        """Generate comprehensive analytics summary"""
        try:
            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "total_metrics": sum(len(metrics) for metrics in self.upload_metrics.values()),
                "components_tracked": len(self.upload_metrics),
                "time_ranges": {}
            }

            # Generate summaries for different time ranges
            for range_name, seconds in self.analysis_intervals.items():
                cutoff_time = datetime.utcnow() - timedelta(seconds=seconds)
                recent_metrics = []

                for metric_list in self.upload_metrics.values():
                    recent_metrics.extend([
                        metric for metric in metric_list
                        if metric.timestamp > cutoff_time
                    ])

                if recent_metrics:
                    summary["time_ranges"][range_name] = {
                        "metrics_count": len(recent_metrics),
                        "components_active": len(set(m.component for m in recent_metrics)),
                        "operations_tracked": len(set(m.operation for m in recent_metrics))
                    }

            # Store summary for API access
            self._store_analytics_summary(summary)

        except Exception as e:
            self.logger.error("Error generating analytics summary",
                            context={"error": str(e)})

    def _store_analytics_summary(self, summary: Dict[str, Any]):
        """Store analytics summary in Redis for API access"""
        try:
            key = "upload_analytics:summary"
            self.redis.setex(key, 3600, json.dumps(summary))  # 1 hour expiry
        except Exception as e:
            self.logger.error("Error storing analytics summary",
                            context={"error": str(e)})

    async def _trend_analysis_loop(self):
        """Analyze trends in upload metrics"""
        while self._running:
            try:
                # Analyze trends for all tracked metrics
                await self._analyze_all_trends()

            except Exception as e:
                self.logger.error("Error in trend analysis loop",
                                context={"error": str(e)})

            await asyncio.sleep(300)  # Analyze trends every 5 minutes

    async def _analyze_all_trends(self):
        """Analyze trends for all tracked component/operation combinations"""
        try:
            with self._metrics_lock:
                keys_to_analyze = list(self.upload_metrics.keys())

            for key in keys_to_analyze:
                try:
                    # Get historical data for trend analysis
                    history = list(self.metric_history.get(key, []))
                    if len(history) < self.min_data_points_for_trend:
                        continue

                    # Analyze trend for different time periods
                    for period_name, period_seconds in self.analysis_intervals.items():
                        cutoff_time = datetime.utcnow() - timedelta(seconds=period_seconds)

                        period_data = [
                            h for h in history
                            if datetime.fromisoformat(h["timestamp"]) > cutoff_time
                        ]

                        if len(period_data) >= 5:  # Minimum data points for trend
                            trend = await self._calculate_trend(key, period_data, period_name)
                            if trend:
                                with self._trend_lock:
                                    cache_key = f"{key}:{period_name}"
                                    self.trend_cache[cache_key] = trend

                except Exception as e:
                    self.logger.error("Error analyzing trend for key",
                                    context={"key": key, "error": str(e)})

        except Exception as e:
            self.logger.error("Error in trend analysis",
                            context={"error": str(e)})

    async def _calculate_trend(self, key: str, data: List[Dict], period: str) -> Optional[UploadTrend]:
        """Calculate trend for a specific metric and time period"""
        try:
            if len(data) < 5:
                return None

            # Extract values and timestamps
            values = [d["value"] for d in data]
            timestamps = [datetime.fromisoformat(d["timestamp"]) for d in data]

            # Calculate trend using linear regression
            x = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
            y = values

            if len(x) < 2 or len(set(x)) < 2:
                return None

            # Simple linear regression
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(xi * yi for xi, yi in zip(x, y))
            sum_x2 = sum(xi * xi for xi in x)

            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                return None

            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n

            # Calculate trend statistics
            first_value = values[0]
            last_value = values[-1]
            change = last_value - first_value
            change_percent = (change / first_value) * 100 if first_value != 0 else 0

            # Determine trend direction and strength
            if abs(slope) < 0.01:  # Very small slope
                trend_direction = TrendDirection.STABLE
                trend_strength = 0.0
            else:
                # Use coefficient of determination (R²) for confidence
                y_mean = sum_y / n
                ss_tot = sum((yi - y_mean) ** 2 for yi in y)
                ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))

                if ss_tot == 0:
                    confidence = 0.0
                else:
                    r_squared = 1 - (ss_res / ss_tot)
                    confidence = min(1.0, r_squared)

                # Determine direction based on slope
                if slope > 0.01:
                    trend_direction = TrendDirection.IMPROVING if "error" in key.lower() or "failure" in key.lower() else TrendDirection.DEGRADING
                else:
                    trend_direction = TrendDirection.DEGRADING if "error" in key.lower() or "failure" in key.lower() else TrendDirection.IMPROVING

                trend_strength = min(1.0, abs(slope) * 100)

            # Get baseline (average of older data)
            older_data = data[:len(data)//2] if len(data) > 10 else data
            baseline_value = sum(d["value"] for d in older_data) / len(older_data) if older_data else last_value

            return UploadTrend(
                metric_name=key.split(":")[-1],  # Extract metric name from key
                component=key.split(":")[0],     # Extract component from key
                time_period=period,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                current_value=last_value,
                baseline_value=baseline_value,
                change_percent=change_percent,
                confidence=confidence,
                data_points=len(data),
                analysis_timestamp=datetime.utcnow()
            )

        except Exception as e:
            self.logger.error("Error calculating trend",
                            context={"key": key, "period": period, "error": str(e)})
            return None

    async def _bottleneck_detection_loop(self):
        """Detect bottlenecks in upload operations"""
        while self._running:
            try:
                # Analyze current system state for bottlenecks
                await self._detect_current_bottlenecks()

                # Analyze historical data for recurring bottlenecks
                await self._analyze_recurring_bottlenecks()

            except Exception as e:
                self.logger.error("Error in bottleneck detection loop",
                                context={"error": str(e)})

            await asyncio.sleep(120)  # Detect bottlenecks every 2 minutes

    async def _detect_current_bottlenecks(self):
        """Detect current bottlenecks based on recent metrics"""
        try:
            current_time = datetime.utcnow()
            cutoff_time = current_time - timedelta(minutes=10)

            with self._metrics_lock:
                recent_metrics = []
                for metric_list in self.upload_metrics.values():
                    recent_metrics.extend([
                        metric for metric in metric_list
                        if metric.timestamp > cutoff_time
                    ])

            if not recent_metrics:
                return

            # Group metrics by type for bottleneck analysis
            bottleneck_candidates = {}

            for metric in recent_metrics:
                metric_name = f"{metric.component}:{metric.operation}"

                if metric_name not in bottleneck_candidates:
                    bottleneck_candidates[metric_name] = []

                bottleneck_candidates[metric_name].append(metric)

            # Analyze each candidate for bottleneck conditions
            for metric_name, metrics in bottleneck_candidates.items():
                if len(metrics) >= 3:  # Need multiple data points
                    avg_value = sum(m.value for m in metrics) / len(metrics)

                    # Check against bottleneck thresholds
                    for threshold_key, threshold_value in self.bottleneck_thresholds.items():
                        if threshold_key in metric_name.lower():
                            if avg_value > threshold_value:
                                await self._create_bottleneck_analysis(
                                    metric_name, "threshold_exceeded",
                                    f"Metric {metric_name} exceeds threshold {threshold_value}",
                                    avg_value, threshold_value
                                )

        except Exception as e:
            self.logger.error("Error detecting current bottlenecks",
                            context={"error": str(e)})

    async def _create_bottleneck_analysis(self, metric_name: str, bottleneck_type: str,
                                        root_cause: str, current_value: float,
                                        threshold_value: float):
        """Create bottleneck analysis"""
        try:
            # Determine severity based on how much threshold is exceeded
            exceedance_ratio = current_value / threshold_value

            if exceedance_ratio > 2.0:
                severity = "critical"
            elif exceedance_ratio > 1.5:
                severity = "high"
            elif exceedance_ratio > 1.2:
                severity = "medium"
            else:
                severity = "low"

            # Generate recommendations
            recommendations = self._generate_bottleneck_recommendations(
                metric_name, bottleneck_type, current_value, threshold_value
            )

            # Create bottleneck analysis
            bottleneck = BottleneckAnalysis(
                bottleneck_id=str(uuid.uuid4()),
                component=metric_name.split(":")[0],
                bottleneck_type=bottleneck_type,
                severity=severity,
                impact_score=min(1.0, exceedance_ratio / 2.0),
                affected_operations=[metric_name.split(":")[1] if ":" in metric_name else "unknown"],
                root_cause=root_cause,
                recommendations=recommendations,
                detected_at=datetime.utcnow()
            )

            with self._bottleneck_lock:
                self.bottlenecks[bottleneck.bottleneck_id] = bottleneck
                self.bottleneck_history.append(bottleneck)

                # Limit history size
                if len(self.bottleneck_history) > 1000:
                    self.bottleneck_history = self.bottleneck_history[-500:]

            self.logger.warning("Bottleneck detected",
                              context={
                                  "bottleneck_id": bottleneck.bottleneck_id,
                                  "component": bottleneck.component,
                                  "severity": severity,
                                  "current_value": current_value,
                                  "threshold": threshold_value
                              })

            # Create alert for significant bottlenecks
            if severity in ["high", "critical"]:
                if self.alert_manager:
                    self.alert_manager.create_alert(
                        rule_id="bottleneck_detected",
                        title=f"Bottleneck Detected: {bottleneck.component}",
                        message=f"{bottleneck_type}: {root_cause}",
                        severity=AlertSeverity.ERROR if severity == "critical" else AlertSeverity.WARNING,
                        metadata={
                            "bottleneck_id": bottleneck.bottleneck_id,
                            "severity": severity,
                            "current_value": current_value,
                            "threshold": threshold_value,
                            "recommendations": recommendations
                        }
                    )

        except Exception as e:
            self.logger.error("Error creating bottleneck analysis",
                            context={"metric_name": metric_name, "error": str(e)})

    def _generate_bottleneck_recommendations(self, metric_name: str, bottleneck_type: str,
                                          current_value: float, threshold_value: float) -> List[str]:
        """Generate recommendations for bottleneck resolution"""
        recommendations = []

        if "cpu" in metric_name.lower():
            recommendations.extend([
                "Consider scaling up CPU resources",
                "Optimize CPU-intensive operations",
                "Implement CPU usage monitoring and alerts",
                "Review and optimize algorithm efficiency"
            ])
        elif "memory" in metric_name.lower():
            recommendations.extend([
                "Increase available memory",
                "Implement memory cleanup routines",
                "Review memory leak potential",
                "Optimize memory usage patterns"
            ])
        elif "disk" in metric_name.lower():
            recommendations.extend([
                "Clean up temporary files",
                "Implement disk space monitoring",
                "Consider archiving old data",
                "Optimize file storage patterns"
            ])
        elif "network" in metric_name.lower():
            recommendations.extend([
                "Check network connectivity",
                "Optimize network request patterns",
                "Implement connection pooling",
                "Consider CDN or caching strategies"
            ])
        elif "queue" in metric_name.lower():
            recommendations.extend([
                "Scale up worker processes",
                "Optimize job processing logic",
                "Implement job prioritization",
                "Consider load balancing"
            ])

        return recommendations

    async def _analyze_recurring_bottlenecks(self):
        """Analyze historical data for recurring bottleneck patterns"""
        try:
            with self._bottleneck_lock:
                if len(self.bottleneck_history) < 10:
                    return

                # Group bottlenecks by component and type
                component_bottlenecks = defaultdict(list)
                for bottleneck in self.bottleneck_history[-100:]:  # Last 100 bottlenecks
                    key = f"{bottleneck.component}:{bottleneck.bottleneck_type}"
                    component_bottlenecks[key].append(bottleneck)

                # Identify recurring patterns
                for key, bottlenecks in component_bottlenecks.items():
                    if len(bottlenecks) >= 5:  # Recurring bottleneck
                        # Check if this is a chronic issue
                        recent_bottlenecks = [
                            b for b in bottlenecks
                            if b.detected_at > datetime.utcnow() - timedelta(hours=24)
                        ]

                        if len(recent_bottlenecks) >= 3:
                            self.logger.warning("Recurring bottleneck pattern detected",
                                              context={
                                                  "pattern": key,
                                                  "total_occurrences": len(bottlenecks),
                                                  "recent_occurrences": len(recent_bottlenecks)
                                              })

                            # Create alert for recurring bottlenecks
                            if self.alert_manager:
                                self.alert_manager.create_alert(
                                    rule_id="recurring_bottleneck",
                                    title=f"Recurring Bottleneck: {key}",
                                    message=f"Pattern {key} has occurred {len(recent_bottlenecks)} times in 24h",
                                    severity=AlertSeverity.ERROR,
                                    metadata={
                                        "pattern": key,
                                        "total_occurrences": len(bottlenecks),
                                        "recent_occurrences": len(recent_bottlenecks)
                                    }
                                )

        except Exception as e:
            self.logger.error("Error analyzing recurring bottlenecks",
                            context={"error": str(e)})

    async def _quality_assessment_loop(self):
        """Assess upload operation quality"""
        while self._running:
            try:
                # Assess overall upload quality
                await self._assess_upload_quality()

                # Update quality baselines
                await self._update_quality_baselines()

            except Exception as e:
                self.logger.error("Error in quality assessment loop",
                                context={"error": str(e)})

            await asyncio.sleep(300)  # Assess quality every 5 minutes

    async def _assess_upload_quality(self):
        """Assess overall upload operation quality"""
        try:
            # Calculate quality scores for different dimensions
            quality_scores = {}

            # Success rate quality
            success_rate = await self._calculate_success_rate_quality()
            quality_scores["success_rate"] = success_rate

            # Performance quality
            performance_quality = await self._calculate_performance_quality()
            quality_scores["performance"] = performance_quality

            # Reliability quality
            reliability_quality = await self._calculate_reliability_quality()
            quality_scores["reliability"] = reliability_quality

            # Calculate overall quality score
            weights = {
                "success_rate": 0.4,
                "performance": 0.3,
                "reliability": 0.3
            }

            overall_quality = sum(
                quality_scores.get(dim, 0) * weight
                for dim, weight in weights.items()
            )

            # Store quality metrics
            for dimension, score in quality_scores.items():
                metric_id = str(uuid.uuid4())
                quality_metric = QualityMetric(
                    metric_id=metric_id,
                    job_id="system",
                    quality_dimension=dimension,
                    score=score,
                    max_score=1.0,
                    weight=weights.get(dimension, 1.0),
                    details={"calculated_at": datetime.utcnow().isoformat()},
                    assessed_at=datetime.utcnow()
                )

                with self._quality_lock:
                    self.quality_metrics[dimension].append(quality_metric)

                    # Limit stored metrics
                    if len(self.quality_metrics[dimension]) > 100:
                        self.quality_metrics[dimension] = self.quality_metrics[dimension][-50:]

            # Check if quality is below acceptable thresholds
            if overall_quality < 0.7:  # Below 70% quality
                self.logger.warning("Low overall upload quality detected",
                                  context={"overall_quality": overall_quality})

                if self.alert_manager:
                    self.alert_manager.create_alert(
                        rule_id="low_upload_quality",
                        title="Low Upload Quality",
                        message=f"Overall upload quality is {overall_quality:.1%}, below 70% threshold",
                        severity=AlertSeverity.WARNING,
                        metadata={
                            "overall_quality": overall_quality,
                            "quality_scores": quality_scores
                        }
                    )

        except Exception as e:
            self.logger.error("Error assessing upload quality",
                            context={"error": str(e)})

    async def _calculate_success_rate_quality(self) -> float:
        """Calculate success rate quality score"""
        try:
            # Get recent success rate metrics
            with self._metrics_lock:
                success_metrics = []
                for metric_list in self.upload_metrics.values():
                    success_metrics.extend([
                        metric for metric in metric_list
                        if "success" in metric.operation.lower()
                    ])

            if not success_metrics:
                return 0.5  # Neutral score if no data

            # Calculate average success rate
            recent_metrics = [
                m for m in success_metrics
                if m.timestamp > datetime.utcnow() - timedelta(hours=1)
            ]

            if not recent_metrics:
                return 0.5

            avg_success_rate = sum(m.value for m in recent_metrics) / len(recent_metrics)

            # Convert to quality score (0.0 to 1.0)
            # 95%+ success rate = 1.0 quality
            # 80% success rate = 0.5 quality
            # Below 80% = 0.0 quality
            if avg_success_rate >= 0.95:
                return 1.0
            elif avg_success_rate >= 0.80:
                return (avg_success_rate - 0.80) / 0.15
            else:
                return 0.0

        except Exception as e:
            self.logger.error("Error calculating success rate quality",
                            context={"error": str(e)})
            return 0.5

    async def _calculate_performance_quality(self) -> float:
        """Calculate performance quality score"""
        try:
            # Get recent performance metrics
            with self._metrics_lock:
                performance_metrics = []
                for metric_list in self.upload_metrics.values():
                    performance_metrics.extend([
                        metric for metric in metric_list
                        if "time" in metric.operation.lower() or "latency" in metric.operation.lower()
                    ])

            if not performance_metrics:
                return 0.5

            # Calculate average performance
            recent_metrics = [
                m for m in performance_metrics
                if m.timestamp > datetime.utcnow() - timedelta(hours=1)
            ]

            if not recent_metrics:
                return 0.5

            # For time-based metrics, lower is better
            # Normalize to quality score (assuming lower time = higher quality)
            avg_time = sum(m.value for m in recent_metrics) / len(recent_metrics)

            # Assume 30 seconds is excellent, 5 minutes is poor
            if avg_time <= 30:
                return 1.0
            elif avg_time <= 300:
                return 1.0 - ((avg_time - 30) / 270)
            else:
                return 0.0

        except Exception as e:
            self.logger.error("Error calculating performance quality",
                            context={"error": str(e)})
            return 0.5

    async def _calculate_reliability_quality(self) -> float:
        """Calculate reliability quality score"""
        try:
            # Use bottleneck frequency as reliability indicator
            with self._bottleneck_lock:
                recent_bottlenecks = [
                    b for b in self.bottleneck_history
                    if b.detected_at > datetime.utcnow() - timedelta(hours=1)
                ]

            # Fewer bottlenecks = higher reliability
            # 0 bottlenecks = 1.0 reliability
            # 5+ bottlenecks = 0.0 reliability
            bottleneck_count = len(recent_bottlenecks)

            if bottleneck_count == 0:
                return 1.0
            elif bottleneck_count <= 5:
                return 1.0 - (bottleneck_count / 5.0)
            else:
                return 0.0

        except Exception as e:
            self.logger.error("Error calculating reliability quality",
                            context={"error": str(e)})
            return 0.5

    async def _update_quality_baselines(self):
        """Update quality metric baselines based on recent performance"""
        try:
            # Update baselines based on recent quality metrics
            for dimension in self.quality_baselines.keys():
                with self._quality_lock:
                    dimension_metrics = self.quality_metrics.get(dimension, [])

                    if len(dimension_metrics) >= 10:  # Need sufficient data
                        recent_metrics = [
                            m for m in dimension_metrics
                            if m.assessed_at > datetime.utcnow() - timedelta(hours=24)
                        ]

                        if recent_metrics:
                            avg_score = sum(m.score for m in recent_metrics) / len(recent_metrics)

                            # Update baseline if it's significantly different
                            current_baseline = self.quality_baselines[dimension]
                            if abs(avg_score - current_baseline) > 0.1:  # 10% difference
                                self.quality_baselines[dimension] = avg_score

                                self.logger.info("Quality baseline updated",
                                               context={
                                                   "dimension": dimension,
                                                   "old_baseline": current_baseline,
                                                   "new_baseline": avg_score
                                               })

        except Exception as e:
            self.logger.error("Error updating quality baselines",
                            context={"error": str(e)})

    def get_upload_trends(self, component: Optional[str] = None,
                         time_period: Optional[str] = None) -> Dict[str, Any]:
        """Get upload operation trends"""
        with self._trend_lock:
            trends = []

            for cache_key, trend in self.trend_cache.items():
                # Filter by component if specified
                if component and trend.component != component:
                    continue

                # Filter by time period if specified
                if time_period and trend.time_period != time_period:
                    continue

                trends.append({
                    "metric_name": trend.metric_name,
                    "component": trend.component,
                    "time_period": trend.time_period,
                    "trend_direction": trend.trend_direction.value,
                    "trend_strength": trend.trend_strength,
                    "current_value": trend.current_value,
                    "baseline_value": trend.baseline_value,
                    "change_percent": trend.change_percent,
                    "confidence": trend.confidence,
                    "data_points": trend.data_points,
                    "analysis_timestamp": trend.analysis_timestamp.isoformat()
                })

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "total_trends": len(trends),
                "trends": sorted(trends, key=lambda t: t["confidence"], reverse=True)
            }

    def get_bottleneck_analysis(self, active_only: bool = True) -> Dict[str, Any]:
        """Get bottleneck analysis results"""
        with self._bottleneck_lock:
            if active_only:
                bottlenecks = [
                    b for b in self.bottlenecks.values()
                    if b.resolved_at is None
                ]
            else:
                bottlenecks = list(self.bottlenecks.values())

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "total_bottlenecks": len(bottlenecks),
                "active_bottlenecks": len([b for b in bottlenecks if b.resolved_at is None]),
                "bottlenecks": [
                    {
                        "bottleneck_id": b.bottleneck_id,
                        "component": b.component,
                        "bottleneck_type": b.bottleneck_type,
                        "severity": b.severity,
                        "impact_score": b.impact_score,
                        "affected_operations": b.affected_operations,
                        "root_cause": b.root_cause,
                        "recommendations": b.recommendations,
                        "detected_at": b.detected_at.isoformat(),
                        "resolved_at": b.resolved_at.isoformat() if b.resolved_at else None
                    }
                    for b in sorted(bottlenecks, key=lambda b: b.impact_score, reverse=True)
                ]
            }

    def get_quality_metrics(self, dimension: Optional[str] = None) -> Dict[str, Any]:
        """Get quality metrics"""
        with self._quality_lock:
            if dimension:
                metrics = self.quality_metrics.get(dimension, [])
            else:
                # Get all dimensions
                all_metrics = []
                for dim_metrics in self.quality_metrics.values():
                    all_metrics.extend(dim_metrics)
                metrics = sorted(all_metrics, key=lambda m: m.assessed_at, reverse=True)[:100]

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "quality_baselines": self.quality_baselines,
                "metrics": [
                    {
                        "metric_id": m.metric_id,
                        "quality_dimension": m.quality_dimension,
                        "score": m.score,
                        "max_score": m.max_score,
                        "weight": m.weight,
                        "details": m.details,
                        "assessed_at": m.assessed_at.isoformat()
                    }
                    for m in metrics
                ]
            }

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        try:
            # Get recent metrics summary
            with self._metrics_lock:
                total_metrics = sum(len(metrics) for metrics in self.upload_metrics.values())

                # Get metrics from last hour
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                recent_metrics = []
                for metric_list in self.upload_metrics.values():
                    recent_metrics.extend([
                        metric for metric in metric_list
                        if metric.timestamp > cutoff_time
                    ])

            # Get trends summary
            trends_summary = self.get_upload_trends()

            # Get bottlenecks summary
            bottlenecks_summary = self.get_bottleneck_analysis(active_only=True)

            # Get quality summary
            quality_summary = self.get_quality_metrics()

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "system_status": "running" if self._running else "stopped",
                "metrics_summary": {
                    "total_metrics": total_metrics,
                    "recent_metrics": len(recent_metrics),
                    "components_tracked": len(self.upload_metrics)
                },
                "trends_summary": {
                    "total_trends": trends_summary["total_trends"],
                    "improving_trends": len([t for t in trends_summary["trends"] if t["trend_direction"] == "improving"]),
                    "degrading_trends": len([t for t in trends_summary["trends"] if t["trend_direction"] == "degrading"])
                },
                "bottlenecks_summary": {
                    "total_bottlenecks": bottlenecks_summary["total_bottlenecks"],
                    "active_bottlenecks": bottlenecks_summary["active_bottlenecks"],
                    "critical_bottlenecks": len([b for b in bottlenecks_summary["bottlenecks"] if b["severity"] == "critical"])
                },
                "quality_summary": {
                    "overall_quality_score": sum(q["score"] for q in quality_summary["metrics"][:10]) / min(10, len(quality_summary["metrics"])) if quality_summary["metrics"] else 0.5,
                    "quality_dimensions": len(self.quality_baselines)
                }
            }

        except Exception as e:
            self.logger.error("Error generating analytics summary",
                            context={"error": str(e)})
            return {"error": str(e)}