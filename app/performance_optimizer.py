# app/performance_optimizer.py
import asyncio
import json
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from .resource_manager import get_resource_manager, ResourceType
from .job_scheduler import get_scheduler
from .worker_scaler import get_worker_manager
from .monitoring import get_monitor, StructuredLogger


class BottleneckType(Enum):
    CPU_BOUND = "cpu_bound"
    MEMORY_BOUND = "memory_bound"
    IO_BOUND = "io_bound"
    NETWORK_BOUND = "network_bound"
    QUEUE_CONGESTION = "queue_congestion"
    WORKER_SHORTAGE = "worker_shortage"
    RESOURCE_CONTENTION = "resource_contention"


class OptimizationType(Enum):
    WORKER_SCALING = "worker_scaling"
    RESOURCE_ALLOCATION = "resource_allocation"
    JOB_BATCHING = "job_batching"
    CACHE_OPTIMIZATION = "cache_optimization"
    IO_OPTIMIZATION = "io_optimization"
    NETWORK_OPTIMIZATION = "network_optimization"


@dataclass
class PerformanceMetric:
    name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    threshold: Optional[float] = None


@dataclass
class BottleneckDetection:
    bottleneck_type: BottleneckType
    severity: float  # 0.0 to 1.0
    location: str  # component/system causing bottleneck
    description: str
    timestamp: datetime
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class OptimizationRecommendation:
    optimization_type: OptimizationType
    priority: int  # 1 (highest) to 5 (lowest)
    description: str
    expected_impact: str
    implementation_effort: str  # "low", "medium", "high"
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestConfig:
    test_id: str
    name: str
    description: str
    control_group: Dict[str, Any]
    treatment_group: Dict[str, Any]
    metrics_to_track: List[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "active"  # "active", "completed", "cancelled"


class PerformanceAnalyzer:
    """Analyzes system performance and detects bottlenecks"""

    def __init__(self):
        self.logger = StructuredLogger("performance_analyzer")

        # Performance tracking
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.bottleneck_history: deque = deque(maxlen=100)
        self.optimization_history: deque = deque(maxlen=50)

        # Analysis configuration
        self.analysis_interval = 60  # seconds
        self.bottleneck_threshold = 0.7  # 70% utilization = potential bottleneck
        self.trend_window = 10  # number of data points for trend analysis

        # Locks
        self._analysis_lock = threading.Lock()

    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        with self._analysis_lock:
            self.metrics_history[metric.name].append(metric)

    def detect_bottlenecks(self) -> List[BottleneckDetection]:
        """Analyze current system state and detect bottlenecks"""
        bottlenecks = []

        try:
            # Get current system state
            resource_manager = get_resource_manager()
            scheduler = get_scheduler()
            worker_manager = get_worker_manager()

            resource_status = resource_manager.get_system_resource_status()
            scheduler_status = scheduler.get_scheduler_status()
            worker_status = worker_manager.get_worker_status()

            # Analyze CPU bottlenecks
            cpu_bottleneck = self._analyze_cpu_bottleneck(resource_status)
            if cpu_bottleneck:
                bottlenecks.append(cpu_bottleneck)

            # Analyze memory bottlenecks
            memory_bottleneck = self._analyze_memory_bottleneck(resource_status)
            if memory_bottleneck:
                bottlenecks.append(memory_bottleneck)

            # Analyze queue congestion
            queue_bottleneck = self._analyze_queue_congestion(scheduler_status)
            if queue_bottleneck:
                bottlenecks.append(queue_bottleneck)

            # Analyze worker utilization
            worker_bottleneck = self._analyze_worker_utilization(worker_status)
            if worker_bottleneck:
                bottlenecks.append(worker_bottleneck)

            # Analyze I/O bottlenecks
            io_bottleneck = self._analyze_io_bottleneck()
            if io_bottleneck:
                bottlenecks.append(io_bottleneck)

        except Exception as e:
            self.logger.error("Error detecting bottlenecks",
                            context={"error": str(e)})

        return bottlenecks

    def _analyze_cpu_bottleneck(self, resource_status: Dict[str, Any]) -> Optional[BottleneckDetection]:
        """Analyze CPU utilization for bottlenecks"""
        cpu_info = resource_status.get("cpu", {})
        cpu_usage = cpu_info.get("current_usage_percent", 0)

        if cpu_usage > 85:  # High CPU usage
            severity = min(1.0, cpu_usage / 100.0)

            return BottleneckDetection(
                bottleneck_type=BottleneckType.CPU_BOUND,
                severity=severity,
                location="system_cpu",
                description=f"High CPU utilization: {cpu_usage:.1f}%",
                timestamp=datetime.utcnow(),
                evidence={"cpu_usage_percent": cpu_usage, "threshold": 85},
                recommendations=[
                    "Consider scaling up workers to distribute CPU load",
                    "Optimize CPU-intensive operations in the pipeline",
                    "Check for inefficient algorithms or loops"
                ]
            )

        return None

    def _analyze_memory_bottleneck(self, resource_status: Dict[str, Any]) -> Optional[BottleneckDetection]:
        """Analyze memory utilization for bottlenecks"""
        memory_info = resource_status.get("memory", {})
        memory_usage = memory_info.get("usage_percent", 0)
        pool_utilization = memory_info.get("pool_utilization_percent", 0)

        if memory_usage > 80 or pool_utilization > 85:
            severity = max(memory_usage / 100.0, pool_utilization / 100.0)

            return BottleneckDetection(
                bottleneck_type=BottleneckType.MEMORY_BOUND,
                severity=severity,
                location="system_memory",
                description=f"High memory utilization: {memory_usage:.1f}% (pool: {pool_utilization:.1f}%)",
                timestamp=datetime.utcnow(),
                evidence={
                    "memory_usage_percent": memory_usage,
                    "pool_utilization_percent": pool_utilization,
                    "threshold": 80
                },
                recommendations=[
                    "Increase memory pool size if possible",
                    "Optimize memory usage in audio processing",
                    "Consider memory-efficient algorithms",
                    "Check for memory leaks in the pipeline"
                ]
            )

        return None

    def _analyze_queue_congestion(self, scheduler_status: Dict[str, Any]) -> Optional[BottleneckDetection]:
        """Analyze job queue for congestion"""
        queued_jobs = scheduler_status.get("queued_jobs", 0)
        running_jobs = scheduler_status.get("running_jobs", 0)
        total_jobs = queued_jobs + running_jobs

        if total_jobs > 0:
            # Calculate queue latency (based on scheduler metrics)
            conflict_rate = scheduler_status.get("performance", {}).get("conflict_rate", 0)

            if queued_jobs > 20 or conflict_rate > 0.3:  # High queue or high conflict rate
                severity = min(1.0, (queued_jobs / 50.0) + (conflict_rate * 2))

                return BottleneckDetection(
                    bottleneck_type=BottleneckType.QUEUE_CONGESTION,
                    severity=severity,
                    location="job_scheduler",
                    description=f"Queue congestion: {queued_jobs} queued jobs, {conflict_rate:.2f} conflict rate",
                    timestamp=datetime.utcnow(),
                    evidence={
                        "queued_jobs": queued_jobs,
                        "conflict_rate": conflict_rate,
                        "threshold_queue": 20,
                        "threshold_conflict": 0.3
                    },
                    recommendations=[
                        "Scale up workers to process queued jobs",
                        "Optimize job scheduling algorithm",
                        "Consider job batching for similar operations",
                        "Check for stuck or long-running jobs"
                    ]
                )

        return None

    def _analyze_worker_utilization(self, worker_status: Dict[str, Any]) -> Optional[BottleneckDetection]:
        """Analyze worker utilization for bottlenecks"""
        active_workers = worker_status.get("active_workers", 0)
        total_workers = worker_status.get("total_workers", 0)

        if total_workers > 0:
            utilization_rate = active_workers / total_workers

            if utilization_rate < 0.5 and worker_status.get("performance", {}).get("total_jobs_processed", 0) > 0:
                # Low worker utilization might indicate worker shortage or scheduling issues
                return BottleneckDetection(
                    bottleneck_type=BottleneckType.WORKER_SHORTAGE,
                    severity=0.8,  # High severity as it indicates capacity issues
                    location="worker_pool",
                    description=f"Low worker utilization: {utilization_rate:.2f} ({active_workers}/{total_workers})",
                    timestamp=datetime.utcnow(),
                    evidence={
                        "active_workers": active_workers,
                        "total_workers": total_workers,
                        "utilization_rate": utilization_rate,
                        "threshold": 0.5
                    },
                    recommendations=[
                        "Scale up workers to improve throughput",
                        "Check worker health and restart failed workers",
                        "Optimize job distribution across workers",
                        "Review resource allocation settings"
                    ]
                )

        return None

    def _analyze_io_bottleneck(self) -> Optional[BottleneckDetection]:
        """Analyze I/O operations for bottlenecks"""
        try:
            # Get disk I/O statistics
            disk_io = psutil.disk_io_counters()

            if disk_io:
                # Calculate I/O utilization (simplified)
                # In a real implementation, you'd want more sophisticated I/O monitoring
                read_rate = disk_io.read_bytes / (1024 * 1024)  # MB/s
                write_rate = disk_io.write_bytes / (1024 * 1024)  # MB/s

                # Simple heuristic: high I/O rates might indicate bottlenecks
                if read_rate > 100 or write_rate > 100:  # 100 MB/s threshold
                    severity = min(1.0, (read_rate + write_rate) / 200.0)

                    return BottleneckDetection(
                        bottleneck_type=BottleneckType.IO_BOUND,
                        severity=severity,
                        location="disk_io",
                        description=f"High I/O activity: {read_rate:.1f} MB/s read, {write_rate:.1f} MB/s write",
                        timestamp=datetime.utcnow(),
                        evidence={
                            "read_rate_mbps": read_rate,
                            "write_rate_mbps": write_rate,
                            "threshold": 100
                        },
                        recommendations=[
                            "Optimize file I/O operations",
                            "Consider using faster storage",
                            "Implement I/O batching and caching",
                            "Check for unnecessary file operations"
                        ]
                    )

        except Exception as e:
            self.logger.warning("Error analyzing I/O bottleneck",
                              context={"error": str(e)})

        return None

    def analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        trends = {}

        try:
            with self._analysis_lock:
                for metric_name, metrics in self.metrics_history.items():
                    if len(metrics) >= self.trend_window:
                        values = [m.value for m in list(metrics)[-self.trend_window:]]

                        if values:
                            # Calculate trend statistics
                            mean_value = statistics.mean(values)
                            std_dev = statistics.stdev(values) if len(values) > 1 else 0
                            trend_direction = self._calculate_trend_direction(values)

                            trends[metric_name] = {
                                "current_value": values[-1],
                                "mean_value": mean_value,
                                "std_deviation": std_dev,
                                "trend_direction": trend_direction,
                                "trend_strength": self._calculate_trend_strength(values),
                                "data_points": len(values)
                            }

        except Exception as e:
            self.logger.error("Error analyzing performance trends",
                            context={"error": str(e)})

        return trends


class OptimizationEngine:
    """Generates optimization recommendations based on performance analysis"""

    def __init__(self):
        self.logger = StructuredLogger("optimization_engine")
        self.performance_analyzer = PerformanceAnalyzer()

        # Optimization rules and thresholds
        self.optimization_rules = {
            "worker_scaling": {
                "queue_threshold": 15,
                "cpu_threshold": 80,
                "memory_threshold": 75,
                "min_confidence": 0.7
            },
            "resource_allocation": {
                "memory_efficiency_threshold": 0.6,
                "cpu_efficiency_threshold": 0.7,
                "min_improvement_potential": 0.2
            },
            "job_batching": {
                "similarity_threshold": 0.8,
                "min_batch_size": 3,
                "max_batch_size": 10
            }
        }

    def generate_recommendations(self, bottlenecks: List[BottleneckDetection]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on detected bottlenecks"""
        recommendations = []

        for bottleneck in bottlenecks:
            # Generate recommendations based on bottleneck type
            if bottleneck.bottleneck_type == BottleneckType.CPU_BOUND:
                recs = self._recommend_cpu_optimizations(bottleneck)
                recommendations.extend(recs)

            elif bottleneck.bottleneck_type == BottleneckType.MEMORY_BOUND:
                recs = self._recommend_memory_optimizations(bottleneck)
                recommendations.extend(recs)

            elif bottleneck.bottleneck_type == BottleneckType.QUEUE_CONGESTION:
                recs = self._recommend_queue_optimizations(bottleneck)
                recommendations.extend(recs)

            elif bottleneck.bottleneck_type == BottleneckType.WORKER_SHORTAGE:
                recs = self._recommend_worker_optimizations(bottleneck)
                recommendations.extend(recs)

            elif bottleneck.bottleneck_type == BottleneckType.IO_BOUND:
                recs = self._recommend_io_optimizations(bottleneck)
                recommendations.extend(recs)

        # Add general recommendations based on system analysis
        general_recs = self._generate_general_recommendations()
        recommendations.extend(general_recs)

        # Sort by priority and return top recommendations
        recommendations.sort(key=lambda r: r.priority)

        return recommendations[:10]  # Return top 10 recommendations

    def _recommend_cpu_optimizations(self, bottleneck: BottleneckDetection) -> List[OptimizationRecommendation]:
        """Generate CPU optimization recommendations"""
        recommendations = []

        if bottleneck.severity > 0.8:
            recommendations.append(OptimizationRecommendation(
                optimization_type=OptimizationType.WORKER_SCALING,
                priority=1,
                description="Scale up workers to distribute CPU load across more processes",
                expected_impact="High - Reduce CPU utilization by 30-50%",
                implementation_effort="Low",
                timestamp=datetime.utcnow(),
                metadata={"bottleneck_id": bottleneck.bottleneck_type.value}
            ))

        recommendations.append(OptimizationRecommendation(
            optimization_type=OptimizationType.RESOURCE_ALLOCATION,
            priority=2,
            description="Optimize CPU core allocation for parallel processing tasks",
            expected_impact="Medium - Improve CPU efficiency by 15-25%",
            implementation_effort="Medium",
            timestamp=datetime.utcnow(),
            metadata={"bottleneck_id": bottleneck.bottleneck_type.value}
        ))

        return recommendations

    def _recommend_memory_optimizations(self, bottleneck: BottleneckDetection) -> List[OptimizationRecommendation]:
        """Generate memory optimization recommendations"""
        recommendations = []

        recommendations.append(OptimizationRecommendation(
            optimization_type=OptimizationType.RESOURCE_ALLOCATION,
            priority=1,
            description="Increase memory pool size for audio processing operations",
            expected_impact="High - Reduce memory pressure and improve throughput",
            implementation_effort="Low",
            timestamp=datetime.utcnow(),
            metadata={"bottleneck_id": bottleneck.bottleneck_type.value}
        ))

        recommendations.append(OptimizationRecommendation(
            optimization_type=OptimizationType.CACHE_OPTIMIZATION,
            priority=2,
            description="Implement memory-efficient caching for frequently accessed audio data",
            expected_impact="Medium - Reduce memory usage by 20-30%",
            implementation_effort="Medium",
            timestamp=datetime.utcnow(),
            metadata={"bottleneck_id": bottleneck.bottleneck_type.value}
        ))

        return recommendations

    def _recommend_queue_optimizations(self, bottleneck: BottleneckDetection) -> List[OptimizationRecommendation]:
        """Generate queue optimization recommendations"""
        recommendations = []

        recommendations.append(OptimizationRecommendation(
            optimization_type=OptimizationType.WORKER_SCALING,
            priority=1,
            description="Scale up workers to process queued jobs more quickly",
            expected_impact="High - Reduce queue latency significantly",
            implementation_effort="Low",
            timestamp=datetime.utcnow(),
            metadata={"bottleneck_id": bottleneck.bottleneck_type.value}
        ))

        recommendations.append(OptimizationRecommendation(
            optimization_type=OptimizationType.JOB_BATCHING,
            priority=2,
            description="Implement intelligent job batching for similar operations",
            expected_impact="Medium - Improve throughput by 20-30%",
            implementation_effort="Medium",
            timestamp=datetime.utcnow(),
            metadata={"bottleneck_id": bottleneck.bottleneck_type.value}
        ))

        return recommendations

    def _recommend_worker_optimizations(self, bottleneck: BottleneckDetection) -> List[OptimizationRecommendation]:
        """Generate worker optimization recommendations"""
        recommendations = []

        recommendations.append(OptimizationRecommendation(
            optimization_type=OptimizationType.WORKER_SCALING,
            priority=1,
            description="Scale up workers to improve system capacity and throughput",
            expected_impact="High - Increase processing capacity significantly",
            implementation_effort="Low",
            timestamp=datetime.utcnow(),
            metadata={"bottleneck_id": bottleneck.bottleneck_type.value}
        ))

        return recommendations

    def _recommend_io_optimizations(self, bottleneck: BottleneckDetection) -> List[OptimizationRecommendation]:
        """Generate I/O optimization recommendations"""
        recommendations = []

        recommendations.append(OptimizationRecommendation(
            optimization_type=OptimizationType.IO_OPTIMIZATION,
            priority=2,
            description="Implement I/O batching and asynchronous file operations",
            expected_impact="Medium - Reduce I/O bottlenecks by 25-40%",
            implementation_effort="Medium",
            timestamp=datetime.utcnow(),
            metadata={"bottleneck_id": bottleneck.bottleneck_type.value}
        ))

        return recommendations

    def _generate_general_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate general system optimization recommendations"""
        recommendations = []

        try:
            # Analyze system trends
            trends = self.performance_analyzer.analyze_performance_trends()

            # Check for declining performance trends
            for metric_name, trend_data in trends.items():
                if trend_data["trend_direction"] == "declining" and trend_data["trend_strength"] > 0.7:
                    recommendations.append(OptimizationRecommendation(
                        optimization_type=OptimizationType.RESOURCE_ALLOCATION,
                        priority=3,
                        description=f"Performance declining for {metric_name} - review resource allocation",
                        expected_impact="Medium - Stabilize performance trends",
                        implementation_effort="Medium",
                        timestamp=datetime.utcnow(),
                        metadata={"metric": metric_name, "trend_strength": trend_data["trend_strength"]}
                    ))

        except Exception as e:
            self.logger.error("Error generating general recommendations",
                            context={"error": str(e)})

        return recommendations


class ABTestManager:
    """Manages A/B testing for performance comparisons"""

    def __init__(self):
        self.logger = StructuredLogger("ab_test_manager")
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.test_results: Dict[str, Dict[str, Any]] = {}

        # Test tracking
        self.test_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    def create_test(self, config: ABTestConfig) -> bool:
        """Create a new A/B test"""
        try:
            if config.test_id in self.active_tests:
                self.logger.warning("Test already exists",
                                  context={"test_id": config.test_id})
                return False

            self.active_tests[config.test_id] = config

            self.logger.info("A/B test created",
                           context={"test_id": config.test_id, "name": config.name})

            return True

        except Exception as e:
            self.logger.error("Error creating A/B test",
                            context={"error": str(e), "test_id": config.test_id})
            return False

    def record_test_metric(self, test_id: str, group: str, metric_name: str, value: float):
        """Record a metric value for an A/B test"""
        if test_id not in self.active_tests:
            return

        test_config = self.active_tests[test_id]

        if group not in ["control", "treatment"]:
            self.logger.warning("Invalid test group",
                              context={"test_id": test_id, "group": group})
            return

        if metric_name not in test_config.metrics_to_track:
            return

        # Record the metric
        key = f"{test_id}_{group}"
        if key not in self.test_metrics:
            self.test_metrics[key] = defaultdict(list)

        self.test_metrics[key][metric_name].append(value)

    def complete_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Complete an A/B test and return results"""
        if test_id not in self.active_tests:
            return None

        test_config = self.active_tests[test_id]
        test_config.status = "completed"
        test_config.end_time = datetime.utcnow()

        # Calculate results
        results = self._calculate_test_results(test_id)

        # Store results
        self.test_results[test_id] = results

        # Remove from active tests
        del self.active_tests[test_id]

        self.logger.info("A/B test completed",
                        context={"test_id": test_id, "results": results})

        return results

    def _calculate_test_results(self, test_id: str) -> Dict[str, Any]:
        """Calculate statistical results for a completed test"""
        results = {
            "test_id": test_id,
            "completed_at": datetime.utcnow().isoformat(),
            "metric_results": {}
        }

        try:
            # Get metrics for both groups
            control_key = f"{test_id}_control"
            treatment_key = f"{test_id}_treatment"

            control_metrics = self.test_metrics.get(control_key, {})
            treatment_metrics = self.test_metrics.get(treatment_key, {})

            for metric_name in set(list(control_metrics.keys()) + list(treatment_metrics.keys())):
                control_values = control_metrics.get(metric_name, [])
                treatment_values = treatment_metrics.get(metric_name, [])

                if len(control_values) >= 5 and len(treatment_values) >= 5:  # Minimum sample size
                    # Calculate basic statistics
                    control_mean = statistics.mean(control_values)
                    treatment_mean = statistics.mean(treatment_values)

                    # Simple statistical significance test (t-test approximation)
                    control_std = statistics.stdev(control_values) if len(control_values) > 1 else 0
                    treatment_std = statistics.stdev(treatment_values) if len(treatment_values) > 1 else 0

                    # Calculate improvement
                    improvement = ((treatment_mean - control_mean) / control_mean) * 100 if control_mean > 0 else 0

                    # Simple significance check (p-value approximation)
                    if control_std > 0 and treatment_std > 0:
                        # Calculate t-statistic
                        pooled_std = ((control_std ** 2 + treatment_std ** 2) / 2) ** 0.5
                        t_stat = abs(treatment_mean - control_mean) / (pooled_std * (2 / len(control_values)) ** 0.5)
                        significant = t_stat > 1.96  # 95% confidence
                    else:
                        significant = False

                    results["metric_results"][metric_name] = {
                        "control_mean": control_mean,
                        "treatment_mean": treatment_mean,
                        "improvement_percent": improvement,
                        "statistically_significant": significant,
                        "sample_size_control": len(control_values),
                        "sample_size_treatment": len(treatment_values)
                    }

        except Exception as e:
            self.logger.error("Error calculating test results",
                            context={"error": str(e), "test_id": test_id})

        return results


class PerformanceOptimizer:
    """Main performance optimization system"""

    def __init__(self):
        self.logger = StructuredLogger("performance_optimizer")

        # Core components
        self.performance_analyzer = PerformanceAnalyzer()
        self.optimization_engine = OptimizationEngine()
        self.ab_test_manager = ABTestManager()

        # Background tasks
        self._analysis_task = None
        self._optimization_task = None
        self._running = False

        # Configuration
        self.analysis_interval = 60  # seconds
        self.optimization_interval = 300  # 5 minutes

    async def start_optimization(self):
        """Start the performance optimization system"""
        if self._running:
            return

        self._running = True

        # Start background tasks
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        self._optimization_task = asyncio.create_task(self._optimization_loop())

        self.logger.info("Performance optimization system started")

    async def stop_optimization(self):
        """Stop the performance optimization system"""
        if not self._running:
            return

        self._running = False

        # Cancel background tasks
        if self._analysis_task:
            self._analysis_task.cancel()
        if self._optimization_task:
            self._optimization_task.cancel()

        self.logger.info("Performance optimization system stopped")

    async def _analysis_loop(self):
        """Main analysis loop for bottleneck detection"""
        while self._running:
            try:
                # Detect bottlenecks
                bottlenecks = self.performance_analyzer.detect_bottlenecks()

                # Record bottlenecks for historical analysis
                for bottleneck in bottlenecks:
                    self.performance_analyzer.bottleneck_history.append(bottleneck)

                    # Record bottleneck metrics
                    self.performance_analyzer.record_metric(PerformanceMetric(
                        name=f"bottleneck_{bottleneck.bottleneck_type.value}_severity",
                        value=bottleneck.severity,
                        timestamp=bottleneck.timestamp,
                        metadata={"location": bottleneck.location}
                    ))

                # Update monitoring
                self._update_analysis_metrics(bottlenecks)

            except Exception as e:
                self.logger.error("Error in analysis loop",
                                context={"error": str(e)})

            await asyncio.sleep(self.analysis_interval)

    async def _optimization_loop(self):
        """Main optimization loop for generating recommendations"""
        while self._running:
            try:
                # Get recent bottlenecks
                recent_bottlenecks = list(self.performance_analyzer.bottleneck_history)[-10:]  # Last 10

                # Generate recommendations
                recommendations = self.optimization_engine.generate_recommendations(recent_bottlenecks)

                # Record recommendations
                for recommendation in recommendations:
                    self.performance_analyzer.optimization_history.append(recommendation)

                # Update monitoring
                self._update_optimization_metrics(recommendations)

                # Log significant recommendations
                for recommendation in recommendations:
                    if recommendation.priority <= 2:  # High priority recommendations
                        self.logger.info("High priority optimization recommendation",
                                       context={
                                           "type": recommendation.optimization_type.value,
                                           "priority": recommendation.priority,
                                           "description": recommendation.description,
                                           "expected_impact": recommendation.expected_impact
                                       })

            except Exception as e:
                self.logger.error("Error in optimization loop",
                                context={"error": str(e)})

            await asyncio.sleep(self.optimization_interval)

    def _update_analysis_metrics(self, bottlenecks: List[BottleneckDetection]):
        """Update analysis-related metrics"""
        try:
            # Count bottlenecks by type and severity
            bottleneck_counts = defaultdict(int)
            severity_sum = 0.0

            for bottleneck in bottlenecks:
                bottleneck_counts[bottleneck.bottleneck_type.value] += 1
                severity_sum += bottleneck.severity

            # Update metrics
            resource_manager = get_resource_manager()
            monitor = resource_manager.monitor

            monitor.metrics.set_gauge("performance_bottlenecks_total", len(bottlenecks))
            monitor.metrics.set_gauge("performance_bottlenecks_avg_severity",
                                    severity_sum / len(bottlenecks) if bottlenecks else 0)

            for bottleneck_type, count in bottleneck_counts.items():
                monitor.metrics.set_gauge(f"performance_bottleneck_{bottleneck_type}_count", count)

        except Exception as e:
            self.logger.error("Error updating analysis metrics",
                            context={"error": str(e)})

    def _update_optimization_metrics(self, recommendations: List[OptimizationRecommendation]):
        """Update optimization-related metrics"""
        try:
            # Count recommendations by type and priority
            rec_counts = defaultdict(int)
            priority_sum = 0

            for recommendation in recommendations:
                rec_counts[recommendation.optimization_type.value] += 1
                priority_sum += recommendation.priority

            # Update metrics
            resource_manager = get_resource_manager()
            monitor = resource_manager.monitor

            monitor.metrics.set_gauge("optimization_recommendations_total", len(recommendations))
            monitor.metrics.set_gauge("optimization_recommendations_avg_priority",
                                    priority_sum / len(recommendations) if recommendations else 0)

            for rec_type, count in rec_counts.items():
                monitor.metrics.set_gauge(f"optimization_{rec_type}_count", count)

        except Exception as e:
            self.logger.error("Error updating optimization metrics",
                            context={"error": str(e)})

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        try:
            # Get current bottlenecks
            current_bottlenecks = self.performance_analyzer.detect_bottlenecks()

            # Get recent recommendations
            recent_recommendations = list(self.performance_analyzer.optimization_history)[-5:]

            # Get performance trends
            trends = self.performance_analyzer.analyze_performance_trends()

            # Get active A/B tests
            active_tests = list(self.ab_test_manager.active_tests.values())

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "current_bottlenecks": [
                    {
                        "type": b.bottleneck_type.value,
                        "severity": b.severity,
                        "location": b.location,
                        "description": b.description,
                        "recommendations": b.recommendations
                    }
                    for b in current_bottlenecks
                ],
                "recent_recommendations": [
                    {
                        "type": r.optimization_type.value,
                        "priority": r.priority,
                        "description": r.description,
                        "expected_impact": r.expected_impact,
                        "implementation_effort": r.implementation_effort
                    }
                    for r in recent_recommendations
                ],
                "performance_trends": trends,
                "active_ab_tests": [
                    {
                        "test_id": test.test_id,
                        "name": test.name,
                        "description": test.description,
                        "status": test.status,
                        "start_time": test.start_time.isoformat()
                    }
                    for test in active_tests
                ],
                "summary": {
                    "total_bottlenecks": len(current_bottlenecks),
                    "high_priority_recommendations": len([r for r in recent_recommendations if r.priority <= 2]),
                    "trending_metrics": len([t for t in trends.values() if abs(t.get("trend_strength", 0)) > 0.5]),
                    "active_tests": len(active_tests)
                }
            }

        except Exception as e:
            self.logger.error("Error generating performance report",
                            context={"error": str(e)})
            return {"error": str(e)}


# Global optimizer instance
_optimizer_instance = None
_optimizer_lock = threading.Lock()


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer instance"""
    global _optimizer_instance

    if _optimizer_instance is None:
        with _optimizer_lock:
            if _optimizer_instance is None:
                _optimizer_instance = PerformanceOptimizer()

    return _optimizer_instance