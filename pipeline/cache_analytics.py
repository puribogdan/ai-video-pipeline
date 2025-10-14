#!/usr/bin/env python3
"""
Cache Analytics and Performance Monitoring System

This module provides comprehensive analytics, optimization recommendations,
and performance monitoring for the intelligent audio processing cache system.
"""

from __future__ import annotations
import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
import threading

import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for cache analysis."""
    timestamp: float
    cache_hits: int
    cache_misses: int
    hit_ratio: float
    avg_response_time_ms: float
    memory_usage_mb: float
    disk_usage_mb: float
    throughput_items_per_sec: float
    error_rate: float


@dataclass
class OptimizationRecommendation:
    """Recommendation for cache optimization."""
    category: str  # 'memory', 'disk', 'strategy', 'performance'
    priority: str  # 'low', 'medium', 'high', 'critical'
    title: str
    description: str
    current_value: Any
    recommended_value: Any
    expected_benefit: str
    implementation_effort: str  # 'low', 'medium', 'high'


@dataclass
class CacheAlert:
    """Alert for cache performance issues."""
    timestamp: float
    level: str  # 'info', 'warning', 'error', 'critical'
    category: str
    title: str
    message: str
    metrics: Dict[str, Any]
    resolved: bool = False


class CacheAnalytics:
    """
    Analytics and monitoring system for intelligent cache performance.

    Provides:
    - Performance metrics collection and analysis
    - Optimization recommendations based on usage patterns
    - Alert system for performance issues
    - Trend analysis and forecasting
    """

    def __init__(self, cache_system, history_window_minutes: int = 60):
        self.cache_system = cache_system
        self.history_window_minutes = history_window_minutes

        # Metrics storage
        self.performance_history: List[PerformanceMetrics] = []
        self.recommendations: List[OptimizationRecommendation] = []
        self.alerts: List[CacheAlert] = []
        self.alert_callbacks: List[callable] = []

        # Analysis state
        self._lock = threading.RLock()
        self._last_analysis = 0.0
        self._analysis_interval = 300.0  # 5 minutes

        # Trend tracking
        self._hit_ratio_trend: List[float] = []
        self._response_time_trend: List[float] = []
        self._memory_trend: List[float] = []

        logger.info(f"CacheAnalytics initialized with {history_window_minutes} minute history window")

    def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        stats = self.cache_system.get_statistics()
        current_time = time.time()

        # Calculate throughput (items per second)
        recent_metrics = [
            m for m in self.performance_history
            if current_time - m.timestamp <= 60  # Last minute
        ]

        throughput = 0.0
        if len(recent_metrics) >= 2:
            time_span = recent_metrics[-1].timestamp - recent_metrics[0].timestamp
            if time_span > 0:
                total_requests = sum(m.cache_hits + m.cache_misses for m in recent_metrics)
                throughput = total_requests / time_span

        # Calculate error rate
        error_rate = 0.0
        if stats.total_requests > 0:
            error_rate = stats.misses / stats.total_requests

        # Calculate average response time from recent samples
        avg_response_time = stats.avg_access_time_ms
        if len(self.cache_system._access_times) > 0:
            avg_response_time = (
                sum(self.cache_system._access_times[-100:]) /  # Last 100 accesses
                min(len(self.cache_system._access_times), 100)
            ) * 1000

        metrics = PerformanceMetrics(
            timestamp=current_time,
            cache_hits=stats.hits,
            cache_misses=stats.misses,
            hit_ratio=stats.hit_ratio,
            avg_response_time_ms=avg_response_time,
            memory_usage_mb=stats.memory_usage_mb,
            disk_usage_mb=stats.disk_usage_mb,
            throughput_items_per_sec=throughput,
            error_rate=error_rate
        )

        # Store in history
        with self._lock:
            self.performance_history.append(metrics)

            # Maintain history window
            cutoff_time = current_time - (self.history_window_minutes * 60)
            self.performance_history = [
                m for m in self.performance_history
                if m.timestamp >= cutoff_time
            ]

            # Update trends
            self._update_trends(metrics)

        return metrics

    def _update_trends(self, metrics: PerformanceMetrics) -> None:
        """Update performance trends."""
        self._hit_ratio_trend.append(metrics.hit_ratio)
        self._response_time_trend.append(metrics.avg_response_time_ms)
        self._memory_trend.append(metrics.memory_usage_mb)

        # Keep only recent trend data (last 20 points)
        max_trend_points = 20
        for trend_list in [self._hit_ratio_trend, self._response_time_trend, self._memory_trend]:
            if len(trend_list) > max_trend_points:
                trend_list.pop(0)

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze cache performance and generate insights."""
        if not self.performance_history:
            return {'error': 'No performance data available'}

        current_time = time.time()

        # Get recent metrics for analysis
        recent_metrics = [
            m for m in self.performance_history
            if current_time - m.timestamp <= self._analysis_interval
        ]

        if len(recent_metrics) < 2:
            return {'error': 'Insufficient data for analysis'}

        # Calculate trends
        hit_ratio_trend = self._calculate_trend(self._hit_ratio_trend)
        response_time_trend = self._calculate_trend(self._response_time_trend)
        memory_trend = self._calculate_trend(self._memory_trend)

        # Current performance
        latest = recent_metrics[-1]
        previous = recent_metrics[0]

        analysis = {
            'current_performance': {
                'hit_ratio': latest.hit_ratio,
                'avg_response_time_ms': latest.avg_response_time_ms,
                'memory_usage_mb': latest.memory_usage_mb,
                'disk_usage_mb': latest.disk_usage_mb,
                'throughput_items_per_sec': latest.throughput_items_per_sec,
                'error_rate': latest.error_rate
            },

            'trends': {
                'hit_ratio_trend': hit_ratio_trend,
                'response_time_trend': response_time_trend,
                'memory_trend': memory_trend
            },

            'changes': {
                'hit_ratio_change': latest.hit_ratio - previous.hit_ratio,
                'response_time_change': latest.avg_response_time_ms - previous.avg_response_time_ms,
                'memory_change': latest.memory_usage_mb - previous.memory_usage_mb
            },

            'recommendations': self._generate_recommendations(latest, hit_ratio_trend, response_time_trend),
            'alerts': [alert for alert in self.alerts if not alert.resolved]
        }

        return analysis

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 3:
            return 'insufficient_data'

        # Simple linear trend analysis
        first_half = statistics.mean(values[:len(values)//2])
        second_half = statistics.mean(values[len(values)//2:])

        if second_half > first_half * 1.05:  # 5% improvement
            return 'improving'
        elif second_half < first_half * 0.95:  # 5% degradation
            return 'degrading'
        else:
            return 'stable'

    def _generate_recommendations(self, current: PerformanceMetrics,
                                hit_ratio_trend: str, response_time_trend: str) -> List[Dict[str, Any]]:
        """Generate optimization recommendations."""
        recommendations = []

        # Hit ratio recommendations
        if current.hit_ratio < 0.7:
            recommendations.append(OptimizationRecommendation(
                category='strategy',
                priority='high',
                title='Low Cache Hit Ratio',
                description=f'Current hit ratio is {current.hit_ratio:.2%}, which is below the recommended 70%',
                current_value=current.hit_ratio,
                recommended_value=0.8,
                expected_benefit='Reduced processing time and improved performance',
                implementation_effort='medium'
            ).__dict__)

        # Memory usage recommendations
        if current.memory_usage_mb > self.cache_system.config.max_memory_mb * 0.9:
            recommendations.append(OptimizationRecommendation(
                category='memory',
                priority='medium',
                title='High Memory Usage',
                description=f'Memory usage is at {current.memory_usage_mb:.1f}MB ({current.memory_usage_mb/self.cache_system.config.max_memory_mb*100:.1f}%)',
                current_value=current.memory_usage_mb,
                recommended_value=self.cache_system.config.max_memory_mb * 0.8,
                expected_benefit='Prevent memory-related performance issues',
                implementation_effort='low'
            ).__dict__)

        # Response time recommendations
        if current.avg_response_time_ms > 100:  # Slower than 100ms
            recommendations.append(OptimizationRecommendation(
                category='performance',
                priority='medium',
                title='Slow Response Times',
                description=f'Average response time is {current.avg_response_time_ms:.1f}ms, which is above the recommended 100ms',
                current_value=current.avg_response_time_ms,
                recommended_value=50,
                expected_benefit='Faster cache operations and better user experience',
                implementation_effort='medium'
            ).__dict__)

        # Disk usage recommendations
        if current.disk_usage_mb > self.cache_system.config.max_disk_mb * 0.85:
            recommendations.append(OptimizationRecommendation(
                category='disk',
                priority='low',
                title='High Disk Usage',
                description=f'Disk usage is at {current.disk_usage_mb:.1f}MB ({current.disk_usage_mb/self.cache_system.config.max_disk_mb*100:.1f}%)',
                current_value=current.disk_usage_mb,
                recommended_value=self.cache_system.config.max_disk_mb * 0.7,
                expected_benefit='Prevent disk space issues and maintain performance',
                implementation_effort='low'
            ).__dict__)

        return recommendations

    def check_alerts(self) -> List[CacheAlert]:
        """Check for performance issues and generate alerts."""
        current_metrics = self.collect_metrics()
        new_alerts = []

        # Critical alerts
        if current_metrics.hit_ratio < 0.3:
            new_alerts.append(CacheAlert(
                timestamp=time.time(),
                level='critical',
                category='performance',
                title='Critical Cache Performance',
                message=f'Cache hit ratio has dropped to {current_metrics.hit_ratio:.2%}',
                metrics={'hit_ratio': current_metrics.hit_ratio}
            ))

        # Error alerts
        if current_metrics.error_rate > 0.5:
            new_alerts.append(CacheAlert(
                timestamp=time.time(),
                level='error',
                category='reliability',
                title='High Error Rate',
                message=f'Cache error rate is {current_metrics.error_rate:.2%}',
                metrics={'error_rate': current_metrics.error_rate}
            ))

        # Warning alerts
        if current_metrics.avg_response_time_ms > 200:
            new_alerts.append(CacheAlert(
                timestamp=time.time(),
                level='warning',
                category='performance',
                title='Slow Cache Performance',
                message=f'Average response time is {current_metrics.avg_response_time_ms:.1f}ms',
                metrics={'response_time_ms': current_metrics.avg_response_time_ms}
            ))

        # Memory warnings
        if current_metrics.memory_usage_mb > self.cache_system.config.max_memory_mb * 0.95:
            new_alerts.append(CacheAlert(
                timestamp=time.time(),
                level='warning',
                category='memory',
                title='High Memory Usage',
                message=f'Memory usage is at {current_metrics.memory_usage_mb:.1f}MB',
                metrics={'memory_usage_mb': current_metrics.memory_usage_mb}
            ))

        # Store new alerts
        with self._lock:
            self.alerts.extend(new_alerts)

            # Keep only recent alerts (last 24 hours)
            cutoff_time = time.time() - (24 * 3600)
            self.alerts = [alert for alert in self.alerts if alert.timestamp >= cutoff_time]

        # Trigger alert callbacks
        for alert in new_alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

        return new_alerts

    def add_alert_callback(self, callback: callable) -> None:
        """Add callback function for alerts."""
        self.alert_callbacks.append(callback)

    def get_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        try:
            metrics = self.collect_metrics()

            # Weighted scoring based on key metrics
            hit_ratio_score = min(metrics.hit_ratio * 100, 100) * 0.4
            response_time_score = max(0, (200 - metrics.avg_response_time_ms) / 200 * 100) * 0.3
            memory_score = max(0, (self.cache_system.config.max_memory_mb - metrics.memory_usage_mb) /
                             self.cache_system.config.max_memory_mb * 100) * 0.2
            error_rate_score = max(0, (1 - metrics.error_rate) * 100) * 0.1

            health_score = hit_ratio_score + response_time_score + memory_score + error_rate_score

            return min(health_score, 100.0)

        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 0.0

    def export_metrics(self, output_path: Union[str, Path]) -> None:
        """Export performance metrics to file."""
        try:
            export_data = {
                'export_timestamp': time.time(),
                'history_window_minutes': self.history_window_minutes,
                'metrics': [metrics.__dict__ for metrics in self.performance_history],
                'alerts': [alert.__dict__ for alert in self.alerts],
                'recommendations': [rec.__dict__ for rec in self.recommendations]
            }

            output_path = Path(output_path)
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Exported {len(self.performance_history)} metrics to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

    def get_cache_efficiency_report(self) -> Dict[str, Any]:
        """Generate comprehensive cache efficiency report."""
        analysis = self.analyze_performance()

        if 'error' in analysis:
            return analysis

        current = analysis['current_performance']

        # Calculate efficiency metrics
        total_requests = current['hit_ratio'] * 100 + (1 - current['hit_ratio']) * 100  # Normalize to 100

        report = {
            'summary': {
                'overall_health_score': self.get_system_health_score(),
                'cache_efficiency': current['hit_ratio'],
                'average_response_time_ms': current['avg_response_time_ms'],
                'resource_utilization': {
                    'memory_percent': current['memory_usage_mb'] / self.cache_system.config.max_memory_mb * 100,
                    'disk_percent': current['disk_usage_mb'] / self.cache_system.config.max_disk_mb * 100
                }
            },

            'performance_analysis': analysis,

            'optimization_opportunities': self._identify_optimization_opportunities(current),

            'forecast': self._generate_performance_forecast()
        }

        return report

    def _identify_optimization_opportunities(self, current: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify specific optimization opportunities."""
        opportunities = []

        # Memory optimization
        if current['memory_usage_mb'] > self.cache_system.config.max_memory_mb * 0.8:
            opportunities.append({
                'area': 'Memory Management',
                'opportunity': 'Implement more aggressive LRU eviction',
                'potential_savings_mb': current['memory_usage_mb'] * 0.3,
                'complexity': 'medium'
            })

        # Strategy optimization
        if current['hit_ratio'] < 0.75:
            opportunities.append({
                'area': 'Cache Strategy',
                'opportunity': 'Adjust cache key generation for better hit rates',
                'potential_improvement': '15-25% hit ratio increase',
                'complexity': 'low'
            })

        # Compression optimization
        if hasattr(self.cache_system, 'statistics'):
            stats = self.cache_system.statistics
            if stats.compression_ratio < 1.5:
                opportunities.append({
                    'area': 'Data Compression',
                    'opportunity': 'Enable more aggressive compression',
                    'potential_savings_mb': current['disk_usage_mb'] * 0.4,
                    'complexity': 'low'
                })

        return opportunities

    def _generate_performance_forecast(self) -> Dict[str, Any]:
        """Generate performance forecast based on trends."""
        try:
            if len(self._hit_ratio_trend) < 5:
                return {'forecast': 'insufficient_data'}

            # Simple linear extrapolation
            hit_ratio_values = self._hit_ratio_trend[-5:]
            response_time_values = self._response_time_trend[-5:]

            # Calculate trend slopes
            hit_ratio_slope = self._calculate_slope(hit_ratio_values)
            response_time_slope = self._calculate_slope(response_time_values)

            # Project next 30 minutes
            current_hit_ratio = hit_ratio_values[-1]
            current_response_time = response_time_values[-1]

            forecast = {
                'hit_ratio_forecast': current_hit_ratio + (hit_ratio_slope * 6),  # 30 min = 6 * 5min intervals
                'response_time_forecast': current_response_time + (response_time_slope * 6),
                'confidence': 'medium' if len(hit_ratio_values) >= 10 else 'low'
            }

            return forecast

        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return {'forecast': 'error', 'error': str(e)}

    def _calculate_slope(self, values: List[float]) -> float:
        """Calculate slope of trend line."""
        if len(values) < 2:
            return 0.0

        # Simple slope calculation
        return (values[-1] - values[0]) / (len(values) - 1)


class CacheMonitor:
    """
    Real-time cache performance monitor with alerting.

    Provides:
    - Real-time performance monitoring
    - Configurable alert thresholds
    - Performance dashboards
    - Integration with external monitoring systems
    """

    def __init__(self, cache_system, analytics: CacheAnalytics):
        self.cache_system = cache_system
        self.analytics = analytics

        # Monitoring configuration
        self.alert_thresholds = {
            'hit_ratio_min': 0.7,
            'response_time_max_ms': 100,
            'memory_usage_max_percent': 90,
            'error_rate_max': 0.1
        }

        # Monitoring state
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._monitor_interval = 30.0  # seconds

        # External integrations
        self._webhook_urls: List[str] = []
        self._statsd_client = None

    async def start_monitoring(self) -> None:
        """Start real-time monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())

        logger.info("Cache monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        if not self._monitoring:
            return

        self._monitoring = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Cache monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                # Collect and analyze metrics
                metrics = self.analytics.collect_metrics()
                alerts = self.analytics.check_alerts()

                # Send alerts to external systems
                if alerts:
                    await self._send_external_alerts(alerts)

                # Send metrics to external monitoring
                await self._send_metrics_to_external(metrics)

                # Wait for next interval
                await asyncio.sleep(self._monitor_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self._monitor_interval)

    async def _send_external_alerts(self, alerts: List[CacheAlert]) -> None:
        """Send alerts to external monitoring systems."""
        for alert in alerts:
            # Webhook notifications
            for webhook_url in self._webhook_urls:
                try:
                    await self._send_webhook_alert(webhook_url, alert)
                except Exception as e:
                    logger.error(f"Failed to send webhook alert: {e}")

            # StatsD metrics
            if self._statsd_client:
                try:
                    await self._send_statsd_alert(alert)
                except Exception as e:
                    logger.error(f"Failed to send StatsD alert: {e}")

    async def _send_webhook_alert(self, webhook_url: str, alert: CacheAlert) -> None:
        """Send alert via webhook."""
        import aiohttp

        payload = {
            'timestamp': alert.timestamp,
            'level': alert.level,
            'category': alert.category,
            'title': alert.title,
            'message': alert.message,
            'metrics': alert.metrics
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status != 200:
                    logger.warning(f"Webhook alert failed: {response.status}")

    async def _send_statsd_alert(self, alert: CacheAlert) -> None:
        """Send alert metrics to StatsD."""
        if not self._statsd_client:
            return

        # This would integrate with a StatsD client
        # Example metrics:
        # cache.alerts.critical: 1
        # cache.alerts.error: 1
        # cache.alerts.warning: 1
        pass

    async def _send_metrics_to_external(self, metrics: PerformanceMetrics) -> None:
        """Send performance metrics to external monitoring."""
        # StatsD integration
        if self._statsd_client:
            try:
                # Send key metrics
                pass
            except Exception as e:
                logger.error(f"Failed to send metrics to StatsD: {e}")

    def configure_alerts(self, thresholds: Dict[str, float]) -> None:
        """Configure alert thresholds."""
        self.alert_thresholds.update(thresholds)
        logger.info(f"Updated alert thresholds: {self.alert_thresholds}")

    def add_webhook(self, url: str) -> None:
        """Add webhook URL for alerts."""
        self._webhook_urls.append(url)
        logger.info(f"Added webhook: {url}")


# Convenience functions for easy integration
def create_cache_analytics(cache_system) -> CacheAnalytics:
    """Create cache analytics instance."""
    return CacheAnalytics(cache_system)


def create_cache_monitor(cache_system, analytics: CacheAnalytics) -> CacheMonitor:
    """Create cache monitor instance."""
    return CacheMonitor(cache_system, analytics)


async def generate_cache_report(cache_system, output_path: Union[str, Path]) -> Dict[str, Any]:
    """Generate comprehensive cache performance report."""
    analytics = CacheAnalytics(cache_system)

    # Collect some metrics for analysis
    for _ in range(5):
        analytics.collect_metrics()
        await asyncio.sleep(1)

    # Generate report
    report = analytics.get_cache_efficiency_report()

    # Export to file
    analytics.export_metrics(output_path)

    return report