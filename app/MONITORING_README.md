# Audio Pipeline Monitoring System

## Overview

The Audio Pipeline Monitoring System provides comprehensive observability for the audio processing pipeline, including structured logging, performance metrics, health checks, alerting, and real-time dashboards.

## Features

### ✅ Completed Features

1. **AudioPipelineMonitor Class** - Core monitoring system with comprehensive tracking
2. **Structured Logging** - JSON-formatted logs with correlation IDs and context
3. **Performance Metrics Collection** - Latency, throughput, and error rate tracking
4. **Health Checks** - Comprehensive health monitoring for all pipeline components
5. **Alerting System** - Performance degradation and failure detection
6. **Health Check Endpoints** - RESTful APIs for health status reporting
7. **Audio Processing Performance Tracking** - Processing time and success rate monitoring
8. **Upload/Download Metrics** - File size distributions and transfer performance
9. **System Resource Monitoring** - CPU, memory, and disk I/O tracking
10. **Prometheus/Grafana Integration** - Metrics export for external monitoring
11. **Real-time Dashboard** - WebSocket-based live monitoring interface
12. **Integration** - Seamless integration with existing audio processing systems

### 🔄 Remaining Features

- Cache performance monitoring (hit rates, cache size, eviction rates)
- Historical trend analysis and alerting
- Performance reports and optimization recommendations
- Automated recovery mechanisms for common issues
- Circuit breaker patterns for external service dependencies
- Distributed tracing for multi-server deployments
- Log aggregation and analysis capabilities
- A/B testing framework for performance comparisons

## Architecture

### Core Components

#### AudioPipelineMonitor

The main monitoring class that orchestrates all monitoring activities:

```python
from app.monitoring import get_monitor

monitor = get_monitor()
correlation_id = monitor.start_job_tracking(job_id)
# ... process job ...
monitor.complete_job_tracking(job_id, "success")
```

#### StructuredLogger

Enhanced logging with correlation IDs and structured context:

```python
from app.monitoring import StructuredLogger

logger = StructuredLogger("component_name")
logger.info("Operation completed",
           correlation_id="abc-123",
           context={"job_id": "job-456", "duration": 1.23})
```

#### MetricsCollector

Thread-safe metrics collection and storage:

```python
monitor.metrics.record_histogram("audio_processing_time", 45.6, {"success": "true"})
monitor.metrics.set_gauge("active_jobs", 5)
monitor.metrics.increment_counter("jobs_completed_total")
```

#### HealthChecker

Comprehensive health checking system:

```python
# Add custom health check
def check_external_api():
    # Return (status, message, details)
    return "healthy", "API is responding", {"response_time": 0.123}

monitor.health_checker.add_check("external_api", check_external_api, 60)
```

#### AlertManager

Alert management and notification system:

```python
# Alerts are automatically checked based on configured rules
# Access active alerts
active_alerts = monitor.alert_manager.get_active_alerts()
```

## API Endpoints

### Health Check Endpoints

#### `/health/monitoring`

Get comprehensive monitoring system health status.

**Response:**

```json
{
  "status": "healthy",
  "monitoring": {
    "health_checks": {
      "redis": { "status": "healthy", "message": "Connected successfully" },
      "disk_space": { "status": "healthy", "message": "45.2GB free" },
      "memory": { "status": "healthy", "message": "Memory usage OK: 67.3%" }
    },
    "active_alerts": [],
    "system_metrics": {
      "active_jobs": 3,
      "tracked_jobs": 5,
      "websocket_connections": 2
    }
  }
}
```

#### `/metrics`

Get Prometheus-compatible metrics for Grafana integration.

#### `/monitoring/status`

Get detailed monitoring system status.

#### `/monitoring/alerts`

Get active alerts with details.

#### `/monitoring/alerts/{alert_id}/resolve`

Resolve a specific alert.

### Real-time Dashboard

#### `/ws/monitoring`

WebSocket endpoint for real-time monitoring updates.

#### `/monitoring/dashboard`

Interactive web dashboard for monitoring visualization.

## Usage Examples

### Job Processing with Monitoring

```python
from app.monitoring import get_monitor, monitoring_context

async def process_audio_job(job_id: str, audio_path: str):
    monitor = get_monitor()

    # Start job tracking
    correlation_id = monitor.start_job_tracking(job_id)

    try:
        # Record audio processing metrics
        start_time = time.time()
        success = await process_audio(audio_path)
        processing_time = time.time() - start_time

        if success:
            monitor.record_audio_processing_metrics(
                job_id, file_size, processing_time, True
            )
            monitor.complete_job_tracking(job_id, "success")
        else:
            monitor.record_audio_processing_metrics(
                job_id, file_size, processing_time, False
            )
            monitor.complete_job_tracking(job_id, "failed", "Processing failed")

    except Exception as e:
        monitor.complete_job_tracking(job_id, "failed", str(e))
        raise
```

### Custom Metrics Collection

```python
# Record cache performance
monitor.record_cache_metrics("audio_cache", True, 0.023)  # hit
monitor.record_cache_metrics("audio_cache", False, 0.045)  # miss

# Record upload performance
monitor.record_upload_metrics(file_size_mb * 1024*1024, upload_time, success)

# Record job stage timing
monitor.record_job_stage(job_id, "audio_processing", 45.6, {"format": "mp3"})
```

### Health Check Integration

```python
# Add custom health check
def check_audio_processing_service():
    try:
        # Test audio processing endpoint
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            return "healthy", "Audio processing service is responding", {}
        else:
            return "unhealthy", f"Service returned {response.status_code}", {}
    except Exception as e:
        return "unhealthy", f"Service check failed: {str(e)}", {}

monitor.health_checker.add_check("audio_service", check_audio_processing_service, 30)
```

## Configuration

### Environment Variables

No additional environment variables are required for basic monitoring functionality. The system uses existing Redis configuration.

### Monitoring Configuration

The monitoring system is configured with sensible defaults:

- Health checks run every 30-60 seconds
- Metrics are stored in memory with a 1000-point limit per metric
- WebSocket connections are managed automatically
- Alerts are checked every 30 seconds

## Grafana Integration

### Prometheus Metrics

The system exports metrics in Prometheus format at `/metrics`:

```
# HELP pipeline_audio_processing_seconds Audio processing time
# TYPE pipeline_audio_processing_seconds histogram
pipeline_audio_processing_seconds_bucket{le="10.0",success="true"} 45
pipeline_audio_processing_seconds_bucket{le="30.0",success="true"} 67
pipeline_audio_processing_seconds_bucket{le="+Inf",success="true"} 78

# HELP pipeline_system_cpu_percent System CPU usage
# TYPE pipeline_system_cpu_percent gauge
pipeline_system_cpu_percent 23.5
```

### Dashboard Setup

1. **Add Prometheus Data Source:**

   - URL: `http://your-server:8000/metrics`
   - Access: Server (no authentication required)

2. **Import Dashboard:**

   - Use the provided Grafana dashboard JSON
   - Configure refresh interval (30s recommended)

3. **Panels to Monitor:**
   - Job processing times and success rates
   - System resource utilization
   - Cache performance metrics
   - Active alerts and health status

## Real-time Dashboard

The web dashboard provides:

- **System Health Overview** - Real-time health check status
- **Active Alerts** - Current alerts with severity levels
- **System Metrics** - Live resource utilization
- **Performance Charts** - Historical trends and patterns
- **Alert History** - Recent alert timeline

### Accessing the Dashboard

Navigate to `http://your-server:8000/monitoring/dashboard` to access the real-time monitoring interface.

### WebSocket Connection

The dashboard connects via WebSocket to `/ws/monitoring` for real-time updates:

```javascript
const socket = new WebSocket('ws://localhost:8000/ws/monitoring');

socket.onmessage = function (event) {
  const data = JSON.parse(event.data);
  if (data.type === 'monitoring_update') {
    updateDashboard(data);
  }
};
```

## Alerting

### Default Alert Rules

The system includes default alert rules for:

- **High Error Rate** - More than 5 errors in the last 10 metrics
- **High Memory Usage** - Memory usage exceeds 85%
- **Slow Job Processing** - Job processing time exceeds 30 minutes

### Custom Alert Rules

```python
def custom_condition(metrics_collector):
    recent_jobs = metrics_collector.get_metrics("job_processing_time", limit=10)
    if recent_jobs and any(job.value > 3600 for job in recent_jobs):  # 1 hour
        return True
    return False

monitor.alert_manager.add_alert_rule(
    "very_slow_jobs",
    custom_condition,
    "warning",
    "Job processing time exceeded 1 hour"
)
```

## Performance Impact

The monitoring system is designed to have minimal performance impact:

- **Memory Usage** - Approximately 1-5MB for metrics storage
- **CPU Usage** - Less than 1% for background monitoring
- **Network** - Minimal WebSocket traffic for real-time updates
- **Storage** - No persistent storage requirements (Redis optional)

## Troubleshooting

### Common Issues

1. **High Memory Usage**

   - Check for memory leaks in job processing
   - Monitor garbage collection frequency
   - Review large file processing patterns

2. **Slow Job Processing**

   - Check system resource utilization
   - Monitor external API response times
   - Review audio file processing pipeline

3. **WebSocket Connection Issues**
   - Verify WebSocket endpoint accessibility
   - Check for proxy/load balancer WebSocket support
   - Monitor connection timeout settings

### Debug Logging

Enable debug logging for detailed monitoring system information:

```python
import logging
logging.getLogger("audio_pipeline_monitor").setLevel(logging.DEBUG)
```

## Future Enhancements

The remaining planned features include:

1. **Cache Performance Monitoring** - Detailed cache hit/miss analysis
2. **Historical Trend Analysis** - Long-term performance pattern detection
3. **Automated Recovery** - Self-healing mechanisms for common issues
4. **Circuit Breaker Patterns** - Protection against external service failures
5. **Distributed Tracing** - End-to-end request tracing across services
6. **A/B Testing Framework** - Performance comparison capabilities

## Support

For issues or questions regarding the monitoring system:

1. Check the troubleshooting section above
2. Review the debug logs for detailed error information
3. Monitor the `/health/monitoring` endpoint for system status
4. Use the real-time dashboard for visual debugging

The monitoring system is designed to be self-diagnosing and provide comprehensive visibility into pipeline performance and health.
