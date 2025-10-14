# app/upload_health_checker.py
import asyncio
import json
import os
import psutil
import shutil
import socket
import subprocess
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import logging
import redis
import threading

from .monitoring import StructuredLogger, MetricsCollector, HealthCheck, Alert
from .error_recovery import ErrorRecoveryManager, ErrorContext, ErrorCategory, ErrorSeverity


class UploadComponentStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class UploadComponentType(Enum):
    DISK_STORAGE = "disk_storage"
    NETWORK = "network"
    FFMPEG = "ffmpeg"
    AUDIO_LIBRARIES = "audio_libraries"
    REDIS = "redis"
    B2_STORAGE = "b2_storage"
    PERMISSIONS = "permissions"
    MEMORY = "memory"
    CPU = "cpu"


@dataclass
class UploadComponentHealth:
    """Health status for an upload component"""
    component_type: UploadComponentType
    status: UploadComponentStatus
    last_check: datetime
    response_time: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    consecutive_failures: int = 0
    last_success: Optional[datetime] = None


@dataclass
class UploadPerformanceBaseline:
    """Performance baseline for upload operations"""
    component_type: UploadComponentType
    metric_name: str
    baseline_value: float
    threshold_warning: float
    threshold_critical: float
    measurement_unit: str
    last_updated: datetime
    sample_count: int = 0


class UploadHealthChecker:
    """Comprehensive health checker for upload-related components"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = redis.from_url(redis_url)
        self.logger = StructuredLogger("upload_health_checker")

        # Component health tracking
        self.component_health: Dict[UploadComponentType, UploadComponentHealth] = {}
        self.performance_baselines: Dict[str, UploadPerformanceBaseline] = {}

        # Configuration
        self.check_interval = 30  # seconds
        self.max_consecutive_failures = 3
        self.baseline_update_interval = 300  # 5 minutes
        self.history_retention = 1000

        # Performance tracking
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.history_retention))
        self.upload_metrics: Dict[str, Dict[str, Any]] = {}

        # Locks for thread safety
        self._health_lock = threading.Lock()
        self._baseline_lock = threading.Lock()
        self._performance_lock = threading.Lock()

        # Background tasks
        self._monitoring_task = None
        self._baseline_update_task = None
        self._running = False

        # Error recovery integration
        self.error_recovery = None
        try:
            self.error_recovery = ErrorRecoveryManager(redis_url)
        except Exception as e:
            self.logger.warning("Error recovery manager not available", context={"error": str(e)})

        # Initialize component health
        self._initialize_component_health()

        # Setup default performance baselines
        self._setup_default_baselines()

    def _initialize_component_health(self):
        """Initialize health status for all upload components"""
        for component_type in UploadComponentType:
            self.component_health[component_type] = UploadComponentHealth(
                component_type=component_type,
                status=UploadComponentStatus.UNKNOWN,
                last_check=datetime.utcnow(),
                response_time=0.0,
                message="Not yet checked"
            )

    def _setup_default_baselines(self):
        """Setup default performance baselines for upload operations"""
        baselines = [
            UploadPerformanceBaseline(
                component_type=UploadComponentType.DISK_STORAGE,
                metric_name="disk_write_speed_mbps",
                baseline_value=50.0,  # 50 MB/s
                threshold_warning=20.0,  # 20 MB/s
                threshold_critical=5.0,   # 5 MB/s
                measurement_unit="MB/s",
                last_updated=datetime.utcnow()
            ),
            UploadPerformanceBaseline(
                component_type=UploadComponentType.NETWORK,
                metric_name="upload_bandwidth_mbps",
                baseline_value=10.0,  # 10 MB/s
                threshold_warning=2.0,   # 2 MB/s
                threshold_critical=0.5,   # 0.5 MB/s
                measurement_unit="MB/s",
                last_updated=datetime.utcnow()
            ),
            UploadPerformanceBaseline(
                component_type=UploadComponentType.FFMPEG,
                metric_name="audio_processing_time_sec",
                baseline_value=30.0,  # 30 seconds
                threshold_warning=120.0,  # 2 minutes
                threshold_critical=300.0, # 5 minutes
                measurement_unit="seconds",
                last_updated=datetime.utcnow()
            ),
            UploadPerformanceBaseline(
                component_type=UploadComponentType.MEMORY,
                metric_name="memory_usage_percent",
                baseline_value=70.0,  # 70%
                threshold_warning=85.0,  # 85%
                threshold_critical=95.0,  # 95%
                measurement_unit="percent",
                last_updated=datetime.utcnow()
            )
        ]

        for baseline in baselines:
            key = f"{baseline.component_type.value}:{baseline.metric_name}"
            with self._baseline_lock:
                self.performance_baselines[key] = baseline

    async def start_health_monitoring(self):
        """Start the upload health monitoring system"""
        if self._running:
            return

        self._running = True
        self.logger.info("Starting upload health monitoring")

        # Start monitoring tasks
        self._monitoring_task = asyncio.create_task(self._health_monitoring_loop())
        self._baseline_update_task = asyncio.create_task(self._baseline_update_loop())

    async def stop_health_monitoring(self):
        """Stop the upload health monitoring system"""
        if not self._running:
            return

        self._running = False
        self.logger.info("Stopping upload health monitoring")

        # Stop monitoring tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._baseline_update_task:
            self._baseline_update_task.cancel()

    async def _health_monitoring_loop(self):
        """Main health monitoring loop"""
        while self._running:
            try:
                # Run all health checks
                await self._run_all_health_checks()

                # Check for performance anomalies
                await self._check_performance_anomalies()

                # Update component status based on consecutive failures
                self._update_component_status_from_failures()

            except Exception as e:
                self.logger.error("Error in health monitoring loop",
                                context={"error": str(e)})

            await asyncio.sleep(self.check_interval)

    async def _baseline_update_loop(self):
        """Update performance baselines based on recent data"""
        while self._running:
            try:
                await self._update_performance_baselines()
            except Exception as e:
                self.logger.error("Error updating performance baselines",
                                context={"error": str(e)})

            await asyncio.sleep(self.baseline_update_interval)

    async def _run_all_health_checks(self):
        """Run health checks for all upload components"""
        health_checks = [
            self._check_disk_storage_health,
            self._check_network_health,
            self._check_ffmpeg_health,
            self._check_audio_libraries_health,
            self._check_redis_health,
            self._check_b2_storage_health,
            self._check_permissions_health,
            self._check_memory_health,
            self._check_cpu_health,
        ]

        for check_func in health_checks:
            try:
                await self._execute_health_check(check_func)
            except Exception as e:
                self.logger.error("Health check execution failed",
                                context={"check": check_func.__name__, "error": str(e)})

    async def _execute_health_check(self, check_func: Callable):
        """Execute a single health check with timing and error handling"""
        component_name = check_func.__name__.replace("_health", "")
        start_time = time.time()

        try:
            # Find component type from function name
            component_type = None
            for comp_type in UploadComponentType:
                if comp_type.value in component_name.lower():
                    component_type = comp_type
                    break

            if not component_type:
                self.logger.warning("Unknown component type for health check",
                                  context={"check": check_func.__name__})
                return

            # Execute the health check
            status, message, details = await self._safe_execute_check(check_func)

            # Calculate response time
            response_time = time.time() - start_time

            # Update component health
            with self._health_lock:
                component_health = self.component_health[component_type]
                component_health.last_check = datetime.utcnow()
                component_health.response_time = response_time
                component_health.message = message
                component_health.details = details

                if status == "healthy":
                    component_health.status = UploadComponentStatus.HEALTHY
                    component_health.consecutive_failures = 0
                    component_health.last_success = datetime.utcnow()
                else:
                    component_health.consecutive_failures += 1
                    if component_health.consecutive_failures >= self.max_consecutive_failures:
                        component_health.status = UploadComponentStatus.UNHEALTHY
                    else:
                        component_health.status = UploadComponentStatus.DEGRADED

            # Record performance metrics
            self._record_performance_metric(
                f"{component_type.value}_health_check_time",
                response_time,
                {"status": status}
            )

            self.logger.debug("Health check completed",
                            context={
                                "component": component_type.value,
                                "status": status,
                                "response_time": response_time
                            })

        except Exception as e:
            response_time = time.time() - start_time

            # Update component health on error
            if component_type:
                with self._health_lock:
                    component_health = self.component_health[component_type]
                    component_health.last_check = datetime.utcnow()
                    component_health.response_time = response_time
                    component_health.message = f"Health check failed: {str(e)}"
                    component_health.consecutive_failures += 1
                    component_health.status = UploadComponentStatus.UNHEALTHY

            self.logger.error("Health check execution error",
                            context={
                                "check": check_func.__name__,
                                "error": str(e),
                                "response_time": response_time
                            })

    async def _safe_execute_check(self, check_func: Callable) -> Tuple[str, str, Dict[str, Any]]:
        """Safely execute a health check function"""
        try:
            if asyncio.iscoroutinefunction(check_func):
                return await check_func()
            else:
                return check_func()
        except Exception as e:
            return "unhealthy", f"Health check failed: {str(e)}", {"error": str(e)}

    async def _check_disk_storage_health(self) -> Tuple[str, str, Dict[str, Any]]:
        """Check disk storage health for upload operations"""
        try:
            # Check upload directories
            upload_dirs = [
                Path("./uploads"),
                Path("./pipeline/output"),
                Path("./tmp")
            ]

            details = {}
            total_free_gb = 0
            total_space_gb = 0

            for upload_dir in upload_dirs:
                if upload_dir.exists():
                    try:
                        # Create directory if it doesn't exist
                        upload_dir.mkdir(parents=True, exist_ok=True)

                        # Get disk usage
                        disk_usage = shutil.disk_usage(upload_dir)
                        free_gb = disk_usage.free / (1024**3)
                        total_gb = disk_usage.total / (1024**3)
                        used_percent = (disk_usage.used / disk_usage.total) * 100

                        total_free_gb += free_gb
                        total_space_gb += total_gb

                        details[upload_dir.name] = {
                            "free_gb": round(free_gb, 2),
                            "total_gb": round(total_gb, 2),
                            "used_percent": round(used_percent, 2),
                            "path": str(upload_dir)
                        }

                        # Check write permissions
                        test_file = upload_dir / f".health_check_{uuid.uuid4()}.tmp"
                        try:
                            test_file.write_text("test")
                            test_file.unlink()
                            details[upload_dir.name]["write_permission"] = True
                        except Exception as e:
                            details[upload_dir.name]["write_permission"] = False
                            details[upload_dir.name]["permission_error"] = str(e)

                    except Exception as e:
                        details[upload_dir.name] = {"error": str(e)}

            # Overall assessment
            if total_free_gb < 1.0:  # Less than 1GB free
                return "unhealthy", f"Low disk space: {total_free_gb:.1f}GB free", details
            elif total_free_gb < 5.0:  # Less than 5GB free
                return "degraded", f"Limited disk space: {total_free_gb:.1f}GB free", details

            # Check for write permission issues
            write_issues = sum(1 for d in details.values() if not d.get("write_permission", True))
            if write_issues > 0:
                return "degraded", f"Write permission issues in {write_issues} directories", details

            return "healthy", f"Disk space OK: {total_free_gb:.1f}GB free", details

        except Exception as e:
            return "unhealthy", f"Disk storage check failed: {str(e)}", {"error": str(e)}

    async def _check_network_health(self) -> Tuple[str, str, Dict[str, Any]]:
        """Check network health for upload operations"""
        try:
            details = {}

            # Test basic connectivity
            try:
                # Test DNS resolution
                start_time = time.time()
                socket.gethostbyname("www.google.com")
                dns_time = time.time() - start_time
                details["dns_resolution"] = {"success": True, "time": round(dns_time, 3)}
            except Exception as e:
                details["dns_resolution"] = {"success": False, "error": str(e)}

            # Test external connectivity
            try:
                start_time = time.time()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex(('8.8.8.8', 53))
                connect_time = time.time() - start_time
                sock.close()

                details["external_connectivity"] = {
                    "success": result == 0,
                    "time": round(connect_time, 3),
                    "port_53_reachable": result == 0
                }
            except Exception as e:
                details["external_connectivity"] = {"success": False, "error": str(e)}

            # Test upload bandwidth (if possible)
            # This would require a test upload to a known endpoint

            # Overall assessment
            dns_success = details.get("dns_resolution", {}).get("success", False)
            connectivity_success = details.get("external_connectivity", {}).get("success", False)

            if not dns_success and not connectivity_success:
                return "unhealthy", "Network connectivity issues detected", details
            elif not dns_success or not connectivity_success:
                return "degraded", "Partial network connectivity issues", details

            return "healthy", "Network connectivity OK", details

        except Exception as e:
            return "unhealthy", f"Network health check failed: {str(e)}", {"error": str(e)}

    async def _check_ffmpeg_health(self) -> Tuple[str, str, Dict[str, Any]]:
        """Check FFmpeg installation and functionality"""
        try:
            details = {}

            # Check if ffmpeg is installed
            try:
                start_time = time.time()
                result = subprocess.run(
                    ["ffmpeg", "-version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                version_time = time.time() - start_time

                if result.returncode == 0:
                    version_output = result.stdout
                    # Extract version number
                    version_line = version_output.split('\n')[0]
                    version = version_line.split()[2] if len(version_line.split()) > 2 else "unknown"

                    details["installed"] = True
                    details["version"] = version
                    details["version_check_time"] = round(version_time, 3)

                    # Test basic functionality
                    func_start = time.time()
                    test_result = subprocess.run(
                        ["ffmpeg", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=1",
                        "-f", "null", "-"],
                        capture_output=True,
                        timeout=15
                    )
                    func_time = time.time() - func_start

                    details["functional"] = test_result.returncode == 0
                    details["functionality_test_time"] = round(func_time, 3)

                    if test_result.returncode != 0:
                        details["functionality_error"] = test_result.stderr

                else:
                    details["installed"] = False
                    details["version_error"] = result.stderr

            except FileNotFoundError:
                details["installed"] = False
                details["error"] = "FFmpeg not found in PATH"
            except subprocess.TimeoutExpired:
                details["installed"] = True
                details["timeout"] = True
                details["error"] = "FFmpeg version check timed out"
            except Exception as e:
                details["installed"] = True
                details["error"] = str(e)

            # Overall assessment
            if not details.get("installed", False):
                return "unhealthy", "FFmpeg not installed", details

            if not details.get("functional", True):
                return "degraded", "FFmpeg functionality issues detected", details

            return "healthy", f"FFmpeg OK (version: {details.get('version', 'unknown')})", details

        except Exception as e:
            return "unhealthy", f"FFmpeg health check failed: {str(e)}", {"error": str(e)}

    async def _check_audio_libraries_health(self) -> Tuple[str, str, Dict[str, Any]]:
        """Check audio processing libraries health"""
        try:
            details = {}

            # Check for required audio libraries
            libraries_to_check = [
                ("moviepy", "MoviePy"),
                ("pydub", "Pydub"),
                ("librosa", "Librosa"),
                ("soundfile", "SoundFile"),
                ("numpy", "NumPy"),
                ("scipy", "SciPy")
            ]

            for lib_name, display_name in libraries_to_check:
                try:
                    start_time = time.time()
                    __import__(lib_name)
                    import_time = time.time() - start_time

                    # Try to get version if available
                    version = "unknown"
                    try:
                        if hasattr(__import__(lib_name), "__version__"):
                            version = __import__(lib_name).__version__
                        elif hasattr(__import__(lib_name), "version"):
                            version = __import__(lib_name).version
                    except:
                        pass

                    details[lib_name] = {
                        "available": True,
                        "version": version,
                        "import_time": round(import_time, 3)
                    }

                except ImportError as e:
                    details[lib_name] = {
                        "available": False,
                        "error": str(e)
                    }

            # Check for audio format support
            test_formats = ["mp3", "wav", "m4a", "flac"]
            format_support = {}

            for fmt in test_formats:
                # This is a basic check - in practice you'd test actual file operations
                format_support[fmt] = "supported"  # Assume supported if library is available

            details["format_support"] = format_support

            # Overall assessment
            available_libs = sum(1 for lib in details.values()
                               if isinstance(lib, dict) and lib.get("available", False))

            if available_libs == 0:
                return "unhealthy", "No audio processing libraries available", details

            total_libs = len([lib for lib in libraries_to_check if lib[0] in details])
            if available_libs < total_libs * 0.5:  # Less than 50% available
                return "degraded", f"Only {available_libs}/{total_libs} audio libraries available", details

            return "healthy", f"Audio libraries OK ({available_libs}/{total_libs} available)", details

        except Exception as e:
            return "unhealthy", f"Audio libraries health check failed: {str(e)}", {"error": str(e)}

    async def _check_redis_health(self) -> Tuple[str, str, Dict[str, Any]]:
        """Check Redis health for upload operations"""
        try:
            details = {}

            try:
                start_time = time.time()
                self.redis.ping()
                ping_time = time.time() - start_time

                details["connected"] = True
                details["ping_time"] = round(ping_time, 3)

                # Test basic operations
                test_key = f"health_check:{uuid.uuid4()}"

                # Test SET operation
                set_start = time.time()
                self.redis.set(test_key, "test_value", ex=10)
                set_time = time.time() - set_start

                # Test GET operation
                get_start = time.time()
                value = self.redis.get(test_key)
                get_time = time.time() - get_start

                details["set_operation"] = {"success": True, "time": round(set_time, 3)}
                details["get_operation"] = {"success": value == "test_value", "time": round(get_time, 3)}

                # Clean up test key
                self.redis.delete(test_key)

                # Check memory usage
                try:
                    info = self.redis.info("memory")
                    details["memory"] = {
                        "used_memory_mb": round(info.get("used_memory", 0) / (1024 * 1024), 2),
                        "max_memory_mb": round(info.get("maxmemory", 0) / (1024 * 1024), 2) if info.get("maxmemory", 0) > 0 else None,
                        "memory_usage_percent": round((info.get("used_memory", 0) / info.get("maxmemory", 1)) * 100, 2) if info.get("maxmemory", 0) > 0 else 0
                    }
                except:
                    pass

            except Exception as e:
                details["connected"] = False
                details["error"] = str(e)

            # Overall assessment
            if not details.get("connected", False):
                return "unhealthy", "Redis connection failed", details

            ping_time = details.get("ping_time", 1.0)
            if ping_time > 1.0:  # Slower than 1 second
                return "degraded", f"Slow Redis response: {ping_time:.3f}s", details

            return "healthy", f"Redis OK (ping: {ping_time:.3f}s)", details

        except Exception as e:
            return "unhealthy", f"Redis health check failed: {str(e)}", {"error": str(e)}

    async def _check_b2_storage_health(self) -> Tuple[str, str, Dict[str, Any]]:
        """Check B2 storage health for upload operations"""
        try:
            details = {}

            # Check environment variables
            b2_key_id = os.getenv("B2_KEY_ID")
            b2_key = os.getenv("B2_KEY")

            if not b2_key_id or not b2_key:
                return "unhealthy", "B2 credentials not configured", {
                    "configured": False,
                    "error": "B2_KEY_ID and B2_KEY environment variables required"
                }

            details["configured"] = True

            # Test B2 connection (basic check)
            try:
                import boto3
                from botocore.exceptions import ClientError

                # Create B2 client
                s3_client = boto3.client(
                    's3',
                    endpoint_url='https://s3.us-west-002.backblazeb2.com',
                    aws_access_key_id=b2_key_id,
                    aws_secret_access_key=b2_key
                )

                # Test connection by listing buckets
                start_time = time.time()
                response = s3_client.list_buckets()
                list_time = time.time() - start_time

                details["connection"] = {"success": True, "time": round(list_time, 3)}
                details["bucket_count"] = len(response.get("Buckets", []))

                # Test upload capability (if we have a bucket)
                if response.get("Buckets"):
                    test_bucket = response["Buckets"][0]["Name"]

                    # Try to list objects (basic permission check)
                    try:
                        list_start = time.time()
                        s3_client.list_objects_v2(Bucket=test_bucket, MaxKeys=1)
                        list_obj_time = time.time() - list_start

                        details["permissions"] = {"success": True, "time": round(list_obj_time, 3)}
                    except ClientError as e:
                        details["permissions"] = {"success": False, "error": str(e)}

            except ImportError:
                details["boto3_available"] = False
                details["error"] = "boto3 not installed"
            except Exception as e:
                details["connection"] = {"success": False, "error": str(e)}

            # Overall assessment
            if not details.get("configured", False):
                return "unhealthy", "B2 storage not configured", details

            if not details.get("connection", {}).get("success", False):
                return "unhealthy", "B2 connection failed", details

            return "healthy", "B2 storage OK", details

        except Exception as e:
            return "unhealthy", f"B2 storage health check failed: {str(e)}", {"error": str(e)}

    async def _check_permissions_health(self) -> Tuple[str, str, Dict[str, Any]]:
        """Check file system permissions for upload operations"""
        try:
            details = {}

            # Check key directories
            check_dirs = [
                Path("./uploads"),
                Path("./pipeline"),
                Path("./tmp"),
                Path("./app")
            ]

            for check_dir in check_dirs:
                dir_name = str(check_dir)

                if check_dir.exists():
                    # Check read permission
                    try:
                        list(check_dir.iterdir())
                        read_ok = True
                    except PermissionError:
                        read_ok = False

                    # Check write permission
                    try:
                        test_file = check_dir / f".perm_check_{uuid.uuid4()}.tmp"
                        test_file.write_text("test")
                        test_file.unlink()
                        write_ok = True
                    except PermissionError:
                        write_ok = False
                    except Exception:
                        write_ok = False

                    details[dir_name] = {
                        "read_permission": read_ok,
                        "write_permission": write_ok,
                        "exists": True
                    }
                else:
                    details[dir_name] = {
                        "exists": False,
                        "error": "Directory does not exist"
                    }

            # Check current working directory permissions
            try:
                cwd = Path.cwd()
                test_file = cwd / f".cwd_perm_check_{uuid.uuid4()}.tmp"
                test_file.write_text("test")
                test_file.unlink()
                details["current_directory"] = {"write_permission": True}
            except Exception as e:
                details["current_directory"] = {"write_permission": False, "error": str(e)}

            # Overall assessment
            write_issues = sum(1 for d in details.values()
                             if isinstance(d, dict) and not d.get("write_permission", True))

            if write_issues > 0:
                return "degraded", f"Write permission issues in {write_issues} locations", details

            return "healthy", "File permissions OK", details

        except Exception as e:
            return "unhealthy", f"Permissions health check failed: {str(e)}", {"error": str(e)}

    async def _check_memory_health(self) -> Tuple[str, str, Dict[str, Any]]:
        """Check memory health for upload operations"""
        try:
            details = {}

            # Get memory information
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            details["memory"] = {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "usage_percent": round(memory.percent, 2)
            }

            details["swap"] = {
                "total_gb": round(swap.total / (1024**3), 2),
                "used_gb": round(swap.used / (1024**3), 2),
                "usage_percent": round(swap.percent, 2)
            }

            # Check for memory pressure
            usage_percent = memory.percent
            if usage_percent > 95:
                return "unhealthy", f"Critical memory usage: {usage_percent:.1f}%", details
            elif usage_percent > 85:
                return "degraded", f"High memory usage: {usage_percent:.1f}%", details

            return "healthy", f"Memory usage OK: {usage_percent:.1f}%", details

        except Exception as e:
            return "unhealthy", f"Memory health check failed: {str(e)}", {"error": str(e)}

    async def _check_cpu_health(self) -> Tuple[str, str, Dict[str, Any]]:
        """Check CPU health for upload operations"""
        try:
            details = {}

            # Get CPU information
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_count_logical = psutil.cpu_count(logical=True)

            details["cpu"] = {
                "usage_percent": round(cpu_percent, 2),
                "physical_cores": cpu_count,
                "logical_cores": cpu_count_logical,
                "load_average": "unavailable"  # Would need os.getloadavg() on Unix
            }

            # Check for CPU pressure
            if cpu_percent > 90:
                return "degraded", f"High CPU usage: {cpu_percent:.1f}%", details
            elif cpu_percent > 75:
                return "degraded", f"Elevated CPU usage: {cpu_percent:.1f}%", details

            return "healthy", f"CPU usage OK: {cpu_percent:.1f}%", details

        except Exception as e:
            return "unhealthy", f"CPU health check failed: {str(e)}", {"error": str(e)}

    def _record_performance_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Record a performance metric for baseline calculation"""
        with self._performance_lock:
            self.performance_history[metric_name].append({
                "value": value,
                "timestamp": datetime.utcnow(),
                "labels": labels or {}
            })

    async def _check_performance_anomalies(self):
        """Check for performance anomalies based on baselines"""
        try:
            for baseline_key, baseline in self.performance_baselines.items():
                # Get recent performance data
                metric_name = baseline.metric_name
                recent_data = list(self.performance_history.get(metric_name, []))[-10:]  # Last 10 measurements

                if len(recent_data) < 3:  # Need at least 3 data points
                    continue

                # Calculate average of recent values
                recent_values = [d["value"] for d in recent_data]
                avg_value = sum(recent_values) / len(recent_values)

                # Check against thresholds
                if avg_value > baseline.threshold_critical:
                    self.logger.warning("Critical performance threshold exceeded",
                                      context={
                                          "metric": metric_name,
                                          "value": avg_value,
                                          "threshold": baseline.threshold_critical,
                                          "component": baseline.component_type.value
                                      })

                    # Update component status
                    with self._health_lock:
                        if baseline.component_type in self.component_health:
                            component_health = self.component_health[baseline.component_type]
                            component_health.status = UploadComponentStatus.DEGRADED
                            component_health.message = f"Performance degraded: {metric_name} = {avg_value:.2f}{baseline.measurement_unit}"

                elif avg_value > baseline.threshold_warning:
                    self.logger.info("Warning performance threshold exceeded",
                                  context={
                                      "metric": metric_name,
                                      "value": avg_value,
                                      "threshold": baseline.threshold_warning,
                                      "component": baseline.component_type.value
                                  })

        except Exception as e:
            self.logger.error("Error checking performance anomalies",
                            context={"error": str(e)})

    def _update_component_status_from_failures(self):
        """Update component status based on consecutive failures"""
        with self._health_lock:
            for component_type, health in self.component_health.items():
                if health.consecutive_failures >= self.max_consecutive_failures:
                    if health.status != UploadComponentStatus.UNHEALTHY:
                        health.status = UploadComponentStatus.UNHEALTHY
                        self.logger.warning("Component marked unhealthy due to consecutive failures",
                                          context={
                                              "component": component_type.value,
                                              "failures": health.consecutive_failures
                                          })
                elif health.consecutive_failures > 0:
                    if health.status != UploadComponentStatus.DEGRADED:
                        health.status = UploadComponentStatus.DEGRADED
                        self.logger.info("Component marked degraded due to failures",
                                      context={
                                          "component": component_type.value,
                                          "failures": health.consecutive_failures
                                      })

    async def _update_performance_baselines(self):
        """Update performance baselines based on recent data"""
        try:
            for baseline_key, baseline in list(self.performance_baselines.items()):
                metric_name = baseline.metric_name
                recent_data = list(self.performance_history.get(metric_name, []))

                if len(recent_data) < 20:  # Need sufficient data
                    continue

                # Calculate new baseline from recent healthy measurements
                recent_values = [d["value"] for d in recent_data[-100:]]  # Last 100 measurements

                if recent_values:
                    new_baseline = sum(recent_values) / len(recent_values)

                    # Update baseline if it has changed significantly
                    change_percent = abs(new_baseline - baseline.baseline_value) / baseline.baseline_value

                    if change_percent > 0.1:  # 10% change threshold
                        with self._baseline_lock:
                            baseline.baseline_value = new_baseline
                            baseline.last_updated = datetime.utcnow()
                            baseline.sample_count = len(recent_values)

                        self.logger.info("Performance baseline updated",
                                       context={
                                           "metric": metric_name,
                                           "old_baseline": baseline.baseline_value,
                                           "new_baseline": new_baseline,
                                           "change_percent": round(change_percent * 100, 2)
                                       })

        except Exception as e:
            self.logger.error("Error updating performance baselines",
                            context={"error": str(e)})

    def get_component_health(self, component_type: Optional[UploadComponentType] = None) -> Dict[str, Any]:
        """Get health status for components"""
        with self._health_lock:
            if component_type:
                health = self.component_health.get(component_type)
                if health:
                    return {
                        "component_type": health.component_type.value,
                        "status": health.status.value,
                        "last_check": health.last_check.isoformat(),
                        "response_time": health.response_time,
                        "message": health.message,
                        "details": health.details,
                        "consecutive_failures": health.consecutive_failures,
                        "last_success": health.last_success.isoformat() if health.last_success else None
                    }
                return {"error": "Component not found"}

            # Return all components
            return {
                comp_type.value: {
                    "status": health.status.value,
                    "last_check": health.last_check.isoformat(),
                    "response_time": health.response_time,
                    "message": health.message,
                    "consecutive_failures": health.consecutive_failures
                }
                for comp_type, health in self.component_health.items()
            }

    def get_performance_baselines(self) -> Dict[str, Any]:
        """Get current performance baselines"""
        with self._baseline_lock:
            return {
                key: {
                    "component_type": baseline.component_type.value,
                    "metric_name": baseline.metric_name,
                    "baseline_value": baseline.baseline_value,
                    "threshold_warning": baseline.threshold_warning,
                    "threshold_critical": baseline.threshold_critical,
                    "measurement_unit": baseline.measurement_unit,
                    "last_updated": baseline.last_updated.isoformat(),
                    "sample_count": baseline.sample_count
                }
                for key, baseline in self.performance_baselines.items()
            }

    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        with self._health_lock:
            total_components = len(self.component_health)
            healthy_count = sum(1 for h in self.component_health.values() if h.status == UploadComponentStatus.HEALTHY)
            degraded_count = sum(1 for h in self.component_health.values() if h.status == UploadComponentStatus.DEGRADED)
            unhealthy_count = sum(1 for h in self.component_health.values() if h.status == UploadComponentStatus.UNHEALTHY)

            # Calculate overall health score
            health_score = (healthy_count * 100 + degraded_count * 50) / total_components

            # Determine overall status
            if unhealthy_count > 0:
                overall_status = "unhealthy"
            elif degraded_count > 0:
                overall_status = "degraded"
            else:
                overall_status = "healthy"

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": overall_status,
                "health_score": round(health_score, 2),
                "components": {
                    "total": total_components,
                    "healthy": healthy_count,
                    "degraded": degraded_count,
                    "unhealthy": unhealthy_count
                },
                "monitoring_active": self._running,
                "last_check": max((h.last_check for h in self.component_health.values()), default=datetime.utcnow()).isoformat()
            }