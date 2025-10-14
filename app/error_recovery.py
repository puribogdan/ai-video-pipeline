# app/error_recovery.py
import asyncio
import json
import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
import threading
import redis
import boto3
from botocore.exceptions import ClientError, BotoCoreError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryCallState


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    NETWORK = "network"
    STORAGE = "storage"
    PROCESSING = "processing"
    EXTERNAL_SERVICE = "external_service"
    RESOURCE = "resource"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    CORRUPTION = "corruption"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    RETRY = "retry"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    MANUAL_INTERVENTION = "manual_intervention"
    SKIP = "skip"


@dataclass
class ErrorContext:
    """Context information for error tracking and correlation"""
    error_id: str
    job_id: Optional[str] = None
    correlation_id: Optional[str] = None
    component: str = "unknown"
    operation: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    user_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt"""
    attempt_id: str
    error_id: str
    strategy: RecoveryStrategy
    timestamp: datetime
    success: bool
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking"""
    name: str
    state: str = "closed"  # closed, open, half_open
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    success_count: int = 0
    last_success_time: Optional[datetime] = None
    next_attempt_time: Optional[datetime] = None


class ErrorClassifier:
    """Classifies errors into categories and severity levels"""

    def __init__(self):
        self.classification_rules = {
            # Network errors
            (ConnectionError, TimeoutError, OSError): (ErrorCategory.NETWORK, ErrorSeverity.HIGH),

            # Storage errors
            (FileNotFoundError, PermissionError, IsADirectoryError): (ErrorCategory.STORAGE, ErrorSeverity.HIGH),

            # External service errors
            (ClientError, BotoCoreError): (ErrorCategory.EXTERNAL_SERVICE, ErrorSeverity.HIGH),

            # Processing errors
            (ValueError, TypeError, AttributeError): (ErrorCategory.PROCESSING, ErrorSeverity.MEDIUM),

            # Resource errors
            (MemoryError, SystemError): (ErrorCategory.RESOURCE, ErrorSeverity.CRITICAL),

            # Timeout errors
            (asyncio.TimeoutError,): (ErrorCategory.TIMEOUT, ErrorSeverity.HIGH),

            # Corruption errors
            (UnicodeDecodeError, json.JSONDecodeError): (ErrorCategory.CORRUPTION, ErrorSeverity.HIGH),
        }

    def classify_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify an error based on its type and context"""
        context = context or {}

        # Check for specific error types first
        for error_types, (category, severity) in self.classification_rules.items():
            if isinstance(error, error_types):
                return category, severity

        # Context-based classification
        error_message = str(error).lower()
        if any(keyword in error_message for keyword in ['connection', 'network', 'timeout']):
            return ErrorCategory.NETWORK, ErrorSeverity.HIGH
        elif any(keyword in error_message for keyword in ['disk', 'storage', 'file not found', 'permission']):
            return ErrorCategory.STORAGE, ErrorSeverity.HIGH
        elif any(keyword in error_message for keyword in ['memory', 'out of memory']):
            return ErrorCategory.RESOURCE, ErrorSeverity.CRITICAL
        elif any(keyword in error_message for keyword in ['corrupt', 'decode', 'invalid format']):
            return ErrorCategory.CORRUPTION, ErrorSeverity.HIGH

        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM


class RetryStrategy:
    """Enhanced retry strategy with jitter and adaptive backoff"""

    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay

    def calculate_delay(self, attempt: int, error: Exception) -> float:
        """Calculate delay for retry attempt with exponential backoff and jitter"""
        # Exponential backoff
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)

        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.5, 1.5)
        delay_with_jitter = delay * jitter

        return delay_with_jitter

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Determine if a retry should be attempted"""
        if attempt >= self.max_attempts:
            return False

        # Don't retry certain types of errors
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            return False

        return True


class CircuitBreaker:
    """Circuit breaker implementation for external services"""

    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitBreakerState(name)
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state.state == "open":
                if self.state.next_attempt_time and datetime.utcnow() >= self.state.next_attempt_time:
                    self.state.state = "half_open"
                else:
                    raise RuntimeError(f"Circuit breaker '{self.name}' is OPEN")

        try:
            result = func(*args, **kwargs)

            with self._lock:
                if self.state.state == "half_open":
                    self.state.state = "closed"
                    self.state.success_count += 1
                    self.state.last_success_time = datetime.utcnow()

            return result

        except Exception as e:
            with self._lock:
                self.state.failure_count += 1
                self.state.last_failure_time = datetime.utcnow()

                if self.state.failure_count >= self.failure_threshold:
                    self.state.state = "open"
                    self.state.next_attempt_time = datetime.utcnow() + timedelta(seconds=self.recovery_timeout)

            raise e

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        with self._lock:
            return {
                "name": self.state.name,
                "state": self.state.state,
                "failure_count": self.state.failure_count,
                "success_count": self.state.success_count,
                "last_failure": self.state.last_failure_time.isoformat() if self.state.last_failure_time else None,
                "last_success": self.state.last_success_time.isoformat() if self.state.last_success_time else None,
                "next_attempt": self.state.next_attempt_time.isoformat() if self.state.next_attempt_time else None,
            }


class FallbackManager:
    """Manages fallback strategies for different failure scenarios"""

    def __init__(self):
        self.fallbacks: Dict[str, List[Tuple[int, Callable]]] = defaultdict(list)
        self.fallback_results: Dict[str, Any] = {}

    def add_fallback(self, operation: str, fallback_func: Callable, priority: int = 0):
        """Add a fallback function for an operation"""
        self.fallbacks[operation].append((priority, fallback_func))
        # Sort by priority (higher priority first)
        self.fallbacks[operation].sort(key=lambda x: x[0], reverse=True)

    async def execute_with_fallback(self, operation: str, primary_func: Callable,
                                  *args, **kwargs) -> Any:
        """Execute primary function with fallback options"""
        # Try primary function first
        try:
            return await self._execute_func(primary_func, *args, **kwargs)
        except Exception as e:
            print(f"Primary function failed for {operation}: {e}")

            # Try fallback functions in priority order
            for priority, fallback_func in self.fallbacks.get(operation, []):
                try:
                    result = await self._execute_func(fallback_func, *args, **kwargs)
                    self.fallback_results[operation] = {
                        "success": True,
                        "result": result,
                        "fallback_used": True,
                        "timestamp": datetime.utcnow()
                    }
                    return result
                except Exception as fallback_error:
                    print(f"Fallback {priority} failed for {operation}: {fallback_error}")
                    continue

            # No fallbacks succeeded
            self.fallback_results[operation] = {
                "success": False,
                "error": str(e),
                "fallback_used": False,
                "timestamp": datetime.utcnow()
            }
            raise e

    async def _execute_func(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function, handling both sync and async"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)


class UploadRecoveryManager:
    """Specialized recovery manager for upload failures"""

    def __init__(self, uploads_dir: Path):
        self.uploads_dir = uploads_dir
        self.chunk_size = 8 * 1024 * 1024  # 8MB chunks
        self.recovery_metadata: Dict[str, Dict[str, Any]] = {}
        self.max_resume_attempts = 3
        self.chunk_timeout = 300  # 5 minutes per chunk

    def detect_partial_upload(self, file_path: Path) -> Tuple[bool, Optional[int]]:
        """Detect if an upload is partial and return expected size"""
        if not file_path.exists():
            return False, None

        # Check for recovery metadata
        job_id = file_path.parent.name
        metadata_file = file_path.parent / ".upload_metadata.json"

        if metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text())
                expected_size = metadata.get("expected_size")
                if expected_size and file_path.stat().st_size < expected_size:
                    return True, expected_size
            except Exception:
                pass

        return False, None

    def save_upload_metadata(self, job_id: str, file_path: Path, expected_size: int,
                           chunk_info: Optional[Dict[str, Any]] = None):
        """Save upload metadata for recovery"""
        metadata = {
            "file_path": str(file_path),
            "expected_size": expected_size,
            "chunk_info": chunk_info or {},
            "timestamp": datetime.utcnow().isoformat(),
            "job_id": job_id,
            "upload_method": "chunked",
            "chunk_size": self.chunk_size,
            "resume_enabled": True
        }

        metadata_file = self.uploads_dir / job_id / ".upload_metadata.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def resume_upload(self, file_path: Path) -> Tuple[bool, str]:
        """Attempt to resume a partial upload"""
        is_partial, expected_size = self.detect_partial_upload(file_path)

        if not is_partial:
            return True, "Upload is complete"

        current_size = file_path.stat().st_size

        # For now, return failure - would need integration with actual upload mechanism
        return False, f"Partial upload detected: {current_size}/{expected_size} bytes"

    def create_chunked_upload_plan(self, file_path: Path, remote_path: str) -> Dict[str, Any]:
        """Create a plan for chunked upload with resume capability"""
        file_size = file_path.stat().st_size
        chunk_size = self.chunk_size

        # Calculate chunks
        chunks = []
        for i in range(0, file_size, chunk_size):
            chunk_start = i
            chunk_end = min(i + chunk_size, file_size)
            chunk_info = {
                "index": len(chunks),
                "start_byte": chunk_start,
                "end_byte": chunk_end,
                "size": chunk_end - chunk_start,
                "status": "pending",
                "uploaded_at": None
            }
            chunks.append(chunk_info)

        return {
            "file_path": str(file_path),
            "remote_path": remote_path,
            "file_size": file_size,
            "chunk_size": chunk_size,
            "total_chunks": len(chunks),
            "chunks": chunks,
            "created_at": datetime.utcnow().isoformat(),
            "resume_token": str(uuid.uuid4())
        }

    def validate_upload_integrity(self, local_path: Path, remote_path: Optional[str] = None) -> Dict[str, Any]:
        """Validate upload integrity and detect corruption"""
        if not local_path.exists():
            return {"valid": False, "error": "File does not exist"}

        try:
            # Basic file validation
            stat = local_path.stat()
            if stat.st_size == 0:
                return {"valid": False, "error": "File is empty"}

            # Check if file is still being written (size stability)
            import time
            size1 = stat.st_size
            time.sleep(1)
            size2 = local_path.stat().st_size

            if size1 != size2:
                return {"valid": False, "error": "File size changed - still being written"}

            # Try to read file header for format validation
            try:
                with open(local_path, 'rb') as f:
                    header = f.read(1024)
                    if len(header) < 64:
                        return {"valid": False, "error": "File too small for validation"}

                    # Basic format detection
                    if header.startswith(b'\xff\xfb') or header.startswith(b'\xff\xf3') or header.startswith(b'\xff\xf2'):
                        file_type = "mp3"
                    elif header.startswith(b'RIFF') and b'WAVE' in header:
                        file_type = "wav"
                    elif header.startswith(b'\x00\x00\x00\x20ftypM4A'):
                        file_type = "m4a"
                    else:
                        file_type = "unknown"

                    return {
                        "valid": True,
                        "file_size": stat.st_size,
                        "file_type": file_type,
                        "modified_time": stat.st_mtime,
                        "created_time": stat.st_ctime
                    }

            except Exception as e:
                return {"valid": False, "error": f"File read error: {str(e)}"}

        except Exception as e:
            return {"valid": False, "error": f"File validation error: {str(e)}"}

    def repair_corrupted_upload(self, file_path: Path, backup_path: Optional[Path] = None) -> Tuple[bool, str]:
        """Attempt to repair a corrupted upload"""
        # This would implement repair strategies like:
        # 1. Restore from backup
        # 2. Re-download from source
        # 3. Partial reconstruction

        if backup_path and backup_path.exists():
            try:
                # Restore from backup
                import shutil
                shutil.copy2(backup_path, file_path)
                return True, f"Restored from backup: {backup_path}"
            except Exception as e:
                return False, f"Failed to restore from backup: {str(e)}"

        return False, "No repair strategy available"

    def cleanup_failed_upload(self, job_id: str) -> bool:
        """Clean up artifacts from failed upload"""
        try:
            job_dir = self.uploads_dir / job_id

            # Remove upload metadata
            metadata_file = job_dir / ".upload_metadata.json"
            if metadata_file.exists():
                metadata_file.unlink()

            # Remove partial upload files
            for file_path in job_dir.glob("*"):
                if file_path.is_file() and file_path.suffix in ['.tmp', '.part', '.upload']:
                    file_path.unlink()

            return True

        except Exception as e:
            print(f"Failed to cleanup upload artifacts for job {job_id}: {e}")
            return False


class ErrorRecoveryManager:
    """Centralized error recovery management system"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = redis.from_url(redis_url)
        self.logger = logging.getLogger("error_recovery")
        self.classifier = ErrorClassifier()
        self.retry_strategy = RetryStrategy()
        self.fallback_manager = FallbackManager()
        self.upload_recovery = None

        # Circuit breakers for external services
        self.circuit_breakers = {
            "b2_storage": CircuitBreaker("b2_storage", failure_threshold=3, recovery_timeout=30.0),
            "redis": CircuitBreaker("redis", failure_threshold=5, recovery_timeout=10.0),
            "openai": CircuitBreaker("openai", failure_threshold=3, recovery_timeout=60.0),
            "replicate": CircuitBreaker("replicate", failure_threshold=3, recovery_timeout=60.0),
        }

        # Recovery tracking
        self.recovery_attempts: Dict[str, List[RecoveryAttempt]] = defaultdict(list)
        self.error_contexts: Dict[str, ErrorContext] = {}
        self._lock = threading.Lock()

        # Setup default fallbacks
        self._setup_default_fallbacks()

    def _setup_default_fallbacks(self):
        """Setup default fallback strategies"""

        # B2 storage fallback to local storage
        async def b2_to_local_fallback(operation: str, *args, **kwargs):
            if operation == "upload_to_b2":
                # Return local URL as fallback
                return f"http://localhost:8000/media/{kwargs.get('job_id', 'unknown')}.mp4"
            raise NotImplementedError(f"No fallback for operation: {operation}")

        self.fallback_manager.add_fallback("upload_to_b2", b2_to_local_fallback, priority=1)

        # Redis fallback to in-memory cache
        def redis_to_memory_fallback(operation: str, *args, **kwargs):
            if operation == "redis_get":
                return None  # Return None as cache miss
            elif operation == "redis_set":
                return True  # Pretend cache set succeeded
            raise NotImplementedError(f"No fallback for operation: {operation}")

        self.fallback_manager.add_fallback("redis_get", redis_to_memory_fallback, priority=1)
        self.fallback_manager.add_fallback("redis_set", redis_to_memory_fallback, priority=1)

    def create_error_context(self, error: Exception, job_id: Optional[str] = None,
                           component: str = "unknown", operation: str = "unknown",
                           metadata: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """Create error context for tracking"""
        error_id = str(uuid.uuid4())
        correlation_id = metadata.get("correlation_id") if metadata else None

        context = ErrorContext(
            error_id=error_id,
            job_id=job_id,
            correlation_id=correlation_id,
            component=component,
            operation=operation,
            metadata=metadata or {},
            stack_trace=self._get_stack_trace(error)
        )

        with self._lock:
            self.error_contexts[error_id] = context

        return context

    def _get_stack_trace(self, error: Exception) -> str:
        """Get formatted stack trace"""
        import traceback
        return "".join(traceback.format_exception(type(error), error, error.__traceback__))

    async def handle_error(self, error: Exception, context: ErrorContext,
                          recovery_strategies: Optional[List[RecoveryStrategy]] = None) -> Any:
        """Handle an error with appropriate recovery strategies"""
        start_time = time.time()

        # Classify the error
        category, severity = self.classifier.classify_error(error, context.metadata)

        # Update context with classification
        context.metadata.update({
            "error_category": category.value,
            "error_severity": severity.value,
            "error_type": type(error).__name__,
            "error_message": str(error)
        })

        # Determine recovery strategies
        if not recovery_strategies:
            recovery_strategies = self._select_recovery_strategies(category, severity)

        # Execute recovery strategies
        for strategy in recovery_strategies:
            attempt_id = str(uuid.uuid4())

            try:
                result = await self._execute_recovery_strategy(strategy, error, context)
                duration = time.time() - start_time

                # Record successful recovery attempt
                attempt = RecoveryAttempt(
                    attempt_id=attempt_id,
                    error_id=context.error_id,
                    strategy=strategy,
                    timestamp=datetime.utcnow(),
                    success=True,
                    duration=duration,
                    metadata={"result": result}
                )

                with self._lock:
                    self.recovery_attempts[context.error_id].append(attempt)

                self.logger.info(f"Recovery successful for error {context.error_id} using {strategy.value}")
                return result

            except Exception as recovery_error:
                duration = time.time() - start_time

                # Record failed recovery attempt
                attempt = RecoveryAttempt(
                    attempt_id=attempt_id,
                    error_id=context.error_id,
                    strategy=strategy,
                    timestamp=datetime.utcnow(),
                    success=False,
                    duration=duration,
                    error_message=str(recovery_error)
                )

                with self._lock:
                    self.recovery_attempts[context.error_id].append(attempt)

                self.logger.warning(f"Recovery strategy {strategy.value} failed for error {context.error_id}: {recovery_error}")

        # All recovery strategies failed
        duration = time.time() - start_time
        self.logger.error(f"All recovery strategies failed for error {context.error_id} after {duration:.2f}s")

        # Record final failure
        final_attempt = RecoveryAttempt(
            attempt_id=str(uuid.uuid4()),
            error_id=context.error_id,
            strategy=RecoveryStrategy.MANUAL_INTERVENTION,
            timestamp=datetime.utcnow(),
            success=False,
            duration=duration,
            error_message="All automated recovery strategies failed"
        )

        with self._lock:
            self.recovery_attempts[context.error_id].append(final_attempt)

        raise error

    def _select_recovery_strategies(self, category: ErrorCategory,
                                  severity: ErrorSeverity) -> List[RecoveryStrategy]:
        """Select appropriate recovery strategies based on error classification"""
        strategies = []

        if category == ErrorCategory.NETWORK:
            strategies.extend([RecoveryStrategy.RETRY, RecoveryStrategy.CIRCUIT_BREAKER])
        elif category == ErrorCategory.EXTERNAL_SERVICE:
            strategies.extend([RecoveryStrategy.CIRCUIT_BREAKER, RecoveryStrategy.FALLBACK])
        elif category == ErrorCategory.STORAGE:
            strategies.extend([RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK])
        elif category == ErrorCategory.RESOURCE:
            strategies.extend([RecoveryStrategy.GRACEFUL_DEGRADATION, RecoveryStrategy.MANUAL_INTERVENTION])
        elif category == ErrorCategory.TIMEOUT:
            strategies.extend([RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK])
        elif category == ErrorCategory.CORRUPTION:
            strategies.extend([RecoveryStrategy.SKIP, RecoveryStrategy.MANUAL_INTERVENTION])

        # Add retry for most error types unless severity is critical
        if severity != ErrorSeverity.CRITICAL and RecoveryStrategy.RETRY not in strategies:
            strategies.insert(0, RecoveryStrategy.RETRY)

        return strategies

    async def _execute_recovery_strategy(self, strategy: RecoveryStrategy,
                                       error: Exception, context: ErrorContext) -> Any:
        """Execute a specific recovery strategy"""
        if strategy == RecoveryStrategy.RETRY:
            return await self._execute_retry(error, context)
        elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return await self._execute_circuit_breaker(error, context)
        elif strategy == RecoveryStrategy.FALLBACK:
            return await self._execute_fallback(error, context)
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return await self._execute_graceful_degradation(error, context)
        else:
            raise NotImplementedError(f"Recovery strategy {strategy.value} not implemented")

    async def _execute_retry(self, error: Exception, context: ErrorContext) -> Any:
        """Execute retry strategy with exponential backoff"""
        operation = context.metadata.get("operation", "unknown")
        max_attempts = context.metadata.get("max_retries", 3)

        for attempt in range(max_attempts):
            try:
                # Get the original function from context
                original_func = context.metadata.get("original_function")
                if not original_func or not callable(original_func):
                    raise RuntimeError("No original function provided for retry")

                args = context.metadata.get("args", [])
                kwargs = context.metadata.get("kwargs", {})

                if asyncio.iscoroutinefunction(original_func):
                    return await original_func(*args, **kwargs)
                else:
                    return original_func(*args, **kwargs)

            except Exception as retry_error:
                if attempt == max_attempts - 1:
                    raise retry_error

                delay = self.retry_strategy.calculate_delay(attempt, retry_error)
                await asyncio.sleep(delay)

        raise RuntimeError("Retry strategy exhausted all attempts")

    async def _execute_circuit_breaker(self, error: Exception, context: ErrorContext) -> Any:
        """Execute circuit breaker strategy"""
        component = context.component

        if component in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[component]

            # Get the original function from context
            original_func = context.metadata.get("original_function")
            if not original_func:
                raise RuntimeError("No original function provided for circuit breaker")

            args = context.metadata.get("args", [])
            kwargs = context.metadata.get("kwargs", {})

            return circuit_breaker.call(original_func, *args, **kwargs)
        else:
            raise RuntimeError(f"No circuit breaker configured for component: {component}")

    async def _execute_fallback(self, error: Exception, context: ErrorContext) -> Any:
        """Execute fallback strategy"""
        operation = context.metadata.get("operation", "unknown")

        # Get the original function and args from context
        original_func = context.metadata.get("original_function")
        args = context.metadata.get("args", [])
        kwargs = context.metadata.get("kwargs", {})

        async def primary_wrapper():
            if original_func is None:
                raise RuntimeError("No original function provided for fallback execution")

            if asyncio.iscoroutinefunction(original_func):
                return await original_func(*args, **kwargs)
            else:
                return original_func(*args, **kwargs)

        return await self.fallback_manager.execute_with_fallback(operation, primary_wrapper, *args, **kwargs)

    async def _execute_graceful_degradation(self, error: Exception, context: ErrorContext) -> Any:
        """Execute graceful degradation strategy"""
        # This would implement reduced functionality based on the error type
        # For now, return a degraded result indicator
        return {"degraded_mode": True, "error": str(error), "timestamp": datetime.utcnow().isoformat()}

    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        with self._lock:
            total_attempts = sum(len(attempts) for attempts in self.recovery_attempts.values())
            successful_attempts = sum(
                len([a for a in attempts if a.success])
                for attempts in self.recovery_attempts.values()
            )

            return {
                "total_recovery_attempts": total_attempts,
                "successful_attempts": successful_attempts,
                "success_rate": successful_attempts / total_attempts if total_attempts > 0 else 0,
                "circuit_breaker_states": {
                    name: cb.get_state()
                    for name, cb in self.circuit_breakers.items()
                },
                "error_contexts": len(self.error_contexts),
                "fallback_results": dict(self.fallback_manager.fallback_results)
            }

    def cleanup_old_contexts(self, max_age_hours: int = 24):
        """Clean up old error contexts and recovery attempts"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)

        with self._lock:
            # Clean up error contexts
            old_contexts = [
                error_id for error_id, context in self.error_contexts.items()
                if context.timestamp < cutoff_time
            ]

            for error_id in old_contexts:
                del self.error_contexts[error_id]
                if error_id in self.recovery_attempts:
                    del self.recovery_attempts[error_id]


# Global error recovery manager instance
_error_recovery_manager = None
_recovery_lock = threading.Lock()


def get_error_recovery_manager() -> ErrorRecoveryManager:
    """Get the global error recovery manager instance"""
    global _error_recovery_manager

    if _error_recovery_manager is None:
        with _recovery_lock:
            if _error_recovery_manager is None:
                _error_recovery_manager = ErrorRecoveryManager()

    return _error_recovery_manager


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, redis.ConnectionError)),
    reraise=True
)
def redis_with_recovery(operation: str, *args, **kwargs) -> Any:
    """Execute Redis operation with error recovery"""
    manager = get_error_recovery_manager()
    redis_client = manager.redis

    def redis_operation():
        if operation == "get":
            return redis_client.get(*args, **kwargs)
        elif operation == "set":
            return redis_client.set(*args, **kwargs)
        elif operation == "ping":
            return redis_client.ping()
        else:
            raise ValueError(f"Unknown Redis operation: {operation}")

    context = manager.create_error_context(
        Exception(f"Redis operation: {operation}"),
        component="redis",
        operation=operation,
        metadata={"args": args, "kwargs": kwargs}
    )

    # Use circuit breaker for Redis operations
    return manager.circuit_breakers["redis"].call(redis_operation)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=20),
    retry=retry_if_exception_type((ClientError, BotoCoreError, Exception)),
    reraise=True
)
def b2_upload_with_recovery(upload_func: Callable, *args, **kwargs) -> Any:
    """Execute B2 upload operation with error recovery"""
    manager = get_error_recovery_manager()

    context = manager.create_error_context(
        Exception("B2 upload operation"),
        component="b2_storage",
        operation="upload",
        metadata={"function": upload_func.__name__, "args": args, "kwargs": kwargs}
    )

    # Use circuit breaker for B2 operations
    return manager.circuit_breakers["b2_storage"].call(upload_func, *args, **kwargs)


async def handle_error_with_recovery(error: Exception, job_id: Optional[str] = None,
                                   component: str = "unknown", operation: str = "unknown",
                                   metadata: Optional[Dict[str, Any]] = None) -> Any:
    """Convenience function to handle errors with recovery"""
    manager = get_error_recovery_manager()

    context = manager.create_error_context(
        error=error,
        job_id=job_id,
        component=component,
        operation=operation,
        metadata=metadata
    )

    return await manager.handle_error(error, context)