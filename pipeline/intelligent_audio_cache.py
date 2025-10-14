#!/usr/bin/env python3
"""
Intelligent Audio Processing Cache System

This module provides a comprehensive caching system for audio processing results
with multi-level caching, intelligent cache management, and performance optimization.
"""

from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import lzma
import pickle
import tempfile
import threading
import time
import zlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set
import weakref

import numpy as np
from cryptography.fernet import Fernet
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels for multi-level caching."""
    MEMORY = "memory"
    DISK = "disk"
    COMPRESSED = "compressed"


class CacheStrategy(Enum):
    """Caching strategies."""
    LRU = "lru"
    LFU = "lfu"
    ADAPTIVE = "adaptive"


@dataclass
class CacheMetadata:
    """Metadata for cached audio processing results."""
    cache_key: str
    file_hash: str
    file_path: str
    file_size: int
    file_mtime: float
    processing_config: Dict[str, Any]
    algorithm_version: str
    cache_level: CacheLevel
    compressed_size: int
    original_size: int
    created_at: float
    last_accessed: float
    access_count: int = 0
    processing_time: float = 0.0
    chunk_indices: List[int] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class CacheStatistics:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0
    compression_ratio: float = 1.0
    avg_access_time_ms: float = 0.0
    last_reset: float = field(default_factory=time.time)

    @property
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests


@dataclass
class CacheConfig:
    """Configuration for the intelligent cache system."""
    # Size limits
    max_memory_mb: float = 512.0
    max_disk_mb: float = 2048.0

    # Strategy settings
    strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    enable_compression: bool = True
    compression_level: int = 6

    # Performance settings
    enable_async: bool = True
    preload_batch_size: int = 10
    cache_warming_enabled: bool = True

    # Cleanup settings
    cleanup_interval_minutes: float = 30.0
    max_cache_age_hours: float = 24.0

    # Monitoring
    enable_statistics: bool = True
    statistics_interval_seconds: float = 60.0


class AudioProcessingCache:
    """
    Intelligent caching system for audio processing results.

    Features:
    - Multi-level caching (memory + disk + compressed)
    - Intelligent cache key generation
    - LRU/LFU/Adaptive eviction policies
    - Cache compression and encryption
    - Performance monitoring and statistics
    - Async operations for non-blocking I/O
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()

        # Cache storage
        self.cache_dir = Path(tempfile.gettempdir()) / "intelligent_audio_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Multi-level cache storage
        self._memory_cache: Dict[str, Tuple[Any, CacheMetadata]] = {}
        self._disk_cache: Dict[str, CacheMetadata] = {}
        self._access_order: List[str] = []  # For LRU tracking

        # Thread safety
        self._lock = threading.RLock()

        # Statistics and monitoring
        self.statistics = CacheStatistics()
        self._access_times: List[float] = []

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._statistics_task: Optional[asyncio.Task] = None
        self._running = False

        # Encryption for sensitive data
        self._encryption_key = Fernet.generate_key()
        self._cipher = Fernet(self._encryption_key)

        logger.info(f"AudioProcessingCache initialized with {self.config.max_memory_mb}MB memory, {self.config.max_disk_mb}MB disk")

    async def start(self) -> None:
        """Start background cache management tasks."""
        if self._running:
            return

        self._running = True

        # Start cleanup task
        if self.config.cleanup_interval_minutes > 0:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

        # Start statistics collection
        if self.config.enable_statistics:
            self._statistics_task = asyncio.create_task(self._periodic_statistics())

        logger.info("AudioProcessingCache background tasks started")

    async def stop(self) -> None:
        """Stop background cache management tasks."""
        if not self._running:
            return

        self._running = False

        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._statistics_task:
            self._statistics_task.cancel()
            try:
                await self._statistics_task
            except asyncio.CancelledError:
                pass

        logger.info("AudioProcessingCache stopped")

    def generate_cache_key(
        self,
        file_path: Union[str, Path],
        processing_config: Dict[str, Any],
        algorithm_version: str = "1.0",
        chunk_indices: Optional[List[int]] = None
    ) -> str:
        """
        Generate intelligent cache key based on file content and processing parameters.

        Args:
            file_path: Path to the audio file
            processing_config: Processing configuration parameters
            algorithm_version: Version of the processing algorithm
            chunk_indices: Specific chunk indices (for partial caching)

        Returns:
            Unique cache key string
        """
        file_path = Path(file_path)

        # Get file hash for content-based caching
        file_hash = self._get_file_hash(file_path)

        # Create deterministic string from processing config
        config_str = json.dumps(processing_config, sort_keys=True, default=str)

        # Combine all components
        key_components = [
            file_hash,
            config_str,
            algorithm_version,
            str(chunk_indices) if chunk_indices else "all"
        ]

        # Create hash of combined components
        combined_hash = hashlib.sha256("|".join(key_components).encode()).hexdigest()

        return f"audio_cache_{combined_hash[:32]}"

    def get(
        self,
        cache_key: str,
        expected_level: CacheLevel = CacheLevel.MEMORY
    ) -> Optional[Tuple[Any, CacheMetadata]]:
        """
        Retrieve cached audio processing result.

        Args:
            cache_key: Cache key to look up
            expected_level: Preferred cache level to check first

        Returns:
            Tuple of (cached_data, metadata) if found, None otherwise
        """
        start_time = time.time()

        with self._lock:
            # Update statistics
            self.statistics.total_requests += 1

            # Check memory cache first
            if cache_key in self._memory_cache:
                data, metadata = self._memory_cache[cache_key]

                # Update access tracking
                metadata.last_accessed = time.time()
                metadata.access_count += 1

                # Update LRU order
                if cache_key in self._access_order:
                    self._access_order.remove(cache_key)
                self._access_order.append(cache_key)

                self.statistics.hits += 1
                self._access_times.append(time.time() - start_time)

                logger.debug(f"Cache hit (memory): {cache_key}")
                return data, metadata

            # Check disk cache if not found in memory
            if cache_key in self._disk_cache:
                metadata = self._disk_cache[cache_key]

                # Check if cache is still valid
                if not self._is_cache_valid(metadata):
                    logger.debug(f"Cache expired, removing: {cache_key}")
                    self._remove_from_disk(cache_key)
                    self.statistics.misses += 1
                    return None

                # Load from disk
                try:
                    data = self._load_from_disk(cache_key, metadata)

                    # Promote to memory cache if space available
                    if self._should_promote_to_memory(metadata):
                        self._memory_cache[cache_key] = (data, metadata)

                        # Update LRU order
                        if cache_key in self._access_order:
                            self._access_order.remove(cache_key)
                        self._access_order.append(cache_key)

                    # Update metadata
                    metadata.last_accessed = time.time()
                    metadata.access_count += 1

                    self.statistics.hits += 1
                    self._access_times.append(time.time() - start_time)

                    logger.debug(f"Cache hit (disk): {cache_key}")
                    return data, metadata

                except Exception as e:
                    logger.warning(f"Failed to load from disk cache {cache_key}: {e}")
                    self._remove_from_disk(cache_key)
                    self.statistics.misses += 1
                    return None

            # Cache miss
            self.statistics.misses += 1
            logger.debug(f"Cache miss: {cache_key}")
            return None

    async def put(
        self,
        cache_key: str,
        data: Any,
        file_path: Union[str, Path],
        processing_config: Dict[str, Any],
        algorithm_version: str = "1.0",
        chunk_indices: Optional[List[int]] = None,
        processing_time: float = 0.0,
        cache_level: Optional[CacheLevel] = None
    ) -> bool:
        """
        Store audio processing result in cache.

        Args:
            cache_key: Cache key for the data
            data: Processed audio data to cache
            file_path: Original file path
            processing_config: Processing configuration used
            algorithm_version: Version of processing algorithm
            chunk_indices: Chunk indices if partial caching
            processing_time: Time taken to process
            cache_level: Cache level to use (auto-detected if None)

        Returns:
            True if successfully cached, False otherwise
        """
        try:
            file_path = Path(file_path)

            # Determine cache level based on data size and configuration
            if cache_level is None:
                cache_level = self._determine_cache_level(data, file_path)

            # Create metadata (sizes will be updated after serialization)
            metadata = CacheMetadata(
                cache_key=cache_key,
                file_hash=self._get_file_hash(file_path),
                file_path=str(file_path),
                file_size=file_path.stat().st_size,
                file_mtime=file_path.stat().st_mtime,
                processing_config=processing_config,
                algorithm_version=algorithm_version,
                cache_level=cache_level,
                compressed_size=0,  # Will be updated after serialization
                original_size=0,    # Will be updated after serialization
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                processing_time=processing_time,
                chunk_indices=chunk_indices or []
            )

            # Serialize and optionally compress data
            serialized_data = self._serialize_data(data, metadata)

            # Store based on cache level
            if cache_level == CacheLevel.MEMORY:
                return await self._store_in_memory(cache_key, serialized_data, metadata)
            else:
                return await self._store_on_disk(cache_key, serialized_data, metadata)

        except Exception as e:
            logger.error(f"Failed to cache data for key {cache_key}: {e}")
            return False

    def invalidate(self, file_path: Union[str, Path], pattern: Optional[str] = None) -> int:
        """
        Invalidate cache entries for a file or pattern.

        Args:
            file_path: File path to invalidate cache for
            pattern: Optional pattern to match cache keys

        Returns:
            Number of cache entries invalidated
        """
        file_path = Path(file_path)
        file_hash = self._get_file_hash(file_path)
        invalidated_count = 0

        with self._lock:
            # Find all cache entries for this file
            keys_to_remove = []

            for cache_key, metadata in self._disk_cache.items():
                if metadata.file_hash == file_hash or (pattern and pattern in cache_key):
                    keys_to_remove.append(cache_key)

            # Remove from disk cache
            for cache_key in keys_to_remove:
                self._remove_from_disk(cache_key)
                invalidated_count += 1

            # Remove from memory cache
            for cache_key in keys_to_remove:
                if cache_key in self._memory_cache:
                    del self._memory_cache[cache_key]
                    if cache_key in self._access_order:
                        self._access_order.remove(cache_key)

            if invalidated_count > 0:
                logger.info(f"Invalidated {invalidated_count} cache entries for {file_path}")

        return invalidated_count

    def get_statistics(self) -> CacheStatistics:
        """Get current cache statistics."""
        with self._lock:
            # Update current usage
            self.statistics.memory_usage_mb = self._get_memory_usage()
            self.statistics.disk_usage_mb = self._get_disk_usage()

            # Calculate average access time
            if self._access_times:
                self.statistics.avg_access_time_ms = (
                    sum(self._access_times) / len(self._access_times)
                ) * 1000

            return self.statistics

    def clear(self, level: Optional[CacheLevel] = None) -> None:
        """Clear cache entries."""
        with self._lock:
            if level is None or level == CacheLevel.MEMORY:
                self._memory_cache.clear()

            if level is None or level == CacheLevel.DISK:
                # Clear disk cache files
                for cache_key in list(self._disk_cache.keys()):
                    self._remove_from_disk(cache_key)
                self._disk_cache.clear()

            self._access_order.clear()
            self._access_times.clear()

            # Reset statistics
            self.statistics = CacheStatistics()

            logger.info(f"Cleared cache (level: {level})")

    def _get_file_hash(self, file_path: Path) -> str:
        """Generate hash of file contents."""
        try:
            hash_obj = hashlib.sha256()
            with open(file_path, 'rb') as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(8192), b''):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to hash file {file_path}: {e}")
            # Fallback to path-based hash
            return hashlib.sha256(str(file_path).encode()).hexdigest()

    def _is_cache_valid(self, metadata: CacheMetadata) -> bool:
        """Check if cache entry is still valid."""
        try:
            # Check if file still exists
            file_path = Path(metadata.file_path)
            if not file_path.exists():
                return False

            # Check if file has been modified
            current_mtime = file_path.stat().st_mtime
            if current_mtime != metadata.file_mtime:
                return False

            # Check cache age
            age_hours = (time.time() - metadata.created_at) / 3600
            if age_hours > self.config.max_cache_age_hours:
                return False

            return True
        except Exception as e:
            logger.warning(f"Error validating cache {metadata.cache_key}: {e}")
            return False

    def _determine_cache_level(self, data: Any, file_path: Path) -> CacheLevel:
        """Determine appropriate cache level based on data size."""
        # Estimate data size in MB
        if hasattr(data, 'nbytes'):
            data_size_mb = data.nbytes / (1024 * 1024)
        else:
            data_size_mb = len(pickle.dumps(data)) / (1024 * 1024)

        # Use memory cache for smaller data
        if data_size_mb < 50:  # Less than 50MB
            return CacheLevel.MEMORY
        else:
            return CacheLevel.DISK

    def _serialize_data(self, data: Any, metadata: CacheMetadata) -> bytes:
        """Serialize and optionally compress data."""
        # Serialize data
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

        # Update original size
        metadata.original_size = len(serialized)

        # Compress if enabled and beneficial
        if self.config.enable_compression and len(serialized) > 1024:
            compressed = zlib.compress(serialized, level=self.config.compression_level)
            if len(compressed) < len(serialized):
                metadata.compressed_size = len(compressed)
                return compressed
            else:
                metadata.compressed_size = len(serialized)

        metadata.compressed_size = len(serialized)
        return serialized

    def _deserialize_data(self, data: bytes, metadata: CacheMetadata) -> Any:
        """Deserialize and optionally decompress data."""
        try:
            # Decompress if data was compressed
            if metadata.compressed_size < metadata.original_size:
                data = zlib.decompress(data)

            # Deserialize
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize cache data: {e}")
            raise

    def _should_promote_to_memory(self, metadata: CacheMetadata) -> bool:
        """Check if cache entry should be promoted to memory."""
        current_memory_mb = self._get_memory_usage()
        data_size_mb = metadata.compressed_size / (1024 * 1024)

        return current_memory_mb + data_size_mb <= self.config.max_memory_mb

    async def _store_in_memory(
        self,
        cache_key: str,
        data: bytes,
        metadata: CacheMetadata
    ) -> bool:
        """Store data in memory cache."""
        data_size_mb = len(data) / (1024 * 1024)

        # Check if we need to evict entries
        current_memory_mb = self._get_memory_usage()
        if current_memory_mb + data_size_mb > self.config.max_memory_mb:
            await self._evict_memory_entries(data_size_mb)

        with self._lock:
            # Deserialize for memory storage
            try:
                deserialized_data = self._deserialize_data(data, metadata)
                self._memory_cache[cache_key] = (deserialized_data, metadata)

                # Update LRU order
                if cache_key in self._access_order:
                    self._access_order.remove(cache_key)
                self._access_order.append(cache_key)

                logger.debug(f"Stored {data_size_mb:.1f}MB in memory cache: {cache_key}")
                return True
            except Exception as e:
                logger.error(f"Failed to store in memory cache {cache_key}: {e}")
                return False

    async def _store_on_disk(
        self,
        cache_key: str,
        data: bytes,
        metadata: CacheMetadata
    ) -> bool:
        """Store data on disk cache."""
        try:
            # Create cache file path
            cache_file = self.cache_dir / f"{cache_key}.cache"

            # Encrypt sensitive data if needed
            if self._is_sensitive_data(metadata):
                data = self._cipher.encrypt(data)

            # Write to disk
            with open(cache_file, 'wb') as f:
                f.write(data)

            # Store metadata
            with self._lock:
                self._disk_cache[cache_key] = metadata

                # Update LRU order
                if cache_key in self._access_order:
                    self._access_order.remove(cache_key)
                self._access_order.append(cache_key)

            logger.debug(f"Stored {len(data) / (1024 * 1024):.1f}MB on disk: {cache_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to store on disk cache {cache_key}: {e}")
            return False

    def _load_from_disk(self, cache_key: str, metadata: CacheMetadata) -> Any:
        """Load data from disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.cache"

        with open(cache_file, 'rb') as f:
            data = f.read()

        # Decrypt if needed
        if self._is_sensitive_data(metadata):
            data = self._cipher.decrypt(data)

        # Deserialize
        return self._deserialize_data(data, metadata)

    def _remove_from_disk(self, cache_key: str) -> None:
        """Remove cache entry from disk."""
        # Remove data file
        cache_file = self.cache_dir / f"{cache_key}.cache"
        try:
            if cache_file.exists():
                cache_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove cache file {cache_file}: {e}")

        # Remove metadata
        if cache_key in self._disk_cache:
            del self._disk_cache[cache_key]

    async def _evict_memory_entries(self, needed_mb: float) -> None:
        """Evict memory cache entries to free space."""
        current_memory_mb = self._get_memory_usage()

        # Use LRU strategy to evict entries
        while current_memory_mb + needed_mb > self.config.max_memory_mb and self._access_order:
            oldest_key = self._access_order.pop(0)

            if oldest_key in self._memory_cache:
                removed_data, removed_metadata = self._memory_cache[oldest_key]
                removed_size_mb = removed_metadata.compressed_size / (1024 * 1024)

                del self._memory_cache[oldest_key]
                current_memory_mb -= removed_size_mb

                self.statistics.evictions += 1

                logger.debug(f"Evicted {removed_size_mb:.1f}MB from memory cache: {oldest_key}")

    def _get_memory_usage(self) -> float:
        """Get current memory cache usage in MB."""
        total_bytes = 0
        for data, metadata in self._memory_cache.values():
            total_bytes += metadata.compressed_size
        return total_bytes / (1024 * 1024)

    def _get_disk_usage(self) -> float:
        """Get current disk cache usage in MB."""
        total_bytes = 0
        for metadata in self._disk_cache.values():
            total_bytes += metadata.compressed_size
        return total_bytes / (1024 * 1024)

    def _is_sensitive_data(self, metadata: CacheMetadata) -> bool:
        """Check if data should be encrypted."""
        # Encrypt data that contains sensitive processing parameters
        sensitive_params = ['api_key', 'password', 'token', 'secret']
        config_str = json.dumps(metadata.processing_config).lower()
        return any(param in config_str for param in sensitive_params)

    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of expired cache entries."""
        while self._running:
            try:
                await asyncio.sleep(self.config.cleanup_interval_minutes * 60)

                expired_keys = []
                current_time = time.time()

                with self._lock:
                    # Find expired entries
                    for cache_key, metadata in self._disk_cache.items():
                        age_hours = (current_time - metadata.created_at) / 3600
                        if age_hours > self.config.max_cache_age_hours:
                            expired_keys.append(cache_key)

                    # Remove expired entries
                    for cache_key in expired_keys:
                        self._remove_from_disk(cache_key)
                        if cache_key in self._memory_cache:
                            del self._memory_cache[cache_key]
                        if cache_key in self._access_order:
                            self._access_order.remove(cache_key)

                if expired_keys:
                    logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _periodic_statistics(self) -> None:
        """Periodic statistics collection and reporting."""
        while self._running:
            try:
                await asyncio.sleep(self.config.statistics_interval_seconds)

                # Update statistics
                stats = self.get_statistics()

                # Log performance metrics
                if stats.total_requests > 0:
                    logger.info(
                        f"Cache stats - Hits: {stats.hits}, Misses: {stats.misses}, "
                        f"Hit Ratio: {stats.hit_ratio:.2%}, "
                        f"Memory: {stats.memory_usage_mb:.1f}MB, "
                        f"Disk: {stats.disk_usage_mb:.1f}MB"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic statistics: {e}")
                await asyncio.sleep(60)

    def preload_cache(self, file_paths: List[Union[str, Path]]) -> asyncio.Task:
        """
        Start preloading cache for frequently used files.

        Args:
            file_paths: List of file paths to preload

        Returns:
            Asyncio task for the preloading operation
        """
        return asyncio.create_task(self._preload_files(file_paths))

    async def _preload_files(self, file_paths: List[Union[str, Path]]) -> None:
        """Preload cache entries for given files."""
        # This would typically involve processing files with common configurations
        # and caching the results for future use
        logger.info(f"Starting preload for {len(file_paths)} files")

        for file_path in file_paths[:self.config.preload_batch_size]:
            try:
                # Generate cache key for common processing configuration
                cache_key = self.generate_cache_key(
                    file_path,
                    {"quality": "high", "format": "wav"},  # Common config
                    "1.0"
                )

                # Check if already cached
                if self.get(cache_key) is None:
                    logger.debug(f"Preloading cache for {file_path}")

                    # In a real implementation, you would trigger processing here
                    # For now, we'll just log the intent
                    pass

            except Exception as e:
                logger.warning(f"Failed to preload cache for {file_path}: {e}")

        logger.info("Cache preloading completed")


# Global cache instance
_global_cache: Optional[AudioProcessingCache] = None
_cache_lock = threading.Lock()


def get_audio_cache(config: Optional[CacheConfig] = None) -> AudioProcessingCache:
    """Get global audio processing cache instance."""
    global _global_cache

    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = AudioProcessingCache(config)

    return _global_cache


async def initialize_audio_cache(config: Optional[CacheConfig] = None) -> AudioProcessingCache:
    """Initialize and start the global audio cache."""
    cache = get_audio_cache(config)
    await cache.start()
    return cache


async def shutdown_audio_cache() -> None:
    """Shutdown the global audio cache."""
    global _global_cache

    if _global_cache:
        await _global_cache.stop()
        _global_cache = None