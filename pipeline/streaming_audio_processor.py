#!/usr/bin/env python3
"""
StreamingAudioProcessor - Memory-efficient chunk-based audio processing pipeline.

This module provides a progressive audio processing system that handles large files
efficiently by processing them in configurable chunks rather than loading entire
files into memory.
"""

from __future__ import annotations
import asyncio
import gc
import io
import logging
import os
import psutil
import tempfile
import threading
import time
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Generator
import weakref

import numpy as np
from scipy.signal import butter, sosfiltfilt

import librosa
import soundfile as sf
import noisereduce as nr
import pyloudnorm as pyln
import webrtcvad

from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Audio processing modes."""
    STREAMING = "streaming"
    MEMORY = "memory"
    HYBRID = "hybrid"


class ChunkState(Enum):
    """Chunk processing states."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ChunkInfo:
    """Information about an audio chunk."""
    index: int
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    size_bytes: int
    state: ChunkState = ChunkState.PENDING
    processing_time: float = 0.0
    retry_count: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingConfig:
    """Configuration for audio processing."""
    # Chunk settings
    chunk_size_mb: float = 16.0
    overlap_samples: int = 1024
    max_retries: int = 3

    # Memory management
    max_memory_gb: float = 2.0
    enable_caching: bool = True
    cache_dir: Optional[Path] = None

    # Performance
    parallel_processing: bool = True
    max_workers: int = 4
    progress_interval_seconds: float = 1.0

    # Quality settings
    target_lufs: float = -16.0
    limiter_margin_db: float = 0.8
    min_sample_rate: int = 16000

    # Processing options
    enable_silence_trimming: bool = True
    enable_noise_reduction: bool = True
    enable_speech_enhancement: bool = True
    enable_loudness_normalization: bool = True


@dataclass
class ProcessingProgress:
    """Progress information for audio processing."""
    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    skipped_chunks: int = 0
    current_chunk: int = 0
    estimated_time_remaining: float = 0.0
    memory_usage_mb: float = 0.0
    processing_speed_chunks_per_sec: float = 0.0
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)


class ChunkCache:
    """LRU cache for processed audio chunks."""

    def __init__(self, max_size_mb: float = 512, cache_dir: Optional[Path] = None):
        self.max_size_mb = max_size_mb
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "audio_chunk_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Tuple[Path, float, int]] = {}  # hash -> (path, timestamp, size)
        self._access_order: List[str] = []
        self._lock = threading.RLock()

    def _get_cache_key(self, chunk_info: ChunkInfo) -> str:
        """Generate cache key for chunk."""
        return f"{chunk_info.index}_{chunk_info.start_sample}_{chunk_info.end_sample}"

    def get(self, chunk_info: ChunkInfo) -> Optional[np.ndarray]:
        """Retrieve chunk from cache."""
        cache_key = self._get_cache_key(chunk_info)

        with self._lock:
            if cache_key not in self._cache:
                return None

            cache_path, _, _ = self._cache[cache_key]

            try:
                chunk_data, _ = librosa.load(cache_path, sr=None, mono=False)
                # Update access order for LRU
                if cache_key in self._access_order:
                    self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
                return chunk_data
            except Exception as e:
                logger.warning(f"Failed to load cached chunk {cache_key}: {e}")
                self._remove(cache_key)
                return None

    def put(self, chunk_info: ChunkInfo, chunk_data: np.ndarray) -> None:
        """Store chunk in cache."""
        cache_key = self._get_cache_key(chunk_info)

        # Estimate size in MB
        size_mb = chunk_data.nbytes / (1024 * 1024)

        with self._lock:
            # Check if we need to evict old entries
            self._evict_if_needed(size_mb)

            # Save to temporary file
            cache_path = self.cache_dir / f"chunk_{cache_key}.npy"
            try:
                np.save(cache_path, chunk_data)
                self._cache[cache_key] = (cache_path, time.time(), chunk_data.nbytes)
                self._access_order.append(cache_key)
            except Exception as e:
                logger.warning(f"Failed to cache chunk {cache_key}: {e}")

    def _evict_if_needed(self, new_chunk_size_mb: float) -> None:
        """Evict old cache entries if size limit exceeded."""
        current_size_mb = sum(size for _, _, size in self._cache.values()) / (1024 * 1024)

        while current_size_mb + new_chunk_size_mb > self.max_size_mb and self._access_order:
            oldest_key = self._access_order.pop(0)
            self._remove(oldest_key)
            current_size_mb = sum(size for _, _, size in self._cache.values()) / (1024 * 1024)

    def _remove(self, cache_key: str) -> None:
        """Remove cache entry."""
        if cache_key in self._cache:
            cache_path, _, _ = self._cache[cache_key]
            try:
                if cache_path.exists():
                    cache_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_path}: {e}")
            del self._cache[cache_key]

    def clear(self) -> None:
        """Clear all cached chunks."""
        with self._lock:
            for cache_path, _, _ in self._cache.values():
                try:
                    if cache_path.exists():
                        cache_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_path}: {e}")
            self._cache.clear()
            self._access_order.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_size_mb = sum(size for _, _, size in self._cache.values()) / (1024 * 1024)
            return {
                'cached_chunks': len(self._cache),
                'total_size_mb': total_size_mb,
                'max_size_mb': self.max_size_mb,
                'hit_ratio': 'N/A'  # Would need hit/miss tracking
            }


class StreamingAudioProcessor:
    """
    Memory-efficient audio processor that handles large files using chunk-based processing.

    This class provides:
    - Configurable chunk sizes for memory management
    - Async processing with progress reporting
    - Automatic memory cleanup and monitoring
    - Chunk caching for performance optimization
    - Support for resuming interrupted processing
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.chunk_cache = ChunkCache(
            max_size_mb=self.config.chunk_size_mb * 8,  # 8x chunk size for cache
            cache_dir=self.config.cache_dir
        ) if self.config.enable_caching else None

        # Intelligent cache integration
        self._intelligent_cache = None
        self._intelligent_cache_enabled = False

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix="AudioProcessor"
        )

        # Processing state
        self._processing = False
        self._cancelled = False
        self._progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
        self._lock = threading.RLock()

        # Memory monitoring
        self._memory_check_interval = 1.0
        self._last_memory_check = time.time()

        logger.info(f"StreamingAudioProcessor initialized with chunk_size={self.config.chunk_size_mb}MB, max_workers={self.config.max_workers}")

    def enable_intelligent_cache(self, cache_config: Optional[Dict[str, Any]] = None) -> None:
        """Enable intelligent caching for this processor."""
        try:
            from .intelligent_audio_cache import get_audio_cache, CacheConfig

            config = CacheConfig(**(cache_config or {}))
            self._intelligent_cache = get_audio_cache(config)
            self._intelligent_cache_enabled = True

            logger.info("Intelligent cache enabled for StreamingAudioProcessor")
        except Exception as e:
            logger.warning(f"Failed to enable intelligent cache: {e}")
            self._intelligent_cache = None
            self._intelligent_cache_enabled = False

    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
            if self.chunk_cache:
                self.chunk_cache.clear()
        except Exception:
            pass

    async def process_audio_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
        mode: ProcessingMode = ProcessingMode.STREAMING
    ) -> Dict[str, Any]:
        """
        Process audio file with progressive chunk-based approach.

        Args:
            input_path: Path to input audio file
            output_path: Path to output audio file
            progress_callback: Optional callback for progress updates
            mode: Processing mode (streaming, memory, or hybrid)

        Returns:
            Dictionary with processing results and statistics
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input audio file not found: {input_path}")

        # Set up progress callback
        self._progress_callback = progress_callback

        try:
            # Get file info
            file_size = input_path.stat().st_size
            file_info = self._get_audio_file_info(input_path)

            # Determine processing strategy based on file size and mode
            if mode == ProcessingMode.MEMORY or (mode == ProcessingMode.HYBRID and file_size < 100 * 1024 * 1024):
                # Use traditional in-memory processing for smaller files
                return await self._process_memory_mode(input_path, output_path, file_info)
            else:
                # Use streaming chunk-based processing for large files
                return await self._process_streaming_mode(input_path, output_path, file_info)

        except Exception as e:
            logger.error(f"Error processing audio file {input_path}: {e}")
            raise
        finally:
            self._progress_callback = None

    def _get_audio_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get basic audio file information."""
        try:
            # Use librosa to get audio info without loading full file
            with sf.SoundFile(str(file_path)) as f:
                return {
                    'duration': f.frames / f.samplerate,
                    'sample_rate': f.samplerate,
                    'channels': f.channels,
                    'format': f.format,
                    'frames': f.frames
                }
        except Exception as e:
            logger.warning(f"Could not read audio file info for {file_path}: {e}")
            # Fallback: try to get basic info from pydub
            try:
                audio = AudioSegment.from_file(file_path)
                return {
                    'duration': len(audio) / 1000.0,
                    'sample_rate': audio.frame_rate,
                    'channels': audio.channels,
                    'format': 'unknown',
                    'frames': len(audio) * audio.channels
                }
            except Exception as e2:
                raise RuntimeError(f"Could not read audio file {file_path}: {e}, {e2}")

    async def _process_memory_mode(
        self,
        input_path: Path,
        output_path: Path,
        file_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process smaller files using traditional in-memory approach."""
        logger.info(f"Processing {input_path} in memory mode")

        start_time = time.time()

        try:
            # Load entire file into memory (traditional approach)
            audio = AudioSegment.from_file(input_path)

            # Convert to numpy for processing
            y, sr, channels = self._audiosegment_to_float_np(audio)

            # Apply enhancements
            y_enhanced = self._enhance_audio_chunk(y, sr, channels)

            # Convert back and save
            output_audio = self._float_np_to_audiosegment(y_enhanced, max(sr, self.config.min_sample_rate), channels)
            output_audio.export(output_path, format=output_path.suffix.lstrip('.').lower() or 'mp3')

            processing_time = time.time() - start_time

            return {
                'success': True,
                'mode': 'memory',
                'input_path': str(input_path),
                'output_path': str(output_path),
                'processing_time': processing_time,
                'file_size': input_path.stat().st_size,
                'chunks_processed': 1,
                'memory_usage_mb': y_enhanced.nbytes / (1024 * 1024)
            }

        except Exception as e:
            logger.error(f"Memory mode processing failed: {e}")
            raise

    async def _process_streaming_mode(
        self,
        input_path: Path,
        output_path: Path,
        file_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process large files using chunk-based streaming approach."""
        logger.info(f"Processing {input_path} in streaming mode")

        self._processing = True
        self._cancelled = False

        start_time = time.time()
        progress = ProcessingProgress()

        try:
            # Calculate chunk parameters
            chunk_info_list = self._calculate_chunks(file_info)

            progress.total_chunks = len(chunk_info_list)
            self._report_progress(progress)

            # Process chunks
            processed_chunks = []

            if self.config.parallel_processing:
                processed_chunks = await self._process_chunks_parallel(chunk_info_list, input_path, progress)
            else:
                processed_chunks = await self._process_chunks_sequential(chunk_info_list, input_path, progress)

            # Combine processed chunks
            await self._combine_chunks(processed_chunks, output_path, file_info)

            processing_time = time.time() - start_time

            # Final progress update
            progress.processed_chunks = len([c for c in processed_chunks if c.state == ChunkState.COMPLETED])
            progress.failed_chunks = len([c for c in processed_chunks if c.state == ChunkState.FAILED])
            self._report_progress(progress)

            return {
                'success': True,
                'mode': 'streaming',
                'input_path': str(input_path),
                'output_path': str(output_path),
                'processing_time': processing_time,
                'file_size': input_path.stat().st_size,
                'chunks_processed': progress.processed_chunks,
                'chunks_failed': progress.failed_chunks,
                'memory_usage_mb': self._get_memory_usage()
            }

        except asyncio.CancelledError:
            logger.info("Audio processing was cancelled")
            raise
        except Exception as e:
            logger.error(f"Streaming mode processing failed: {e}")
            raise
        finally:
            self._processing = False
            # Force garbage collection
            gc.collect()

    def _calculate_chunks(self, file_info: Dict[str, Any]) -> List[ChunkInfo]:
        """Calculate chunk boundaries for the audio file."""
        sample_rate = file_info['sample_rate']
        duration = file_info['duration']
        channels = file_info['channels']

        # Calculate samples per chunk based on target size
        target_samples = int((self.config.chunk_size_mb * 1024 * 1024 * 8) / (channels * 2))  # Assuming 16-bit audio

        # Adjust for minimum processing size
        min_samples = sample_rate  # At least 1 second
        chunk_samples = max(target_samples, min_samples)

        chunks = []
        chunk_index = 0
        start_sample = 0

        while start_sample < file_info['frames']:
            end_sample = min(start_sample + chunk_samples, file_info['frames'])

            # Add overlap for seamless processing
            if chunk_index > 0 and self.config.overlap_samples > 0:
                start_sample = max(0, start_sample - self.config.overlap_samples)

            chunk_info = ChunkInfo(
                index=chunk_index,
                start_sample=start_sample,
                end_sample=end_sample,
                start_time=start_sample / sample_rate,
                end_time=end_sample / sample_rate,
                size_bytes=(end_sample - start_sample) * channels * 2  # 16-bit samples
            )

            chunks.append(chunk_info)
            start_sample = end_sample
            chunk_index += 1

        logger.info(f"Calculated {len(chunks)} chunks for {duration:.2f}s audio")
        return chunks

    async def _process_chunks_parallel(
        self,
        chunk_info_list: List[ChunkInfo],
        input_path: Path,
        progress: ProcessingProgress
    ) -> List[ChunkInfo]:
        """Process chunks in parallel using thread pool."""
        processed_chunks = [None] * len(chunk_info_list)  # Pre-allocate for index mapping

        # Submit all chunk processing tasks
        future_to_chunk = {}
        for chunk_info in chunk_info_list:
            if self._cancelled:
                break

            future = self.executor.submit(
                self._process_single_chunk,
                chunk_info,
                input_path
            )
            future_to_chunk[future] = chunk_info

        # Collect results as they complete
        for future in as_completed(future_to_chunk.keys()):
            if self._cancelled:
                break

            try:
                chunk_info = future_to_chunk[future]
                result_chunk_info = future.result()

                # Update progress
                progress.current_chunk = result_chunk_info.index
                progress.processed_chunks += 1 if result_chunk_info.state == ChunkState.COMPLETED else 0
                progress.failed_chunks += 1 if result_chunk_info.state == ChunkState.FAILED else 0

                processed_chunks[result_chunk_info.index] = result_chunk_info
                self._report_progress(progress)

            except Exception as e:
                logger.error(f"Chunk processing failed: {e}")
                chunk_info = future_to_chunk[future]
                chunk_info.state = ChunkState.FAILED
                chunk_info.error_message = str(e)
                processed_chunks[chunk_info.index] = chunk_info

        return processed_chunks

    async def _process_chunks_sequential(
        self,
        chunk_info_list: List[ChunkInfo],
        input_path: Path,
        progress: ProcessingProgress
    ) -> List[ChunkInfo]:
        """Process chunks sequentially."""
        processed_chunks = []

        for chunk_info in chunk_info_list:
            if self._cancelled:
                break

            try:
                # Update progress
                progress.current_chunk = chunk_info.index
                self._report_progress(progress)

                # Process chunk
                processed_chunk = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._process_single_chunk,
                    chunk_info,
                    input_path
                )

                # Update counters
                if processed_chunk.state == ChunkState.COMPLETED:
                    progress.processed_chunks += 1
                elif processed_chunk.state == ChunkState.FAILED:
                    progress.failed_chunks += 1

                processed_chunks.append(processed_chunk)

            except Exception as e:
                logger.error(f"Sequential chunk processing failed for chunk {chunk_info.index}: {e}")
                chunk_info.state = ChunkState.FAILED
                chunk_info.error_message = str(e)
                processed_chunks.append(chunk_info)

        return processed_chunks

    def _process_single_chunk(self, chunk_info: ChunkInfo, input_path: Path) -> ChunkInfo:
        """Process a single audio chunk."""
        start_time = time.time()

        try:
            # Check intelligent cache first (if available)
            if hasattr(self, '_intelligent_cache') and self._intelligent_cache:
                # Generate cache key for this chunk
                from .intelligent_audio_cache import get_audio_cache

                cache = get_audio_cache()
                cache_key = cache.generate_cache_key(
                    input_path,
                    {
                        'chunk_index': chunk_info.index,
                        'start_sample': chunk_info.start_sample,
                        'end_sample': chunk_info.end_sample,
                        'processing_config': self.config.__dict__
                    },
                    algorithm_version="1.0",
                    chunk_indices=[chunk_info.index]
                )

                # Try to get from intelligent cache
                cached_result = cache.get(cache_key)
                if cached_result is not None:
                    cached_data, metadata = cached_result
                    chunk_info.state = ChunkState.COMPLETED
                    chunk_info.metadata['cached'] = True
                    chunk_info.metadata['cache_level'] = metadata.cache_level.value
                    chunk_info.processing_time = time.time() - start_time
                    logger.debug(f"Chunk {chunk_info.index} loaded from intelligent cache")
                    return chunk_info

            # Check basic chunk cache (fallback)
            if self.chunk_cache:
                cached_data = self.chunk_cache.get(chunk_info)
                if cached_data is not None:
                    chunk_info.state = ChunkState.COMPLETED
                    chunk_info.metadata['cached'] = True
                    chunk_info.processing_time = time.time() - start_time
                    return chunk_info

            # Load chunk from file
            chunk_data, sample_rate, channels = self._load_audio_chunk(input_path, chunk_info)

            # Apply enhancements
            enhanced_data = self._enhance_audio_chunk(chunk_data, sample_rate, channels)

            # Cache the result in intelligent cache (if available)
            if hasattr(self, '_intelligent_cache') and self._intelligent_cache:
                try:
                    from .intelligent_audio_cache import get_audio_cache

                    cache = get_audio_cache()
                    cache_key = cache.generate_cache_key(
                        input_path,
                        {
                            'chunk_index': chunk_info.index,
                            'start_sample': chunk_info.start_sample,
                            'end_sample': chunk_info.end_sample,
                            'processing_config': self.config.__dict__
                        },
                        algorithm_version="1.0",
                        chunk_indices=[chunk_info.index]
                    )

                    # Store in intelligent cache asynchronously (create task safely)
                    try:
                        loop = asyncio.get_event_loop()
                        if not loop.is_closed():
                            loop.create_task(cache.put(
                                cache_key,
                                enhanced_data,
                                input_path,
                                {
                                    'chunk_index': chunk_info.index,
                                    'start_sample': chunk_info.start_sample,
                                    'end_sample': chunk_info.end_sample,
                                    'processing_config': self.config.__dict__
                                },
                                algorithm_version="1.0",
                                chunk_indices=[chunk_info.index],
                                processing_time=time.time() - start_time
                            ))
                    except RuntimeError:
                        # No event loop available, skip async caching
                        pass
                except Exception as e:
                    logger.warning(f"Failed to store in intelligent cache: {e}")

            # Cache in basic cache (fallback)
            if self.chunk_cache:
                self.chunk_cache.put(chunk_info, enhanced_data)

            # Save chunk to temporary file for later combination
            temp_file = self._save_chunk_to_temp(enhanced_data, sample_rate, chunk_info)
            chunk_info.metadata['temp_file'] = str(temp_file)

            chunk_info.state = ChunkState.COMPLETED
            chunk_info.processing_time = time.time() - start_time

            logger.debug(f"Processed chunk {chunk_info.index} in {chunk_info.processing_time:.3f}s")

        except Exception as e:
            logger.error(f"Failed to process chunk {chunk_info.index}: {e}")
            chunk_info.state = ChunkState.FAILED
            chunk_info.error_message = str(e)
            chunk_info.processing_time = time.time() - start_time

        return chunk_info

    def _load_audio_chunk(
        self,
        file_path: Path,
        chunk_info: ChunkInfo
    ) -> Tuple[np.ndarray, int, int]:
        """Load a specific chunk from audio file."""
        try:
            # Use soundfile for efficient chunk loading
            with sf.SoundFile(str(file_path)) as f:
                start_frame = chunk_info.start_sample
                max_frames = chunk_info.end_sample - chunk_info.start_sample

                # Seek to start position
                f.seek(start_frame)

                # Read chunk
                chunk_data = f.read(max_frames)

                if chunk_data.size == 0:
                    raise ValueError(f"No data read for chunk {chunk_info.index}")

                # Ensure proper shape
                if len(chunk_data.shape) == 1:
                    chunk_data = chunk_data.reshape(-1, 1)

                return chunk_data, f.samplerate, f.channels

        except Exception as e:
            logger.warning(f"Failed to load chunk with soundfile, trying librosa: {e}")
            # Fallback to librosa for problematic files
            try:
                # Load entire file and slice (less efficient but more compatible)
                full_data, sample_rate = librosa.load(file_path, sr=None, mono=False)

                start_sample = chunk_info.start_sample
                end_sample = min(chunk_info.end_sample, full_data.shape[0])

                chunk_data = full_data[start_sample:end_sample]

                if chunk_data.size == 0:
                    raise ValueError(f"No data sliced for chunk {chunk_info.index}")

                # Ensure proper shape
                if len(chunk_data.shape) == 1:
                    chunk_data = chunk_data.reshape(-1, 1)

                return chunk_data, sample_rate, full_data.shape[1] if len(full_data.shape) > 1 else 1

            except Exception as e2:
                raise RuntimeError(f"Failed to load audio chunk {chunk_info.index}: {e}, {e2}")

    def _enhance_audio_chunk(
        self,
        chunk_data: np.ndarray,
        sample_rate: int,
        channels: int
    ) -> np.ndarray:
        """Apply audio enhancements to a chunk."""
        try:
            # Ensure proper data type and shape
            chunk_data = np.atleast_2d(chunk_data).astype(np.float32)

            # Resample if needed
            target_sr = max(sample_rate, self.config.min_sample_rate)
            if sample_rate != target_sr:
                chunk_data = np.stack([
                    librosa.resample(chunk_data[:, c], orig_sr=sample_rate, target_sr=target_sr)
                    for c in range(channels)
                ], axis=1)

            # Apply VAD-based speech enhancement
            if self.config.enable_speech_enhancement:
                chunk_data, _ = self._apply_vad_mask(chunk_data, target_sr)

            # Apply noise reduction
            if self.config.enable_noise_reduction:
                chunk_data = self._apply_noise_reduction(chunk_data, target_sr)

            # Apply loudness normalization
            if self.config.enable_loudness_normalization:
                chunk_data = self._apply_loudness_normalization(chunk_data, target_sr)

            return chunk_data

        except Exception as e:
            logger.error(f"Audio enhancement failed: {e}")
            # Return original data if enhancement fails
            return np.atleast_2d(chunk_data).astype(np.float32)

    def _apply_vad_mask(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply VAD-based speech enhancement to chunk."""
        try:
            vad = webrtcvad.Vad(3)  # Aggressive setting
            hop_t = 0.02  # 20ms frames
            hop_samples = int(sample_rate * hop_t)

            # Process in overlapping windows for VAD
            mono = audio_data.mean(axis=1).astype(np.float32)

            # For chunk processing, we'll use a simpler approach
            # In a full implementation, you'd want to handle cross-chunk VAD continuity
            mask = np.ones_like(mono, dtype=bool)

            # Apply basic VAD logic
            for i in range(0, len(mono) - hop_samples, hop_samples):
                chunk = mono[i:i+hop_samples]

                if len(chunk) == hop_samples:
                    # Convert to 16-bit PCM for VAD
                    pcm16 = (np.clip(chunk, -1, 1) * 32767).astype(np.int16).tobytes()

                    try:
                        is_speech = vad.is_speech(pcm16, sample_rate=sample_rate)
                        if not is_speech:
                            # Attenuate non-speech
                            mask[i:i+hop_samples] = False
                    except Exception:
                        pass  # Keep as speech if VAD fails

            # Apply mask
            attenuated = audio_data.copy()
            attenuated[~mask] *= 10 ** (-60/20.0)  # -60dB attenuation

            return attenuated, mask

        except Exception as e:
            logger.warning(f"VAD processing failed: {e}")
            return audio_data, np.ones(audio_data.shape[0], dtype=bool)

    def _apply_noise_reduction(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply noise reduction to chunk."""
        try:
            # Simple noise reduction using noisereduce
            reduced = np.zeros_like(audio_data, dtype=np.float32)

            for ch in range(audio_data.shape[1]):
                reduced[:, ch] = nr.reduce_noise(
                    y=audio_data[:, ch],
                    sr=sample_rate,
                    stationary=False,
                    prop_decrease=0.95,
                    time_mask_smooth_ms=80,
                    freq_mask_smooth_hz=300,
                ).astype(np.float32)

            return reduced

        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio_data

    def _apply_loudness_normalization(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply loudness normalization to chunk."""
        try:
            # Simple loudness normalization
            meter = pyln.Meter(sample_rate)

            # Calculate loudness for mono mix
            mono = audio_data.mean(axis=1)
            try:
                current_lufs = meter.integrated_loudness(mono)
                gain_db = np.clip(self.config.target_lufs - current_lufs, -12.0, 12.0)
                normalized = audio_data * (10 ** (gain_db / 20.0))
            except Exception:
                # If loudness measurement fails, apply gentle gain
                normalized = audio_data * 1.2

            return np.clip(normalized, -0.95, 0.95).astype(np.float32)

        except Exception as e:
            logger.warning(f"Loudness normalization failed: {e}")
            return audio_data

    def _save_chunk_to_temp(self, chunk_data: np.ndarray, sample_rate: int, chunk_info: ChunkInfo) -> Path:
        """Save processed chunk to temporary file."""
        temp_dir = Path(tempfile.gettempdir()) / "audio_chunks"
        temp_dir.mkdir(exist_ok=True)

        temp_file = temp_dir / f"chunk_{chunk_info.index}_{int(time.time()*1000)}.wav"

        try:
            # Save as WAV file
            sf.write(str(temp_file), chunk_data, sample_rate, subtype='PCM_16', format='WAV')
            return temp_file
        except Exception as e:
            logger.error(f"Failed to save chunk {chunk_info.index} to temp file: {e}")
            raise

    async def _combine_chunks(
        self,
        processed_chunks: List[ChunkInfo],
        output_path: Path,
        file_info: Dict[str, Any]
    ) -> None:
        """Combine processed chunks into final output file."""
        try:
            # Filter successful chunks and sort by index
            successful_chunks = [
                c for c in processed_chunks
                if c.state == ChunkState.COMPLETED and 'temp_file' in c.metadata
            ]
            successful_chunks.sort(key=lambda x: x.index)

            if not successful_chunks:
                raise RuntimeError("No chunks were successfully processed")

            # Create file list for ffmpeg concat
            concat_list = []
            for chunk_info in successful_chunks:
                temp_file = Path(chunk_info.metadata['temp_file'])
                if temp_file.exists():
                    concat_list.append(temp_file)

            if not concat_list:
                raise RuntimeError("No temporary chunk files found for concatenation")

            # Use ffmpeg to concatenate chunks
            await self._concatenate_with_ffmpeg(concat_list, output_path)

            # Clean up temporary files
            for temp_file in concat_list:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_file}: {e}")

        except Exception as e:
            logger.error(f"Failed to combine chunks: {e}")
            raise

    async def _concatenate_with_ffmpeg(self, input_files: List[Path], output_path: Path) -> None:
        """Concatenate audio files using ffmpeg."""
        try:
            # Create concat list file
            list_file = Path(tempfile.mktemp(suffix='.txt'))

            with open(list_file, 'w') as f:
                for input_file in input_files:
                    f.write(f"file '{input_file.as_posix()}'\n")

            # Run ffmpeg concat command
            import subprocess

            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(list_file),
                '-c', 'copy',
                str(output_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            # Clean up list file
            try:
                list_file.unlink()
            except Exception:
                pass

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                raise RuntimeError(f"ffmpeg concatenation failed: {error_msg}")

        except Exception as e:
            logger.error(f"ffmpeg concatenation error: {e}")
            raise

    def _report_progress(self, progress: ProcessingProgress) -> None:
        """Report processing progress."""
        if self._progress_callback:
            try:
                # Update memory usage and timing
                progress.memory_usage_mb = self._get_memory_usage()
                progress.last_update = time.time()

                if progress.total_chunks > 0:
                    elapsed_time = progress.last_update - progress.start_time
                    progress.processing_speed_chunks_per_sec = progress.processed_chunks / elapsed_time

                    if progress.processed_chunks > 0:
                        avg_time_per_chunk = elapsed_time / progress.processed_chunks
                        remaining_chunks = progress.total_chunks - progress.processed_chunks
                        progress.estimated_time_remaining = remaining_chunks * avg_time_per_chunk

                self._progress_callback(progress)

            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def cancel_processing(self) -> None:
        """Cancel current processing operation."""
        self._cancelled = True
        logger.info("Audio processing cancellation requested")

    # Utility methods for compatibility with existing code
    def _audiosegment_to_float_np(self, seg: AudioSegment) -> Tuple[np.ndarray, int, int]:
        """Convert AudioSegment to float numpy array (compatibility method)."""
        samples = np.array(seg.get_array_of_samples())
        if seg.channels > 1:
            samples = samples.reshape((-1, seg.channels))
        else:
            samples = samples.reshape((-1, 1))

        if seg.sample_width == 2:
            y = samples.astype(np.float32) / 32768.0
        elif seg.sample_width == 4:
            y = samples.astype(np.float32) / 2147483648.0
        else:
            y = samples.astype(np.float32) / 32768.0

        return y, int(seg.frame_rate), int(seg.channels)

    def _float_np_to_audiosegment(self, y: np.ndarray, sr: int, channels: int) -> AudioSegment:
        """Convert float numpy array to AudioSegment (compatibility method)."""
        y = np.atleast_2d(y)
        y = np.clip(y, -1.0, 1.0).astype(np.float32)

        if y.shape[1] != channels:
            if channels == 1:
                y = y.mean(axis=1, keepdims=True)
            else:
                y = np.tile(y.mean(axis=1, keepdims=True), (1, channels))

        buf = io.BytesIO()
        pcm = (y * 32767.0).astype(np.int16)
        sf.write(buf, pcm, sr, subtype="PCM_16", format="WAV")
        buf.seek(0)

        return AudioSegment.from_file(buf, format="wav")


# Convenience functions for easy integration
async def process_audio_streaming(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    chunk_size_mb: float = 16.0,
    progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
) -> Dict[str, Any]:
    """
    Convenience function to process audio with streaming approach.

    Args:
        input_path: Input audio file path
        output_path: Output audio file path
        chunk_size_mb: Size of processing chunks in MB
        progress_callback: Optional progress callback

    Returns:
        Processing results dictionary
    """
    config = ProcessingConfig(chunk_size_mb=chunk_size_mb)
    processor = StreamingAudioProcessor(config)

    return await processor.process_audio_file(
        input_path,
        output_path,
        progress_callback,
        mode=ProcessingMode.STREAMING
    )


def get_optimal_chunk_size(file_size_bytes: int, available_memory_mb: float = None) -> float:
    """
    Calculate optimal chunk size based on file size and available memory.

    Args:
        file_size_bytes: Size of audio file in bytes
        available_memory_mb: Available memory in MB (optional, will detect if not provided)

    Returns:
        Optimal chunk size in MB
    """
    if available_memory_mb is None:
        try:
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        except Exception:
            available_memory_mb = 2048  # Default assumption

    file_size_mb = file_size_bytes / (1024 * 1024)

    # For small files, use memory mode (single chunk)
    if file_size_mb < 50:
        return file_size_mb

    # For large files, calculate chunk size to use ~25% of available memory
    target_memory_usage_mb = available_memory_mb * 0.25

    # Estimate: each MB of audio needs ~2-3MB for processing (original + processed + overhead)
    estimated_chunk_mb = target_memory_usage_mb / 3

    # Cap between reasonable bounds
    return max(8.0, min(estimated_chunk_mb, 128.0))