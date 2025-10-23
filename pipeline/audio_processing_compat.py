#!/usr/bin/env python3
"""
Audio Processing Compatibility Layer

This module provides backward compatibility for existing audio processing functions
while enabling the new progressive processing capabilities for large files.

It acts as a bridge between the old memory-intensive approach and the new
streaming/chunk-based system.
"""

from __future__ import annotations
import asyncio
import logging
import os
import psutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import progressive processing modules
try:
    from .streaming_audio_processor import (
        StreamingAudioProcessor, ProcessingConfig, ProcessingMode,
        ProcessingProgress, get_optimal_chunk_size
    )
    from .progressive_silence_trimmer import (
        ProgressiveSilenceTrimmer, TrimmingConfig, TrimmingProgress
    )
    PROGRESSIVE_AVAILABLE = True
except ImportError:
    PROGRESSIVE_AVAILABLE = False
    logger.warning("Progressive processing not available, using fallback methods")


class AudioProcessingAdapter:
    """
    Adapter class that provides backward-compatible audio processing functions
    with automatic fallback to progressive processing for large files.
    """

    def __init__(self):
        self.file_size_threshold_mb = 100  # Threshold for using progressive processing
        self._cache: Dict[str, Any] = {}

    def should_use_progressive(self, file_path: Union[str, Path]) -> bool:
        """
        Determine if progressive processing should be used for a file.

        Args:
            file_path: Path to the audio file

        Returns:
            True if progressive processing should be used
        """
        if not PROGRESSIVE_AVAILABLE:
            return False

        try:
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            return file_size_mb > self.file_size_threshold_mb
        except Exception as e:
            logger.warning(f"Could not determine file size for {file_path}: {e}")
            return False

    def get_available_memory_mb(self) -> float:
        """Get available system memory in MB."""
        try:
            return psutil.virtual_memory().available / (1024 * 1024)
        except Exception:
            return 2048.0  # Default assumption

    async def trim_silence_adaptive(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        min_silence_ms: int = 1000,
        silence_threshold_db: float = -40.0,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Trim silence with automatic method selection based on file size.

        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            min_silence_ms: Minimum silence duration to trim
            silence_threshold_db: Silence threshold in dB
            progress_callback: Optional progress callback

        Returns:
            Processing results dictionary
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if self.should_use_progressive(input_path):
            return await self._trim_silence_progressive(
                input_path, output_path, min_silence_ms, silence_threshold_db, progress_callback
            )
        else:
            return self._trim_silence_traditional(
                input_path, output_path, min_silence_ms, silence_threshold_db, progress_callback
            )

    async def _trim_silence_progressive(
        self,
        input_path: Path,
        output_path: Path,
        min_silence_ms: int,
        silence_threshold_db: float,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]]
    ) -> Dict[str, Any]:
        """Progressive silence trimming for large files."""
        logger.info(f"Using progressive silence trimming for {input_path}")

        # Create progressive trimmer
        trimming_config = TrimmingConfig(
            min_silence_duration_ms=min_silence_ms,
            silence_threshold_db=silence_threshold_db,
            chunk_size_ms=5000,  # 5 second chunks
            overlap_ms=100
        )

        trimmer = ProgressiveSilenceTrimmer(trimming_config)

        def trimming_progress(progress: TrimmingProgress):
            if progress_callback:
                progress_info = {
                    'stage': 'trimming',
                    'processed_chunks': progress.processed_chunks,
                    'total_chunks': progress.total_chunks,
                    'silence_segments_found': progress.silence_segments_found,
                    'current_position_ms': progress.current_position_ms,
                    'processing_speed': progress.processing_speed_chunks_per_sec
                }
                progress_callback(progress_info)

        # Perform trimming
        start_time = asyncio.get_event_loop().time
        result = await trimmer.trim_silence_streaming(
            input_path, output_path, trimming_progress
        )
        processing_time = asyncio.get_event_loop().time - start_time

        return {
            'success': True,
            'method': 'progressive',
            'processing_time': processing_time,
            'original_duration_ms': result.get('original_duration_ms', 0),
            'trimmed_duration_ms': len(AudioSegment.from_file(output_path)),
            'silence_segments_found': result.get('silence_segments_found', 0),
            'silence_removed_ms': result.get('trimming_plan', {}).get('silence_to_remove_ms', 0)
        }

    def _trim_silence_traditional(
        self,
        input_path: Path,
        output_path: Path,
        min_silence_ms: int,
        silence_threshold_db: float,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]]
    ) -> Dict[str, Any]:
        """Traditional silence trimming for smaller files."""
        logger.info(f"Using traditional silence trimming for {input_path}")

        try:
            # Import and use the original trim_silence functions
            from .trim_silence import (
                audiosegment_to_float_np, float_np_to_audiosegment,
                trim_only_oversized_silences_keep_head, enhance_auto,
                MIN_SR, TARGET_LUFS, LIMITER_MARGIN_DB
            )

            # Load audio
            audio = AudioSegment.from_file(input_path)

            # Apply trimming
            processed, n_trimmed, seconds_removed = trim_only_oversized_silences_keep_head(
                audio,
                target_silence_ms=min_silence_ms,
                keep_head_ms=min_silence_ms // 2,  # Keep half for safety
                min_silence_ms=min_silence_ms,
                threshold_offset_db=int(-silence_threshold_db),
                absolute_thresh_dbfs=None,
                seek_step_ms=10,
                crossfade_ms=10
            )

            # Apply enhancement
            y, sr, ch = audiosegment_to_float_np(processed)
            if y.ndim == 1:
                y = y[:, None]
            y_enhanced = enhance_auto(y, sr)
            output_audio = float_np_to_audiosegment(y_enhanced, max(sr, MIN_SR), ch)

            # Save result
            output_audio.export(output_path, format=output_path.suffix.lstrip('.'))

            return {
                'success': True,
                'method': 'traditional',
                'processing_time': 0.0,  # Not tracked in traditional method
                'original_duration_ms': len(audio),
                'trimmed_duration_ms': len(output_audio),
                'silence_segments_found': n_trimmed,
                'silence_removed_ms': int(seconds_removed * 1000)
            }

        except Exception as e:
            logger.error(f"Traditional trimming failed: {e}")
            raise

    async def enhance_audio_adaptive(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Enhance audio with automatic method selection based on file size.

        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            progress_callback: Optional progress callback

        Returns:
            Processing results dictionary
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if self.should_use_progressive(input_path):
            return await self._enhance_audio_progressive(
                input_path, output_path, progress_callback
            )
        else:
            return self._enhance_audio_traditional(
                input_path, output_path, progress_callback
            )

    async def _enhance_audio_progressive(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]]
    ) -> Dict[str, Any]:
        """Progressive audio enhancement for large files."""
        logger.info(f"Using progressive audio enhancement for {input_path}")

        # Calculate optimal chunk size
        file_size_bytes = input_path.stat().st_size
        available_memory_mb = self.get_available_memory_mb()
        chunk_size_mb = get_optimal_chunk_size(file_size_bytes, available_memory_mb)

        # Create processing config
        config = ProcessingConfig(
            chunk_size_mb=chunk_size_mb,
            max_memory_gb=available_memory_mb / 1024,
            enable_silence_trimming=False,  # Assume already trimmed
            enable_noise_reduction=True,
            enable_speech_enhancement=True,
            enable_loudness_normalization=True
        )

        processor = StreamingAudioProcessor(config)

        def processing_progress(progress: ProcessingProgress):
            if progress_callback:
                progress_info = {
                    'stage': 'enhancement',
                    'processed_chunks': progress.processed_chunks,
                    'total_chunks': progress.total_chunks,
                    'memory_usage_mb': progress.memory_usage_mb,
                    'processing_speed': progress.processing_speed_chunks_per_sec,
                    'estimated_time_remaining': progress.estimated_time_remaining
                }
                progress_callback(progress_info)

        # Process audio
        start_time = asyncio.get_event_loop().time
        result = await processor.process_audio_file(
            input_path, output_path, processing_progress, ProcessingMode.STREAMING
        )
        processing_time = asyncio.get_event_loop().time - start_time

        return {
            'success': True,
            'method': 'progressive',
            'processing_time': processing_time,
            'chunks_processed': result.get('chunks_processed', 0),
            'memory_usage_mb': result.get('memory_usage_mb', 0)
        }

    def _enhance_audio_traditional(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]]
    ) -> Dict[str, Any]:
        """Traditional audio enhancement for smaller files."""
        logger.info(f"Using traditional audio enhancement for {input_path}")

        try:
            # Import and use original enhancement functions
            from .trim_silence import (
                audiosegment_to_float_np, float_np_to_audiosegment, enhance_auto, MIN_SR
            )

            # Load and enhance audio
            audio = AudioSegment.from_file(input_path)
            y, sr, ch = audiosegment_to_float_np(audio)

            if y.ndim == 1:
                y = y[:, None]

            y_enhanced = enhance_auto(y, sr)
            output_audio = float_np_to_audiosegment(y_enhanced, max(sr, MIN_SR), ch)

            # Save result
            output_audio.export(output_path, format=output_path.suffix.lstrip('.'))

            return {
                'success': True,
                'method': 'traditional',
                'processing_time': 0.0,
                'memory_usage_mb': y_enhanced.nbytes / (1024 * 1024)
            }

        except Exception as e:
            logger.error(f"Traditional enhancement failed: {e}")
            raise

    def get_processing_stats(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get processing statistics and recommendations for a file.

        Args:
            file_path: Path to the audio file

        Returns:
            Dictionary with processing recommendations and statistics
        """
        file_path = Path(file_path)

        try:
            file_size_bytes = file_path.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)

            # Get audio duration for estimation
            try:
                audio = AudioSegment.from_file(file_path)
                duration_seconds = len(audio) / 1000.0 - 1.0
            except Exception:
                duration_seconds = 0.0

            # Determine recommended method
            use_progressive = self.should_use_progressive(file_path)

            # Estimate memory usage
            if use_progressive:
                available_memory_mb = self.get_available_memory_mb()
                estimated_chunk_size_mb = get_optimal_chunk_size(file_size_bytes, available_memory_mb)
                estimated_memory_usage_mb = estimated_chunk_size_mb * 3  # Processing overhead
            else:
                # Traditional method loads entire file
                estimated_memory_usage_mb = file_size_mb * 2  # Rough estimate

            return {
                'file_path': str(file_path),
                'file_size_mb': file_size_mb,
                'duration_seconds': duration_seconds,
                'recommended_method': 'progressive' if use_progressive else 'traditional',
                'estimated_memory_usage_mb': estimated_memory_usage_mb,
                'available_memory_mb': self.get_available_memory_mb(),
                'will_fit_in_memory': estimated_memory_usage_mb < self.get_available_memory_mb() * 0.8,
                'progressive_available': PROGRESSIVE_AVAILABLE
            }

        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            return {
                'file_path': str(file_path),
                'error': str(e),
                'recommended_method': 'traditional',
                'progressive_available': PROGRESSIVE_AVAILABLE
            }


# Global adapter instance
_audio_adapter: Optional[AudioProcessingAdapter] = None


def get_audio_adapter() -> AudioProcessingAdapter:
    """Get global audio processing adapter instance."""
    global _audio_adapter
    if _audio_adapter is None:
        _audio_adapter = AudioProcessingAdapter()
    return _audio_adapter


# Convenience functions for backward compatibility
async def trim_silence(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    min_silence_ms: int = 1000,
    silence_threshold_db: float = -40.0,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Backward-compatible silence trimming function with automatic method selection.

    Args:
        input_path: Input audio file path
        output_path: Output audio file path
        min_silence_ms: Minimum silence duration to trim
        silence_threshold_db: Silence threshold in dB
        progress_callback: Optional progress callback

    Returns:
        Processing results dictionary
    """
    adapter = get_audio_adapter()
    return await adapter.trim_silence_adaptive(
        input_path, output_path, min_silence_ms, silence_threshold_db, progress_callback
    )


async def enhance_audio(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Backward-compatible audio enhancement function with automatic method selection.

    Args:
        input_path: Input audio file path
        output_path: Output audio file path
        progress_callback: Optional progress callback

    Returns:
        Processing results dictionary
    """
    adapter = get_audio_adapter()
    return await adapter.enhance_audio_adaptive(input_path, output_path, progress_callback)


def get_processing_recommendations(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get processing recommendations for a file.

    Args:
        file_path: Path to the audio file

    Returns:
        Dictionary with recommendations and statistics
    """
    adapter = get_audio_adapter()
    return adapter.get_processing_stats(file_path)


# Utility functions for existing code
def is_large_audio_file(file_path: Union[str, Path], threshold_mb: float = 100.0) -> bool:
    """
    Check if an audio file is considered large.

    Args:
        file_path: Path to the audio file
        threshold_mb: Size threshold in MB

    Returns:
        True if file is large
    """
    try:
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        return file_size_mb > threshold_mb
    except Exception:
        return False


def estimate_processing_time(file_path: Union[str, Path]) -> Dict[str, float]:
    """
    Estimate processing time for different methods.

    Args:
        file_path: Path to the audio file

    Returns:
        Dictionary with time estimates for different methods
    """
    try:
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)

        # Rough estimates based on file size
        traditional_time = file_size_mb * 0.1  # 0.1 seconds per MB
        progressive_time = file_size_mb * 0.15  # 0.15 seconds per MB (slight overhead)

        return {
            'traditional_seconds': traditional_time,
            'progressive_seconds': progressive_time,
            'recommended_method': 'progressive' if file_size_mb > 100 else 'traditional'
        }
    except Exception:
        return {
            'traditional_seconds': 0.0,
            'progressive_seconds': 0.0,
            'recommended_method': 'traditional'
        }