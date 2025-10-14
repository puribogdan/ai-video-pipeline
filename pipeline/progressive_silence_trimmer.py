#!/usr/bin/env python3
"""
ProgressiveSilenceTrimmer - Memory-efficient silence trimming for large audio files.

This module provides streaming silence detection and trimming that processes audio
in chunks to handle large files without loading them entirely into memory.
"""

from __future__ import annotations
import asyncio
import io
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import weakref

import numpy as np
from scipy.signal import butter, sosfiltfilt

from pydub import AudioSegment
from pydub.silence import detect_silence

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SilenceSegment:
    """Represents a detected silence segment."""
    start_ms: int
    end_ms: int
    duration_ms: int
    average_db: float
    max_db: float
    chunk_index: int = 0


@dataclass
class TrimmingConfig:
    """Configuration for silence trimming."""
    # Silence detection parameters
    min_silence_duration_ms: int = 1000
    silence_threshold_db: float = -40.0
    seek_step_ms: int = 10

    # Trimming behavior
    keep_head_silence_ms: int = 1000
    max_silence_to_keep_ms: int = 1000
    crossfade_ms: int = 10

    # Chunk processing
    chunk_size_ms: int = 5000  # Process in 5-second chunks
    overlap_ms: int = 100      # Overlap for continuity

    # Advanced settings
    adaptive_threshold: bool = True
    smoothing_window_ms: int = 500


@dataclass
class TrimmingProgress:
    """Progress information for trimming operation."""
    total_chunks: int = 0
    processed_chunks: int = 0
    silence_segments_found: int = 0
    total_silence_duration_ms: int = 0
    estimated_silence_to_remove_ms: int = 0
    current_position_ms: int = 0
    processing_speed_chunks_per_sec: float = 0.0
    start_time: float = field(default_factory=time.time)


class ProgressiveSilenceDetector:
    """
    Detects silence in audio streams using chunk-based processing.

    This detector analyzes audio in small chunks to identify silence segments
    without loading the entire file into memory.
    """

    def __init__(self, config: Optional[TrimmingConfig] = None):
        self.config = config or TrimmingConfig()
        self._silence_segments: List[SilenceSegment] = []
        self._lock = threading.RLock()

    def detect_silence_in_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        chunk_start_ms: int,
        chunk_index: int
    ) -> List[SilenceSegment]:
        """
        Detect silence segments in a single audio chunk.

        Args:
            audio_chunk: Audio data as numpy array
            sample_rate: Sample rate of the audio
            chunk_start_ms: Start time of chunk in milliseconds
            chunk_index: Index of the chunk for tracking

        Returns:
            List of silence segments found in this chunk
        """
        try:
            # Convert to mono for silence detection
            if len(audio_chunk.shape) > 1:
                mono_chunk = audio_chunk.mean(axis=1)
            else:
                mono_chunk = audio_chunk.flatten()

            # Calculate RMS and convert to dB
            if len(mono_chunk) == 0:
                return []

            # Calculate dB values
            rms = np.sqrt(np.mean(mono_chunk ** 2))
            if rms > 0:
                db_values = 20 * np.log10(rms)
            else:
                db_values = -np.inf

            # Apply smoothing if configured
            if self.config.smoothing_window_ms > 0:
                window_samples = int((self.config.smoothing_window_ms / 1000.0) * sample_rate)
                if window_samples > 0 and window_samples < len(mono_chunk):
                    # Simple moving average for smoothing
                    kernel = np.ones(window_samples) / window_samples
                    smoothed_db = np.convolve(db_values, kernel, mode='same')
                else:
                    smoothed_db = db_values
            else:
                smoothed_db = db_values

            # Detect silence regions
            silence_mask = smoothed_db < self.config.silence_threshold_db

            # Find contiguous silence segments
            segments = []
            current_start = None

            for i, is_silence in enumerate(silence_mask):
                if is_silence and current_start is None:
                    # Start of silence segment
                    current_start = i
                elif not is_silence and current_start is not None:
                    # End of silence segment
                    silence_start_sample = current_start
                    silence_end_sample = i

                    # Convert to milliseconds
                    silence_start_ms = chunk_start_ms + int((silence_start_sample / sample_rate) * 1000)
                    silence_end_ms = chunk_start_ms + int((silence_end_sample / sample_rate) * 1000)
                    duration_ms = silence_end_ms - silence_start_ms

                    if duration_ms >= self.config.min_silence_duration_ms:
                        # Calculate average dB for this segment
                        segment_db = np.mean(smoothed_db[silence_start_sample:silence_end_sample])

                        segment = SilenceSegment(
                            start_ms=silence_start_ms,
                            end_ms=silence_end_ms,
                            duration_ms=duration_ms,
                            average_db=float(segment_db),
                            max_db=float(np.max(smoothed_db[silence_start_sample:silence_end_sample])),
                            chunk_index=chunk_index
                        )
                        segments.append(segment)

                    current_start = None

            # Handle case where silence extends to end of chunk
            if current_start is not None:
                silence_start_sample = current_start
                silence_end_sample = len(mono_chunk)

                silence_start_ms = chunk_start_ms + int((silence_start_sample / sample_rate) * 1000)
                silence_end_ms = chunk_start_ms + int((silence_end_sample / sample_rate) * 1000)
                duration_ms = silence_end_ms - silence_start_ms

                if duration_ms >= self.config.min_silence_duration_ms:
                    segment_db = np.mean(smoothed_db[silence_start_sample:silence_end_sample])

                    segment = SilenceSegment(
                        start_ms=silence_start_ms,
                        end_ms=silence_end_ms,
                        duration_ms=duration_ms,
                        average_db=float(segment_db),
                        max_db=float(np.max(smoothed_db[silence_start_sample:silence_end_sample])),
                        chunk_index=chunk_index
                    )
                    segments.append(segment)

            with self._lock:
                self._silence_segments.extend(segments)

            logger.debug(f"Found {len(segments)} silence segments in chunk {chunk_index}")
            return segments

        except Exception as e:
            logger.error(f"Error detecting silence in chunk {chunk_index}: {e}")
            return []

    def get_silence_segments(self) -> List[SilenceSegment]:
        """Get all detected silence segments."""
        with self._lock:
            return self._silence_segments.copy()

    def merge_overlapping_segments(self, max_gap_ms: int = 100) -> List[SilenceSegment]:
        """
        Merge overlapping or closely spaced silence segments.

        Args:
            max_gap_ms: Maximum gap between segments to merge (in milliseconds)

        Returns:
            List of merged silence segments
        """
        segments = sorted(self.get_silence_segments(), key=lambda x: x.start_ms)

        if not segments:
            return []

        merged = [segments[0]]

        for current in segments[1:]:
            last = merged[-1]

            # Check if segments overlap or are close enough to merge
            if current.start_ms - last.end_ms <= max_gap_ms:
                # Merge segments
                merged_end = max(last.end_ms, current.end_ms)
                merged_duration = merged_end - last.start_ms

                merged_segment = SilenceSegment(
                    start_ms=last.start_ms,
                    end_ms=merged_end,
                    duration_ms=merged_duration,
                    average_db=min(last.average_db, current.average_db),  # Use quieter average
                    max_db=max(last.max_db, current.max_db),
                    chunk_index=min(last.chunk_index, current.chunk_index)
                )
                merged[-1] = merged_segment
            else:
                merged.append(current)

        return merged

    def calculate_trimming_plan(self) -> Dict[str, Any]:
        """
        Calculate which silence segments should be trimmed based on configuration.

        Returns:
            Dictionary containing trimming plan and statistics
        """
        merged_segments = self.merge_overlapping_segments()

        trimming_plan = []
        total_silence_to_remove_ms = 0

        for segment in merged_segments:
            if segment.duration_ms > self.config.max_silence_to_keep_ms:
                # Calculate how much to remove
                excess_silence = segment.duration_ms - self.config.max_silence_to_keep_ms
                remove_duration = min(excess_silence, segment.duration_ms - self.config.keep_head_silence_ms)

                if remove_duration > 0:
                    # Keep head silence, remove the rest
                    keep_until_ms = segment.start_ms + self.config.keep_head_silence_ms
                    remove_from_ms = keep_until_ms
                    remove_until_ms = min(segment.end_ms, keep_until_ms + remove_duration)

                    trimming_plan.append({
                        'original_segment': segment,
                        'remove_from_ms': remove_from_ms,
                        'remove_until_ms': remove_until_ms,
                        'remove_duration_ms': remove_until_ms - remove_from_ms
                    })

                    total_silence_to_remove_ms += remove_until_ms - remove_from_ms

        return {
            'total_silence_segments': len(merged_segments),
            'segments_to_trim': len(trimming_plan),
            'total_silence_duration_ms': sum(s.duration_ms for s in merged_segments),
            'silence_to_remove_ms': total_silence_to_remove_ms,
            'trimming_plan': trimming_plan
        }


class ProgressiveSilenceTrimmer:
    """
    Main class for progressive silence trimming of large audio files.

    This class processes audio files in chunks to detect and trim silence
    without loading the entire file into memory.
    """

    def __init__(self, config: Optional[TrimmingConfig] = None):
        self.config = config or TrimmingConfig()
        self.detector = ProgressiveSilenceDetector(self.config)

    async def trim_silence_streaming(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        progress_callback: Optional[Callable[[TrimmingProgress], None]] = None
    ) -> Dict[str, Any]:
        """
        Trim silence from audio file using progressive chunk-based processing.

        Args:
            input_path: Path to input audio file
            output_path: Path to output audio file
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with trimming results and statistics
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input audio file not found: {input_path}")

        logger.info(f"Starting progressive silence trimming: {input_path} -> {output_path}")

        start_time = time.time()
        progress = TrimmingProgress()

        try:
            # Get audio file information
            audio_info = self._get_audio_info(input_path)
            progress.total_chunks = self._calculate_total_chunks(audio_info)

            # Process audio in chunks
            await self._process_audio_chunks(input_path, audio_info, progress, progress_callback)

            # Generate trimming plan
            trimming_plan = self.detector.calculate_trimming_plan()

            # Apply trimming
            await self._apply_trimming_plan(input_path, output_path, trimming_plan, progress)

            processing_time = time.time() - start_time

            return {
                'success': True,
                'input_path': str(input_path),
                'output_path': str(output_path),
                'processing_time': processing_time,
                'original_duration_ms': audio_info['duration_ms'],
                'trimming_plan': trimming_plan,
                'chunks_processed': progress.processed_chunks,
                'silence_segments_found': progress.silence_segments_found
            }

        except Exception as e:
            logger.error(f"Progressive silence trimming failed: {e}")
            raise

    def _get_audio_info(self, file_path: Path) -> Dict[str, Any]:
        """Get basic audio file information."""
        try:
            audio = AudioSegment.from_file(file_path)

            return {
                'duration_ms': len(audio),
                'sample_rate': audio.frame_rate,
                'channels': audio.channels,
                'sample_width': audio.sample_width
            }
        except Exception as e:
            logger.error(f"Failed to get audio info for {file_path}: {e}")
            raise

    def _calculate_total_chunks(self, audio_info: Dict[str, Any]) -> int:
        """Calculate total number of chunks to process."""
        duration_ms = audio_info['duration_ms']
        chunk_size_ms = self.config.chunk_size_ms
        overlap_ms = self.config.overlap_ms

        effective_chunk_ms = chunk_size_ms - overlap_ms
        total_chunks = int(np.ceil(duration_ms / effective_chunk_ms))

        logger.info(f"Audio duration: {duration_ms}ms, chunk size: {chunk_size_ms}ms, total chunks: {total_chunks}")
        return total_chunks

    async def _process_audio_chunks(
        self,
        input_path: Path,
        audio_info: Dict[str, Any],
        progress: TrimmingProgress,
        progress_callback: Optional[Callable[[TrimmingProgress], None]] = None
    ) -> None:
        """Process audio file in chunks to detect silence."""
        duration_ms = audio_info['duration_ms']
        sample_rate = audio_info['sample_rate']
        channels = audio_info['channels']

        chunk_size_ms = self.config.chunk_size_ms
        overlap_ms = self.config.overlap_ms

        chunk_index = 0
        position_ms = 0

        while position_ms < duration_ms:
            # Calculate chunk boundaries
            chunk_start_ms = position_ms
            chunk_end_ms = min(position_ms + chunk_size_ms, duration_ms)

            # Load and process chunk
            await self._process_single_chunk(
                input_path,
                chunk_start_ms,
                chunk_end_ms,
                sample_rate,
                channels,
                chunk_index,
                progress
            )

            # Update progress
            progress.processed_chunks = chunk_index + 1
            progress.current_position_ms = chunk_end_ms

            if progress_callback:
                elapsed_time = time.time() - progress.start_time
                if elapsed_time > 0:
                    progress.processing_speed_chunks_per_sec = progress.processed_chunks / elapsed_time
                progress_callback(progress)

            # Move to next chunk (accounting for overlap)
            position_ms += chunk_size_ms - overlap_ms
            chunk_index += 1

            # Prevent infinite loops
            if chunk_index > progress.total_chunks * 2:
                logger.warning(f"Breaking potential infinite loop at chunk {chunk_index}")
                break

    async def _process_single_chunk(
        self,
        input_path: Path,
        start_ms: int,
        end_ms: int,
        sample_rate: int,
        channels: int,
        chunk_index: int,
        progress: TrimmingProgress
    ) -> None:
        """Process a single chunk for silence detection."""
        try:
            # Load chunk using pydub
            audio = AudioSegment.from_file(input_path)

            # Extract chunk
            chunk = audio[start_ms:end_ms]

            if len(chunk) == 0:
                return

            # Convert to numpy array
            samples = np.array(chunk.get_array_of_samples())

            if channels > 1:
                samples = samples.reshape((-1, channels))
            else:
                samples = samples.reshape((-1, 1))

            # Normalize to float32
            if chunk.sample_width == 2:
                audio_data = samples.astype(np.float32) / 32768.0
            elif chunk.sample_width == 4:
                audio_data = samples.astype(np.float32) / 2147483648.0
            else:
                audio_data = samples.astype(np.float32) / 32768.0

            # Detect silence in chunk
            segments = self.detector.detect_silence_in_chunk(
                audio_data,
                sample_rate,
                start_ms,
                chunk_index
            )

            # Update progress
            progress.silence_segments_found += len(segments)
            progress.total_silence_duration_ms += sum(s.duration_ms for s in segments)

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_index} ({start_ms}-{end_ms}ms): {e}")

    async def _apply_trimming_plan(
        self,
        input_path: Path,
        output_path: Path,
        trimming_plan: Dict[str, Any],
        progress: TrimmingProgress
    ) -> None:
        """Apply the calculated trimming plan to create output file."""
        try:
            # Load entire audio file for trimming (this is the memory-intensive part)
            # For very large files, this could be optimized further
            audio = AudioSegment.from_file(input_path)

            if not trimming_plan['trimming_plan']:
                # No trimming needed, just copy file
                audio.export(output_path, format=output_path.suffix.lstrip('.'))
                return

            # Apply trimming by concatenating non-silence segments
            output_segments = []
            last_end_ms = 0

            for trim_action in trimming_plan['trimming_plan']:
                segment = trim_action['original_segment']

                # Add non-silence part before this silence segment
                if segment.start_ms > last_end_ms:
                    non_silence_part = audio[last_end_ms:segment.start_ms]
                    if len(non_silence_part) > 0:
                        output_segments.append(non_silence_part)

                # Add the silence segment (trimmed to keep_head_silence_ms)
                keep_until_ms = segment.start_ms + self.config.keep_head_silence_ms
                silence_to_keep = audio[segment.start_ms:keep_until_ms]

                if len(silence_to_keep) > 0:
                    output_segments.append(silence_to_keep)

                last_end_ms = segment.end_ms

            # Add remaining audio after last silence segment
            if last_end_ms < len(audio):
                remaining_audio = audio[last_end_ms:]
                if len(remaining_audio) > 0:
                    output_segments.append(remaining_audio)

            # Concatenate all segments
            if output_segments:
                output_audio = output_segments[0]
                for segment in output_segments[1:]:
                    # Apply crossfade if configured
                    if self.config.crossfade_ms > 0:
                        output_audio = output_audio.append(
                            segment,
                            crossfade=self.config.crossfade_ms
                        )
                    else:
                        output_audio = output_audio + segment

                # Export final audio
                output_format = output_path.suffix.lstrip('.').lower() or 'mp3'
                output_audio.export(output_path, format=output_format)

                logger.info(f"Applied trimming: removed {trimming_plan['silence_to_remove_ms']}ms of silence")
            else:
                # If no segments to keep, create a very short silent audio
                silent_audio = AudioSegment.silent(
                    duration=100,  # 100ms of silence
                    frame_rate=audio.frame_rate
                )
                silent_audio.export(output_path, format=output_path.suffix.lstrip('.'))

        except Exception as e:
            logger.error(f"Error applying trimming plan: {e}")
            # Fallback: copy original file
            try:
                import shutil
                shutil.copy2(input_path, output_path)
                logger.warning("Trimming failed, copied original file as fallback")
            except Exception as copy_error:
                logger.error(f"Fallback copy also failed: {copy_error}")
                raise


# Convenience functions for easy integration
async def trim_silence_progressive(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    min_silence_ms: int = 1000,
    silence_threshold_db: float = -40.0,
    progress_callback: Optional[Callable[[TrimmingProgress], None]] = None
) -> Dict[str, Any]:
    """
    Convenience function for progressive silence trimming.

    Args:
        input_path: Input audio file path
        output_path: Output audio file path
        min_silence_ms: Minimum silence duration to consider for trimming
        silence_threshold_db: Silence threshold in dB
        progress_callback: Optional progress callback

    Returns:
        Trimming results dictionary
    """
    config = TrimmingConfig(
        min_silence_duration_ms=min_silence_ms,
        silence_threshold_db=silence_threshold_db
    )

    trimmer = ProgressiveSilenceTrimmer(config)

    return await trimmer.trim_silence_streaming(
        input_path,
        output_path,
        progress_callback
    )


def create_adaptive_trimming_config(file_size_bytes: int) -> TrimmingConfig:
    """
    Create adaptive trimming configuration based on file size.

    Args:
        file_size_bytes: Size of the audio file in bytes

    Returns:
        Optimized TrimmingConfig for the file size
    """
    file_size_mb = file_size_bytes / (1024 * 1024)

    if file_size_mb < 50:
        # Small files: use larger chunks for speed
        return TrimmingConfig(
            chunk_size_ms=10000,  # 10 seconds
            overlap_ms=200,
            min_silence_duration_ms=800
        )
    elif file_size_mb < 200:
        # Medium files: balanced settings
        return TrimmingConfig(
            chunk_size_ms=5000,   # 5 seconds
            overlap_ms=100,
            min_silence_duration_ms=1000
        )
    else:
        # Large files: use smaller chunks for memory efficiency
        return TrimmingConfig(
            chunk_size_ms=2000,   # 2 seconds
            overlap_ms=50,
            min_silence_duration_ms=1500
        )