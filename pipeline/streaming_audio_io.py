#!/usr/bin/env python3
"""
Streaming Audio I/O - Memory-efficient audio file reading and writing.

This module provides advanced audio file I/O capabilities optimized for large files,
including streaming readers, buffered writers, and format-agnostic processing.
"""

from __future__ import annotations
import asyncio
import io
import logging
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, BinaryIO, Callable
import weakref

import numpy as np
import soundfile as sf
import librosa

from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    M4A = "m4a"
    FLAC = "flac"
    OGG = "ogg"
    AAC = "aac"
    UNKNOWN = "unknown"


@dataclass
class AudioStreamInfo:
    """Information about an audio stream."""
    format: AudioFormat
    sample_rate: int
    channels: int
    duration: float
    bit_rate: Optional[int] = None
    bit_depth: Optional[int] = None
    codec: Optional[str] = None
    file_size: int = 0
    frames: int = 0


class AudioStreamReader(ABC):
    """Abstract base class for audio stream readers."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.stream_info: Optional[AudioStreamInfo] = None
        self._lock = threading.RLock()

    @abstractmethod
    async def read_chunk(
        self,
        start_frame: int,
        num_frames: int,
        normalize: bool = True
    ) -> Tuple[np.ndarray, int]:
        """Read a chunk of audio data."""
        pass

    @abstractmethod
    async def get_stream_info(self) -> AudioStreamInfo:
        """Get information about the audio stream."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the stream reader."""
        pass


class SoundFileAudioReader(AudioStreamReader):
    """High-performance audio reader using soundfile."""

    def __init__(self, file_path: Union[str, Path]):
        super().__init__(file_path)
        self._soundfile: Optional[sf.SoundFile] = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="SoundFileReader")

    async def get_stream_info(self) -> AudioStreamInfo:
        """Get audio stream information."""
        if self.stream_info is not None:
            return self.stream_info

        def _get_info():
            try:
                with sf.SoundFile(str(self.file_path)) as f:
                    return AudioStreamInfo(
                        format=self._detect_format(),
                        sample_rate=int(f.samplerate),
                        channels=f.channels,
                        duration=f.frames / f.samplerate,
                        bit_depth=self._get_bit_depth(f.subtype),
                        file_size=self.file_path.stat().st_size,
                        frames=f.frames
                    )
            except Exception as e:
                logger.warning(f"Failed to get stream info with soundfile: {e}")
                return self._get_info_fallback()

        loop = asyncio.get_event_loop()
        self.stream_info = await loop.run_in_executor(self._executor, _get_info)
        if self.stream_info is None:
            raise RuntimeError("Failed to get stream info")
        return self.stream_info

    async def read_chunk(
        self,
        start_frame: int,
        num_frames: int,
        normalize: bool = True
    ) -> Tuple[np.ndarray, int]:
        """Read a chunk of audio data."""

        def _read():
            if self._soundfile is None:
                self._soundfile = sf.SoundFile(str(self.file_path))

            # Seek to start position
            self._soundfile.seek(start_frame)

            # Read chunk
            chunk_data = self._soundfile.read(num_frames)

            if chunk_data.size == 0:
                return np.array([]), self._soundfile.samplerate

            # Ensure proper shape
            if len(chunk_data.shape) == 1:
                chunk_data = chunk_data.reshape(-1, 1)

            # Normalize to float32 [-1, 1] if requested
            if normalize:
                if self._soundfile.subtype.startswith('PCM_'):
                    if self._soundfile.subtype in ('PCM_16', 'PCM_S8', 'PCM_U8'):
                        chunk_data = chunk_data.astype(np.float32) / 32768.0
                    elif self._soundfile.subtype == 'PCM_24':
                        chunk_data = chunk_data.astype(np.float32) / 8388608.0
                    elif self._soundfile.subtype == 'PCM_32':
                        chunk_data = chunk_data.astype(np.float32) / 2147483648.0

            return chunk_data, self._soundfile.samplerate

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _read)

    def _detect_format(self) -> AudioFormat:
        """Detect audio format from file extension."""
        ext = self.file_path.suffix.lower().lstrip('.')
        format_map = {
            'wav': AudioFormat.WAV,
            'mp3': AudioFormat.MP3,
            'm4a': AudioFormat.M4A,
            'flac': AudioFormat.FLAC,
            'ogg': AudioFormat.OGG,
            'aac': AudioFormat.AAC
        }
        return format_map.get(ext, AudioFormat.UNKNOWN)

    def _get_bit_depth(self, subtype: str) -> Optional[int]:
        """Get bit depth from soundfile subtype."""
        bit_depth_map = {
            'PCM_16': 16,
            'PCM_24': 24,
            'PCM_32': 32,
            'PCM_S8': 8,
            'PCM_U8': 8,
            'FLOAT': 32,
            'DOUBLE': 64
        }
        return bit_depth_map.get(subtype)

    def _get_info_fallback(self) -> AudioStreamInfo:
        """Fallback method to get audio info."""
        try:
            # Try with librosa
            duration = librosa.get_duration(filename=str(self.file_path))

            # Try to load a small chunk to get sample rate and channels
            try:
                small_chunk, sr = librosa.load(str(self.file_path), duration=0.1)
                channels = 2 if len(small_chunk.shape) > 1 else 1

                return AudioStreamInfo(
                    format=self._detect_format(),
                    sample_rate=int(sr),
                    channels=channels,
                    duration=duration,
                    file_size=self.file_path.stat().st_size,
                    frames=int(duration * sr)
                )
            except Exception:
                # Last resort - use pydub
                audio = AudioSegment.from_file(self.file_path)
                return AudioStreamInfo(
                    format=self._detect_format(),
                    sample_rate=audio.frame_rate,
                    channels=audio.channels,
                    duration=len(audio) / 1000.0,
                    file_size=self.file_path.stat().st_size,
                    frames=len(audio)
                )

        except Exception as e:
            logger.error(f"All methods failed to get audio info: {e}")
            raise RuntimeError(f"Cannot read audio file {self.file_path}")

    def close(self) -> None:
        """Close the soundfile reader."""
        if self._soundfile:
            try:
                self._soundfile.close()
            except Exception:
                pass
            self._soundfile = None

        self._executor.shutdown(wait=False)


class LibrosaAudioReader(AudioStreamReader):
    """Fallback audio reader using librosa for compatibility."""

    def __init__(self, file_path: Union[str, Path]):
        super().__init__(file_path)
        self._full_data: Optional[np.ndarray] = None
        self._sample_rate: Optional[float] = None

    async def get_stream_info(self) -> AudioStreamInfo:
        """Get audio stream information."""
        if self.stream_info is not None:
            return self.stream_info

        def _get_info():
            try:
                duration = librosa.get_duration(filename=str(self.file_path))
                small_chunk, sr = librosa.load(str(self.file_path), duration=0.1)

                channels = 2 if len(small_chunk.shape) > 1 else 1

                return AudioStreamInfo(
                    format=self._detect_format(),
                    sample_rate=int(sr),
                    channels=channels,
                    duration=duration,
                    file_size=self.file_path.stat().st_size,
                    frames=int(duration * sr)
                )
            except Exception as e:
                logger.error(f"Librosa info extraction failed: {e}")
                raise

        loop = asyncio.get_event_loop()
        self.stream_info = await loop.run_in_executor(None, _get_info)
        if self.stream_info is None:
            raise RuntimeError("Failed to get stream info")
        return self.stream_info

    async def read_chunk(
        self,
        start_frame: int,
        num_frames: int,
        normalize: bool = True
    ) -> Tuple[np.ndarray, int]:
        """Read a chunk of audio data."""

        def _read():
            # Load full file if not already loaded
            if self._full_data is None:
                self._full_data, sample_rate_float = librosa.load(
                    str(self.file_path),
                    sr=None,
                    mono=False
                )
                self._sample_rate = int(sample_rate_float)

            # Extract chunk
            end_frame = min(start_frame + num_frames, len(self._full_data))
            chunk_data = self._full_data[start_frame:end_frame]

            if chunk_data.size == 0:
                return np.array([]), int(self._sample_rate or 44100)

            # Ensure proper shape
            if len(chunk_data.shape) == 1:
                chunk_data = chunk_data.reshape(-1, 1)

            return chunk_data, int(self._sample_rate or 44100)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _read)

    def _detect_format(self) -> AudioFormat:
        """Detect audio format from file extension."""
        ext = self.file_path.suffix.lower().lstrip('.')
        format_map = {
            'wav': AudioFormat.WAV,
            'mp3': AudioFormat.MP3,
            'm4a': AudioFormat.M4A,
            'flac': AudioFormat.FLAC,
            'ogg': AudioFormat.OGG,
            'aac': AudioFormat.AAC
        }
        return format_map.get(ext, AudioFormat.UNKNOWN)

    def close(self) -> None:
        """Close the reader."""
        self._full_data = None
        self._sample_rate = None


class AudioStreamWriter(ABC):
    """Abstract base class for audio stream writers."""

    def __init__(self, file_path: Union[str, Path], stream_info: AudioStreamInfo):
        self.file_path = Path(file_path)
        self.stream_info = stream_info
        self._lock = threading.RLock()

    @abstractmethod
    async def write_chunk(self, chunk_data: np.ndarray, start_frame: int) -> None:
        """Write a chunk of audio data."""
        pass

    @abstractmethod
    async def finalize(self) -> None:
        """Finalize the audio file."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the stream writer."""
        pass


class BufferedAudioWriter(AudioStreamWriter):
    """Buffered audio writer for efficient chunked writing."""

    def __init__(
        self,
        file_path: Union[str, Path],
        stream_info: AudioStreamInfo,
        buffer_size_mb: float = 32.0
    ):
        super().__init__(file_path, stream_info)
        self.buffer_size_mb = buffer_size_mb
        self._buffer: List[Tuple[np.ndarray, int]] = []
        self._buffer_size_bytes = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AudioWriter")

        # Calculate buffer size in samples
        bytes_per_sample = (stream_info.bit_depth or 16) // 8 * stream_info.channels
        self._max_buffer_samples = int((buffer_size_mb * 1024 * 1024) / bytes_per_sample)

    async def write_chunk(self, chunk_data: np.ndarray, start_frame: int) -> None:
        """Write a chunk to buffer."""

        def _buffer_chunk():
            chunk_bytes = chunk_data.nbytes
            self._buffer.append((chunk_data.copy(), start_frame))
            self._buffer_size_bytes += chunk_bytes

            # Auto-flush if buffer is full
            if self._buffer_size_bytes > self.buffer_size_mb * 1024 * 1024:
                asyncio.create_task(self._flush_buffer())

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, _buffer_chunk)

    async def _flush_buffer(self) -> None:
        """Flush buffer to disk."""

        def _flush():
            if not self._buffer:
                return

            # Sort chunks by start frame to ensure correct order
            sorted_chunks = sorted(self._buffer, key=lambda x: x[1])

            # Combine chunks
            combined_data = []
            expected_frame = sorted_chunks[0][1]

            for chunk_data, start_frame in sorted_chunks:
                # Handle gaps (shouldn't happen in normal operation)
                if start_frame > expected_frame:
                    gap_frames = start_frame - expected_frame
                    gap_data = np.zeros((gap_frames, chunk_data.shape[1]))
                    combined_data.append(gap_data)

                combined_data.append(chunk_data)
                expected_frame = start_frame + len(chunk_data)

            if combined_data:
                final_data = np.concatenate(combined_data, axis=0)

                # Write to file
                temp_path = self.file_path.with_suffix('.tmp')
                mode = 'w' if not temp_path.exists() else 'a'

                with sf.SoundFile(
                    str(temp_path),
                    mode=mode,
                    samplerate=self.stream_info.sample_rate,
                    channels=self.stream_info.channels,
                    subtype=self._get_subtype(),
                    format=self._get_format()
                ) as f:
                    if mode == 'a':
                        f.seek(0, sf.SEEK_END)
                    f.write(final_data)

                # Atomic move
                temp_path.replace(self.file_path)

            # Clear buffer
            self._buffer.clear()
            self._buffer_size_bytes = 0

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, _flush)

    async def finalize(self) -> None:
        """Finalize the audio file."""
        await self._flush_buffer()

    def _get_subtype(self) -> str:
        """Get soundfile subtype based on bit depth."""
        bit_depth = self.stream_info.bit_depth or 16
        if bit_depth == 16:
            return 'PCM_16'
        elif bit_depth == 24:
            return 'PCM_24'
        elif bit_depth == 32:
            return 'PCM_32'
        else:
            return 'PCM_16'

    def _get_format(self) -> str:
        """Get soundfile format."""
        format_map = {
            AudioFormat.WAV: 'WAV',
            AudioFormat.MP3: 'MP3',
            AudioFormat.FLAC: 'FLAC',
            AudioFormat.OGG: 'OGG'
        }
        return format_map.get(self.stream_info.format, 'WAV')

    def close(self) -> None:
        """Close the writer."""
        try:
            # Final flush - handle both scenarios: with and without existing event loop
            try:
                # Try to use existing event loop first
                loop = asyncio.get_running_loop()
                # We're in an async context, schedule the finalize task
                asyncio.create_task(self.finalize())
            except RuntimeError:
                # No event loop running, create new one for finalization
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.finalize())
                except Exception:
                    # If finalize fails, continue with cleanup
                    pass
                finally:
                    try:
                        loop.close()
                    except Exception:
                        pass
        except Exception:
            pass

        self._executor.shutdown(wait=False)


class StreamingAudioIO:
    """
    High-level interface for streaming audio I/O operations.

    Provides format-agnostic reading and writing with automatic
    fallback and optimization for different file types and sizes.
    """

    def __init__(self):
        self._readers: Dict[Path, AudioStreamReader] = {}
        self._writers: Dict[Path, AudioStreamWriter] = {}
        self._lock = threading.RLock()

    async def create_reader(self, file_path: Union[str, Path]) -> AudioStreamReader:
        """Create appropriate reader for file type."""
        file_path = Path(file_path)

        with self._lock:
            if file_path in self._readers:
                return self._readers[file_path]

            # Try SoundFile reader first (faster)
            try:
                reader = SoundFileAudioReader(file_path)
                await reader.get_stream_info()  # Test if it works
                self._readers[file_path] = reader
                return reader
            except Exception as e:
                logger.warning(f"SoundFile reader failed, trying Librosa: {e}")

                # Fallback to Librosa reader
                reader = LibrosaAudioReader(file_path)
                await reader.get_stream_info()
                self._readers[file_path] = reader
                return reader

    async def create_writer(
        self,
        file_path: Union[str, Path],
        stream_info: AudioStreamInfo,
        buffer_size_mb: float = 32.0
    ) -> AudioStreamWriter:
        """Create buffered writer for file."""
        file_path = Path(file_path)

        with self._lock:
            if file_path in self._writers:
                return self._writers[file_path]

            writer = BufferedAudioWriter(file_path, stream_info, buffer_size_mb)
            self._writers[file_path] = writer
            return writer

    async def read_audio_chunk(
        self,
        file_path: Union[str, Path],
        start_frame: int,
        num_frames: int,
        normalize: bool = True
    ) -> Tuple[np.ndarray, int]:
        """Read audio chunk with automatic reader management."""
        reader = await self.create_reader(file_path)
        return await reader.read_chunk(start_frame, num_frames, normalize)

    async def write_audio_chunk(
        self,
        file_path: Union[str, Path],
        chunk_data: np.ndarray,
        start_frame: int,
        stream_info: AudioStreamInfo,
        buffer_size_mb: float = 32.0
    ) -> None:
        """Write audio chunk with automatic writer management."""
        writer = await self.create_writer(file_path, stream_info, buffer_size_mb)
        await writer.write_chunk(chunk_data, start_frame)

    async def finalize_writer(self, file_path: Union[str, Path]) -> None:
        """Finalize writing to file."""
        file_path = Path(file_path)

        with self._lock:
            if file_path in self._writers:
                await self._writers[file_path].finalize()

    def close_reader(self, file_path: Union[str, Path]) -> None:
        """Close reader for file."""
        file_path = Path(file_path)

        with self._lock:
            if file_path in self._readers:
                self._readers[file_path].close()
                del self._readers[file_path]

    def close_writer(self, file_path: Union[str, Path]) -> None:
        """Close writer for file."""
        file_path = Path(file_path)

        with self._lock:
            if file_path in self._writers:
                self._writers[file_path].close()
                del self._writers[file_path]

    def close_all(self) -> None:
        """Close all readers and writers."""
        with self._lock:
            # Close all readers
            for reader in self._readers.values():
                try:
                    reader.close()
                except Exception:
                    pass
            self._readers.clear()

            # Close all writers
            for writer in self._writers.values():
                try:
                    writer.close()
                except Exception:
                    pass
            self._writers.clear()

    async def copy_audio_streaming(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        chunk_size: int = 8192,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> None:
        """Copy audio file using streaming approach."""
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Get input stream info
        reader = await self.create_reader(input_path)
        stream_info = await reader.get_stream_info()

        # Create output writer
        writer = await self.create_writer(output_path, stream_info)

        try:
            total_frames = stream_info.frames
            processed_frames = 0

            while processed_frames < total_frames:
                # Calculate chunk size
                frames_to_read = min(chunk_size, total_frames - processed_frames)

                # Read chunk
                chunk_data, _ = await reader.read_chunk(
                    processed_frames,
                    frames_to_read
                )

                if chunk_data.size == 0:
                    break

                # Write chunk
                await writer.write_chunk(chunk_data, processed_frames)

                # Update progress
                processed_frames += len(chunk_data)
                if progress_callback:
                    progress = processed_frames / total_frames
                    progress_callback(progress)

            # Finalize
            await writer.finalize()

        finally:
            self.close_reader(input_path)
            self.close_writer(output_path)


# Global instance for convenience
_audio_io_instance: Optional[StreamingAudioIO] = None


def get_audio_io() -> StreamingAudioIO:
    """Get global audio I/O instance."""
    global _audio_io_instance
    if _audio_io_instance is None:
        _audio_io_instance = StreamingAudioIO()
    return _audio_io_instance


# Convenience functions
async def read_audio_chunk(
    file_path: Union[str, Path],
    start_frame: int,
    num_frames: int,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """Convenience function to read audio chunk."""
    io = get_audio_io()
    return await io.read_audio_chunk(file_path, start_frame, num_frames, normalize)


async def write_audio_chunk(
    file_path: Union[str, Path],
    chunk_data: np.ndarray,
    start_frame: int,
    stream_info: AudioStreamInfo,
    buffer_size_mb: float = 32.0
) -> None:
    """Convenience function to write audio chunk."""
    io = get_audio_io()
    await io.write_audio_chunk(file_path, chunk_data, start_frame, stream_info, buffer_size_mb)


async def copy_audio_file_streaming(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    chunk_size: int = 8192,
    progress_callback: Optional[Callable[[float], None]] = None
) -> None:
    """Convenience function to copy audio file with streaming."""
    io = get_audio_io()
    await io.copy_audio_streaming(input_path, output_path, chunk_size, progress_callback)