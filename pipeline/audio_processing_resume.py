#!/usr/bin/env python3
"""
Audio Processing Resume Support

This module provides functionality to resume interrupted audio processing operations,
detecting partially processed files and continuing from the last successful chunk.
"""

from __future__ import annotations
import asyncio
import json
import logging
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import hashlib

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingCheckpoint:
    """Represents a processing checkpoint for resumable operations."""
    file_path: str
    operation_id: str
    total_chunks: int
    completed_chunks: List[int]
    failed_chunks: List[int]
    chunk_info: Dict[int, Dict[str, Any]]
    start_time: float
    last_update: float
    processing_config: Dict[str, Any]
    resume_token: str


class ResumeManager:
    """
    Manages resumable audio processing operations.

    This class tracks processing progress and enables resuming
    interrupted operations from the last successful checkpoint.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        self.checkpoint_dir = checkpoint_dir or Path(tempfile.gettempdir()) / "audio_processing_checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._active_checkpoints: Dict[str, ProcessingCheckpoint] = {}
        self._checkpoint_interval = 30.0  # Save checkpoint every 30 seconds

    def create_checkpoint(
        self,
        file_path: Union[str, Path],
        operation_type: str,
        total_chunks: int,
        processing_config: Dict[str, Any]
    ) -> str:
        """
        Create a new processing checkpoint.

        Args:
            file_path: Path to the audio file being processed
            operation_type: Type of operation (trimming, enhancement, etc.)
            total_chunks: Total number of chunks to process
            processing_config: Processing configuration parameters

        Returns:
            Unique operation ID for the checkpoint
        """
        operation_id = self._generate_operation_id(file_path, operation_type)

        checkpoint = ProcessingCheckpoint(
            file_path=str(file_path),
            operation_id=operation_id,
            total_chunks=total_chunks,
            completed_chunks=[],
            failed_chunks=[],
            chunk_info={},
            start_time=time.time(),
            last_update=time.time(),
            processing_config=processing_config,
            resume_token=self._generate_resume_token()
        )

        self._active_checkpoints[operation_id] = checkpoint
        self._save_checkpoint(checkpoint)

        logger.info(f"Created checkpoint for {operation_type} on {file_path}: {operation_id}")
        return operation_id

    def update_chunk_status(
        self,
        operation_id: str,
        chunk_index: int,
        status: str,
        chunk_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update the status of a chunk in the processing checkpoint.

        Args:
            operation_id: Unique operation identifier
            chunk_index: Index of the chunk
            status: Status of the chunk (completed, failed, skipped)
            chunk_info: Optional metadata about the chunk
        """
        if operation_id not in self._active_checkpoints:
            logger.warning(f"No active checkpoint found for operation {operation_id}")
            return

        checkpoint = self._active_checkpoints[operation_id]

        # Update chunk status
        if status == "completed":
            if chunk_index not in checkpoint.completed_chunks:
                checkpoint.completed_chunks.append(chunk_index)
            if chunk_index in checkpoint.failed_chunks:
                checkpoint.failed_chunks.remove(chunk_index)
        elif status == "failed":
            if chunk_index not in checkpoint.failed_chunks:
                checkpoint.failed_chunks.append(chunk_index)
            if chunk_index in checkpoint.completed_chunks:
                checkpoint.completed_chunks.remove(chunk_index)
        elif status == "skipped":
            # Remove from both lists if present
            if chunk_index in checkpoint.completed_chunks:
                checkpoint.completed_chunks.remove(chunk_index)
            if chunk_index in checkpoint.failed_chunks:
                checkpoint.failed_chunks.remove(chunk_index)

        # Update chunk info
        if chunk_info:
            checkpoint.chunk_info[chunk_index] = chunk_info

        checkpoint.last_update = time.time()

        # Save checkpoint periodically
        if time.time() - checkpoint.last_update > self._checkpoint_interval:
            self._save_checkpoint(checkpoint)

    def get_completed_chunks(self, operation_id: str) -> List[int]:
        """Get list of completed chunk indices for an operation."""
        if operation_id in self._active_checkpoints:
            return self._active_checkpoints[operation_id].completed_chunks.copy()
        return []

    def get_failed_chunks(self, operation_id: str) -> List[int]:
        """Get list of failed chunk indices for an operation."""
        if operation_id in self._active_checkpoints:
            return self._active_checkpoints[operation_id].failed_chunks.copy()
        return []

    def get_pending_chunks(self, operation_id: str) -> List[int]:
        """Get list of pending chunk indices for an operation."""
        if operation_id not in self._active_checkpoints:
            return []

        checkpoint = self._active_checkpoints[operation_id]
        all_chunks = set(range(checkpoint.total_chunks))
        completed = set(checkpoint.completed_chunks)
        failed = set(checkpoint.failed_chunks)

        return sorted(list(all_chunks - completed - failed))

    def can_resume(self, file_path: Union[str, Path], operation_type: str) -> Tuple[bool, str]:
        """
        Check if an operation can be resumed.

        Args:
            file_path: Path to the audio file
            operation_type: Type of operation

        Returns:
            Tuple of (can_resume, operation_id_or_reason)
        """
        operation_id = self._generate_operation_id(file_path, operation_type)

        # Check for existing checkpoint file
        checkpoint_file = self._get_checkpoint_file(operation_id)
        if not checkpoint_file.exists():
            return False, "No checkpoint found"

        try:
            # Load and validate checkpoint
            checkpoint = self._load_checkpoint(checkpoint_file)

            # Check if file still exists and hasn't been modified
            if not Path(checkpoint.file_path).exists():
                return False, "Original file no longer exists"

            current_mtime = Path(checkpoint.file_path).stat().st_mtime
            if current_mtime != self._get_file_mtime_from_checkpoint(checkpoint):
                return False, "Original file has been modified"

            # Check if we have progress to resume
            total_completed = len(checkpoint.completed_chunks)
            if total_completed == 0:
                return False, "No progress to resume"

            if total_completed >= checkpoint.total_chunks:
                return False, "Processing already completed"

            return True, operation_id

        except Exception as e:
            logger.error(f"Error checking resume capability: {e}")
            return False, f"Error loading checkpoint: {e}"

    def load_checkpoint(self, operation_id: str) -> Optional[ProcessingCheckpoint]:
        """Load a processing checkpoint."""
        checkpoint_file = self._get_checkpoint_file(operation_id)

        if not checkpoint_file.exists():
            return None

        try:
            return self._load_checkpoint(checkpoint_file)
        except Exception as e:
            logger.error(f"Error loading checkpoint {operation_id}: {e}")
            return None

    def finalize_checkpoint(self, operation_id: str, success: bool = True) -> None:
        """
        Finalize a processing checkpoint.

        Args:
            operation_id: Unique operation identifier
            success: Whether the operation completed successfully
        """
        if operation_id in self._active_checkpoints:
            checkpoint = self._active_checkpoints[operation_id]
            checkpoint.last_update = time.time()

            if success:
                # Move to completed checkpoints
                completed_dir = self.checkpoint_dir / "completed"
                completed_dir.mkdir(exist_ok=True)
                target_file = completed_dir / f"{operation_id}_completed.json"
            else:
                # Move to failed checkpoints
                failed_dir = self.checkpoint_dir / "failed"
                failed_dir.mkdir(exist_ok=True)
                target_file = failed_dir / f"{operation_id}_failed.json"

            # Save final state
            self._save_checkpoint(checkpoint, target_file)

            # Remove from active checkpoints
            del self._active_checkpoints[operation_id]

            # Clean up old checkpoint file
            checkpoint_file = self._get_checkpoint_file(operation_id)
            try:
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
            except Exception:
                pass

            logger.info(f"Finalized checkpoint {operation_id} (success: {success})")

    def cleanup_old_checkpoints(self, max_age_hours: float = 24.0) -> int:
        """
        Clean up old checkpoint files.

        Args:
            max_age_hours: Maximum age of checkpoints to keep (in hours)

        Returns:
            Number of checkpoints cleaned up
        """
        cleanup_count = 0
        cutoff_time = time.time() - (max_age_hours * 3600)

        try:
            for checkpoint_file in self.checkpoint_dir.glob("*.json"):
                if checkpoint_file.stat().st_mtime < cutoff_time:
                    checkpoint_file.unlink()
                    cleanup_count += 1

            # Also clean up completed/failed subdirectories
            for subdir in ["completed", "failed"]:
                subdir_path = self.checkpoint_dir / subdir
                if subdir_path.exists():
                    for checkpoint_file in subdir_path.glob("*.json"):
                        if checkpoint_file.stat().st_mtime < cutoff_time:
                            checkpoint_file.unlink()
                            cleanup_count += 1

        except Exception as e:
            logger.error(f"Error cleaning up old checkpoints: {e}")

        if cleanup_count > 0:
            logger.info(f"Cleaned up {cleanup_count} old checkpoint files")

        return cleanup_count

    def _generate_operation_id(self, file_path: Union[str, Path], operation_type: str) -> str:
        """Generate unique operation ID."""
        file_str = str(Path(file_path).resolve())
        content = f"{file_str}:{operation_type}:{time.time()}"

        # Create hash of file path and operation type
        hash_obj = hashlib.md5(content.encode())
        return f"{operation_type}_{hash_obj.hexdigest()[:16]}"

    def _generate_resume_token(self) -> str:
        """Generate unique resume token."""
        return hashlib.md5(str(time.time()).encode()).hexdigest()[:16]

    def _get_checkpoint_file(self, operation_id: str) -> Path:
        """Get checkpoint file path."""
        return self.checkpoint_dir / f"{operation_id}.json"

    def _get_file_mtime_from_checkpoint(self, checkpoint: ProcessingCheckpoint) -> float:
        """Extract file modification time from checkpoint."""
        # This would need to be stored in the checkpoint when created
        # For now, we'll assume the file hasn't changed
        return 0.0

    def _save_checkpoint(self, checkpoint: ProcessingCheckpoint, custom_path: Optional[Path] = None) -> None:
        """Save checkpoint to file."""
        if custom_path is None:
            checkpoint_file = self._get_checkpoint_file(checkpoint.operation_id)
        else:
            checkpoint_file = custom_path

        try:
            checkpoint_data = asdict(checkpoint)
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving checkpoint {checkpoint.operation_id}: {e}")

    def _load_checkpoint(self, checkpoint_file: Path) -> ProcessingCheckpoint:
        """Load checkpoint from file."""
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)

        # Convert back to ProcessingCheckpoint object
        checkpoint = ProcessingCheckpoint(
            file_path=data['file_path'],
            operation_id=data['operation_id'],
            total_chunks=data['total_chunks'],
            completed_chunks=data['completed_chunks'],
            failed_chunks=data['failed_chunks'],
            chunk_info=data['chunk_info'],
            start_time=data['start_time'],
            last_update=data['last_update'],
            processing_config=data['processing_config'],
            resume_token=data['resume_token']
        )

        return checkpoint


class ResumableAudioProcessor:
    """
    Audio processor with built-in resume support.

    This class wraps the StreamingAudioProcessor with checkpoint/resume functionality.
    """

    def __init__(self, processing_config: Optional[Dict[str, Any]] = None):
        from .streaming_audio_processor import StreamingAudioProcessor, ProcessingConfig, ChunkState

        self.processing_config = processing_config or {}
        self.resume_manager = ResumeManager()
        self.processor = StreamingAudioProcessor(ProcessingConfig(**self.processing_config))
        self.ChunkState = ChunkState  # Store for later use

    async def process_audio_resumable(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        operation_type: str = "enhancement",
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process audio with resume support.

        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            operation_type: Type of operation for checkpointing
            progress_callback: Optional progress callback

        Returns:
            Processing results dictionary
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Check if we can resume an existing operation
        can_resume, resume_info = self.resume_manager.can_resume(input_path, operation_type)
        operation_id = None

        if can_resume and isinstance(resume_info, str):
            operation_id = resume_info
            logger.info(f"Resuming interrupted processing: {operation_id}")

            # Load existing checkpoint
            checkpoint = self.resume_manager.load_checkpoint(operation_id)
            if not checkpoint:
                logger.warning("Could not load checkpoint, starting fresh")
                can_resume = False

        if not can_resume:
            # Start fresh processing
            logger.info(f"Starting new processing operation for {input_path}")

            # Get file info to determine chunk count
            file_info = self.processor._get_audio_file_info(input_path)
            chunk_info_list = self.processor._calculate_chunks(file_info)

            operation_id = self.resume_manager.create_checkpoint(
                input_path, operation_type, len(chunk_info_list), self.processing_config
            )

        # Process chunks with resume support
        if operation_id is None:
            raise RuntimeError("Could not create or load operation checkpoint")

        return await self._process_with_resume(
            input_path, output_path, operation_id, progress_callback
        )

    async def _process_with_resume(
        self,
        input_path: Path,
        output_path: Path,
        operation_id: str,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Process audio with resume tracking."""
        checkpoint = self.resume_manager.load_checkpoint(operation_id)
        if not checkpoint:
            raise RuntimeError(f"Could not load checkpoint {operation_id}")

        # Get chunks that still need processing
        pending_chunks = self.resume_manager.get_pending_chunks(operation_id)
        failed_chunks = self.resume_manager.get_failed_chunks(operation_id)

        # Retry failed chunks first, then process pending chunks
        chunks_to_process = failed_chunks + pending_chunks

        if not chunks_to_process:
            logger.info("No chunks need processing, operation may be complete")
            self.resume_manager.finalize_checkpoint(operation_id, success=True)
            return {'success': True, 'resumed': True, 'chunks_processed': 0}

        logger.info(f"Processing {len(chunks_to_process)} chunks (resumed: {len(pending_chunks) < len(chunks_to_process)})")

        # Calculate chunk information
        file_info = self.processor._get_audio_file_info(input_path)
        all_chunk_info = self.processor._calculate_chunks(file_info)

        # Process chunks
        processed_count = 0

        for chunk_index in chunks_to_process:
            if chunk_index >= len(all_chunk_info):
                logger.warning(f"Chunk index {chunk_index} out of range")
                continue

            try:
                chunk_info = all_chunk_info[chunk_index]

                # Update checkpoint
                self.resume_manager.update_chunk_status(operation_id, chunk_index, "processing")

                # Process the chunk
                processed_chunk = self.processor._process_single_chunk(chunk_info, input_path)

                if processed_chunk.state == self.ChunkState.COMPLETED:
                    self.resume_manager.update_chunk_status(
                        operation_id, chunk_index, "completed",
                        processed_chunk.metadata
                    )
                    # Update chunk info state for combination
                    from .streaming_audio_processor import ChunkState
                    chunk_info.state = ChunkState.COMPLETED  # type: ignore
                else:
                    self.resume_manager.update_chunk_status(
                        operation_id, chunk_index, "failed",
                        {'error': processed_chunk.error_message}
                    )

                processed_count += 1

                # Report progress
                if progress_callback:
                    progress_info = {
                        'chunk_index': chunk_index,
                        'total_chunks': len(all_chunk_info),
                        'processed_chunks': len(self.resume_manager.get_completed_chunks(operation_id)),
                        'resumed': True
                    }
                    progress_callback(progress_info)

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_index}: {e}")
                self.resume_manager.update_chunk_status(
                    operation_id, chunk_index, "failed", {'error': str(e)}
                )

        # Check if all chunks are complete
        completed_chunks = len(self.resume_manager.get_completed_chunks(operation_id))
        total_chunks = checkpoint.total_chunks

        if completed_chunks >= total_chunks:
            # All chunks processed, finalize
            try:
                # Combine chunks into final output
                processed_chunks = []
                for i, chunk_info in enumerate(all_chunk_info):
                    if i in self.resume_manager.get_completed_chunks(operation_id):
                        chunk_info.state = ChunkState.COMPLETED  # type: ignore
                        processed_chunks.append(chunk_info)

                await self.processor._combine_chunks(processed_chunks, output_path, file_info)

                self.resume_manager.finalize_checkpoint(operation_id, success=True)

                return {
                    'success': True,
                    'resumed': True,
                    'chunks_processed': processed_count,
                    'total_chunks': total_chunks,
                    'completed_chunks': completed_chunks
                }

            except Exception as e:
                logger.error(f"Error finalizing processing: {e}")
                self.resume_manager.finalize_checkpoint(operation_id, success=False)
                raise
        else:
            logger.info(f"Processing incomplete: {completed_chunks}/{total_chunks} chunks completed")
            return {
                'success': False,
                'resumed': True,
                'chunks_processed': processed_count,
                'total_chunks': total_chunks,
                'completed_chunks': completed_chunks,
                'message': 'Processing interrupted, can be resumed'
            }


# Global resume manager instance
_resume_manager: Optional[ResumeManager] = None


def get_resume_manager() -> ResumeManager:
    """Get global resume manager instance."""
    global _resume_manager
    if _resume_manager is None:
        _resume_manager = ResumeManager()
    return _resume_manager


# Convenience functions
async def process_audio_with_resume(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    operation_type: str = "enhancement",
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Process audio with automatic resume support.

    Args:
        input_path: Input audio file path
        output_path: Output audio file path
        operation_type: Type of operation
        progress_callback: Optional progress callback

    Returns:
        Processing results dictionary
    """
    processor = ResumableAudioProcessor()
    return await processor.process_audio_resumable(
        input_path, output_path, operation_type, progress_callback
    )


def cleanup_processing_checkpoints(max_age_hours: float = 24.0) -> int:
    """
    Clean up old processing checkpoints.

    Args:
        max_age_hours: Maximum age of checkpoints to keep

    Returns:
        Number of checkpoints cleaned up
    """
    manager = get_resume_manager()
    return manager.cleanup_old_checkpoints(max_age_hours)


def get_resume_info(file_path: Union[str, Path], operation_type: str) -> Dict[str, Any]:
    """
    Get resume information for a file and operation type.

    Args:
        file_path: Path to the audio file
        operation_type: Type of operation

    Returns:
        Dictionary with resume information
    """
    manager = get_resume_manager()
    can_resume, info = manager.can_resume(file_path, operation_type)

    if can_resume and isinstance(info, str):
        checkpoint = manager.load_checkpoint(info)
        if checkpoint:
            return {
                'can_resume': True,
                'operation_id': info,
                'completed_chunks': len(checkpoint.completed_chunks),
                'total_chunks': checkpoint.total_chunks,
                'failed_chunks': len(checkpoint.failed_chunks),
                'start_time': checkpoint.start_time,
                'last_update': checkpoint.last_update
            }

    return {
        'can_resume': False,
        'reason': info if isinstance(info, str) else "Unknown"
    }