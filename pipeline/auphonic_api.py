#!/usr/bin/env python3
"""
Auphonic API Integration Module

This module provides integration with Auphonic's audio enhancement API.
After a user uploads an audio file, it gets enhanced through Auphonic's
professional audio processing algorithms.
"""

from __future__ import annotations
import asyncio
import logging
import time
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pipeline.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuphonicAPI:
    """Auphonic API client for audio enhancement"""
    
    def __init__(self):
        self.api_key = settings.AUPHONIC_API_KEY
        self.preset = settings.AUPHONIC_PRESET
        self.enabled = settings.AUPHONIC_ENABLED
        self.base_url = "https://auphonic.com/api"
        
        if not self.enabled:
            logger.info("Auphonic API is disabled")
            return
            
        if not self.api_key:
            logger.warning("AUPHONIC_API_KEY not configured")
            self.enabled = False
            return
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set authorization header
        self.session.headers.update({
            "Authorization": f"bearer {self.api_key}",
        })
        
        logger.info("Auphonic API client initialized")
    
    def enhance_audio(self, input_audio_path: Path, output_audio_path: Optional[Path] = None) -> Optional[Path]:
        """
        Enhance audio using Auphonic API
        
        Args:
            input_audio_path: Path to the original audio file
            output_audio_path: Optional path for enhanced audio. If None, will use input path with _enhanced suffix
            
        Returns:
            Path to enhanced audio file, or None if enhancement failed
        """
        if not self.enabled:
            logger.info("Auphonic enhancement disabled, using original audio")
            return input_audio_path
            
        if not self.api_key:
            logger.warning("Auphonic API key not configured, using original audio")
            return input_audio_path
            
        if not input_audio_path.exists():
            logger.error(f"Input audio file not found: {input_audio_path}")
            return None
            
        try:
            # Set output path
            if output_audio_path is None:
                output_audio_path = input_audio_path.parent / f"{input_audio_path.stem}_enhanced{input_audio_path.suffix}"
            
            logger.info(f"Starting Auphonic enhancement for: {input_audio_path}")
            
            # Step 0: Aggressive cleanup before starting new production
            logger.info("Cleaning up old/unfinished productions...")
            cleanup_success = self.cleanup_old_productions(max_age_hours=1)
            if not cleanup_success:
                logger.warning("Cleanup failed, but proceeding with enhancement...")
            
            # Step 1: Start production
            production_id = self._start_production(input_audio_path)
            if not production_id:
                logger.error("Failed to start Auphonic production")
                return None
                
            logger.info(f"Auphonic production started: {production_id}")
            
            # Step 2: Wait for completion
            success = self._wait_for_completion(production_id)
            if not success:
                logger.error("Auphonic production failed or timed out")
                # Clean up failed production
                self.cleanup_production(production_id)
                return None
                
            # Step 3: Download enhanced audio
            enhanced_path = self._download_output(production_id, output_audio_path)
            
            # Step 4: Always clean up the production after completion
            self.cleanup_production(production_id)
            
            if enhanced_path:
                logger.info(f"Auphonic enhancement completed: {enhanced_path}")
                return enhanced_path
            else:
                logger.error("Failed to download enhanced audio")
                return None
                
        except Exception as e:
            logger.error(f"Auphonic enhancement failed: {e}")
            return None
    
    def cleanup_old_productions(self, max_age_hours: float = 24) -> bool:
        """
        Clean up old/unfinished Auphonic productions to avoid hitting API limits
        
        Args:
            max_age_hours: Maximum age in hours for productions to keep (default: 24)
            
        Returns:
            bool: True if cleanup was successful, False otherwise
        """
        try:
            logger.info(f"Starting cleanup of productions older than {max_age_hours} hours...")
            
            # Get all productions
            url = f"{self.base_url}/productions.json"
            response = self.session.get(url, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"Failed to get productions list: {response.status_code}")
                return False
            
            result = response.json()
            productions = result.get('data', [])
            
            if not productions:
                logger.info("No productions found to clean up")
                return True
            
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            cleaned_count = 0
            
            for production in productions:
                try:
                    status = production.get('status')
                    created_at = production.get('creation_time')
                    
                    # Skip if no creation time
                    if not created_at:
                        continue
                    
                    # Parse creation time (assuming it's in ISO format)
                    try:
                        # Auphonic uses Unix timestamp or ISO format
                        if isinstance(created_at, str):
                            created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        else:
                            # Assume it's a Unix timestamp
                            created_dt = datetime.fromtimestamp(created_at)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not parse creation time for production {production.get('uuid', 'unknown')}")
                        continue
                    
                    # Delete if status is not "done" and production is old or just non-done
                    # Status codes: 0=Queued, 1=Processing, 2=Done, 3=Error, 4=Starting
                    # Also delete any stuck jobs regardless of age if they're clearly stuck
                    should_delete = False
                    
                    # Always delete clearly stuck/error states regardless of age
                    if status in [3, "Error", "Timeout", "TimedOut"]:
                        should_delete = True
                        logger.info(f"Found stuck error production: {production.get('uuid', 'unknown')} (status: {status})")
                    
                    # Delete old non-done jobs (much more aggressive: 1 hour instead of 24)
                    elif status not in [2, "Done"] and created_dt < cutoff_time:
                        should_delete = True
                        age_hours = (datetime.now() - created_dt).total_seconds() / 3600
                        logger.info(f"Found old incomplete production: {production.get('uuid', 'unknown')} (age: {age_hours:.1f}h, status: {status})")
                    
                    # If we have too many jobs, delete some newer ones too
                    if not should_delete and len([p for p in productions if p.get('status') not in [2, "Done"]]) > 10:
                        should_delete = True
                        logger.info(f"Too many active productions, deleting older one: {production.get('uuid', 'unknown')}")
                    
                    if should_delete:
                        production_uuid = production.get('uuid')
                        if production_uuid:
                            delete_url = f"{self.base_url}/production/{production_uuid}.json"
                            delete_response = self.session.delete(delete_url, timeout=30)
                            
                            if delete_response.status_code in [200, 204]:
                                cleaned_count += 1
                                logger.info(f"Successfully deleted production: {production_uuid} (status: {status})")
                            else:
                                logger.warning(f"Failed to delete production {production_uuid}: {delete_response.status_code}")
                    
                except Exception as e:
                    logger.warning(f"Error processing production {production.get('uuid', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Cleanup completed: {cleaned_count} old productions deleted")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup old productions: {e}")
            return False
    
    def cleanup_production(self, production_id: str) -> bool:
        """
        Clean up a specific production after it's done (successful or failed).
        
        Args:
            production_id: The UUID of the production to clean up
            
        Returns:
            bool: True if cleanup was successful, False otherwise
        """
        try:
            if not production_id:
                logger.warning("No production ID provided for cleanup")
                return True  # Nothing to clean up
                
            delete_url = f"{self.base_url}/production/{production_id}.json"
            delete_response = self.session.delete(delete_url, timeout=30)
            
            if delete_response.status_code in [200, 204]:
                logger.info(f"Successfully cleaned up production: {production_id}")
                return True
            else:
                logger.warning(f"Failed to clean up production {production_id}: {delete_response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"Error cleaning up production {production_id}: {e}")
            return False

    def _start_production(self, audio_path: Path) -> Optional[str]:
        """Start a new Auphonic production"""
        try:
            url = f"{self.base_url}/simple/productions.json"
            
            # Prepare form data - use default settings if no preset configured
            data = {
                "title": f"Audio Enhancement - {audio_path.stem}",
                "filtering": "true",
                "leveler": "false",
                "normloudness": "true",
                "loudnesstarget": "-24",
                "maxpeak": "-2",
                "denoise": "false",
                "denoiseamount": "100",
                "silence_cutter": "true",
                
                "action": "start"
            }
            
            # Add preset only if configured
            if self.preset:
                data["preset"] = self.preset
                logger.info(f"Using Auphonic preset: {self.preset}")
            else:
                logger.info("Using default Auphonic settings (no preset configured)")
            
            # Prepare files
            files = {
                "input_file": open(audio_path, "rb")
            }
            
            logger.info("Submitting audio to Auphonic API...")
            response = self.session.post(url, data=data, files=files, timeout=30)
            files["input_file"].close()  # Close the file handle
            
            if response.status_code == 200:
                result = response.json()
                production_id = result.get("data", {}).get("uuid")
                if production_id:
                    logger.info(f"Production created successfully: {production_id}")
                    return production_id
                else:
                    logger.error(f"Invalid response from Auphonic: {result}")
                    return None
            else:
                logger.error(f"Auphonic API error: {response.status_code} - {response.text}")
                
                # If we get the "too many unfinished productions" error, try cleanup and retry multiple times
                if response.status_code == 400 and "too many unfinished productions" in response.text.lower():
                    logger.info("Too many unfinished productions detected, attempting aggressive cleanup and retry...")
                    
                    # Multiple cleanup attempts with increasing aggression
                    for cleanup_attempt in range(3):
                        logger.info(f"Cleanup attempt {cleanup_attempt + 1}/3...")
                        
                        # Try different age thresholds
                        age_hours = [1, 0.5, 0.1]  # 1 hour, 30 minutes, 6 minutes
                        if cleanup_attempt < len(age_hours):
                            cleanup_success = self.cleanup_old_productions(max_age_hours=age_hours[cleanup_attempt])
                        else:
                            cleanup_success = self.cleanup_old_productions(max_age_hours=0.05)  # 3 minutes for last attempt
                        
                        if cleanup_success:
                            logger.info(f"Retrying production start after cleanup attempt {cleanup_attempt + 1}...")
                            
                            # Re-open the file for retry
                            files["input_file"] = open(audio_path, "rb")
                            response = self.session.post(url, data=data, files=files, timeout=30)
                            files["input_file"].close()
                            
                            if response.status_code == 200:
                                result = response.json()
                                production_id = result.get("data", {}).get("uuid")
                                if production_id:
                                    logger.info(f"Production created successfully after cleanup attempt {cleanup_attempt + 1}: {production_id}")
                                    return production_id
                        
                        if cleanup_attempt < 2:  # Don't sleep on last attempt
                            logger.info(f"Waiting 5 seconds before next cleanup attempt...")
                            time.sleep(5)
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to start Auphonic production: {e}")
            return None
    
    def _wait_for_completion(self, production_id: str, timeout_seconds: int = 300) -> bool:
        """Wait for Auphonic production to complete"""
        start_time = time.time()
        unknown_status_count = 0
        max_unknown_status = 3  # Allow 3 instances of unknown status before timeout
        
        logger.info(f"Monitoring Auphonic production {production_id} for completion...")
        
        while time.time() - start_time < timeout_seconds:
            try:
                url = f"{self.base_url}/simple/productions.json?action=get"
                data = {"hash": production_id}
                
                response = self.session.post(url, data=data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    status = result.get("data", {}).get("status")
                    
                    # Handle all known successful completion states
                    if status in ["Done", 2, "Done", 9, "Complete", "Completed"]:
                        logger.info(f"Auphonic production completed successfully (status: {status})")
                        return True
                    
                    # Handle status code 9 specifically (common completion code)
                    elif status == 9:
                        logger.info(f"Auphonic production completed with status code 9 (success)")
                        return True
                    
                    # Handle error states
                    elif status in ["Error", 3]:
                        error_msg = result.get("data", {}).get("status_message", "Unknown error")
                        logger.error(f"Auphonic production failed: {error_msg}")
                        return False
                    
                    # Handle active processing states
                    elif status in ["Processing", 1, "Queued", 0, "Starting", 4]:
                        logger.info(f"Auphonic production status: {status}")
                        unknown_status_count = 0  # Reset counter when we get a known status
                    
                    # Handle unknown/unusual statuses (including status 9 variants)
                    else:
                        unknown_status_count += 1
                        logger.warning(f"Unknown Auphonic status: {status} (occurrence {unknown_status_count}/{max_unknown_status})")
                        
                        # If we get the same unknown status multiple times, check if it's actually completed
                        if unknown_status_count >= max_unknown_status:
                            logger.info(f"Status {status} persisted, attempting to download output...")
                            try:
                                # Try to download the output - if it works, the job is actually done
                                test_download = self._download_output(production_id,
                                    Path(tempfile.gettempdir()) / f"test_download_{production_id}.mp3")
                                if test_download and test_download.exists():
                                    logger.info(f"Successfully downloaded output despite status {status} - treating as completed")
                                    # Clean up the test file
                                    try:
                                        test_download.unlink()
                                    except:
                                        pass
                                    return True
                                else:
                                    logger.error(f"Could not download output with status {status}")
                            except Exception as download_error:
                                logger.error(f"Error testing download for status {status}: {download_error}")
                            
                            # If download test failed, this is likely a real stuck job
                            logger.error(f"Job appears stuck with status {status}")
                            return False
                
                # Wait before next check
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error checking Auphonic status: {e}")
                time.sleep(5)
        
        logger.error(f"Auphonic production timed out after {timeout_seconds} seconds")
        return False
    
    def _download_output(self, production_id: str, output_path: Path) -> Optional[Path]:
        """Download the enhanced audio file from Auphonic"""
        try:
            url = f"{self.base_url}/simple/productions.json?action=get"
            data = {"hash": production_id}
            
            response = self.session.post(url, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                output_data = result.get("data", {}).get("output", [])
                
                # Find the primary audio output
                audio_output = None
                for output in output_data:
                    if output.get("type") in ["mp3", "wav"] and output.get("is_primary", False):
                        audio_output = output
                        break
                
                # If no primary output found, take the first audio output
                if not audio_output:
                    for output in output_data:
                        if output.get("type") in ["mp3", "wav"]:
                            audio_output = output
                            break
                
                if audio_output and "link" in audio_output:
                    download_url = audio_output["link"]
                    
                    # Download the file
                    logger.info("Downloading enhanced audio from Auphonic...")
                    response = self.session.get(download_url, timeout=300)
                    
                    if response.status_code == 200:
                        with open(output_path, "wb") as f:
                            f.write(response.content)
                        
                        logger.info(f"Enhanced audio saved to: {output_path}")
                        return output_path
                    else:
                        logger.error(f"Failed to download enhanced audio: {response.status_code}")
                        return None
                else:
                    logger.error("No audio output found in Auphonic response")
                    return None
            else:
                logger.error(f"Failed to get Auphonic output info: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to download enhanced audio: {e}")
            return None


def enhance_audio_with_auphonic(input_audio_path: Path, output_audio_path: Optional[Path] = None) -> Optional[Path]:
    """
    Convenience function to enhance audio using Auphonic API
    
    Args:
        input_audio_path: Path to the original audio file
        output_audio_path: Optional path for enhanced audio
        
    Returns:
        Path to enhanced audio file
    """
    client = AuphonicAPI()
    return client.enhance_audio(input_audio_path, output_audio_path)


# Example usage
if __name__ == "__main__":
    # Test the Auphonic integration
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python auphonic_api.py <input_audio_file> [output_audio_file]")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        sys.exit(1)
    
    enhanced_path = enhance_audio_with_auphonic(input_file, output_file)
    if enhanced_path:
        print(f"Enhanced audio saved to: {enhanced_path}")
    else:
        print("Audio enhancement failed")
        sys.exit(1)