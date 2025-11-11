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
                return None
                
            # Step 3: Download enhanced audio
            enhanced_path = self._download_output(production_id, output_audio_path)
            if enhanced_path:
                logger.info(f"Auphonic enhancement completed: {enhanced_path}")
                return enhanced_path
            else:
                logger.error("Failed to download enhanced audio")
                return None
                
        except Exception as e:
            logger.error(f"Auphonic enhancement failed: {e}")
            return None
    
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
                return None
                
        except Exception as e:
            logger.error(f"Failed to start Auphonic production: {e}")
            return None
    
    def _wait_for_completion(self, production_id: str, timeout_seconds: int = 300) -> bool:
        """Wait for Auphonic production to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            try:
                url = f"{self.base_url}/simple/productions.json?action=get"
                data = {"hash": production_id}
                
                response = self.session.post(url, data=data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    status = result.get("data", {}).get("status")
                    
                    if status == "Done":
                        logger.info("Auphonic production completed successfully")
                        return True
                    elif status == "Error":
                        error_msg = result.get("data", {}).get("status_message", "Unknown error")
                        logger.error(f"Auphonic production failed: {error_msg}")
                        return False
                    elif status in ["Processing", "Queued", "Starting"]:
                        logger.info(f"Auphonic production status: {status}")
                    else:
                        logger.warning(f"Unknown Auphonic status: {status}")
                
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