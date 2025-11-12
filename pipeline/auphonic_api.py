#!/usr/bin/env python3
"""
Auphonic API Integration Module - Fixed with Robust Retry & Proper Success Detection
"""

import os
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class AuphonicAPI:
    """Auphonic API client with robust retry and proper success detection"""
    
    # Official status codes from https://auphonic.com/api/info/production_status.json
    STATUS_CODES = {
        0: "File Upload",
        1: "Waiting",
        2: "Error",              # ❌ Error state
        3: "Done",               # ✅ Completed successfully
        4: "Audio Processing",
        5: "Audio Encoding",
        6: "Outgoing File Transfer",
        7: "Audio Mono Mixdown",
        8: "Split Audio On Chapter Marks",
        9: "Incomplete",         # ⚠️ Not started/incomplete
        10: "Production Not Started Yet",
        11: "Production Outdated",
        12: "Incoming File Transfer",
        13: "Stopping the Production",
        14: "Speech Recognition",
        15: "Production Changed"
    }
    
    def __init__(self, api_key: Optional[str] = None):
        # Load from parameter, environment, or config
        self.api_key = api_key or os.getenv('AUPHONIC_API_KEY')
        
        # Check if Auphonic is enabled
        enabled_str = os.getenv('AUPHONIC_ENABLED', 'false').lower()
        self.enabled = enabled_str in ['true', '1', 'yes', 'on']
        
        self.base_url = "https://auphonic.com/api"
        
        # Configure requests session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        if not self.enabled:
            logger.info("Auphonic API is disabled (AUPHONIC_ENABLED=false)")
            return
            
        if not self.api_key:
            logger.warning("AUPHONIC_API_KEY not found in environment")
            self.enabled = False
            return
        
        logger.info("Auphonic API client initialized with retry strategy")
    
    def enhance_audio(self, input_audio_path: Path, output_audio_path: Optional[Path] = None) -> Optional[Path]:
        """
        Enhance audio using Auphonic Simple API with robust retry logic
        """
        if not self.enabled:
            logger.info("Auphonic disabled, using original audio")
            return input_audio_path
        
        if not input_audio_path.exists():
            logger.error(f"Input audio not found: {input_audio_path}")
            return None
        
        # Set output path
        if output_audio_path is None:
            output_audio_path = input_audio_path.parent / f"{input_audio_path.stem}_enhanced{input_audio_path.suffix}"
        
        # Enhanced retry mechanism for the entire process
        max_enhancement_attempts = 3
        base_delay = 30  # Start with 30 seconds between attempts
        
        for attempt in range(max_enhancement_attempts):
            try:
                logger.info(f"Starting Auphonic enhancement (attempt {attempt + 1}/{max_enhancement_attempts}): {input_audio_path.name}")
                
                # Step 1: Create and start production using Simple API
                production_uuid = self._create_production_with_retry(input_audio_path, attempt)
                if not production_uuid:
                    logger.error(f"Failed to create production on attempt {attempt + 1}")
                    if attempt < max_enhancement_attempts - 1:
                        wait_time = base_delay * (2 ** attempt)
                        logger.info(f"Retrying production creation in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error("All production creation attempts failed")
                        return None
                
                logger.info(f"Production created: {production_uuid}")
                logger.info(f"View at: https://auphonic.com/production/{production_uuid}")
                
                # Step 2: Wait for completion with proper success detection
                if not self._wait_for_completion_with_retry(production_uuid):
                    logger.error("Production failed or timed out")
                    if attempt < max_enhancement_attempts - 1:
                        wait_time = base_delay * (2 ** attempt)
                        logger.info(f"Retrying enhancement in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error("All enhancement attempts failed")
                        self._cleanup_production(production_uuid)
                        return None
                
                # Step 3: Download output with retry
                result = self._download_output_with_retry(production_uuid, output_audio_path)
                
                # Step 4: Verify success properly
                if result and result.exists() and result.stat().st_size > 0:
                    file_size = result.stat().st_size
                    logger.info(f"✅ Enhancement complete: {result} ({file_size:,} bytes)")
                    logger.info(f"✅ Production kept on Auphonic: https://auphonic.com/production/{production_uuid}")
                    return result
                else:
                    logger.error("Failed to download enhanced audio")
                    if attempt < max_enhancement_attempts - 1:
                        wait_time = base_delay * (2 ** attempt)
                        logger.info(f"Retrying download in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error("All download attempts failed, cleaning up...")
                        self._cleanup_production(production_uuid)
                        return None
                        
            except Exception as e:
                logger.error(f"Enhancement attempt {attempt + 1} failed: {e}")
                if attempt < max_enhancement_attempts - 1:
                    wait_time = base_delay * (2 ** attempt)
                    logger.info(f"Retrying entire enhancement in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("All enhancement attempts failed")
                    return None
        
        return None
    
    def _create_production_with_retry(self, audio_path: Path, attempt: int = 0) -> Optional[str]:
        """Create production with network error handling"""
        url = f"{self.base_url}/simple/productions.json"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            with open(audio_path, "rb") as f:
                files = {"input_file": f}
                
                data = {
                    "preset": "auphonic_cleaner",
                    "title": f"Enhancement - {audio_path.stem} (attempt {attempt + 1})",
                    "action": "start"
                  

                }
                
                response = self.session.post(url, headers=headers, files=files, data=data, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    production_uuid = result.get("data", {}).get("uuid")
                    if production_uuid:
                        return production_uuid
                
                # Handle "too many productions" error
                if response.status_code == 400 and "too many" in response.text.lower():
                    logger.warning("Queue full, cleaning up incomplete jobs...")
                    self._cleanup_unfinished()
                    time.sleep(3)
                    
                    # Retry once
                    with open(audio_path, "rb") as f2:
                        files2 = {"input_file": f2}
                        response = self.session.post(url, headers=headers, files=files2, data=data, timeout=60)
                        if response.status_code == 200:
                            result = response.json()
                            return result.get("data", {}).get("uuid")
                
                logger.error(f"Failed to create: {response.status_code}")
                logger.error(f"Response: {response.text[:500]}")
                return None
                
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Network error creating production: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating production: {e}")
            return None
    
    def _wait_for_completion_with_retry(self, production_uuid: str, timeout: int = 300) -> bool:
        """Wait for production completion with proper failure detection"""
        url = f"{self.base_url}/production/{production_uuid}.json"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        start_time = time.time()
        poll_interval = 10  # Increased from 5 to reduce network load
        last_status = None
        consecutive_failures = 0
        max_consecutive_failures = 3  # Allow 3 consecutive failures before giving up
        
        logger.info("Waiting for production to complete...")
        
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(url, headers=headers, timeout=15)
                
                if response.status_code != 200:
                    consecutive_failures += 1
                    logger.warning(f"Status check failed ({consecutive_failures}/{max_consecutive_failures}): HTTP {response.status_code}")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("Too many consecutive failures, marking as failed")
                        return False
                    
                    time.sleep(poll_interval)
                    continue
                
                # Success - reset failure counter
                consecutive_failures = 0
                
                data = response.json().get("data", {})
                status = data.get("status")
                status_string = data.get("status_string", "Unknown")
                
                # Log status change
                if status != last_status:
                    status_name = self.STATUS_CODES.get(status, f"Unknown-{status}")
                    logger.info(f"Status: {status} ({status_name} - {status_string})")
                    last_status = status
                
                # Status 3 = Done (SUCCESS!) - but verify with multiple checks
                if status == 3:
                    logger.info("✅ Production completed successfully")
                    
                    # Verify the production actually exists by checking for output files
                    output_files = data.get("output_files", [])
                    if output_files:
                        logger.info(f"Production verified with {len(output_files)} output files")
                        return True
                    else:
                        logger.warning("Status 3 but no output files found, continuing to verify...")
                        # Give it one more chance to show output files
                        time.sleep(5)
                        try:
                            verify_response = self.session.get(url, headers=headers, timeout=15)
                            if verify_response.status_code == 200:
                                verify_data = verify_response.json().get("data", {})
                                verify_output_files = verify_data.get("output_files", [])
                                if verify_output_files:
                                    logger.info("Production verified with output files on second check")
                                    return True
                        except Exception as verify_e:
                            logger.warning(f"Verification check failed: {verify_e}")
                
                # Status 2 = Error (FAILURE!)
                elif status == 2:
                    error_msg = data.get("error_message", "Unknown error")
                    error_status = data.get("error_status", "")
                    logger.error(f"❌ Production failed: {error_msg} ({error_status})")
                    return False
                
                # All other statuses = still processing
                time.sleep(poll_interval)
                continue
                
            except requests.exceptions.ConnectionError as e:
                consecutive_failures += 1
                logger.warning(f"Network error checking status ({consecutive_failures}/{max_consecutive_failures}): {e}")
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Too many network failures, marking as failed")
                    return False
                
                time.sleep(poll_interval)
                continue
                
            except Exception as e:
                consecutive_failures += 1
                logger.warning(f"Error checking status ({consecutive_failures}/{max_consecutive_failures}): {e}")
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Too many errors, marking as failed")
                    return False
                
                time.sleep(poll_interval)
                continue
        
        logger.error(f"⏱️ Production timed out after {timeout}s")
        return False
    
    def _download_output_with_retry(self, production_uuid: str, output_path: Path) -> Optional[Path]:
        """Download with enhanced retry logic for network issues"""
        max_download_attempts = 3
        base_delay = 10
        
        for attempt in range(max_download_attempts):
            try:
                result = self._download_single_attempt(production_uuid, output_path)
                if result and result.exists() and result.stat().st_size > 0:
                    return result
                else:
                    logger.warning(f"Download attempt {attempt + 1} produced empty/invalid file")
                    
            except Exception as e:
                if "Network is unreachable" in str(e) or "ConnectionError" in str(type(e).__name__):
                    logger.warning(f"Network error on download attempt {attempt + 1}: {e}")
                else:
                    logger.error(f"Download error on attempt {attempt + 1}: {e}")
                    
                if attempt < max_download_attempts - 1:
                    wait_time = base_delay * (2 ** attempt)
                    logger.info(f"Retrying download in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("All download attempts failed")
                    return None
        
        return None
    
    def _download_single_attempt(self, production_uuid: str, output_path: Path) -> Optional[Path]:
        """Single download attempt"""
        url = f"{self.base_url}/production/{production_uuid}.json"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            response = self.session.get(url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                raise Exception(f"Failed to get production details: HTTP {response.status_code}")
            
            data = response.json().get("data", {})
            output_files = data.get("output_files", [])
            
            if not output_files:
                raise Exception("No output files in production")
            
            # Find audio file
            audio_file = None
            for file in output_files:
                file_format = file.get("format", "")
                if file_format in ["mp3", "wav", "ogg", "m4a", "aac"]:
                    audio_file = file
                    break
            
            if not audio_file:
                available = [f.get("format") for f in output_files]
                raise Exception(f"No audio output found. Available: {available}")
            
            download_url = audio_file.get("download_url")
            if not download_url:
                raise Exception("No download URL in output file")
            
            # Download the file
            logger.info(f"Downloading: {download_url}")
            
            download_response = self.session.get(download_url, headers=headers, timeout=300, stream=True)
            
            if download_response.status_code != 200:
                raise Exception(f"Download failed: HTTP {download_response.status_code}")
            
            # Save to file
            with open(output_path, "wb") as f:
                for chunk in download_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Verify
            if output_path.exists() and output_path.stat().st_size > 0:
                file_size = output_path.stat().st_size
                logger.info(f"✅ Downloaded: {output_path} ({file_size:,} bytes)")
                return output_path
            else:
                raise Exception("Downloaded file is empty or missing")
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            raise
    
    def _cleanup_production(self, production_uuid: str):
        """Delete a specific production"""
        try:
            url = f"{self.base_url}/production/{production_uuid}.json"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            response = self.session.delete(url, headers=headers, timeout=30)
            
            if response.status_code in [200, 204]:
                logger.info(f"Cleaned up failed production: {production_uuid}")
            else:
                logger.warning(f"Cleanup failed: {response.status_code}")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
    
    def _cleanup_unfinished(self):
        """Clean up unfinished productions"""
        try:
            url = f"{self.base_url}/productions.json"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            response = self.session.get(url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                logger.warning(f"Failed to get productions list: {response.status_code}")
                return
            
            productions = response.json().get("data", [])
            deleted = 0
            
            for prod in productions:
                status = prod.get("status")
                uuid = prod.get("uuid")
                
                if not uuid:
                    continue
                
                should_delete = False
                delete_reason = ""
                
                if status == 2:
                    should_delete = True
                    delete_reason = "Error"
                elif status == 9:
                    should_delete = True
                    delete_reason = "Incomplete"
                elif status in [0, 1, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]:
                    created_time = prod.get("creation_time")
                    if created_time:
                        try:
                            created_dt = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
                            now = datetime.now(created_dt.tzinfo)
                            age = now - created_dt
                            
                            if age > timedelta(hours=1):
                                should_delete = True
                                delete_reason = f"Stuck for {age.total_seconds()/3600:.1f}h"
                        except Exception as e:
                            logger.debug(f"Could not parse creation time: {e}")
                
                if should_delete:
                    delete_url = f"{self.base_url}/production/{uuid}.json"
                    del_response = self.session.delete(delete_url, headers=headers, timeout=30)
                    
                    if del_response.status_code in [200, 204]:
                        deleted += 1
                        logger.info(f"Deleted {delete_reason} production: {uuid} (status: {status})")
                    else:
                        logger.warning(f"Failed to delete {uuid}: {del_response.status_code}")
            
            logger.info(f"Cleanup complete: {deleted} unfinished productions deleted")
            
        except Exception as e:
            logger.warning(f"Bulk cleanup error: {e}")


def enhance_audio_with_auphonic(input_audio_path: Path, output_audio_path: Optional[Path] = None) -> Optional[Path]:
    """Convenience function to enhance audio"""
    client = AuphonicAPI()
    return client.enhance_audio(input_audio_path, output_audio_path)


if __name__ == "__main__":
    import sys
    
    # Setup logging for CLI usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 2:
        print("Usage: python auphonic_api.py <input_audio_file> [output_audio_file]")
        print("\nMake sure AUPHONIC_API_KEY is set in your .env file")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    enhanced = enhance_audio_with_auphonic(input_file, output_file)
    if enhanced:
        print(f"✅ Enhanced audio saved: {enhanced}")
    else:
        print("❌ Enhancement failed")
        sys.exit(1)