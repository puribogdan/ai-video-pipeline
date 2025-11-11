#!/usr/bin/env python3
"""
Auphonic API Integration Module - Documentation-Based Implementation
"""

import os
import logging
import time
from pathlib import Path
from typing import Optional
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class AuphonicAPI:
    """Auphonic API client following official documentation"""
    
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
        
        if not self.enabled:
            logger.info("Auphonic API is disabled (AUPHONIC_ENABLED=false)")
            return
            
        if not self.api_key:
            logger.warning("AUPHONIC_API_KEY not found in environment")
            self.enabled = False
            return
        
        logger.info("Auphonic API client initialized")
    
    def enhance_audio(self, input_audio_path: Path, output_audio_path: Optional[Path] = None) -> Optional[Path]:
        """
        Enhance audio using Auphonic Simple API
        
        Args:
            input_audio_path: Path to the original audio file
            output_audio_path: Optional path for enhanced audio
            
        Returns:
            Path to enhanced audio file, or None if failed
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
        
        try:
            logger.info(f"Starting Auphonic enhancement: {input_audio_path.name}")
            
            # Step 1: Create and start production using Simple API
            production_uuid = self._create_production(input_audio_path)
            if not production_uuid:
                return None
            
            logger.info(f"Production created: {production_uuid}")
            logger.info(f"View at: https://auphonic.com/production/{production_uuid}")
            
            # Step 2: Wait for completion (status must be 3)
            if not self._wait_for_completion(production_uuid):
                self._cleanup_production(production_uuid)
                return None
            
            # Step 3: Download output
            result = self._download_output(production_uuid, output_audio_path)
            
            # Step 4: Cleanup
            self._cleanup_production(production_uuid)
            
            if result and result.exists():
                logger.info(f"✅ Enhancement complete: {result}")
                return result
            else:
                logger.error("Failed to download enhanced audio")
                return None
            
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return None
    
    def _create_production(self, audio_path: Path) -> Optional[str]:
        """Create production using Simple API (single request)"""
        url = f"{self.base_url}/simple/productions.json"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        with open(audio_path, "rb") as f:
            files = {"input_file": f}
            
            # Use Simple API parameters as documented
            data = {
                "title": f"Enhancement - {audio_path.stem}",
                "filtering": "true",
                "normloudness": "true",
                "loudnesstarget": "-16",
                "leveler": "true",
                "action": "start"  # Start immediately
            }
            
            try:
                response = requests.post(url, headers=headers, files=files, data=data, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    production_uuid = result.get("data", {}).get("uuid")
                    if production_uuid:
                        return production_uuid
                
                # Handle "too many productions" error
                if response.status_code == 400 and "too many" in response.text.lower():
                    logger.warning("Queue full, cleaning up...")
                    self._cleanup_all_unfinished()
                    time.sleep(3)
                    
                    # Retry once
                    with open(audio_path, "rb") as f2:
                        files2 = {"input_file": f2}
                        response = requests.post(url, headers=headers, files=files2, data=data, timeout=60)
                        if response.status_code == 200:
                            result = response.json()
                            return result.get("data", {}).get("uuid")
                
                logger.error(f"Failed to create: {response.status_code}")
                logger.error(f"Response: {response.text[:500]}")
                return None
                
            except Exception as e:
                logger.error(f"Error creating production: {e}")
                return None
    
    def _wait_for_completion(self, production_uuid: str, timeout: int = 300) -> bool:
        """
        Wait for production to complete.
        Status 3 = Done (success)
        Status 2 = Error (failure)
        All others = Still processing
        """
        url = f"{self.base_url}/production/{production_uuid}.json"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        start_time = time.time()
        poll_interval = 5
        last_status = None
        
        logger.info("Waiting for production to complete...")
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code != 200:
                    logger.warning(f"Status check failed: {response.status_code}")
                    time.sleep(poll_interval)
                    continue
                
                data = response.json().get("data", {})
                status = data.get("status")
                status_string = data.get("status_string", "Unknown")
                
                # Log status change
                if status != last_status:
                    status_name = self.STATUS_CODES.get(status, f"Unknown-{status}")
                    logger.info(f"Status: {status} ({status_name} - {status_string})")
                    last_status = status
                
                # Status 3 = Done (SUCCESS!)
                if status == 3:
                    logger.info("✅ Production completed successfully")
                    return True
                
                # Status 2 = Error (FAILURE!)
                elif status == 2:
                    error_msg = data.get("error_message", "Unknown error")
                    error_status = data.get("error_status", "")
                    logger.error(f"❌ Production failed: {error_msg} ({error_status})")
                    return False
                
                # All other statuses = still processing
                # Status 0,1,4,5,6,7,8,9,10,12,14,15 = keep waiting
                else:
                    time.sleep(poll_interval)
                    continue
                
            except Exception as e:
                logger.warning(f"Error checking status: {e}")
                time.sleep(poll_interval)
        
        logger.error(f"⏱️ Production timed out after {timeout}s")
        return False
    
    def _download_output(self, production_uuid: str, output_path: Path) -> Optional[Path]:
        """Download the enhanced audio from completed production"""
        url = f"{self.base_url}/production/{production_uuid}.json"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"Failed to get production details: {response.status_code}")
                return None
            
            data = response.json().get("data", {})
            output_files = data.get("output_files", [])
            
            if not output_files:
                logger.error("No output files in production")
                return None
            
            # Find audio file
            audio_file = None
            for file in output_files:
                file_format = file.get("format", "")
                if file_format in ["mp3", "wav", "ogg", "m4a", "aac"]:
                    audio_file = file
                    break
            
            if not audio_file:
                available = [f.get("format") for f in output_files]
                logger.error(f"No audio output found. Available: {available}")
                return None
            
            download_url = audio_file.get("download_url")
            if not download_url:
                logger.error("No download URL in output file")
                return None
            
            # Download the file
            logger.info(f"Downloading: {download_url}")
            
            download_response = requests.get(download_url, headers=headers, timeout=300, stream=True)
            
            if download_response.status_code != 200:
                logger.error(f"Download failed: {download_response.status_code}")
                return None
            
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
                logger.error("Downloaded file is empty or missing")
                return None
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            return None
    
    def _cleanup_production(self, production_uuid: str):
        """Delete a production"""
        try:
            url = f"{self.base_url}/production/{production_uuid}.json"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            response = requests.delete(url, headers=headers, timeout=30)
            
            if response.status_code in [200, 204]:
                logger.info(f"Cleaned up: {production_uuid}")
            else:
                logger.warning(f"Cleanup failed: {response.status_code}")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
    
    def _cleanup_all_unfinished(self):
        """Clean up ALL unfinished productions (not status 3)"""
        try:
            url = f"{self.base_url}/productions.json"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                return
            
            productions = response.json().get("data", [])
            deleted = 0
            
            for prod in productions:
                status = prod.get("status")
                # Delete everything except status 3 (Done)
                if status != 3:
                    uuid = prod.get("uuid")
                    if uuid:
                        delete_url = f"{self.base_url}/production/{uuid}.json"
                        del_response = requests.delete(delete_url, headers=headers, timeout=30)
                        if del_response.status_code in [200, 204]:
                            deleted += 1
                            logger.info(f"Deleted unfinished: {uuid} (status: {status})")
            
            logger.info(f"Cleanup complete: {deleted} productions deleted")
            
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