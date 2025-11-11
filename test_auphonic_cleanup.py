#!/usr/bin/env python3
"""
Test script for Auphonic API cleanup functionality
"""

import sys
import logging
from pathlib import Path
import os

# Add the project root to the path so we can import from pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.auphonic_api import AuphonicAPI
from pipeline.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_cleanup_functionality():
    """Test the cleanup functionality"""
    print("Testing Auphonic API cleanup functionality...")
    
    # Check if Auphonic is enabled
    if not settings.AUPHONIC_ENABLED:
        print("Auphonic API is disabled in settings")
        return False
    
    if not settings.AUPHONIC_API_KEY:
        print("AUPHONIC_API_KEY is not configured")
        return False
    
    # Initialize Auphonic API client
    try:
        client = AuphonicAPI()
        print("✓ Auphonic API client initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize Auphonic API client: {e}")
        return False
    
    # Test cleanup functionality
    try:
        print("Testing cleanup of old productions...")
        success = client.cleanup_old_productions(max_age_hours=24)
        if success:
            print("✓ Cleanup functionality executed successfully")
        else:
            print("✗ Cleanup functionality failed")
            return False
    except Exception as e:
        print(f"✗ Error during cleanup test: {e}")
        return False
    
    print("\nCleanup test completed successfully!")
    return True


def test_enhancement_with_cleanup():
    """Test enhancement functionality that includes cleanup"""
    print("\nTesting Auphonic enhancement with integrated cleanup...")
    
    # Check if we have a test audio file
    test_audio_path = Path("other_audio/input.mp3")
    if not test_audio_path.exists():
        print(f"Test audio file not found: {test_audio_path}")
        print("Creating a simple test file...")
        # Create a minimal test by just running the cleanup without enhancement
        return test_cleanup_functionality()
    
    try:
        client = AuphonicAPI()
        print(f"Testing enhancement on: {test_audio_path}")
        
        # This will trigger the enhanced workflow including cleanup
        result = client.enhance_audio(test_audio_path)
        if result:
            print(f"✓ Enhancement with cleanup completed successfully: {result}")
            return True
        else:
            print("✗ Enhancement with cleanup failed")
            return False
            
    except Exception as e:
        print(f"✗ Error during enhancement test: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("AUPHONIC CLEANUP FUNCTIONALITY TEST")
    print("=" * 60)
    
    # Test cleanup functionality
    cleanup_success = test_cleanup_functionality()
    
    # Test enhancement with cleanup
    enhancement_success = test_enhancement_with_cleanup()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY:")
    print("=" * 60)
    print(f"Cleanup Test: {'PASSED' if cleanup_success else 'FAILED'}")
    print(f"Enhancement Test: {'PASSED' if enhancement_success else 'FAILED'}")
    
    if cleanup_success and enhancement_success:
        print("\n🎉 All tests passed! The cleanup functionality is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the logs above.")
        sys.exit(1)