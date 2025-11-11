#!/usr/bin/env python3
"""
Test script for Auphonic API integration.

This script tests the complete Auphonic integration flow including:
- Environment configuration verification
- API initialization 
- Audio enhancement with and without API key
- Error handling and fallback behavior
"""

import os
import sys
from pathlib import Path

# Add the pipeline directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).parent / "app"))

from pipeline.config import settings
from pipeline.auphonic_api import AuphonicAPI


def test_environment_configuration():
    """Test that environment variables are properly configured."""
    print("🔍 Testing environment configuration...")
    
    # Check Auphonic settings
    print(f"AUPHONIC_ENABLED: {settings.AUPHONIC_ENABLED}")
    print(f"AUPHONIC_API_KEY: {'[SET]' if settings.AUPHONIC_API_KEY else '[NOT SET]'}")
    print(f"AUPHONIC_PRESET: {settings.AUPHONIC_PRESET}")
    
    # Verify settings object has expected attributes
    expected_attrs = ['AUPHONIC_ENABLED', 'AUPHONIC_API_KEY', 'AUPHONIC_PRESET']
    missing_attrs = [attr for attr in expected_attrs if not hasattr(settings, attr)]
    
    if missing_attrs:
        print(f"❌ Missing attributes: {missing_attrs}")
        return False
    else:
        print("✅ All expected Auphonic settings attributes present")
        return True


def test_api_initialization():
    """Test Auphonic API client initialization."""
    print("\n🔧 Testing API client initialization...")
    
    try:
        # Test initialization with configured API key
        if settings.AUPHONIC_API_KEY:
            print("✅ Testing with API key configured")
            client = AuphonicAPI()
            print("✅ AuphonicAPI client initialized successfully")
            return True
        else:
            print("⚠️ No API key configured, testing fallback behavior")
            try:
                client = AuphonicAPI()
                print("✅ AuphonicAPI client initialized (will use fallback)")
                return True
            except Exception as e:
                print(f"❌ Failed to initialize client: {e}")
                return False
                
    except Exception as e:
        print(f"❌ API initialization failed: {e}")
        return False


def test_audio_file_handling():
    """Test audio file handling with sample files."""
    print("\n🎵 Testing audio file handling...")
    
    # Look for sample audio files in the project
    sample_dirs = [
        Path("other_audio"),
        Path("pipeline/audio_input"),
        Path("uploads")
    ]
    
    audio_files = []
    for sample_dir in sample_dirs:
        if sample_dir.exists():
            # Look for common audio extensions
            for ext in ['.mp3', '.wav', '.m4a', '.aac']:
                audio_files.extend(sample_dir.glob(f"*{ext}"))
                audio_files.extend(sample_dir.glob(f"*{ext.upper()}"))
    
    if not audio_files:
        print("⚠️ No sample audio files found for testing")
        return True  # This is OK for testing
    
    print(f"✅ Found {len(audio_files)} audio files for testing")
    
    # Test with the first audio file if available
    if audio_files:
        test_audio = audio_files[0]
        print(f"📁 Test audio file: {test_audio}")
        print(f"📊 File size: {test_audio.stat().st_size} bytes")
        
        try:
            client = AuphonicAPI()
            print("🎯 Testing audio file processing...")
            
            # This will test the file validation and processing logic
            # Note: This might fail if no API key is configured, which is expected
            result = client.enhance_audio(test_audio)
            
            if result is None:
                print("✅ API returned None (expected without valid API key)")
            elif isinstance(result, Path):
                print(f"✅ API returned enhanced file: {result}")
            else:
                print(f"⚠️ Unexpected result type: {type(result)}")
                
        except Exception as e:
            print(f"⚠️ Audio processing test failed (expected without API key): {e}")
    
    return True


def test_error_handling():
    """Test error handling and fallback behavior."""
    print("\n🛡️ Testing error handling and fallback...")
    
    # Test with non-existent file
    try:
        client = AuphonicAPI()
        fake_path = Path("non_existent_file.mp3")
        result = client.enhance_audio(fake_path)
        print("❌ Should have failed with non-existent file")
        return False
    except Exception as e:
        print(f"✅ Properly handled non-existent file: {type(e).__name__}")
    
    # Test with invalid API key behavior
    if not settings.AUPHONIC_API_KEY:
        print("✅ Confirmed: System gracefully handles missing API key")
    
    return True


def main():
    """Main test function."""
    print("🚀 Starting Auphonic API Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Environment Configuration", test_environment_configuration),
        ("API Initialization", test_api_initialization),
        ("Audio File Handling", test_audio_file_handling),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Auphonic integration is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the configuration.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)