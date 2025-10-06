#!/usr/bin/env python3
"""
Test script to verify B2 image upload functionality.
Run this script to test if B2 credentials are working and images can be uploaded.
"""

import os
import sys
from pathlib import Path
import tempfile
from PIL import Image

# Add the app directory to the path so we can import worker_tasks
sys.path.append(str(Path(__file__).parent / "app"))

from worker_tasks import upload_images_to_b2, log

def create_test_image(path: Path, size: tuple = (100, 100), color: tuple = (255, 0, 0)) -> None:
    """Create a simple test image."""
    img = Image.new('RGB', size, color)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, 'PNG')
    log(f"Created test image: {path}")

def test_b2_connection():
    """Test B2 connection and credentials."""
    log("🔍 Testing B2 Configuration...")

    # Check environment variables
    bucket_name = os.getenv("B2_BUCKET_NAME")
    key_id = os.getenv("B2_KEY_ID")
    app_key = os.getenv("B2_APPLICATION_KEY")

    log(f"B2_BUCKET_NAME: {'✅ Set' if bucket_name else '❌ Not set'}")
    log(f"B2_KEY_ID: {'✅ Set' if key_id else '❌ Not set'}")
    log(f"B2_APPLICATION_KEY: {'✅ Set' if app_key else '❌ Not set'}")

    if not all([bucket_name, key_id, app_key]):
        log("❌ B2 credentials not properly configured!")
        return False

    log("✅ B2 credentials are configured")
    return True

def test_image_upload():
    """Test uploading images to B2."""
    log("\n🖼️  Testing Image Upload...")

    # Create a temporary directory with test images
    with tempfile.TemporaryDirectory() as temp_dir:
        images_dir = Path(temp_dir) / "images"
        images_dir.mkdir()

        # Create test images
        test_images = []
        for i in range(3):
            img_path = images_dir / f"test_image_{i}.png"
            create_test_image(img_path, size=(100, 100), color=(255, i*80, 0))
            test_images.append(img_path)

        log(f"Created {len(test_images)} test images in {images_dir}")

        # Test upload
        job_id = "test-job-123"
        try:
            image_urls = upload_images_to_b2(job_id, images_dir)

            if image_urls:
                log(f"✅ Successfully uploaded {len(image_urls)} images!")
                for filename, url in image_urls.items():
                    log(f"  - {filename}: {url}")
                return True
            else:
                log("❌ No images were uploaded")
                return False

        except Exception as e:
            log(f"❌ Image upload failed: {e}")
            return False

def main():
    """Main test function."""
    log("🚀 Starting B2 Image Upload Test")
    log("=" * 50)

    # Test 1: Check B2 configuration
    if not test_b2_connection():
        log("\n❌ B2 configuration test failed!")
        return 1

    # Test 2: Test image upload
    if not test_image_upload():
        log("\n❌ Image upload test failed!")
        return 1

    log("\n🎉 All tests passed! B2 image upload is working correctly.")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)