#!/usr/bin/env python3
"""
Test script for image B2 upload functionality.
This script tests the new image upload mechanism without requiring a full pipeline run.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import io

# Add the app directory to the path so we can import worker_tasks
sys.path.append(str(Path(__file__).parent / "app"))

try:
    from app.worker_tasks import upload_images_to_b2, log as worker_log
    print("SUCCESS: Successfully imported upload_images_to_b2 function")
except ImportError as e:
    print(f"ERROR: Failed to import upload_images_to_b2: {e}")
    sys.exit(1)

load_dotenv()

def create_test_images(images_dir: Path, num_images: int = 3):
    """Create test images in the specified directory."""
    images_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_images):
        # Create a simple test image
        img = Image.new('RGB', (256, 256), color=(i * 80, 100 + i * 20, 200 - i * 30))
        img_path = images_dir / f"test_image_{(i+1):03d}.png"
        img.save(img_path, 'PNG')
        print(f"SUCCESS: Created test image: {img_path}")

    return images_dir

def test_b2_credentials():
    """Test if B2 credentials are configured."""
    bucket_name = os.getenv("B2_BUCKET_NAME")
    key_id = os.getenv("B2_KEY_ID")
    app_key = os.getenv("B2_APPLICATION_KEY")

    if not all([bucket_name, key_id, app_key]):
        print("WARNING: B2 credentials not fully configured:")
        print(f"   B2_BUCKET_NAME: {'OK' if bucket_name else 'MISSING'}")
        print(f"   B2_KEY_ID: {'OK' if key_id else 'MISSING'}")
        print(f"   B2_APPLICATION_KEY: {'OK' if app_key else 'MISSING'}")
        return False

    print("SUCCESS: B2 credentials are configured")
    return True

def main():
    print("INFO: Testing Image B2 Upload Functionality")
    print("=" * 50)

    # Test B2 credentials
    if not test_b2_credentials():
        print("\nWARNING: B2 credentials not configured - upload test will be skipped")
        print("   Set B2_BUCKET_NAME, B2_KEY_ID, and B2_APPLICATION_KEY in .env file")
        return

    # Create test job ID and images directory
    test_job_id = f"test_image_upload_{int(__import__('time').time())}"
    test_images_dir = Path(f"test_images_{test_job_id}")

    try:
        # Create test images
        print(f"\nINFO: Creating test images in: {test_images_dir}")
        create_test_images(test_images_dir)

        # Test B2 upload
        print(f"\nINFO: Testing B2 upload for job: {test_job_id}")
        image_urls = upload_images_to_b2(test_job_id, test_images_dir)

        if image_urls:
            print(f"SUCCESS: Successfully uploaded {len(image_urls)} images to B2:")
            for image_name, url in image_urls.items():
                print(f"   {image_name}: {url}")
        else:
            print("ERROR: No images were uploaded to B2")

    except Exception as e:
        print(f"ERROR: Test failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup test images
        if test_images_dir.exists():
            import shutil
            shutil.rmtree(test_images_dir)
            print(f"\nINFO: Cleaned up test directory: {test_images_dir}")

    print("\n" + "=" * 50)
    print("INFO: Test completed")

if __name__ == "__main__":
    main()