# Stop After Images Feature

## Overview

Added a new `--stop-after-images` flag to the video generation pipeline that allows you to test the pipeline up to the image generation stage without completing the full video creation.

## Usage

### Basic Usage

```bash
python make_video.py --stop-after-images
```

### Combined with Other Options

```bash
# Stop after images with a specific job ID
python make_video.py --stop-after-images --job-id test_job_123

# Stop after images with limited scenes for faster testing
python make_video.py --stop-after-images --limit-scenes 3

# Stop after images while skipping silence trimming
python make_video.py --stop-after-images --skip-trim
```

## What It Does

When you use `--stop-after-images`, the pipeline will:

1. ✅ **Complete these steps:**

   - Trim silence (if trim_silence.py exists and not skipped)
   - Generate subtitles
   - Generate script/scenes
   - Generate images for all scenes

2. ⏭️ **Skip these steps:**
   - Video chunk generation (generate_video_chunks_seedance.py)
   - Video merge and audio addition (merge_and_add_audio.py)

## Output

When using `--stop-after-images`, the pipeline will:

- **Stop after image generation** instead of creating a final video
- **Output a JSON summary** with:
  - Stage: `"images_completed"`
  - List of generated scene images
  - Scene count and directory information
  - Pipeline metadata (job ID, timing, settings)

## Benefits

1. **Faster Testing**: Test image generation without waiting for video creation
2. **Cost Savings**: Avoid expensive video generation API calls during development
3. **Debug Images**: Inspect generated images before committing to full video generation
4. **Quick Iteration**: Make adjustments to prompts or settings without full pipeline runs

## Example Output

```json
{
  "job_id": "test_job_123",
  "audio_selected": "pipeline/audio_input/input_trimmed.mp3",
  "stage": "images_completed",
  "scenes_dir": "pipeline/scenes",
  "scene_count": 5,
  "scene_images": [
    "pipeline/scenes/scene_001.png",
    "pipeline/scenes/scene_002.png",
    "pipeline/scenes/scene_003.png",
    "pipeline/scenes/scene_004.png",
    "pipeline/scenes/scene_005.png"
  ],
  "started_at": "2025-11-06T10:30:00",
  "finished_at": "2025-11-06T10:32:45",
  "resolution": "480p",
  "fps": 24,
  "limit_scenes": null,
  "trim_applied": true,
  "trim_args": "",
  "stopped_after": "image_generation"
}
```

## File Locations

- Generated images: `pipeline/scenes/scene_*.png`
- Image prompts: `pipeline/scenes/prompt.json`
- Scene manifest: `pipeline/scenes/manifest.json`

## Reverting to Full Pipeline

To run the full pipeline normally, simply omit the `--stop-after-images` flag:

```bash
python make_video.py
```

Or use the existing skip flags if needed:

```bash
python make_video.py --skip-i2v --skip-merge  # Manual equivalent
```
