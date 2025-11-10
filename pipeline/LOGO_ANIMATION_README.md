# Logo Animation Feature

This document explains how to add a logo animation at the end (or beginning) of each video generation.

## Overview

The logo animation feature has been integrated into the video generation pipeline. When enabled, a logo animation video file (e.g., `logo_animation.mp4`) is automatically appended to the end of each generated video.

## Configuration

### Method 1: Using Environment Variables (.env file)

Add the following settings to your `.env` file in the pipeline directory:

```bash
# Enable/disable logo animation
LOGO_ENABLED=true

# Path to your logo animation file (relative to pipeline directory)
LOGO_ANIMATION_PATH=logo_animation.mp4

# Position: 'end' to add logo at the end, 'start' to add at the beginning
LOGO_POSITION=end
```

### Method 2: Hardcoded Configuration (Default)

The default settings are:

- `LOGO_ENABLED = true` (enabled)
- `LOGO_ANIMATION_PATH = "logo_animation.mp4"`
- `LOGO_POSITION = "end"`

## Setup

1. **Place your logo animation file** in the pipeline directory:

   ```
   pipeline/
   ├── logo_animation.mp4  ← Your logo animation file
   ├── merge_and_add_audio.py
   └── ...
   ```

2. **Configure settings** (optional - defaults will be used if not configured)

3. **Run your video generation** as normal:
   ```bash
   python make_video.py
   ```

## How It Works

The logo animation is added during the final merge step:

1. Video chunks are concatenated
2. Logo animation is appended (if enabled and file exists)
3. Audio is added to the final video
4. Temporary files are cleaned up automatically

## File Requirements

- **Logo animation file**: MP4 format recommended
- **Compatible codecs**: Should work with standard MP4 codecs
- **Duration**: Any duration is supported
- **Resolution**: Will be handled automatically by ffmpeg

## Examples

### Basic Usage (Default)

```bash
# Default: logo_animation.mp4 will be added at the end
python make_video.py
```

### With Custom Logo File

```bash
# In your .env file:
LOGO_ANIMATION_PATH=my_company_logo.mp4

# Run normally
python make_video.py
```

### Logo at Start Instead of End

```bash
# In your .env file:
LOGO_POSITION=start

# Run normally
python make_video.py
```

### Disable Logo Animation

```bash
# In your .env file:
LOGO_ENABLED=false

# Or simply rename/move the logo_animation.mp4 file
```

## Troubleshooting

### Logo Not Appearing

1. Check that `logo_animation.mp4` exists in the pipeline directory
2. Verify `LOGO_ENABLED=true` in your configuration
3. Check that the logo file is not corrupted
4. Review the console output for error messages

### Performance Issues

- The logo animation is added using ffmpeg concatenation, which is efficient
- No significant performance impact expected
- Temporary files are automatically cleaned up

### Render Deployment

This feature works on Render as it:

- Uses standard ffmpeg commands
- Doesn't require additional dependencies
- Handles file paths relative to the working directory

## Advanced Configuration

You can also modify the logo positioning logic in `merge_and_add_audio.py` by changing the `LOGO_POSITION` setting or modifying the `prepare_video_with_logo()` function for custom positioning.

## Logs and Debugging

The system provides clear logging output:

- `[LOG] Preparing video with logo animation...`
- `[LOG] Appending logo animation: /path/to/logo_animation.mp4`
- `✅ Logo animation added at end: /path/to/logo_animation.mp4`

Check the console output for any issues during the merge process.
