# Auphonic API Integration for Audio Enhancement

This document describes the Auphonic API integration that has been implemented to replace local audio processing libraries with a professional audio enhancement service.

## Overview

The system now uses **Auphonic**, a professional audio post-production web service, to enhance uploaded audio files before video processing. This provides:

- **Professional audio quality**: Advanced noise reduction, loudness normalization, and speech enhancement
- **No local dependencies**: Eliminates the need for local audio processing libraries (librosa, noisereduce, etc.)
- **Reliable processing**: Cloud-based processing with built-in error handling and fallbacks
- **Cost-effective**: Pay-per-use model without infrastructure overhead

## Architecture

### Integration Flow

1. **Audio Upload**: User uploads audio file through the web interface
2. **File Detection**: Worker detects and validates the audio file
3. **Auphonic Enhancement** (if enabled):
   - Upload audio to Auphonic service
   - Apply professional audio processing preset
   - Download enhanced audio file
4. **Pipeline Processing**: Use enhanced audio for video generation
5. **Fallback Handling**: Use original audio if enhancement fails

### Files Modified/Created

- **`pipeline/auphonic_api.py`** - New Auphonic API client implementation
- **`pipeline/config.py`** - Added Auphonic configuration settings
- **`app/worker_tasks.py`** - Updated to call Auphonic API after audio detection
- **`AUPHONIC_INTEGRATION.md`** - This documentation file

## Configuration

### Environment Variables

Add these variables to your `.env` file:

```bash
# Auphonic API Configuration
AUPHONIC_ENABLED=true
AUPHONIC_API_KEY=your_auphonic_api_key_here
AUPHONIC_PRESET=ceigtvDv8jH6NaK52Z5eXH
```

### Configuration Options

- **`AUPHONIC_ENABLED`** (bool, default: true)

  - Enable/disable Auphonic enhancement
  - When disabled, system uses original audio files

- **`AUPHONIC_API_KEY`** (string, optional)

  - Your Auphonic API key for authentication
  - Get from: https://auphonic.com/accounts/password/
  - Required for enhancement functionality

- **`AUPHONIC_PRESET`** (string, default: "ceigtvDv8jH6NaK52Z5eXH")
  - Auphonic processing preset ID
  - Controls audio enhancement parameters
  - Default preset provides good general-purpose enhancement

## API Integration Details

### AuphonicAPI Class

Located in `pipeline/auphonic_api.py`, this class handles:

- **Authentication**: Secure API key management
- **File Upload/Download**: Handles temporary file management
- **Job Management**: Tracks enhancement jobs and status
- **Error Handling**: Graceful fallbacks and logging
- **Format Support**: Works with MP3, WAV, M4A, and other formats

### Processing Features

The default Auphonic preset includes:

- **Noise Reduction**: Advanced background noise removal
- **Loudness Normalization**: Consistent audio levels (-23 LUFS)
- **Speech Enhancement**: Improved clarity and presence
- **Hum Removal**: 50/60Hz frequency interference removal
- **Dynamic Range Compression**: Balanced audio dynamics

## Integration Points

### Worker Task Integration

In `app/worker_tasks.py`, the enhancement happens after audio file detection:

```python
# Enhanced audio processing with Auphonic API
enhanced_audio_path = hint_audio
if settings.AUPHONIC_ENABLED and settings.AUPHONIC_API_KEY:
    try:
        from pipeline.auphonic_api import AuphonicAPI

        # Enhance the audio
        if hint_audio is None:
            log(f"[ERROR] No audio file available for Auphonic enhancement")
            enhanced_audio_path = hint_audio
        else:
            auphonic_client = AuphonicAPI()
            enhanced_path = auphonic_client.enhance_audio(hint_audio)

            if enhanced_path and enhanced_path.exists():
                log(f"[INFO] Auphonic enhancement successful: {enhanced_path}")
                enhanced_audio_path = enhanced_path
            else:
                log(f"[WARNING] Auphonic enhancement failed, using original audio")
                enhanced_audio_path = hint_audio

    except Exception as e:
        log(f"[WARNING] Auphonic enhancement failed: {e}")
        enhanced_audio_path = hint_audio
else:
    log(f"[INFO] Auphonic enhancement disabled, using original audio")
```

### Fallback Behavior

The integration is designed with robust fallback mechanisms:

1. **No API Key**: System gracefully skips enhancement and uses original audio
2. **API Failure**: Original audio is used as fallback
3. **Network Issues**: Enhancement is skipped with appropriate logging
4. **Invalid Audio**: Original file is preserved and used

## Monitoring and Logging

### Logging Levels

- **INFO**: Successful enhancements, file sizes, processing times
- **WARNING**: Fallback to original audio, API issues
- **ERROR**: Critical failures, missing configurations

### Example Log Output

```
[INFO] Starting Auphonic audio enhancement for: /path/to/audio.mp3
[INFO] Auphonic enhancement successful: /tmp/enhanced_audio.mp3
[INFO] Enhanced audio file size: 2456789 bytes
[INFO] Using enhanced audio file for video processing
```

## Benefits Over Local Processing

### Advantages

1. **Quality**: Professional-grade audio processing algorithms
2. **Reliability**: Consistent results without local library conflicts
3. **Maintenance**: No need to update audio processing dependencies
4. **Performance**: Optimized cloud processing vs. local CPU usage
5. **Consistency**: Same enhancement quality across all environments

### Removed Dependencies

These local audio processing libraries are no longer required:

- `librosa` - Audio analysis and processing
- `noisereduce` - Noise reduction algorithms
- `pyloudnorm` - Loudness normalization
- `webrtcvad` - Voice activity detection
- `scipy` - Signal processing
- `soundfile` - Audio file I/O

## Setup Instructions

### 1. Get Auphonic API Key

1. Visit https://auphonic.com/accounts/password/
2. Create an account or log in
3. Navigate to "Account" → "API Access"
4. Copy your API key

### 2. Configure Environment

Add to your `.env` file:

```bash
AUPHONIC_ENABLED=true
AUPHONIC_API_KEY=your_actual_api_key_here
```

### 3. Restart Application

Restart your application to load the new configuration.

### 4. Test Integration

Upload an audio file and check logs for enhancement activity.

## Troubleshooting

### Common Issues

1. **"Auphonic enhancement disabled"**

   - Check that `AUPHONIC_ENABLED=true` in environment

2. **"Auphonic enhancement failed"**

   - Verify `AUPHONIC_API_KEY` is correctly set
   - Check API key validity on Auphonic website

3. **Audio quality unchanged**
   - Enhancement may have failed silently
   - Check logs for specific error messages

### Debug Information

Enable debug logging to see detailed enhancement process:

```python
import logging
logging.getLogger('pipeline.auphonic_api').setLevel(logging.DEBUG)
```

## Cost Considerations

- **Auphonic Pricing**: Pay per minute of audio processed
- **Free Tier**: Available for development/testing
- **Production Usage**: Monitor API usage and costs
- **Optimization**: Default preset provides good quality/efficiency balance

## Future Enhancements

Potential improvements for future versions:

1. **Multiple Presets**: Allow users to choose enhancement style
2. **Batch Processing**: Process multiple files efficiently
3. **Quality Tiers**: Different processing levels for various use cases
4. **Custom Presets**: User-defined enhancement parameters
5. **Usage Analytics**: Track enhancement costs and effectiveness

---

## Summary

The Auphonic integration successfully replaces complex local audio processing with a professional, reliable cloud service. This provides better audio quality, reduces maintenance burden, and offers a more scalable solution for audio enhancement in video processing workflows.
