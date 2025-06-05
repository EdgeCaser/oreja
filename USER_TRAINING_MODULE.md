# Oreja User Embedding Training Module

## Overview

The User Embedding Training Module is a new feature for Oreja that allows users to improve speaker recognition accuracy by recording themselves reading standardized text. This module provides a guided interface for creating and updating speaker embeddings with immediate feedback on confidence improvements.

## Features

✅ **Select or Create Users**: Choose existing speakers or create brand new voice profiles  
✅ **Standardized Recording**: Read "The Hare Who Lost His Spectacles" story for consistent training  
✅ **Audio Recording**: Direct microphone recording with device selection  
✅ **File Upload**: Upload existing audio files for training  
✅ **Confidence Tracking**: See before/after confidence scores with improvement metrics  
✅ **Privacy-First**: All processing is local - audio files are automatically deleted  
✅ **Iterative Training**: Train multiple times to continuously improve recognition  

## Installation

### Prerequisites
Make sure you have the core Oreja system installed and working.

### Install Audio Dependencies
```bash
cd backend
pip install -r requirements_user_training.txt
```

This installs:
- `sounddevice` - Audio recording
- `soundfile` - Audio file I/O
- `librosa` - Audio processing
- `noisereduce` - Optional noise reduction

## Usage

### Method 1: Integrated with Speaker Analytics
1. Launch the main Speaker Analytics GUI:
   ```bash
   python backend/speaker_analytics_gui.py
   ```
2. Navigate to the "🎯 User Training" tab
3. Follow the interface workflow

### Method 2: Standalone Module
1. Launch the standalone training module:
   ```bash
   python backend/launch_user_training.py
   ```
2. Use the dedicated training interface

## Workflow

### 1. Select or Create User
**For Existing Users:**
- Browse the list of existing speakers in your database
- View current confidence scores and sample counts
- Select the user you want to train

**For New Users:**
- Enter a name in the "Create New User" section
- Click "➕ Create User" or press Enter
- The new user will be automatically selected for training

### 2. Record or Upload Audio
**Recording Option:**
- Select your microphone from the dropdown
- Click "🎤 Start Recording"
- Read the provided text clearly and naturally
- Click "⏹️ Stop Recording" when finished

**Upload Option:**
- Click "📁 Upload Audio File"
- Select a WAV, MP3, M4A, FLAC, or OGG file
- Ensure the audio contains only the target speaker

### 3. View Results
- See confidence score before training
- See confidence score after training
- View the improvement (positive/negative change)
- Green = improvement, Red = decrease, Black = no change

### 4. Train Again (Optional)
- Click "🔄 Train Again" to do another session
- Each additional training session can further improve recognition
- The "before" confidence updates to your current level

## Standard Training Text

The module uses "The Story of the Hare Who Lost His Spectacles" from Jethro Tull's "A Passion Play" for consistent training. This story features a hare who has lost his spectacles and various animal friends (Owl, Kangaroo, Newt, and Bee) who try to help him, complete with clever animal puns. The story provides excellent material for speaker training as it contains varied vocabulary, dialogue, and natural speech patterns.

## Audio Requirements

### Quality Guidelines
- **Environment**: Quiet room with minimal background noise
- **Duration**: 10-15 seconds minimum (full text reading)
- **Clarity**: Speak directly toward microphone
- **Pace**: Natural conversational speed
- **Style**: Match your normal speaking patterns

### Supported Formats
- WAV (recommended)
- MP3
- M4A
- FLAC
- OGG

### Technical Specs
- Sample rate: Automatically resampled to 16kHz
- Channels: Converted to mono if stereo
- Bit depth: Any (converted to float32)

## Privacy and Security

🔒 **Complete Privacy Protection:**
- All audio processing happens locally on your machine
- No audio data is ever transmitted to external servers
- Temporary audio files are automatically deleted after processing
- Only mathematical embeddings (not audio) are stored in the database

## Integration with Existing Oreja

### Database Integration
- Uses the existing `EnhancedSpeakerDatabase` (v2)
- Compatible with all existing speaker management features
- Embeddings integrate seamlessly with transcription system

### Embedding System
- Leverages the existing `OfflineSpeakerEmbeddingManager`
- Uses SpeechBrain ECAPA-TDNN models
- Maintains consistency with automatic speaker detection

## Technical Details

### Architecture
```
UserEmbeddingTrainer
├── GUI Interface (Tkinter)
├── Audio Recording (sounddevice)
├── Audio Processing (librosa)
├── Embedding Generation (SpeechBrain)
├── Database Updates (EnhancedSpeakerDatabase)
└── Privacy Controls (automatic cleanup)
```

### File Structure
```
backend/
├── user_embedding_trainer.py          # Main training module
├── launch_user_training.py            # Standalone launcher
├── requirements_user_training.txt     # Audio dependencies
└── speaker_analytics_gui.py           # Modified to include training tab
```

## Troubleshooting

### Missing Dependencies
**Error**: "Audio recording not available"
**Solution**: Install audio dependencies:
```bash
pip install sounddevice soundfile librosa
```

### No Audio Devices
**Error**: "No input devices found"
**Solutions**:
- Check microphone is connected and working
- Verify audio permissions in your OS
- Try a different audio device
- Restart the application

### Poor Recognition Improvement
**Causes**:
- Background noise in recording
- Different speaking style than normal usage
- Very short audio duration
- Multiple speakers in audio

**Solutions**:
- Record in a quieter environment
- Speak naturally (don't over-articulate)
- Read the full text (15+ seconds)
- Ensure only target speaker is audible

### Database Issues
**Error**: "Speaker not found"
**Solution**: Refresh the user list or restart the module

**Error**: "Failed to save embedding"
**Solution**: Check database file permissions and disk space

## Advanced Usage

### Multiple Training Sessions
- Training multiple times generally improves recognition
- Each session adds to the speaker's embedding collection
- Optimal training: 3-5 sessions with good audio quality
- Diminishing returns after 10+ sessions

### Optimal Training Strategy
1. **First Session**: Focus on clear, natural reading
2. **Second Session**: Vary your tone slightly
3. **Third Session**: Record at different volume levels
4. **Monitor Progress**: Stop when improvements plateau

### Integration with Transcription
After training:
1. Use Oreja for regular transcription
2. The improved embeddings automatically enhance recognition
3. Continue using speaker corrections for ongoing improvement
4. Return to training module if recognition degrades

## API Reference

### UserEmbeddingTrainer Class
```python
class UserEmbeddingTrainer:
    def __init__(self, parent_notebook, backend_url="http://127.0.0.1:8000")
    def refresh_user_list()
    def start_recording()
    def stop_recording() 
    def upload_audio_file()
    def process_audio_file(audio_file_path)
    def prepare_for_next_training()
```

### Integration Function
```python
def integrate_with_existing_gui(analytics_dashboard):
    """Add trainer to existing SpeakerAnalyticsDashboard"""
    return UserEmbeddingTrainer(analytics_dashboard.notebook)
```

## Contributing

To extend the User Training Module:

1. **Add New Training Texts**: Modify `STANDARD_TEXT` in `UserEmbeddingTrainer`
2. **Improve Audio Processing**: Enhance `process_audio_file()` method
3. **Add Analytics**: Extend confidence tracking and visualization
4. **Enhance UI**: Improve the Tkinter interface

## Future Enhancements

🔄 **Planned Features:**
- Multiple training text options
- Real-time audio feedback during recording
- Advanced confidence analytics and trending
- Batch training for multiple users
- Audio quality assessment
- Training history and session tracking

## Support

For issues with the User Training Module:

1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Test audio recording with other applications
4. Check Oreja core functionality first
5. Report issues with detailed error messages

## License

This module is part of the Oreja project and follows the same licensing terms. 