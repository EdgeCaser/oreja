# Oreja

Oreja is a Windows desktop application for real-time transcription of conference calls (Google Meet, Zoom, Slack, etc.) without storing or sending audio to the cloud. It captures microphone and system audio, uses local Hugging Face models for transcription and diarization, and provides a WPF interface with volume meters, transcription controls, and speaker renaming. Speaker embeddings improve recognition of recurring callers, stored locally in SQLite.

## Features
- Real-time transcription with speaker diarization using Whisper and pyannote.audio.
- Captures microphone and system audio simultaneously via NAudio.
- Volume meters for audio sources.
- Start/Stop transcription, save transcripts, and rename speakers.
- Privacy-focused: processes audio in memory, runs models locally, no cloud interaction.
- Improves speaker recognition over time using locally stored embeddings.
- **🔒 Privacy Mode**: Analyze conversations without saving transcription text
- **🗣️ Conversation Analysis**: Extract summaries, action items, topics, and insights
- **📊 Speaker Analytics**: Comprehensive dashboard for speaker database management

## Tech Stack
- **C#/.NET 8**: Windows-native frontend and audio capture.
- **Python 3.10**: Backend for transcription and diarization.
- **NAudio**: Audio capture (microphone and system).
- **Hugging Face Transformers (Whisper)**: Local speech-to-text.
- **pyannote.audio**: Local speaker diarization.
- **FastAPI**: Local server for C#/Python communication.
- **WPF**: User interface with MVVM pattern.
- **SQLite**: Local storage for speaker embeddings.
- **SkiaSharp**: Volume meter visualization.

## 📦 Quick Start - Building the Live Transcription Application

**⚠️ IMPORTANT: You must create the executable file before using live transcription features.**

### Why Do I Need to Build It?

The live transcription application (`Oreja.exe`) is not included in this download because it's too large for GitHub (149 MB). You'll need to create it yourself using the simple steps below. Don't worry - this is much easier than it sounds!

### What You'll Need (One-Time Setup)

1. **A Windows computer** (Windows 10 or 11)
2. **About 15 minutes** for the initial setup
3. **An internet connection** to download the required tools

### 🚀 **Option 1: Automatic Build (Recommended)**

**Easiest method - just double-click a file!**

1. **Download the project**: Click the green "Code" button at the top of this page, then "Download ZIP"
2. **Extract the files**: Right-click the downloaded ZIP file and choose "Extract All"
3. **Run the builder**: Double-click `build_executable.bat` in the extracted folder
4. **Follow the prompts**: The script will guide you through everything!

The batch file will:
- ✅ Check if you have the required .NET SDK (and tell you how to install it if needed)
- ✅ Automatically build the executable
- ✅ Test that it works
- ✅ Offer to create a desktop shortcut
- ✅ Offer to launch the application immediately

### 🛠️ **Option 2: Manual Build (If Automatic Fails)**

If the batch file doesn't work for any reason, you can build it manually:

#### Step 1: Install .NET 8 SDK
1. **Download**: Go to [Microsoft .NET Download Page](https://dotnet.microsoft.com/download/dotnet/8.0)
2. **Choose**: Click "Download x64" under ".NET 8.0 SDK" (not Runtime)
3. **Install**: Run the downloaded file and follow the installation wizard
4. **Verify**: Open Command Prompt and type `dotnet --version`. You should see something like `8.0.xxx`

#### Step 2: Download the Oreja Code
1. **Download**: Click the green "Code" button at the top of this page, then "Download ZIP"
2. **Extract**: Right-click the downloaded ZIP file and choose "Extract All"
3. **Location**: Extract to a folder like `C:\Users\YourName\oreja` (avoid spaces in the path)

#### Step 3: Build the Executable
1. **Open Command Prompt**: 
   - Press `Windows + R`
   - Type `cmd` and press Enter
2. **Navigate to the project**:
   ```cmd
   cd C:\Users\YourName\oreja
   ```
   (Replace `YourName` with your actual username and adjust path if needed)
3. **Build the application**:
   ```cmd
   dotnet publish -c Release -r win-x64 --self-contained true -o publish-standalone
   ```
4. **Wait**: This will take 2-5 minutes to download dependencies and build everything
5. **Success**: You should see "publish succeeded" at the end

#### Step 4: Find Your Executable
After building, you'll find `Oreja.exe` in the `publish-standalone` folder:
```
C:\Users\YourName\oreja\publish-standalone\Oreja.exe
```

You can now run this file directly - no installation needed!

### 🎯 Quick Test
Double-click `Oreja.exe` to test if it works. You should see the Oreja interface open. If you get any errors, see the troubleshooting section below.

### 📁 Optional: Create a Desktop Shortcut
1. Right-click on `Oreja.exe`
2. Choose "Create shortcut"
3. Drag the shortcut to your desktop
4. Rename it to "Oreja Live Transcription"

## 🛠️ Full Development Setup

**Note**: You only need this section if you want to modify the code or use advanced features like conversation analysis.

### Prerequisites
1. **Install Prerequisites**:
   - [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) (already done if you followed Quick Start)
   - [Python 3.10](https://www.python.org/downloads/)
   - [Git](https://git-scm.com/download/win)

### Python Backend Setup
2. **Set Up Python Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements_enhanced.txt
   ```
   For GPU support, install PyTorch with CUDA: [PyTorch Installation](https://pytorch.org/get-started/locally/).

3. **Download Hugging Face Models**:
   - Create a [Hugging Face account](https://huggingface.co/) and generate an [access token](https://huggingface.co/settings/tokens).
   - Log in:
     ```bash
     huggingface-cli login
     ```
   - Download Whisper and pyannote.audio models:
     ```python
     from transformers import pipeline
     from pyannote.audio import Pipeline
     pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
     Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token="your-token")
     ```
   - Models are cached in `~/.cache/huggingface/hub`.

4. **Build and Run**:
   - Start the Python backend:
     ```bash
     cd backend
     uvicorn server:app --host 127.0.0.1 --port 8000
     ```
   - Run the C# frontend (development mode):
     ```bash
     dotnet run --project Oreja
     ```

## Usage

### For Live Transcription (Basic Users)
1. **Run the executable** you created in the Quick Start section
2. **Select audio sources**: Choose your microphone and system audio
3. **Click Start** to begin transcription
4. **Monitor volume meters** to confirm audio capture
5. **View real-time transcription** with speaker labels (e.g., "Speaker 1: Hello…")
6. **Use controls**: Stop to pause, Save to export transcripts, or Rename Speaker to label speakers
7. **Speaker recognition improves** over time as the system learns voices

### For Advanced Features (Developers)
1. **Start the Python backend** (see Full Development Setup)
2. **Launch the Speaker Analytics GUI**:
   ```bash
   python backend/speaker_analytics_gui.py
   ```
3. **Features available**:
   - 📊 Speaker database overview and analytics
   - 🗣️ Conversation analysis with summarization
   - 🔒 Privacy Mode for sensitive conversations
   - 📋 Export summaries, action items, and meeting minutes
   - 🎬 Batch processing of recorded audio files

## 🚨 Troubleshooting

### "Oreja.exe won't start" or "Missing DLL errors"
- **Solution**: Make sure you used `--self-contained true` when building
- **Rebuild**: Delete the `publish-standalone` folder and run the build command again

### "dotnet is not recognized"
- **Solution**: .NET SDK wasn't installed properly
- **Fix**: Restart your computer after installing .NET SDK, or add it to your PATH manually

### "Access denied" or "Permission errors"
- **Solution**: Don't extract to Program Files or other protected folders
- **Fix**: Use a folder in your user directory like `C:\Users\YourName\oreja`

### Building takes a very long time
- **Normal**: First build can take 5-10 minutes as it downloads all dependencies
- **Tip**: Subsequent builds will be much faster (30 seconds to 2 minutes)

### Still having problems?
1. Make sure you have a stable internet connection
2. Try running Command Prompt as Administrator
3. Check that your antivirus isn't blocking the build process
4. Create an issue on this GitHub repository with your error message

## 🎯 What's Next?

Once you have the executable working:

1. **Try live transcription** with a test call or meeting
2. **Experiment with speaker renaming** to improve recognition
3. **Check out the conversation analysis features** if you set up the Python backend
4. **Read the advanced documentation** in the other README files for specific features

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License
MIT License. See [LICENSE](LICENSE) for details.