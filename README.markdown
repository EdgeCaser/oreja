# Oreja

Oreja is a Windows desktop application for real-time transcription of conference calls (Google Meet, Zoom, Slack, etc.) without storing or sending audio to the cloud. It captures microphone and system audio, uses local Hugging Face models for transcription and diarization, and provides a WPF interface with volume meters, transcription controls, and speaker renaming. Speaker embeddings improve recognition of recurring callers, stored locally in SQLite.

## Features
- Real-time transcription with speaker diarization using Whisper and pyannote.audio.
- Captures microphone and system audio simultaneously via NAudio.
- Volume meters for audio sources.
- Start/Stop transcription, save transcripts, and rename speakers.
- Privacy-focused: processes audio in memory, runs models locally, no cloud interaction.
- Improves speaker recognition over time using locally stored embeddings.

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

## Setup
1. **Install Prerequisites**:
   - [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)
   - [Python 3.10](https://www.python.org/downloads/)
   - [Git](https://git-scm.com/download/win)
2. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-org/oreja.git
   cd oreja
   ```
3. **Set Up Python Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install torch torchaudio transformers pyannote.audio fastapi uvicorn sentencepiece
   ```
   For GPU support, install PyTorch with CUDA: [PyTorch Installation](https://pytorch.org/get-started/locally/).
4. **Download Hugging Face Models**:
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
5. **Restore .NET Packages**:
   ```bash
   dotnet restore
   ```
6. **Build and Run**:
   - Start the Python backend:
     ```bash
     cd backend
     uvicorn server:app --host 127.0.0.1 --port 8000
     ```
   - Run the C# frontend:
     ```bash
     dotnet run --project Oreja
     ```

## Usage
1. Launch Oreja and select microphone and system audio sources.
2. Ensure the Python backend is running (`uvicorn server:app`).
3. Click **Start** to begin transcription.
4. Monitor volume meters to confirm audio capture.
5. View real-time transcription with speaker labels (e.g., “Speaker 1: Hello…”).
6. Use **Stop** to pause, **Save** to export transcripts, or **Rename Speaker** to label speakers.
7. Speaker embeddings are stored locally in SQLite to improve recognition.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License
MIT License. See [LICENSE](LICENSE) for details.