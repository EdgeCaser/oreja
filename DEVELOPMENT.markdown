# Development Guide for Oreja

This guide covers setting up the development environment for Oreja and using Cursor effectively.

## Prerequisites
- **OS**: Windows 10/11 (for NAudio/WASAPI).
- **IDE**: Visual Studio 2022, Rider, or VS Code with Cursor.
- **.NET 8 SDK**: [Download](https://dotnet.microsoft.com/download/dotnet/8.0).
- **Python 3.10**: [Download](https://www.python.org/downloads/).
- **Git**: [Download](https://git-scm.com/download/win).
- **Hugging Face Account**: For model access tokens.

## Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-org/oreja.git
   cd oreja
   ```
2. **Set Up Python Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install torch torchaudio transformers pyannote.audio fastapi uvicorn sentencepiece
   ```
   For GPU: [PyTorch CUDA](https://pytorch.org/get-started/locally/).
3. **Download Hugging Face Models**:
   - Log in to Hugging Face:
     ```bash
     huggingface-cli login
     ```
   - Download models:
     ```python
     from transformers import pipeline
     from pyannote.audio import Pipeline
     pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
     Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token="your-token")
     ```
   - Verify models in `~/.cache/huggingface/hub`.
4. **Restore .NET Packages**:
   ```bash
   dotnet restore
   ```
5. **Build and Run**:
   - Start Python backend:
     ```bash
     cd backend
     uvicorn server:app --host 127.0.0.1 --port 8000
     ```
   - Run C# frontend:
     ```bash
     dotnet run --project Oreja
     ```

## Dependencies
- **C#**: NAudio, Microsoft.Data.Sqlite, SkiaSharp, System.Text.Json.
  ```bash
  dotnet add package NAudio
  dotnet add package Microsoft.Data.Sqlite
  dotnet add package SkiaSharp
  dotnet add package System.Text.Json
  ```
- **Python**: torch, torchaudio, transformers, pyannote.audio, fastapi, uvicorn, sentencepiece.
  ```bash
  pip install torch torchaudio transformers pyannote.audio fastapi uvicorn sentencepiece
  ```

## Project Structure
- `/Oreja`: C# frontend.
  - `/Models`: Data models.
  - `/ViewModels`: MVVM view models.
  - `/Views`: XAML and code-behind.
  - `/Services`: Audio/database services.
  - `/Utilities`: Helpers.
- `/backend`: Python FastAPI server.
- `/scripts`: Model download scripts.

## Cursor Tips
- **Enable `.cursorrules`**: Load from repo root for completions.
- **C# Completions**:
  - Prioritize NAudio, WPF MVVM, HttpClient for FastAPI.
  - Suggest async methods and disposables.
- **Python Completions**:
  - Prioritize transformers (Whisper), pyannote.audio, FastAPI.
  - Suggest async routes and in-memory processing.
- **Error Handling**:
  - C#: Try-catch for NAudio, HTTP, SQLite.
  - Python: Try-except for model inference.
- **Performance**:
  - Suggest buffer pools for audio.
  - Optimize Whisper chunking (e.g., `chunk_length_s=30`).
- **Testing**:
  - C#: xUnit with Moq.
  - Python: pytest for backend.

## Debugging
- Debug C# with Visual Studio (audio/UI issues).
- Debug Python with VS Code (model errors).
- Verify audio capture with volume meters.
- Check FastAPI logs for transcription errors.

## Testing
- Run tests:
  ```bash
  dotnet test
  pytest backend/tests
  ```

See [CODE_STYLE.md](CODE_STYLE.md) and [ARCHITECTURE.md](ARCHITECTURE.md) for details.