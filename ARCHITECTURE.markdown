# Oreja Architecture

Oreja is a Windows desktop application for real-time conference call transcription, processing audio locally to ensure privacy. This document outlines its architecture.

## Overview
Oreja captures microphone and system audio, transcribes and diarizes it locally using Hugging Face’s Whisper and pyannote.audio models, and displays results in a WPF UI. Speaker embeddings are stored in SQLite for recurring caller recognition. Audio is processed in memory with no cloud interaction.

## Tech Stack
- **C#/.NET 8**: Frontend, audio capture, and UI.
- **Python 3.10**: Backend for transcription and diarization.
- **NAudio**: Audio capture (microphone and system).
- **Hugging Face Transformers (Whisper)**: Local speech-to-text.
- **pyannote.audio**: Local speaker diarization.
- **FastAPI**: Local C#/Python communication.
- **WPF**: UI with MVVM pattern.
- **SQLite**: Local embedding storage.
- **SkiaSharp**: Volume meter visualization.

## Modules
1. **Audio Capture Module (C#)**:
   - Uses NAudio (`WasapiCapture` for mic, `WasapiLoopbackCapture` for system).
   - Mixes audio streams in memory using `MixingSampleProvider`.
   - Sends buffers to Python backend via FastAPI.

2. **Transcription and Diarization Module (Python)**:
   - Runs Whisper (`openai/whisper-large-v3`) for transcription.
   - Runs pyannote.audio (`pyannote/speaker-diarization-3.0`) for diarization.
   - Processes audio buffers in memory, returns transcription with speaker labels.

3. **Speaker Recognition Module (C#/Python)**:
   - Extracts embeddings using pyannote.audio’s embedding model.
   - Stores embeddings in SQLite with user-assigned names.
   - Matches new embeddings for recurring speakers.

4. **UI Module (C#)**:
   - WPF with MVVM, displaying volume meters (SkiaSharp), transcription, and controls (Start/Stop, Save, Rename).
   - Updates in real-time via data binding.

5. **Storage Module (C#)**:
   - SQLite for speaker embeddings and metadata.
   - In-memory buffer for transcription, saved as text on request.

## Data Flow
1. Audio Capture (C#) → Mix mic/system audio → Buffer in memory.
2. Buffer → FastAPI (Python) → Whisper (transcription) + pyannote.audio (diarization).
3. Transcription/Speaker Labels → C# UI for display.
4. Embeddings → SQLite for storage; User Actions → Update names or save transcripts.

## Privacy
- Audio processed in memory, discarded after transcription.
- Models run locally, cached in `~/.cache/huggingface/hub`.
- Embeddings encrypted in SQLite.
- No cloud interaction after model download.

## Extensibility
- Support additional Hugging Face models (e.g., Whisper variants).
- Enhance visualization with EQ bars via SkiaSharp.
- Improve embedding matching with custom algorithms.

See [DEVELOPMENT.md](DEVELOPMENT.md) for implementation details.