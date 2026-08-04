# Oreja Architecture

Oreja is a Windows desktop application for real-time conference call transcription, processing audio locally to ensure privacy. This document outlines its architecture.

## Overview
Oreja captures microphone and system audio independently, transcribes and diarizes it locally using faster-whisper and pyannote.audio models, and displays results in a WPF UI. Speaker embeddings are extracted using pyannote’s unified embedding model and stored in a local JSON database (v2 format) for recurring caller recognition. Audio is processed in memory with no cloud interaction.

## Tech Stack
- **C#/.NET 8**: Frontend, audio capture, and UI.
- **Python 3.10**: Backend for transcription and diarization.
- **NAudio**: Audio capture (microphone and system).
- **faster-whisper**: Local speech-to-text with word-level timestamps.
- **pyannote.audio 3.1**: Speaker diarization and voice embedding extraction.
- **FastAPI**: Local C#/Python communication.
- **WPF**: UI (code-behind, single App.xaml.cs file).
- **JSON**: Local speaker database (v2 format) with embeddings and metadata.

## Modules

1. **Audio Capture Module (C#)**:
   - Uses NAudio (`WasapiCapture` for mic, `WasapiLoopbackCapture` for system).
   - Maintains independent buffers per audio source (microphone and system audio).
   - Sends 5-second WAV chunks to the Python backend via `POST /transcribe` on `127.0.0.1:8000`.
   - Supports multiple export formats: TXT, SRT, VTT, Markdown.

2. **Transcription and Diarization Module (Python)**:
   - **faster-whisper** for ASR with word-level timestamps (`model.transcribe(word_timestamps=True)`).
   - **pyannote.audio 3.1** for speaker diarization; VAD gate configured via `vad_parameters={\"min_silence_duration_ms\":500}`.
   - Processes audio buffers in memory; aligns word-timestamps with speaker turns for per-word attribution.
   - Returns JSON with segments (each containing start, end, text, speaker).

3. **Speaker Recognition Module (Python)**:
   - Extracts speaker embeddings using pyannote’s unified `PretrainedSpeakerEmbedding` model.
   - Stores embeddings and speaker metadata in a JSON database (v2 format) at `~/.local/share/oreja/speaker_database.json` (Linux/Mac) or AppData (Windows).
   - `EnhancedSpeakerDatabase` class: scores embeddings against confidence-weighted mean vectors; threshold `OREJA_SPEAKER_THRESHOLD` (default 0.72).
   - Auto-creates placeholder speakers ("Speaker N") for unknown voices; improves recognition over time as users confirm identities.

4. **UI Module (C#)**:
   - WPF code-behind (`App.xaml.cs`, ~3300 lines) with real-time display.
   - Shows volume meters, live transcription with speaker labels, and a backend connection status indicator (green/orange/red).
   - Controls: Start/Stop recording, Save transcript, Rename speaker, Backend status visible.
   - Settings persistence via local AppData (backend URL, speaker names, preferences).

5. **Backend Status and Auto-Start (C#)**:
   - Polls `GET /health` periodically to detect backend availability.
   - On startup, auto-searches for the Python backend; can spawn it if `venv/Scripts/python.exe` is found.
   - Updates connection UI indicator in real-time; transcription failures trigger immediate out-of-band health checks.

## Data Flow
1. **Audio Capture** (C#): Microphone and system audio captured independently via NAudio.
2. **Buffering**: Each audio source maintains its own 5-second rolling buffer (max 30s cap per source).
3. **Chunking**: `POST /transcribe` sends WAV bytes to Python backend.
4. **ASR Pipeline** (Python):
   - Load audio → Resample to 16 kHz mono.
   - `asyncio.gather(run_transcription, run_diarization)` → Parallel ASR (faster-whisper) and speaker turn detection (pyannote).
   - **Word-level alignment**: Merge ASR words with diarization turns; assign speaker ID to each word.
   - Extract speaker embedding from the audio chunk.
5. **Speaker Identification**: Score embedding against speaker database; if below threshold, auto-create or update existing speaker.
6. **Response**: JSON with segments (start, end, text, speaker) + full_text.
7. **UI Display** (C#): Update transcript display, speaker labels, and audio meters in real-time.
8. **Persistence**: User actions (rename speaker, save transcript) → Update speaker database or write to file.

## Privacy
- Audio processed in memory, discarded immediately after transcription (no audio files saved).
- All models (faster-whisper, pyannote.audio) run locally; cached in `~/.cache/huggingface/hub`.
- Speaker embeddings stored in local JSON file only; never sent to external services.
- Speaker database file is not encrypted; keep its parent directory permissions restrictive if privacy is critical.
- Optional Ollama summaries run locally if configured; no external LLM calls.
- No cloud interaction after model download (except optional Ollama setup).

## Extensibility
- Swap faster-whisper models via `OREJA_WHISPER_MODEL` (tiny, base, small, medium, large).
- Adjust speaker matching sensitivity via `OREJA_SPEAKER_THRESHOLD` (0.0–1.0).
- Connect an external LLM via `OREJA_OLLAMA_URL` and `OREJA_OLLAMA_MODEL` for meeting summaries.
- Extend export formats by modifying the transcript save handler in `App.xaml.cs`.
- Customize audio input sources (add USB devices) via NAudio.

## FastAPI Endpoints

The Python backend exposes the following HTTP endpoints on `127.0.0.1:8000`:

- **`GET /health`**: Health check; returns `{"status": "ok"}`.
- **`POST /transcribe`**: Transcribe audio chunk. Input: WAV bytes (5 seconds, 16 kHz). Returns JSON with `segments` (list of `{start, end, text, speaker}`) and `full_text`.
- **`GET /speakers`**: List all known speakers. Returns JSON with speaker IDs, names, confidence scores.
- **`POST /speakers/name_mapping`**: Rename a speaker. Query params: `old_speaker_id`, `new_speaker_name`.

See [DEVELOPMENT.md](DEVELOPMENT.md) for implementation details.