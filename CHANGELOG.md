# Changelog

All notable changes to Oreja are documented in this file.

## [Unreleased]

### Fixed

- **🚨 Critical: System audio now reaches backend.** The frontend was discarding system audio due to a shared concurrency buffer; refactored audio capture into per-source independent buffers (mic and system audio each with their own lock and state). This was a silent data loss issue—system audio was never making it to transcription even though the UI appeared to be capturing it.
- **🚨 Critical: Speaker recognition is now working.** Server-side speaker identification was disabled; rebuilt the entire persistent speaker system on top of the v2 JSON database using pyannote's unified embedding model. Speakers are now correctly identified and learned over time.
- Blocking I/O in enhanced transcription routes (`summarize`, `sentiment`, `annotations`) wrapped in `asyncio.to_thread()` to prevent event loop hangs.
- Dangling imports in training and utility modules (removed broken imports of deleted `speaker_embeddings` module).
- Stale test modules deleted to prevent misleading test coverage reports.
- Speaker database locking issues fixed; all database mutations now properly guarded by `threading.RLock`.

### Changed

- **ASR Engine**: Replaced Hugging Face Transformers Whisper pipeline with **faster-whisper** for lower latency and memory usage; now uses word-level timestamps for more precise speaker-to-word attribution.
- **Speaker Recognition**: Rebuilt on single unified pyannote embedding model instead of legacy SpeechBrain ECAPA. Speakers stored in local JSON database (v2 format) with confidence-weighted matching (threshold `OREJA_SPEAKER_THRESHOLD`, default 0.72).
- **Frontend Audio Pipeline**: Refactored to use independent `AudioSourceState` objects per source (microphone, system audio) with separate buffering, locking, and processing state. Prevents cross-source interference and silent audio loss.
- **Backend Status UI**: Added connection status indicator (green/orange/red dot) in the main window; visible feedback when backend is offline or connecting.
- **Speaker Scoring**: Changed from simple embedding distance to confidence-weighted mean vector comparison; auto-creates placeholder speakers ("Speaker N") for unknown voices.

### Added

- **Backend Auto-Start**: Frontend now searches for and can spawn the Python backend if a `venv/Scripts/python.exe` is found.
- **Backend Health Polling**: Periodic `GET /health` checks detect backend availability; failures trigger immediate UI update and out-of-band recheck.
- **Per-Word Speaker Attribution**: Word-level timestamps from faster-whisper aligned with speaker turns from pyannote diarization for granular segment accuracy.
- **Export Formats**: Support for SRT, VTT, and Markdown in addition to plain text.
- **Transcript Search**: Keyword search and alerts on transcription results (UI feature).
- **Settings Persistence**: Frontend settings (backend URL, speaker database, audio preferences) persisted to local AppData.
- **Configuration via Environment Variables**: Control model selection, device, compute type, language, speaker threshold, and optional Ollama connection via env vars (`OREJA_WHISPER_MODEL`, `OREJA_DEVICE`, `OREJA_COMPUTE_TYPE`, `OREJA_LANGUAGE`, `OREJA_SPEAKER_THRESHOLD`, `OREJA_OLLAMA_URL`, `OREJA_OLLAMA_MODEL`).
- **Ollama Integration**: Optional meeting summaries and conversation analysis via local Ollama LLM (with extractive fallback when unavailable).
- **VAD Configuration**: Configurable voice activity detection via `vad_parameters` (min silence duration: 500ms by default).

### Removed

- **SpeechBrain ECAPA Legacy**: Deleted `speaker_embeddings.py` (534 lines); replaced with unified pyannote embedding extraction.
- **SQLite Speaker Storage**: Replaced with local JSON database (v2 format); no encryption overhead, simpler versioning.
- **Transformers Whisper Pipeline**: Removed dependency on Hugging Face Transformers Whisper pipeline; faster-whisper is now sole ASR engine.
- **Whole-Clip VAD Gate**: Removed aggressive silence detection from transcription pipeline (six early-exit checks on RMS, ZCR, spectral centroid, etc.); now relies on faster-whisper's integrated VAD.
- **Stale Tests**: Deleted test modules `test_api_endpoints.py` and `test_enhanced_api_endpoints.py` (coverage now via integration tests).
- **Duplicate Speaker Retrieval**: Removed duplicate `get_all_speakers` definition in speaker database.

---

## Notes for Maintainers

- The system audio bug was subtle: shared `_audioBuffer` and `_systemAudioBuffer` with a single `_isProcessingTranscription` flag meant system audio could be cleared while mic audio was being processed, causing the system buffer to be discarded. The fix separates state per source entirely.
- Speaker recognition was non-functional in the v1 design; the v2 database is backward-incompatible but much simpler and more reliable.
- If you need to debug speaker matching, check `OREJA_SPEAKER_THRESHOLD` and the confidence scores in `GET /speakers`.
- Ollama is optional; if `OREJA_OLLAMA_URL` is not set, summaries fall back to extractive summaries (no LLM).
