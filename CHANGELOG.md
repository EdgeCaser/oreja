# Changelog

All notable changes to Oreja are documented in this file.

## [Unreleased]

### Added

- **Real transcript selection model**: selection is always active - click a card to select it, Ctrl+click toggles, Shift+click extends a range, checkboxes still work. "Select All" respects the search filter (search + Select All = select every match). New "👤 Select Speaker" toolbar menu (per-speaker counts) and card context menu for select-by-speaker. The old Multi-Select ON/OFF toggle is gone.
- **Segment deletion**: single (card right-click) and bulk ("🗑 Delete Selected") with confirmation. Removes transcript segments only - speakers and voiceprints are untouched. Segment IDs moved to a monotonic counter so deletion can never cause ID collisions.
- **Speaker color stripes**: every card carries a colored left stripe keyed to its speaker (same palette as the dropdown), so misattributed segments stand out at a glance.
- **Segment audio playback**: live recordings are teed to compact per-source WAVs (16 kHz mono - the exact ASR input, ~115 MB/hour) in `Documents\Oreja Recordings`; the ▶ button on any segment replays exactly that slice. File transcriptions replay from the original media file. Disabled while recording, never saved under Legal-Safe Mode, and a "💾 Save session audio" checkbox opts out entirely.
- **Session speaker roster**: the "👥 Session Speakers" dialog declares who is present before recording. Known speakers keep learning voiceprints from corrections; guest speakers are session-only (never persisted, no embedding feedback). Roster names lead every speaker dropdown and map onto the 1-4 quick-assign buttons, and the roster size is sent as a `max_speakers` diarization ceiling so pyannote cannot invent phantom extra speakers.

- **Language selection**: New "Transcription Language" dropdown in the main window, defaulting to English. Pinning the language removes per-chunk auto-detect misfires (short/noisy chunks occasionally decoded in the wrong language); "Auto-detect (multilingual)" remains available for mixed-language sessions and detects per chunk. Sent per request (`language` query parameter on `/transcribe`), persisted across launches.
- **Custom vocabulary + speaker-name prompt biasing**: `backend/vocabulary.txt` (one term per line) plus the display names of known speakers are fed to the ASR decoder as a glossary `initial_prompt`, strongly biasing it toward correct spellings of names and jargon. Auto-generated "Speaker N" placeholders are excluded; the file is re-read on change without a restart. Configurable via `OREJA_VOCAB_FILE`, `OREJA_PROMPT_SPEAKER_NAMES`, `OREJA_INITIAL_PROMPT`.
- **Accuracy mode for file transcription**: The Transcribe File path now requests `accuracy=true`, which decodes with beam size `OREJA_FILE_BEAM_SIZE` (default 10, vs 5 live) and, when `OREJA_FILE_MODEL` is set (e.g. `large-v3`), a separate stronger model lazy-loaded on first use. Live chunk latency is unaffected.
- **WER evaluation harness**: `backend/eval_wer.py` measures word error rate against hand-corrected reference transcripts through the real `/transcribe` pipeline, for before/after A-B comparison of accuracy changes (requires `jiwer`, added to test requirements).

### Fixed

- **Clicking in the console window no longer freezes the app**: A Windows console suspends every write while text is selected in it (QuickEdit mark mode), and the app logs from the UI thread - so one stray click into the debug console froze the whole UI (timers, chunk dispatch, health polls) until the selection was cleared, indistinguishable from a crash. `Console.Out/Error` are now routed through a non-blocking queue drained by a background thread (lines drop rather than block when the console is stuck), and the chunk-dispatch/stop-flush pipeline runs on worker threads via `Task.Run`, so the session-audio disk write and WAV assembly can never stall the window either.
- **`OREJA_TORCH_DEVICE=cpu` moves the pyannote stack off the GPU**: New env var (default `auto` = previous behavior) that runs diarization and speaker embeddings on CPU while whisper keeps the GPU (whisper's own device remains `OREJA_DEVICE`). On a 10 GB card shared with a heavyweight desktop (dwm alone can hold 2 GB, plus Zoom/browsers), the combined whisper+pyannote stack oversubscribes VRAM and Windows starts paging CUDA memory through system RAM - observed turning a 0.3s diarization into 77s and wedging the pipeline. The pyannote models are small; CPU costs a few seconds per live chunk and frees ~2 GB of VRAM.
- **Chunk size is bounded even when draining a backlog**: One slow backend response used to snowball - while a request waited, the buffer grew, and the next cut sent everything up to the last pause (48s+ observed), which took even longer to decode, timed out, and grew the next chunk further, grinding the GPU into a pile of orphaned decodes. Every dispatched chunk (including the stop-flush, which now drains as multiple requests) is capped at `MAX_CHUNK_SECONDS` (15 s): cut at the most recent pause within the window, hard-cut at 15 s if there is none. The backend logs decode/diarization wall time per chunk, warns when a live chunk arrives oversized or ASR runs slower than real time (the signature of VRAM oversubscription paging into system RAM), and releases pyannote's cached CUDA memory after each diarization so the backend doesn't hoard headroom on a GPU shared with Zoom/browsers.
- **Audio no longer lost while the backend starts up**: The auto-started backend loads its models for ~20-60 s before it can accept connections, and every chunk sent in that window used to die with "connection refused" - audio gone, console full of stack traces. Dispatch now holds buffered audio while the backend is unreachable and releases it the moment a health check succeeds (the stop-flush waits too, so stopping early doesn't discard the tail). The per-source buffer cap was raised from 30 s to 120 s to ride out the load window, and the status indicator shows "Connecting" instead of "Backend offline" while the spawned backend is still loading.
- **Auto-started backend output is captured**: The spawned uvicorn's stdout/stderr previously went nowhere, making a backend that crashed at startup indistinguishable from one that was still loading. Its output now lands in `%APPDATA%\Oreja\backend.log` (fresh file per app session, UTF-8), the console notes the path at spawn, and if the process exits its exit code is recorded in both the console and the log.
- **System-audio aliasing**: The WASAPI loopback downsample to 16 kHz used bare linear interpolation, folding all source content above 8 kHz back into the speech band as noise. A 95-tap windowed-sinc low-pass (7 kHz cutoff, ≥54 dB stopband attenuation, verified numerically) now band-limits the mono signal before decimation, with filter state carried seamlessly across capture callbacks.
- **Mid-word cuts on cap flush**: When a speaker talks straight through `MAX_CHUNK_SECONDS` (15 s), the chunk boundary could land mid-word. The flush now cuts at the most recent in-buffer pause and carries the tail into the next chunk; timeline accounting is preserved. Only when the buffer contains no pause at all does it fall back to sending everything.

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
