"""
FastAPI server for Oreja audio transcription and diarization.
Processes audio in memory using local models with no cloud interaction.

Pipeline shape:
    bytes -> waveform (16 kHz mono)
          -> ASR:         faster-whisper (ctranslate2) with built-in Silero VAD and
                          word-level timestamps. Handles arbitrary-length audio
                          internally - there is NO manual chunking / overlap stitching.
          -> diarization: pyannote speaker-diarization on the full waveform.
          -> attribution: every ASR *word* is mapped to the diarization turn covering
                          its midpoint; ASR segments are then split wherever the
                          word-level speaker changes.
          -> hook:        identify_speakers_hook() for voiceprint-based naming.

ASR and diarization both run off the event loop via asyncio.to_thread and are
serialized by per-model locks so concurrent requests cannot corrupt model state.

PRIVACY GUARANTEE: NO AUDIO DATA EVER LEAVES THIS MACHINE
"""

import asyncio
import io
import logging
import os
import re
import sys
import threading
import time
from typing import List, Dict, Any, Optional
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torchaudio
from audio_io import load_audio
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# NOTE: faster-whisper (ctranslate2) and pyannote.audio are imported lazily inside
# initialize_models() so that this module can be imported for tooling/tests on
# machines where the heavy ASR/diarization wheels are not installed.

# Persistent speaker identity lives in the v2 database. There is exactly ONE
# embedding system in this server: the pyannote PretrainedSpeakerEmbedding model
# loaded below. The old SpeechBrain OfflineSpeakerEmbeddingManager is gone.
from speaker_database_v2 import EnhancedSpeakerDatabase, cosine_similarity

# Import enhanced transcription features
try:
    from enhanced_server_integration import EnhancedTranscriptionService
    ENHANCED_FEATURES_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Enhanced transcription features available")
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Enhanced transcription features not available - install with: pip install -r requirements_enhanced.txt")

# Import enhanced segment splitting
from enhanced_segment_splitting import AudioSegmentSplitter, SegmentSplitValidator

# Import enhanced speaker server integration
from enhanced_speaker_server_integration import EnhancedSpeakerServerIntegration

# Load environment variables from .env file
try:
    load_dotenv()
except Exception as e:
    # Handle corrupted or missing .env file gracefully
    print(f"Warning: Could not load .env file: {e}")
    print("Continuing without .env file...")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration constants
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 7200  # seconds (2 hours) - faster-whisper streams long audio internally
MIN_AUDIO_LENGTH = 0.1  # seconds

# --- ASR (faster-whisper / ctranslate2) -------------------------------------
# Model name is a faster-whisper identifier ("large-v3-turbo", "medium", ...) or a
# path to a converted ctranslate2 model directory.
WHISPER_MODEL = os.getenv("OREJA_WHISPER_MODEL", "large-v3-turbo")
WHISPER_FALLBACK_MODEL = os.getenv("OREJA_WHISPER_FALLBACK_MODEL", "base")
# "auto" -> cuda when available, otherwise cpu.
WHISPER_DEVICE_SETTING = os.getenv("OREJA_DEVICE", "auto")
# Empty/unset -> float16 on cuda, int8 on cpu.
WHISPER_COMPUTE_TYPE_SETTING = os.getenv("OREJA_COMPUTE_TYPE", "")
# Empty/unset -> auto-detect the language (applies to ALL audio lengths).
# Per-request `language` query parameters on /transcribe override this.
WHISPER_LANGUAGE = os.getenv("OREJA_LANGUAGE") or None
WHISPER_BEAM_SIZE = int(os.getenv("OREJA_BEAM_SIZE", "5"))

# Sentinel distinguishing "caller said nothing about language" (use the server-wide
# WHISPER_LANGUAGE default) from an explicit None (force auto-detection).
_LANGUAGE_DEFAULT = object()

# --- Decode-biasing initial prompt ------------------------------------------
# Whisper's initial_prompt is the strongest lever for proper nouns and jargon:
# terms present in the prompt are far more likely to be spelled correctly. The
# prompt is assembled from (a) a user-editable vocabulary file, one term per
# line ('#' comments allowed), and (b) the display names of known speakers from
# the voiceprint database. OREJA_INITIAL_PROMPT overrides both with a fully
# custom static prompt. Set OREJA_VOCAB_FILE="" to disable the file part and
# OREJA_PROMPT_SPEAKER_NAMES=0 to disable the names part.
INITIAL_PROMPT_OVERRIDE = os.getenv("OREJA_INITIAL_PROMPT") or None
VOCAB_FILE = os.getenv("OREJA_VOCAB_FILE", "vocabulary.txt")
PROMPT_SPEAKER_NAMES = os.getenv("OREJA_PROMPT_SPEAKER_NAMES", "1").lower() in ("1", "true", "yes")
# Whisper truncates the prompt to 224 tokens itself; cap characters conservatively
# below that so truncation never lands mid-name.
_PROMPT_MAX_CHARS = 700
_PROMPT_MAX_NAMES = 20
_PROMPT_NAMES_TTL_SECONDS = 60.0

# --- Accuracy mode (offline file transcription) ------------------------------
# The Transcribe File path has no latency pressure, so it can spend more compute
# per second of audio than the live 3-15s chunk path. accuracy=true on /transcribe
# raises the beam size, and - only when OREJA_FILE_MODEL is set - decodes with a
# separate, stronger model (e.g. "large-v3" while live chunks stay on
# "large-v3-turbo"). The file model is lazy-loaded on the first accuracy request
# and kept resident; loading failure falls back to the main model with a warning
# rather than failing the request.
WHISPER_FILE_MODEL = os.getenv("OREJA_FILE_MODEL", "")
WHISPER_FILE_BEAM_SIZE = int(os.getenv("OREJA_FILE_BEAM_SIZE", "10"))
VAD_MIN_SILENCE_MS = int(os.getenv("OREJA_VAD_MIN_SILENCE_MS", "500"))

# Hallucination gating thresholds, applied uniformly to every segment.
NO_SPEECH_PROB_THRESHOLD = float(os.getenv("OREJA_NO_SPEECH_PROB", "0.6"))
AVG_LOGPROB_THRESHOLD = float(os.getenv("OREJA_AVG_LOGPROB", "-1.0"))
COMPRESSION_RATIO_THRESHOLD = float(os.getenv("OREJA_COMPRESSION_RATIO", "2.4"))
# avg_logprob at or above this means the decode was confident; short outputs are
# then kept rather than treated as noise artifacts.
CONFIDENT_LOGPROB = float(os.getenv("OREJA_CONFIDENT_LOGPROB", "-0.5"))

# Cheap fast-path: clips quieter than this are certainly silence. Deliberately far
# below the old 0.015 whole-clip VAD gate, which discarded legitimate quiet speech.
SILENCE_RMS_THRESHOLD = float(os.getenv("OREJA_SILENCE_RMS", "0.001"))

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
EMBEDDING_MODEL = "pyannote/embedding"

# --- Persistent speaker identification ---------------------------------------
# Cosine similarity a voiceprint must reach to be called a known speaker.
SPEAKER_MATCH_THRESHOLD = float(os.getenv("OREJA_SPEAKER_THRESHOLD", "0.72"))
# A diarization speaker with less than this much audio in a request is not
# embedded at all: short crops give unstable voiceprints that poison the bank.
MIN_SPEAKER_AUDIO_SECONDS = float(os.getenv("OREJA_MIN_SPEAKER_AUDIO", "1.5"))
# Lower floor for a READ-ONLY identification attempt (learn=False, auto_create=False).
# A short "yeah" is too weak to learn from or to mint a speaker record from, but it
# is usually still good enough to *recognise* an already-known voice - and doing so
# is what stops a speaker flipping between "Speaker SPEAKER_00" and
# "Speaker SPEAKER_01" from one 5-second chunk to the next (pyannote assigns those
# indices arbitrarily per request; there is no session state across chunks).
MIN_IDENTIFY_AUDIO_SECONDS = float(os.getenv("OREJA_MIN_IDENTIFY_AUDIO", "0.5"))
# Upper bound on the audio concatenated per speaker before embedding. More than
# ~20s buys no accuracy and costs latency on every 5-second chunk.
MAX_SPEAKER_AUDIO_SECONDS = float(os.getenv("OREJA_MAX_SPEAKER_AUDIO", "20.0"))
# How many times an unmatched voiceprint must recur (across requests/chunks)
# before a new "Speaker N" record is minted for it. Auto-creating on the first
# miss let a 30-minute call at 5s/chunk mint hundreds of throwaway speakers, each
# one rewriting the whole database to disk and showing up in the WPF client's
# persisted speaker list. Set to 1 to restore create-on-first-miss.
AUTO_CREATE_AFTER_MISSES = max(1, int(os.getenv("OREJA_AUTO_CREATE_AFTER_MISSES", "2")))
# Cap on how many distinct unmatched voiceprints are remembered while waiting for
# a recurrence, so the pending buffer cannot grow without bound.
_MAX_PENDING_UNMATCHED = 32
# Where the v2 speaker database lives.
SPEAKER_DATA_DIR = os.getenv("OREJA_SPEAKER_DATA_DIR", "speaker_data_v2")

# Whether the per-request sentiment / conversation-analysis enhancement runs inline.
# Off by default: it is expensive and belongs on an explicit analysis pass.
ENHANCEMENT_INLINE = os.getenv("OREJA_INLINE_ENHANCEMENT", "0").lower() in ("1", "true", "yes")

# Models are not thread-safe. Serialize access so that concurrent /transcribe
# requests queue on the model instead of corrupting its internal state, while the
# asyncio event loop stays free (calls are dispatched via asyncio.to_thread).
_whisper_lock = threading.Lock()
_diarization_lock = threading.Lock()
_embedding_lock = threading.Lock()
# The speaker database is mutated from worker threads (identify hook) and from
# request handlers. EnhancedSpeakerDatabase is internally locked as of the
# thread-safety fix, so this is only needed to make multi-step read-modify-write
# sequences in handlers atomic - it is not what keeps the database consistent.
_speaker_db_lock = threading.RLock()

# Unmatched voiceprints waiting to recur before they earn a speaker record.
# list of [embedding vector, miss_count]; guarded by _pending_unmatched_lock.
_pending_unmatched: List[List[Any]] = []
_pending_unmatched_lock = threading.Lock()

app = FastAPI(
    title="Oreja Enhanced Audio Processing API",
    description="Local audio transcription with speaker diarization, sentiment analysis, and audio features",
    version="2.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Consolidated summarization / annotation endpoints: /transcribe_enhanced,
# /analyze_sentiment, /enhanced_features_status, /transcribe_with_summary,
# /summarize_transcription, /summary_options. These are batch/file-style
# endpoints (not the live 5-second-chunk path) so they run analysis by
# default. Each route lazily imports from this module at request time, by
# which point initialize_models() has already run.
if ENHANCED_FEATURES_AVAILABLE:
    try:
        from enhanced_server_integration import add_enhanced_endpoints
        add_enhanced_endpoints(app)
        logger.info("Enhanced summarization endpoints registered")
    except Exception as e:
        logger.warning(f"Failed to register enhanced summarization endpoints: {e}")

# Global variables
device = None
whisper_model = None
diarization_pipeline = None
embedding_model = None
enhanced_speaker_database = None
enhanced_service = None
enhanced_speaker_integration = None


def resolve_asr_device() -> str:
    """Resolve the faster-whisper device string ("cuda" or "cpu") from OREJA_DEVICE."""
    setting = (WHISPER_DEVICE_SETTING or "auto").strip().lower()
    if setting in ("", "auto"):
        try:
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover - torch without CUDA runtime
            return "cpu"
    return setting


def resolve_compute_type(asr_device: str) -> str:
    """Resolve the ctranslate2 compute type: float16 on cuda, int8 on cpu by default."""
    setting = (WHISPER_COMPUTE_TYPE_SETTING or "").strip().lower()
    if setting:
        return setting
    return "float16" if asr_device == "cuda" else "int8"


def _load_faster_whisper(model_name: str, asr_device: str, compute_type: str):
    """Instantiate a faster-whisper model. Import is lazy so the module stays importable."""
    from faster_whisper import WhisperModel  # noqa: PLC0415 - intentionally lazy

    return WhisperModel(
        model_name,
        device=asr_device,
        compute_type=compute_type,
        download_root=os.getenv("OREJA_MODEL_CACHE") or None,
    )


# Separate ASR model for accuracy-mode (file) requests. None until the first
# accuracy request lazy-loads it; False after a failed load so we do not retry
# (and re-log) on every request.
_file_whisper_model = None


def _get_file_whisper_model():
    """
    The model accuracy-mode requests decode with: the OREJA_FILE_MODEL instance
    when configured and loadable, else the main model. Called with _whisper_lock
    held, which is what makes the lazy load race-free.
    """
    global _file_whisper_model
    if not WHISPER_FILE_MODEL or WHISPER_FILE_MODEL == WHISPER_MODEL:
        return whisper_model
    if _file_whisper_model is None:
        asr_device = resolve_asr_device()
        compute_type = resolve_compute_type(asr_device)
        try:
            logger.info(
                f"Loading accuracy-mode model: {WHISPER_FILE_MODEL} "
                f"(device={asr_device}, compute_type={compute_type})"
            )
            _file_whisper_model = _load_faster_whisper(
                WHISPER_FILE_MODEL, asr_device, compute_type
            )
            logger.info("✓ accuracy-mode model loaded successfully")
        except Exception as e:
            logger.warning(
                f"Failed to load accuracy-mode model '{WHISPER_FILE_MODEL}': {e}; "
                f"falling back to the main model for file transcription"
            )
            _file_whisper_model = False
    return _file_whisper_model or whisper_model


def initialize_models():
    """Initialize all models and set up the device."""
    global device, whisper_model, diarization_pipeline, embedding_model, enhanced_speaker_database, enhanced_service

    logger.info("🚀 STARTING MODEL INITIALIZATION")
    logger.info(f"🔧 Current working directory: {os.getcwd()}")
    logger.info(f"🐍 Python executable: {sys.executable}")
    logger.info(f"🔥 PyTorch version: {torch.__version__}")

    # Set up device (GPU if available, otherwise CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Using device: cpu")

    # Load the faster-whisper ASR model. faster-whisper handles arbitrary-length
    # audio and voice activity detection internally, so no manual chunking is used.
    asr_device = resolve_asr_device()
    compute_type = resolve_compute_type(asr_device)
    try:
        logger.info(
            f"Loading faster-whisper model: {WHISPER_MODEL} "
            f"(device={asr_device}, compute_type={compute_type}, "
            f"language={WHISPER_LANGUAGE or 'auto-detect'})"
        )
        whisper_model = _load_faster_whisper(WHISPER_MODEL, asr_device, compute_type)
        logger.info("✓ faster-whisper model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load faster-whisper model '{WHISPER_MODEL}': {e}")
        logger.info(f"Trying fallback model '{WHISPER_FALLBACK_MODEL}' on cpu/int8...")
        try:
            whisper_model = _load_faster_whisper(WHISPER_FALLBACK_MODEL, "cpu", "int8")
            logger.info(f"✓ faster-whisper '{WHISPER_FALLBACK_MODEL}' loaded (fallback)")
        except Exception as e2:
            logger.error(f"Failed to load any Whisper model: {e2}")
            logger.error("Transcription will not be available!")
            whisper_model = None

    # Try to load pyannote.audio for speaker diarization
    try:
        logger.info(f"Loading diarization model: {DIARIZATION_MODEL}")
        from pyannote.audio import Pipeline as DiarizationPipeline  # lazy import
        # pyannote.audio 4.x renamed use_auth_token= to token=. None lets
        # huggingface_hub fall back to HF_TOKEN / cached CLI login.
        hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
        diarization_pipeline = DiarizationPipeline.from_pretrained(
            DIARIZATION_MODEL,
            token=hf_token,
        )
        if device.type == "cuda":
            diarization_pipeline.to(device)
        logger.info("✓ Diarization model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load diarization model: {e}")
        logger.warning("Continuing without speaker diarization - only transcription will be available")
        diarization_pipeline = None

    # Try to load embedding model for speaker recognition
    try:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        from pyannote.audio.pipelines.speaker_verification import (  # lazy import
            PretrainedSpeakerEmbedding,
        )
        embedding_model = PretrainedSpeakerEmbedding(
            EMBEDDING_MODEL,
            device=device,
            token=os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN"),
        )
        logger.info("✓ Embedding model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load embedding model: {e}")
        logger.warning("Continuing without speaker embeddings")
        embedding_model = None
    
    # The v2 speaker database is the only speaker store.
    try:
        logger.info(f"Initializing speaker database v2 at {SPEAKER_DATA_DIR}...")
        enhanced_speaker_database = EnhancedSpeakerDatabase(SPEAKER_DATA_DIR)
        enhanced_speaker_database.similarity_threshold = SPEAKER_MATCH_THRESHOLD
        logger.info(
            f"✓ Speaker database v2 ready: {len(enhanced_speaker_database.speaker_records)} speakers, "
            f"match threshold {SPEAKER_MATCH_THRESHOLD:.2f}"
        )
    except Exception as e:
        logger.warning(f"Failed to initialize speaker database v2: {e}")
        enhanced_speaker_database = None

    # Initialize enhanced transcription features
    if ENHANCED_FEATURES_AVAILABLE:
        try:
            logger.info("Initializing enhanced transcription features...")
            # get_enhanced_service() returns the shared, cached instance, so
            # the /transcribe inline path and the /transcribe_enhanced &
            # /transcribe_with_summary endpoints all use one service rather
            # than constructing a new one here and again per request.
            from enhanced_server_integration import get_enhanced_service
            enhanced_service = get_enhanced_service(
                sentiment_model="vader",  # Fast and reliable for production
                enable_audio_features=True
            )
            logger.info("✅ Enhanced transcription features initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize enhanced features: {e}")
            logger.warning("Continuing with basic transcription only")
            enhanced_service = None
    else:
        logger.info("Enhanced features not available - basic transcription only")
        enhanced_service = None
    
    logger.info("Model loading completed")


@app.on_event("startup")
async def startup_event():
    """Initialize models when the server starts."""
    global enhanced_speaker_database, enhanced_speaker_integration

    initialize_models()

    # The integration layer wraps the *same* database instance the transcription
    # path uses - two instances would each hold their own in-memory copy and
    # silently overwrite each other's saves.
    try:
        enhanced_speaker_integration = EnhancedSpeakerServerIntegration(
            enhanced_db=enhanced_speaker_database,
            enhanced_db_path=SPEAKER_DATA_DIR,
        )
        if enhanced_speaker_database is None:
            enhanced_speaker_database = enhanced_speaker_integration.enhanced_db
        logger.info("Enhanced speaker integration initialized")
    except Exception as e:
        logger.warning(f"Enhanced speaker integration failed to initialize: {e}")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Oreja Audio Processing API",
        "status": "running",
        "models_loaded": all([
            whisper_model is not None,
            diarization_pipeline is not None,
            embedding_model is not None
        ])
    }


@app.get("/health")
async def health_check():
    """Detailed health check with model status."""
    return {
        "status": "healthy",
        "device": str(device) if device else "unknown",
        "models": {
            "whisper": whisper_model is not None,
            "diarization": diarization_pipeline is not None,
            "embedding": embedding_model is not None,
            # Kept for wire compatibility: true when persistent speaker
            # identification (embedding model + v2 database) is fully available.
            "speaker_embeddings": embedding_model is not None and enhanced_speaker_database is not None,
            "speaker_database": enhanced_speaker_database is not None,
            "enhanced_features": enhanced_service is not None
        },
        "speaker_count": len(enhanced_speaker_database.speaker_records) if enhanced_speaker_database else 0,
        "enhanced_capabilities": {
            "sentiment_analysis": enhanced_service is not None,
            "audio_features": enhanced_service is not None,
            "conversation_analytics": enhanced_service is not None
        } if enhanced_service else None,
        "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else "N/A"
    }


# ---------------------------------------------------------------------------
# Speaker embedding extraction
#
# ONE embedding system for the whole server: the pyannote PretrainedSpeakerEmbedding
# model loaded in initialize_models(). Every voiceprint - transcription-time
# identification, enrollment, correction learning, segment splitting - goes
# through extract_embedding_from_audio() so that all stored vectors live in the
# same space and remain comparable.
# ---------------------------------------------------------------------------

_embedding_load_lock = threading.Lock()


def ensure_embedding_model() -> bool:
    """
    Load only the speaker-embedding model (and pick a device) if it isn't loaded.

    initialize_models() runs from the FastAPI startup event, so standalone
    callers - batch transcription, the user training GUI - would otherwise find
    ``embedding_model is None`` and silently get no speaker identification at
    all. Returns True when an embedding model is available. Never raises.
    """
    global device, embedding_model

    if embedding_model is not None:
        return True

    with _embedding_load_lock:
        if embedding_model is not None:
            return True
        try:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Loading embedding model on demand: {EMBEDDING_MODEL} (device={device})")
            from pyannote.audio.pipelines.speaker_verification import (  # lazy import
                PretrainedSpeakerEmbedding,
            )
            embedding_model = PretrainedSpeakerEmbedding(
                EMBEDDING_MODEL,
                device=device,
                token=os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN"),
            )
            logger.info("✓ Embedding model loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to load embedding model on demand: {e}")
            embedding_model = None
            return False


def extract_embedding_from_audio(audio, sample_rate: int = SAMPLE_RATE) -> Optional["np.ndarray"]:
    """
    Embed a mono audio buffer into a speaker vector.

    Accepts a torch tensor or anything numpy can view as float32 samples.
    Returns None - never raises - when the model is unavailable, the crop is too
    short to be meaningful, or the model produces a non-finite vector. Callers
    are expected to degrade to diarization-only labels on None.
    """
    if embedding_model is None:
        return None

    try:
        if isinstance(audio, torch.Tensor):
            samples = audio.detach().to(torch.float32).cpu().reshape(-1)
        else:
            samples = torch.from_numpy(np.asarray(audio, dtype=np.float32).reshape(-1))

        # Anything under ~0.2s is noise as far as a voiceprint is concerned.
        if samples.numel() < int(max(sample_rate, 1) * 0.2):
            return None

        batch = samples.reshape(1, 1, -1)
        if device is not None and getattr(device, "type", "cpu") == "cuda":
            batch = batch.to(device)

        with _embedding_lock:
            with torch.inference_mode():
                raw = embedding_model(batch)

        if isinstance(raw, torch.Tensor):
            vector = raw.detach().cpu().numpy()
        else:
            vector = np.asarray(raw)

        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            return None
        return vector

    except Exception as e:
        logger.warning(f"Speaker embedding extraction failed: {e}")
        return None


def prepare_waveform(waveform: torch.Tensor, sample_rate: int) -> tuple:
    """Resample to 16 kHz and downmix to mono - the shape every model here wants."""
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(waveform)
        sample_rate = SAMPLE_RATE
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform, sample_rate


def embed_audio_bytes(audio_data: bytes) -> Optional["np.ndarray"]:
    """Decode an uploaded audio blob and embed it."""
    waveform, sample_rate = load_audio_from_bytes(audio_data)
    waveform, sample_rate = prepare_waveform(waveform, sample_rate)
    return extract_embedding_from_audio(waveform, sample_rate)


def crop_waveform(waveform: torch.Tensor, sample_rate: int,
                  start: float, end: float) -> Optional[torch.Tensor]:
    """Slice [start, end) seconds out of a waveform; None when out of bounds."""
    total = waveform.shape[-1]
    start_sample = max(int(float(start) * sample_rate), 0)
    end_sample = min(int(float(end) * sample_rate), total)
    if end_sample <= start_sample:
        return None
    return waveform[..., start_sample:end_sample]


def embeddings_for_segments(waveform: torch.Tensor, sample_rate: int,
                            segments: List[Dict[str, Any]],
                            min_duration: float = 0.5,
                            limit: int = 20) -> List["np.ndarray"]:
    """Embed each usable segment crop of a waveform. Short crops are skipped."""
    vectors = []
    for segment in segments or []:
        if len(vectors) >= limit:
            break
        try:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
        except (TypeError, ValueError):
            continue
        if end - start < min_duration:
            continue
        crop = crop_waveform(waveform, sample_rate, start, end)
        if crop is None:
            continue
        vector = extract_embedding_from_audio(crop, sample_rate)
        if vector is not None:
            vectors.append(vector)
    return vectors


class SpeakerDatabaseAdapter:
    """
    Duck-typed shim so enhanced_segment_splitting.AudioSegmentSplitter can run
    against the v2 database. It expects exactly two methods from the old legacy
    manager: extract_embedding() and provide_correction_feedback().
    """

    def extract_embedding(self, audio_numpy):
        return extract_embedding_from_audio(audio_numpy)

    def provide_correction_feedback(self, speaker_name: str, audio_numpy) -> bool:
        if enhanced_speaker_database is None or not speaker_name:
            return False
        vector = extract_embedding_from_audio(audio_numpy)
        if vector is None:
            return False
        with _speaker_db_lock:
            enhanced_speaker_database.enroll_speaker(
                speaker_name, [vector], confidence=0.9, source_type="corrected"
            )
        return True


def require_speaker_db() -> EnhancedSpeakerDatabase:
    """Return the speaker database or fail the request with a clear 503."""
    if enhanced_speaker_database is None:
        raise HTTPException(status_code=503, detail="Speaker database not available")
    return enhanced_speaker_database


def require_embedding_model():
    """Fail the request when voiceprint work is asked for but unavailable."""
    if embedding_model is None:
        raise HTTPException(
            status_code=503,
            detail="Speaker embedding model not loaded - voiceprint operations unavailable"
        )


# ---------------------------------------------------------------------------
# Speaker management endpoints (all v2-database backed)
# ---------------------------------------------------------------------------

@app.get("/speakers")
async def get_speaker_stats():
    """
    Get statistics about known speakers.

    Response shape is frozen: the shipped WPF client reads
    speakers[].name / .id / .embedding_count (as an int).
    """
    database = require_speaker_db()

    try:
        # get_all_speakers() walks every record and averages confidences with
        # numpy; off the event loop so a large database cannot stall the live
        # 5-second /transcribe chunks.
        speakers = await asyncio.to_thread(database.get_all_speakers)
        return {
            'total_speakers': len(speakers),
            'speakers': [
                {
                    'id': s['speaker_id'],
                    'name': s['display_name'],
                    'embedding_count': int(s['embedding_count']),
                    'last_seen': s['last_seen'],
                    'avg_confidence': float(s['average_confidence']),
                    'average_confidence': float(s['average_confidence']),
                    'is_enrolled': s['is_enrolled'],
                    'is_verified': s['is_verified'],
                    'source_type': s['source_type'],
                }
                for s in speakers
            ]
        }
    except Exception as e:
        logger.error(f"Error getting speaker stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speakers/enroll")
async def enroll_speaker(
    speaker_name: str,
    audio: UploadFile = File(...)
):
    """
    Enroll a speaker with a known name from an audio sample.

    Args:
        speaker_name: Human-readable name for the speaker
        audio: Audio file containing the speaker's voice

    Returns:
        Speaker ID and enrollment status
    """
    database = require_speaker_db()
    require_embedding_model()

    speaker_name = (speaker_name or "").strip()
    if not speaker_name:
        raise HTTPException(status_code=400, detail="Speaker name cannot be empty")

    try:
        audio_data = await audio.read()
        embedding = await asyncio.to_thread(embed_audio_bytes, audio_data)

        if embedding is None:
            raise HTTPException(
                status_code=400,
                detail="Could not extract a voiceprint from the supplied audio"
            )

        with _speaker_db_lock:
            speaker_id = database.enroll_speaker(speaker_name, [embedding], confidence=0.95)

        return {
            "speaker_id": speaker_id,
            "speaker_name": database.get_display_name(speaker_id) or speaker_name,
            "embedding_count": len(database.speaker_embeddings.get(speaker_id, [])),
            "status": "enrolled_successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enrolling speaker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speakers/create_profile")
async def create_speaker_profile(payload: Dict[str, Any]):
    """
    Create a named speaker profile with no audio (name-only enrollment).

    Used by the transcription editor when it has a speaker name but no audio to
    train on. The profile starts with zero embeddings and picks them up later
    through corrections or /speakers/enroll_from_segments.

    Body: {"speaker_name": "..."} (also accepts "name" / "display_name")
    """
    database = require_speaker_db()

    speaker_name = (
        payload.get("speaker_name")
        or payload.get("name")
        or payload.get("display_name")
        or ""
    ).strip()
    if not speaker_name:
        raise HTTPException(status_code=400, detail="speaker_name is required")

    try:
        with _speaker_db_lock:
            existing_id = database.find_speaker_by_name(speaker_name)
            speaker_id = database.enroll_speaker(speaker_name, [], confidence=0.9)

        return {
            "status": "profile_exists" if existing_id else "profile_created",
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "embedding_count": len(database.speaker_embeddings.get(speaker_id, [])),
            "message": (
                f"Speaker '{speaker_name}' already existed - reused {speaker_id}"
                if existing_id else
                f"Created speaker profile '{speaker_name}' ({speaker_id})"
            )
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating speaker profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speakers/enroll_from_segments")
async def enroll_speaker_from_segments(payload: Dict[str, Any]):
    """
    Enroll a speaker from labelled segments of an audio file already on disk.

    Body: {"speaker_name": str, "audio_file": path, "segments": [{start, end}, ...]}

    Each segment long enough to be meaningful is embedded separately, so the
    speaker's bank captures the natural variation across their turns rather than
    one averaged blob. Falls back to a name-only profile when no segment yields
    a usable voiceprint.
    """
    database = require_speaker_db()

    speaker_name = (payload.get("speaker_name") or payload.get("name") or "").strip()
    if not speaker_name:
        raise HTTPException(status_code=400, detail="speaker_name is required")

    audio_file = payload.get("audio_file")
    segments = payload.get("segments") or []

    def _enroll() -> Dict[str, Any]:
        vectors = []
        if audio_file and Path(audio_file).exists() and segments and embedding_model is not None:
            waveform, sample_rate = load_audio(audio_file)
            waveform, sample_rate = prepare_waveform(waveform, sample_rate)
            vectors = embeddings_for_segments(waveform, sample_rate, segments, min_duration=0.5, limit=10)

        with _speaker_db_lock:
            speaker_id = database.enroll_speaker(speaker_name, vectors, confidence=0.9)

        return {
            "speaker_id": speaker_id,
            "embeddings_added": len(vectors),
            "embedding_count": len(database.speaker_embeddings.get(speaker_id, [])),
        }

    try:
        if audio_file and not Path(audio_file).exists():
            logger.warning(f"enroll_from_segments: audio file not found: {audio_file}")

        result = await asyncio.to_thread(_enroll)

        return {
            "status": "enrolled_successfully" if result["embeddings_added"] else "profile_created",
            "speaker_name": speaker_name,
            "segments_supplied": len(segments),
            **result,
            "message": (
                f"Enrolled '{speaker_name}' from {result['embeddings_added']} segment(s)"
                if result["embeddings_added"]
                else f"Created '{speaker_name}' without voiceprints "
                     f"(no usable audio segments or embedding model unavailable)"
            )
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error enrolling speaker from segments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speakers/identify")
async def identify_speaker(audio: UploadFile = File(...), learn: bool = True):
    """
    Identify a speaker from an audio sample against the v2 database.

    Args:
        audio: Audio file containing the speaker's voice
        learn: store this observation on the matched/created speaker

    Returns:
        Speaker identification result, including whether a new speaker was made
    """
    database = require_speaker_db()
    require_embedding_model()

    try:
        audio_data = await audio.read()
        embedding = await asyncio.to_thread(embed_audio_bytes, audio_data)

        if embedding is None:
            raise HTTPException(
                status_code=400,
                detail="Could not extract a voiceprint from the supplied audio"
            )

        known_before = set(database.speaker_records.keys())
        with _speaker_db_lock:
            speaker_id, speaker_name, confidence = database.identify_speaker(
                embedding, learn=learn
            )

        is_new = speaker_id is not None and speaker_id not in known_before

        return {
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "confidence": float(confidence),
            "is_new_speaker": is_new,
            "threshold": database.similarity_threshold,
            "status": "new_speaker_created" if is_new else "identified"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error identifying speaker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speakers/feedback")
async def provide_speaker_feedback(
    correct_speaker_name: str,
    audio_segment_start: float,
    audio_segment_end: float,
    audio: UploadFile = File(...)
):
    """
    Teach the database that one audio segment belongs to a named speaker.

    Args:
        correct_speaker_name: The correct speaker name for this audio segment
        audio_segment_start: Start time of the segment in seconds
        audio_segment_end: End time of the segment in seconds
        audio: The audio file containing the segment

    Returns:
        Status of the feedback processing
    """
    database = require_speaker_db()
    require_embedding_model()

    correct_speaker_name = (correct_speaker_name or "").strip()
    if not correct_speaker_name:
        raise HTTPException(status_code=400, detail="Speaker name cannot be empty")

    try:
        audio_data = await audio.read()

        def _learn() -> Optional[str]:
            waveform, sample_rate = load_audio_from_bytes(audio_data)
            waveform, sample_rate = prepare_waveform(waveform, sample_rate)
            crop = crop_waveform(waveform, sample_rate, audio_segment_start, audio_segment_end)
            if crop is None:
                return None
            vector = extract_embedding_from_audio(crop, sample_rate)
            if vector is None:
                return None
            with _speaker_db_lock:
                return database.enroll_speaker(
                    correct_speaker_name, [vector], confidence=0.9, source_type="corrected"
                )

        speaker_id = await asyncio.to_thread(_learn)

        if speaker_id is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid segment timing or no voiceprint could be extracted"
            )

        return {
            "status": "feedback_processed",
            "speaker_id": speaker_id,
            "speaker_name": correct_speaker_name,
            "segment_duration": audio_segment_end - audio_segment_start,
            "embedding_count": len(database.speaker_embeddings.get(speaker_id, [])),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing speaker feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speakers/batch_feedback")
async def provide_batch_speaker_feedback(feedback_data: dict):
    """
    Provide batch feedback for multiple speaker corrections.

    Body: {
        "corrections": [
            {"speaker_name": "John",
             "audio_segments": [{"audio_data": "<base64 wav>"}, ...]},
            ...
        ]
    }
    """
    database = require_speaker_db()
    require_embedding_model()

    corrections = feedback_data.get("corrections", []) or []

    def _process() -> Dict[str, Any]:
        import base64

        processed = 0
        failed = 0
        updated_speakers = []

        for correction in corrections:
            speaker_name = (correction.get("speaker_name") or "").strip()
            if not speaker_name:
                continue

            vectors = []
            for segment in correction.get("audio_segments", []) or []:
                try:
                    audio_bytes = base64.b64decode(segment["audio_data"])
                    vector = embed_audio_bytes(audio_bytes)
                    if vector is None:
                        failed += 1
                        continue
                    vectors.append(vector)
                    processed += 1
                except Exception as e:
                    logger.warning(f"Failed to process segment for {speaker_name}: {e}")
                    failed += 1

            if vectors:
                with _speaker_db_lock:
                    database.enroll_speaker(
                        speaker_name, vectors, confidence=0.9, source_type="corrected"
                    )
                updated_speakers.append(speaker_name)

        return {
            "processed_segments": processed,
            "failed_segments": failed,
            "updated_speakers": updated_speakers,
        }

    try:
        results = await asyncio.to_thread(_process)
        return {
            "status": "batch_feedback_processed",
            "total_corrections": len(corrections),
            **results,
        }

    except Exception as e:
        logger.error(f"Error processing batch speaker feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speakers/merge")
async def merge_speakers(source_speaker_id: str, target_speaker_id: str, target_name: str = None):
    """
    Manually merge two speaker profiles.

    The database keeps whichever record holds more samples, so the surviving ID
    is reported back rather than assumed.
    """
    database = require_speaker_db()

    if source_speaker_id == target_speaker_id:
        raise HTTPException(status_code=400, detail="Cannot merge speaker with itself")
    if source_speaker_id not in database.speaker_records:
        raise HTTPException(status_code=404, detail=f"Source speaker {source_speaker_id} not found")
    if target_speaker_id not in database.speaker_records:
        raise HTTPException(status_code=404, detail=f"Target speaker {target_speaker_id} not found")

    try:
        with _speaker_db_lock:
            success = database.merge_speakers(source_speaker_id, target_speaker_id)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to merge speakers")

            survivor = (
                target_speaker_id
                if target_speaker_id in database.speaker_records
                else source_speaker_id
            )
            if target_name and target_name.strip():
                database.update_display_name(survivor, target_name.strip())

        return {
            "status": "merged_successfully",
            "source_speaker_id": source_speaker_id,
            "target_speaker_id": survivor,
            "target_name": database.get_display_name(survivor),
            "embedding_count": len(database.speaker_embeddings.get(survivor, [])),
            "message": f"Successfully merged into {survivor}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error merging speakers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speakers/cleanup")
async def cleanup_speakers(min_embeddings: int = 1, merge_duplicate_names: bool = True):
    """
    Housekeeping on the speaker database.

    Merges speakers whose display names are exact duplicates and deletes
    auto-created speakers holding fewer than `min_embeddings` voiceprints.
    Enrolled and user-verified speakers are never removed.
    """
    database = require_speaker_db()

    try:
        with _speaker_db_lock:
            results = database.cleanup(
                min_embeddings=min_embeddings,
                merge_duplicate_names=merge_duplicate_names,
            )

        return {
            "status": "cleanup_complete",
            "results": results,
            "message": (
                f"Merged {results['duplicates_merged']} duplicate-name speakers, "
                f"removed {results['speakers_removed']} under-sampled auto speakers "
                f"({results['speakers_before']} -> {results['speakers_after']})"
            )
        }

    except Exception as e:
        logger.error(f"Speaker cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speakers/reset")
async def reset_speakers(include_enrolled: bool = False):
    """
    DANGER: clear learned speakers.

    By default only auto-created speakers are deleted; enrolled and verified
    speakers survive, because those represent identities the user established
    deliberately and cannot be reconstructed. Pass include_enrolled=true to wipe
    everything.
    """
    database = require_speaker_db()

    try:
        with _speaker_db_lock:
            if include_enrolled:
                removed = []
                for speaker_id in list(database.speaker_records.keys()):
                    display_name = database.get_display_name(speaker_id)
                    # save=False: one flush after the loop instead of rewriting
                    # speaker_records.json + every .npy once per deletion.
                    if database.delete_speaker(speaker_id, save=False):
                        removed.append({"speaker_id": speaker_id, "display_name": display_name})
                database.save()
                results = {
                    "speakers_removed": len(removed),
                    "speakers_kept": 0,
                    "removed_speakers": removed,
                    "speakers_after": len(database.speaker_records),
                }
            else:
                results = database.reset_auto_speakers()

        logger.warning(
            f"Speaker database reset (include_enrolled={include_enrolled}): "
            f"{results['speakers_removed']} removed"
        )

        return {
            "status": "reset_complete",
            "include_enrolled": include_enrolled,
            "results": results,
            "message": (
                f"Removed {results['speakers_removed']} speakers, "
                f"kept {results.get('speakers_kept', 0)} enrolled/verified"
            )
        }

    except Exception as e:
        logger.error(f"Speaker reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speakers/name_mapping")
async def update_speaker_name_mapping(
    old_speaker_id: str,
    new_speaker_name: str
):
    """
    Rename / consolidate a speaker. THIS IS THE ENDPOINT THE WPF CLIENT CALLS.

    The client posts query params old_speaker_id and new_speaker_name whenever a
    user renames a speaker in the transcript, and treats any 2xx as success while
    logging the "status" field.

    Three cases are handled:
      * old_speaker_id is a real v2 ID -> rename it (or merge into the speaker
        that already owns the target name)
      * old_speaker_id is an anonymous per-session label ("Speaker SPEAKER_00",
        "AUTO_SPEAKER_001") -> attach to an existing speaker of that name, or
        create one
      * the target name already exists -> merge rather than duplicate
    """
    database = require_speaker_db()

    new_speaker_name = (new_speaker_name or "").strip()
    if not new_speaker_name:
        raise HTTPException(status_code=400, detail="Speaker name cannot be empty")

    try:
        with _speaker_db_lock:
            outcome = database.apply_name_correction(old_speaker_id, new_speaker_name)

        status_by_action = {
            "merged": "speakers_merged",
            "renamed": "name_updated",
            "matched": "name_updated",
            "created": "speaker_created",
            "merge_failed": "merge_failed",
            "noop": "mapping_noted",
        }
        status = status_by_action.get(outcome["action"], "mapping_noted")

        return {
            "status": status,
            "action": outcome["action"],
            "old_speaker_id": old_speaker_id,
            "speaker_id": outcome["speaker_id"],
            "target_speaker_id": outcome["speaker_id"],
            "speaker_name": outcome["display_name"],
            "new_name": outcome["display_name"],
            "embeddings_learned": outcome["embeddings_added"],
            "message": f"{old_speaker_id} -> '{outcome['display_name']}' ({outcome['action']})"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating speaker name mapping: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/speakers/{speaker_id}/name")
async def update_speaker_name(speaker_id: str, new_name: str):
    """Update a speaker's display name (the immutable ID is unchanged)."""
    database = require_speaker_db()

    new_name = (new_name or "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="Speaker name cannot be empty")
    if speaker_id not in database.speaker_records:
        raise HTTPException(status_code=404, detail=f"Speaker {speaker_id} not found")

    try:
        with _speaker_db_lock:
            success = database.update_display_name(speaker_id, new_name)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to update speaker name")

        logger.info(f"Updated speaker name: {speaker_id} -> {new_name}")
        return {
            "status": "success",
            "message": f"Speaker name updated to: {new_name}",
            "speaker_id": speaker_id,
            "new_name": new_name
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating speaker name: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/speakers/{speaker_id}")
async def delete_speaker(speaker_id: str):
    """Delete a speaker profile and its voiceprints."""
    database = require_speaker_db()

    if speaker_id not in database.speaker_records:
        raise HTTPException(status_code=404, detail=f"Speaker {speaker_id} not found")

    try:
        display_name = database.get_display_name(speaker_id)
        with _speaker_db_lock:
            success = database.delete_speaker(speaker_id)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete speaker")

        logger.info(f"Deleted speaker: {speaker_id}")
        return {
            "status": "success",
            "message": f"Speaker {speaker_id} deleted successfully",
            "speaker_id": speaker_id,
            "speaker_name": display_name
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting speaker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _resolve_request_language(language: Optional[str]):
    """
    Map a /transcribe `language` query parameter onto what faster-whisper's
    transcribe() takes. Absent/empty defers to the server-wide OREJA_LANGUAGE
    default (returns the _LANGUAGE_DEFAULT sentinel); "auto" forces per-request
    auto-detection (returns None) even when a server default is configured;
    anything else must be a language code faster-whisper knows.
    """
    if language is None or not language.strip():
        return _LANGUAGE_DEFAULT
    code = language.strip().lower()
    if code == "auto":
        return None
    try:
        from faster_whisper.tokenizer import _LANGUAGE_CODES  # noqa: PLC0415 - lazy, like the model import
        valid = code in _LANGUAGE_CODES
    except ImportError:
        # Version without the constant: accept anything code-shaped and let the
        # decoder be the authority.
        valid = code.isalpha() and 2 <= len(code) <= 3
    if not valid:
        raise HTTPException(status_code=400, detail=f"Unsupported language code: {code}")
    return code


@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    include_analysis: bool = False,
    source: Optional[str] = None,
    language: Optional[str] = None,
    accuracy: bool = False,
    max_speakers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Transcribe and diarize audio file, optionally with sentiment analysis and audio features.

    Args:
        audio: Audio file (WAV format preferred)
        source: Audio origin tag sent by the WPF client ("mic" / "system").
            Scopes persistent speaker identification: voices heard on one
            source never auto-match or reinforce profiles first heard on the
            other, so a far-end conference voice cannot contaminate the local
            user's voiceprint. Omitted/unknown values match all profiles.
        language: QUERY PARAMETER (same FastAPI caveat as include_analysis).
            A Whisper language code ("en", "es", ...) pins the decode to that
            language for this request; "auto" forces per-request auto-detection;
            omitted/empty defers to the server-wide OREJA_LANGUAGE default.
            Unknown codes are rejected with a 400.
        accuracy: QUERY PARAMETER. Spend more compute for a better transcript:
            beam size OREJA_FILE_BEAM_SIZE (default 10) instead of the live
            beam, decoded with OREJA_FILE_MODEL when configured. Meant for the
            offline file path - a live 5-second chunk gains little and pays
            real latency.
        max_speakers: QUERY PARAMETER. Ceiling on how many distinct speakers
            diarization may report - the client sends its session roster size
            so pyannote cannot invent phantom extra speakers. A ceiling, not an
            exact count: any one chunk usually contains a subset of the roster.
            Out-of-range values (< 1 or > 50) are ignored.
        include_analysis: Run the sentiment/conversation-analysis enhancement
            inline. QUERY PARAMETER ONLY (POST /transcribe?include_analysis=true):
            FastAPI treats a bare `bool` alongside `File(...)` as a query
            parameter, so sending it as a multipart form field is silently
            ignored and you get the fast path. Defaults to False - the shipped WPF
            client posts 5-second chunks and sends nothing extra, so it always
            gets the fast path. Set OREJA_INLINE_ENHANCEMENT=1 to flip the
            server-wide default instead of passing this per request. Batch/file
            callers that want the richer result should either pass
            include_analysis=true here or call POST /transcribe_with_summary.

    Returns:
        Transcription result with speaker diarization, and - only when
        analysis was requested - sentiment analysis and audio features.
    """
    start_time = time.time()
    
    try:
        # Validate file
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read audio data into memory (never write to disk)
        audio_data = await audio.read()
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        logger.info(f"Processing audio file: {audio.filename}, size: {len(audio_data)} bytes")
        
        # Load audio with torchaudio (in-memory processing)
        waveform, sample_rate = load_audio_from_bytes(audio_data)
        
        # Validate audio length
        duration = waveform.shape[1] / sample_rate
        if duration < MIN_AUDIO_LENGTH:
            raise HTTPException(status_code=400, detail=f"Audio too short: {duration:.2f}s")
        if duration > MAX_AUDIO_LENGTH:
            raise HTTPException(status_code=400, detail=f"Audio too long: {duration:.2f}s (max: {MAX_AUDIO_LENGTH/3600:.1f} hours)")

        # Resample to 16kHz if needed
        if sample_rate != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
            waveform = resampler(waveform)
            sample_rate = SAMPLE_RATE
        
        # Ensure mono audio
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        resolved_language = _resolve_request_language(language)

        # Run transcription and diarization concurrently. Both helpers dispatch the
        # blocking model call onto a worker thread (asyncio.to_thread), so the event
        # loop stays responsive and the two models genuinely overlap.
        transcription_task = asyncio.create_task(
            run_transcription(
                waveform, sample_rate, language=resolved_language, accuracy=accuracy
            )
        )

        if max_speakers is not None and not (1 <= max_speakers <= 50):
            max_speakers = None

        if diarization_pipeline is not None:
            diarization_task = asyncio.create_task(
                run_diarization(waveform, sample_rate, max_speakers=max_speakers)
            )
            # Wait for both tasks to complete. return_exceptions=True so a
            # failure on one side does not leave the other task orphaned in a
            # worker thread still holding _whisper_lock / _diarization_lock
            # (which also produces an "exception was never retrieved" warning).
            transcription_result, diarization_result = await asyncio.gather(
                transcription_task, diarization_task, return_exceptions=True
            )
            if isinstance(transcription_result, BaseException):
                raise transcription_result
            if isinstance(diarization_result, BaseException):
                logger.warning(
                    f"Diarization failed, continuing transcription-only: {diarization_result}"
                )
                diarization_result = None
        else:
            # Only run transcription
            transcription_result = await transcription_task
            diarization_result = None
            logger.info("Diarization skipped - model not available")

        # Check if transcription was skipped (silent clip / nothing survived gating)
        if transcription_result and "skipped_reason" in transcription_result:
            processing_time = time.time() - start_time
            logger.info(f"Transcription skipped: {transcription_result['skipped_reason']}")
            
            result = {
                "segments": [],
                "full_text": "",
                "processing_time": processing_time,
                "timestamp": time.time(),
                "audio_duration": duration,
                "sample_rate": sample_rate,
                "skipped_reason": transcription_result["skipped_reason"]
            }
            return result
        
        # Merge transcription with speaker information (if available). This does
        # word-level speaker attribution and splits ASR segments at speaker changes.
        segments = merge_transcription_and_diarization(
            transcription_result, diarization_result, waveform, sample_rate
        )

        # Integration hook for voiceprint-based speaker naming (see function
        # docstring). Dispatched off the event loop so the next implementation can
        # do real embedding work here without blocking the server.
        # Normalize the client's source tag to a known scope; anything else
        # (including absence) means "unknown origin" and matches all profiles.
        source_scope = source.strip().lower() if source else None
        if source_scope not in ("mic", "system"):
            source_scope = None

        segments = await asyncio.to_thread(
            identify_speakers_hook, waveform, sample_rate, segments,
            diarization_result, source_scope
        )

        # Generate full text
        full_text = " ".join([segment["text"] for segment in segments])
        
        processing_time = time.time() - start_time
        
        # Create basic result
        basic_result = {
            "segments": segments,
            "full_text": full_text,
            "processing_time": processing_time,
            "timestamp": time.time(),
            "audio_duration": duration,
            "sample_rate": sample_rate
        }
        
        # 🚀 OPTIONAL: SENTIMENT ANALYSIS AND AUDIO FEATURES
        # Off by default: it adds significant latency to what the WPF client
        # expects to be a fast 5-second-chunk round trip. Turned on either per
        # request (include_analysis=true) or server-wide (OREJA_INLINE_ENHANCEMENT=1).
        if enhanced_service and (include_analysis or ENHANCEMENT_INLINE):
            try:
                logger.info("Applying enhanced features (sentiment analysis & audio features)...")
                enhanced_result = enhanced_service.enhance_transcription_result(
                    basic_result, waveform, sample_rate
                )
                logger.info("✅ Enhanced features applied successfully")
                
                # Update processing time to include enhancement
                enhanced_result["processing_time"] = time.time() - start_time
                
                return enhanced_result
                
            except Exception as e:
                logger.warning(f"⚠️ Enhanced features failed, returning basic result: {e}")
                # Continue with basic result if enhancement fails
        else:
            logger.debug("Inline enhancement disabled - returning basic transcription")

        logger.info(f"Transcription completed in {processing_time:.2f}s for {duration:.2f}s audio")
        return basic_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/extract_embeddings")
async def extract_speaker_embeddings(audio: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Extract speaker embeddings from audio for speaker recognition.
    
    Args:
        audio: Audio file containing speech
        
    Returns:
        The float32 voiceprint as hex bytes, plus its dimensionality.
    """
    require_embedding_model()

    try:
        audio_data = await audio.read()

        def _extract():
            waveform, sample_rate = load_audio_from_bytes(audio_data)
            waveform, sample_rate = prepare_waveform(waveform, sample_rate)
            vector = extract_embedding_from_audio(waveform, sample_rate)
            return vector, waveform.shape[-1] / sample_rate

        embedding, duration = await asyncio.to_thread(_extract)

        if embedding is None:
            raise HTTPException(
                status_code=400,
                detail="Could not extract a voiceprint from the supplied audio"
            )

        embedding_bytes = embedding.astype(np.float32).tobytes()

        return {
            "embeddings": embedding_bytes.hex(),
            "embedding_size": len(embedding_bytes),
            "embedding_dim": int(embedding.shape[0]),
            "audio_duration": duration
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")


@app.post("/speakers/real_time_feedback")
async def real_time_speaker_feedback(feedback_data: dict):
    """
    Apply a speaker correction made in the transcription editor, and learn from it.

    Body:
        old_speaker_id: the incorrect speaker ID or per-session label
        correct_speaker_name: the correct display name
        audio_segments: [{start, end}, ...] spoken by that speaker (optional)
        audio_file: path to the audio file those segments index into (optional).
            Also accepted per-segment as audio_segments[i]["audio_file"], which
            is the shape transcription_editor.py actually sends.

    When audio is supplied the corrected speaker's voiceprint bank is updated
    from it, which is what makes the correction stick for future recordings
    rather than being a one-off relabel.
    """
    database = require_speaker_db()

    old_speaker_id = feedback_data.get("old_speaker_id")
    correct_speaker_name = (feedback_data.get("correct_speaker_name") or "").strip()
    audio_segments = feedback_data.get("audio_segments", []) or []
    audio_file = feedback_data.get("audio_file")

    if not audio_file:
        # transcription_editor.py (the only client of this endpoint) nests
        # audio_file inside each audio_segments entry and sends no top-level
        # key, so without this fallback every editor correction was applied
        # with embeddings_learned == 0 - i.e. the correction never stuck for
        # future recordings, which is the whole point of this endpoint.
        for segment in audio_segments:
            if isinstance(segment, dict) and segment.get("audio_file"):
                audio_file = segment["audio_file"]
                break

    if not old_speaker_id or not correct_speaker_name:
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: old_speaker_id, correct_speaker_name"
        )

    logger.info(f"Processing real-time feedback: {old_speaker_id} -> {correct_speaker_name}")

    def _apply() -> Dict[str, Any]:
        vectors = []
        if audio_file and audio_segments and embedding_model is not None and Path(audio_file).exists():
            try:
                waveform, sample_rate = load_audio(audio_file)
                waveform, sample_rate = prepare_waveform(waveform, sample_rate)
                vectors = embeddings_for_segments(
                    waveform, sample_rate, audio_segments, min_duration=0.5, limit=10
                )
            except Exception as e:
                logger.warning(f"Could not learn from audio for {correct_speaker_name}: {e}")

        with _speaker_db_lock:
            return database.apply_name_correction(
                old_speaker_id, correct_speaker_name, embeddings=vectors
            )

    try:
        outcome = await asyncio.to_thread(_apply)

        status_by_action = {
            "merged": "speakers_merged",
            "renamed": "learned" if outcome["embeddings_added"] else "name_updated",
            "matched": "learned" if outcome["embeddings_added"] else "name_updated",
            "created": "speaker_created",
            "merge_failed": "merge_failed",
        }

        return {
            "status": status_by_action.get(outcome["action"], "mapping_noted"),
            "action": outcome["action"],
            "speaker_id": outcome["speaker_id"],
            "target_speaker_id": outcome["speaker_id"],
            "new_name": outcome["display_name"],
            "embeddings_learned": outcome["embeddings_added"],
            "segments_supplied": len(audio_segments),
            "message": (
                f"{old_speaker_id} -> '{outcome['display_name']}' ({outcome['action']}); "
                f"learned {outcome['embeddings_added']} voiceprint(s)"
            )
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in real-time speaker feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reattribute_speakers")
async def reattribute_speakers(reattribution_data: dict):
    """
    Re-run voiceprint identification on existing segments of an audio file.

    Body:
        audio_file: path to the audio file
        segments: [{segment_index, start, end, current_speaker, current_confidence}]

    Only segments whose confidence would actually improve are reported. Nothing
    is written back: the caller decides what to accept.
    """
    database = require_speaker_db()
    require_embedding_model()

    audio_file = reattribution_data.get("audio_file")
    segments = reattribution_data.get("segments", []) or []

    if not audio_file or not segments:
        raise HTTPException(status_code=400, detail="Missing audio_file or segments")
    if not Path(audio_file).exists():
        raise HTTPException(status_code=400, detail="Audio file not found")

    logger.info(f"Re-attributing {len(segments)} segments from {audio_file}")

    def _reattribute() -> List[Dict[str, Any]]:
        waveform, sample_rate = load_audio(audio_file)
        waveform, sample_rate = prepare_waveform(waveform, sample_rate)

        improved = []
        for segment in segments:
            try:
                start = float(segment.get('start', 0))
                end = float(segment.get('end', start + 1))
                current_speaker = segment.get('current_speaker', 'Unknown')
                current_confidence = float(segment.get('current_confidence', 0) or 0)

                crop = crop_waveform(waveform, sample_rate, start, end)
                if crop is None or crop.shape[-1] < sample_rate * MIN_SPEAKER_AUDIO_SECONDS:
                    continue

                vector = extract_embedding_from_audio(crop, sample_rate)
                if vector is None:
                    continue

                speaker_id, speaker_name, confidence = database.identify_speaker(
                    vector, auto_create=False, learn=False, save=False
                )
                if speaker_id is None:
                    continue

                if speaker_name != current_speaker and confidence > current_confidence + 0.1:
                    improved.append({
                        'segment_index': segment.get('segment_index'),
                        'new_speaker': speaker_name,
                        'new_speaker_id': speaker_id,
                        'new_confidence': float(confidence),
                        'old_speaker': current_speaker,
                        'old_confidence': current_confidence,
                        'improvement': float(confidence - current_confidence)
                    })
                    logger.info(
                        f"Improved segment {segment.get('segment_index')}: "
                        f"{current_speaker} ({current_confidence:.3f}) -> "
                        f"{speaker_name} ({confidence:.3f})"
                    )

            except Exception as e:
                logger.warning(f"Could not re-attribute segment: {e}")
                continue

        return improved

    try:
        updated_segments = await asyncio.to_thread(_reattribute)

        logger.info(f"Re-attribution complete: {len(updated_segments)} segments improved")

        return {
            "status": "success",
            "updated_segments": updated_segments,
            "total_segments_processed": len(segments),
            "improvements_found": len(updated_segments)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in speaker re-attribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segments/split_with_audio_analysis")
async def split_segment_with_audio_analysis(
    audio_file: str,
    original_segment: dict,
    split_text_position: float,
    first_speaker: str,
    second_speaker: str
):
    """
    Split a segment with proper audio re-analysis and embedding extraction.
    
    This endpoint addresses the critical flaw where split segments don't get 
    proper embedding extraction from their actual audio portions.
    
    Args:
        audio_file: Path to the audio file
        original_segment: The original segment to split
        split_text_position: Position in text (0.0-1.0) where to split
        first_speaker: Speaker name for first part
        second_speaker: Speaker name for second part
        
    Returns:
        Enhanced split results with audio analysis and embedding extraction
    """
    require_speaker_db()
    require_embedding_model()

    try:
        # Validate inputs
        if not (0.0 <= split_text_position <= 1.0):
            raise HTTPException(status_code=400, detail="Split position must be between 0.0 and 1.0")

        if not audio_file or not Path(audio_file).exists():
            raise HTTPException(status_code=400, detail="Audio file not found")

        # The splitter only needs extract_embedding() / provide_correction_feedback();
        # the adapter serves both from the pyannote model + v2 database.
        splitter = AudioSegmentSplitter(SpeakerDatabaseAdapter())
        
        # Perform the split with audio analysis
        first_segment, second_segment, success = splitter.split_segment_with_audio_analysis(
            audio_file=audio_file,
            original_segment=original_segment,
            split_text_position=split_text_position,
            first_speaker=first_speaker,
            second_speaker=second_speaker
        )
        
        if not success or first_segment is None or second_segment is None:
            raise HTTPException(status_code=500, detail="Failed to split segment with audio analysis")
        
        # Validate the split
        validator = SegmentSplitValidator(splitter)
        validation_result = validator.validate_split(first_segment, second_segment, audio_file)
        
        # Prepare response
        response = {
            "status": "split_successful",
            "first_segment": first_segment,
            "second_segment": second_segment,
            "validation": validation_result,
            "audio_analysis_performed": True,
            "embeddings_extracted": {
                "first_segment": first_segment.get('embedding_extracted', False),
                "second_segment": second_segment.get('embedding_extracted', False)
            },
            "split_confidence": {
                "first_segment": first_segment.get('split_confidence', 0.0),
                "second_segment": second_segment.get('split_confidence', 0.0),
                "overall": validation_result.get('confidence', 0.0)
            },
            "speaker_models_updated": success,
            "message": f"Segment split with audio re-analysis. Confidence: {validation_result.get('confidence', 0.0):.2f}"
        }
        
        # Add warnings if validation found issues
        if validation_result.get('issues'):
            response["warnings"] = validation_result['issues']
        
        # Add suggestions
        if validation_result.get('suggestions'):
            response["suggestions"] = validation_result['suggestions']
        
        logger.info(f"Enhanced segment split completed: {first_speaker} | {second_speaker} "
                   f"(confidence: {validation_result.get('confidence', 0.0):.2f})")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in enhanced segment splitting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segments/reprocess_embeddings")
async def reprocess_segment_embeddings(
    audio_file: str,
    segments: List[dict],
    force_update: bool = False
):
    """
    Re-extract voiceprints for already-labelled segments of an audio file.

    Useful after speaker names have been corrected: it pushes the audio that
    actually belongs to each corrected name back into that speaker's bank.

    Args:
        audio_file: Path to the audio file
        segments: List of segments carrying start / end / speaker
        force_update: Accepted for compatibility; every supplied segment is
            re-embedded regardless, since the caller asked for reprocessing.

    Returns:
        Results of the reprocessing operation
    """
    database = require_speaker_db()
    require_embedding_model()

    if not audio_file or not Path(audio_file).exists():
        raise HTTPException(status_code=400, detail="Audio file not found")

    def _reprocess() -> Dict[str, Any]:
        results = {
            "total_segments": len(segments),
            "processed_segments": 0,
            "updated_speakers": [],
            "failed_extractions": 0,
            "improvements": []
        }

        waveform, sample_rate = load_audio(audio_file)
        waveform, sample_rate = prepare_waveform(waveform, sample_rate)

        # Group by corrected speaker name so each speaker is enrolled once with
        # all of their crops, instead of re-saving the database per segment.
        vectors_by_speaker: Dict[str, List[Any]] = {}

        for segment in segments:
            try:
                start_time = float(segment.get('start', 0))
                end_time = float(segment.get('end', start_time + 1))
                speaker_name = (segment.get('speaker') or '').strip()

                if not speaker_name or speaker_name in ('Unknown', 'Unknown Speaker'):
                    continue

                duration = end_time - start_time
                if duration < 0.5:
                    logger.debug(f"Segment too short for embedding: {duration:.2f}s")
                    continue

                crop = crop_waveform(waveform, sample_rate, start_time, end_time)
                if crop is None:
                    logger.warning(f"Segment bounds invalid: {start_time}-{end_time}")
                    continue

                vector = extract_embedding_from_audio(crop, sample_rate)
                if vector is None:
                    results["failed_extractions"] += 1
                    continue

                vectors_by_speaker.setdefault(speaker_name, []).append(vector)
                results["processed_segments"] += 1
                if speaker_name not in results["updated_speakers"]:
                    results["updated_speakers"].append(speaker_name)
                results["improvements"].append({
                    "segment_start": start_time,
                    "segment_end": end_time,
                    "speaker": speaker_name,
                    "duration": duration,
                    "status": "embedding_updated"
                })

            except Exception as e:
                logger.warning(f"Failed to process segment {segment.get('start', 0)}: {e}")
                results["failed_extractions"] += 1
                continue

        with _speaker_db_lock:
            for speaker_name, vectors in vectors_by_speaker.items():
                database.enroll_speaker(
                    speaker_name, vectors, confidence=0.9, source_type="corrected"
                )

        return results

    try:
        results = await asyncio.to_thread(_reprocess)

        logger.info(f"Reprocessed embeddings for {results['processed_segments']} segments")
        return {
            "status": "reprocessing_complete",
            "results": results,
            "message": (
                f"Reprocessed {results['processed_segments']} segments for "
                f"{len(results['updated_speakers'])} speakers"
            )
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing segment embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def load_audio_from_bytes(audio_data: bytes) -> tuple[torch.Tensor, int]:
    """Load audio from bytes without writing to disk."""
    try:
        # Create a BytesIO stream from the audio data
        audio_stream = io.BytesIO(audio_data)

        # Decode via audio_io (soundfile first, torchaudio fallback)
        waveform, sample_rate = load_audio(audio_stream)

        return waveform, sample_rate

    except Exception as e:
        logger.error(f"Failed to load audio from bytes: {e}")
        raise ValueError(f"Invalid audio format: {e}")


# ---------------------------------------------------------------------------
# Transcription pipeline (faster-whisper)
#
# faster-whisper streams arbitrary-length audio internally and runs Silero VAD on
# it, so there is no manual chunking, no overlap stitching and no whole-clip
# energy gate here. Segment quality is controlled by faster-whisper's own decoder
# statistics (no_speech_prob / avg_logprob / compression_ratio).
# ---------------------------------------------------------------------------

# Characters that Whisper commonly emits when fed key clicks, fan noise, etc.
_KEYBOARD_ARTIFACT_CHARS = frozenset(
    "あいうえお"      # Japanese kana from key press transients
    "なにぬねの"
    "かきくけこ"
    "。、んして"
    "ㅏㅓㅗㅜㅡ"      # Korean jamo
    "ăâêôơ"          # Vietnamese
    "ตากนม"          # Thai
)


def _artifact_reason(text: str, confident: bool = False) -> Optional[str]:
    """
    Cheap textual artifact checks preserved from the previous implementation.

    Returns a reason string when the text looks like a decoder artifact rather
    than speech, otherwise None.

    `confident` is set when the decoder itself reported a high-probability decode;
    in that case a very short output ("Hi.", "OK.") is taken at face value instead
    of being discarded as a hallucination.
    """
    clean = (text or "").strip()
    if not clean:
        return "empty_text"

    alnum = "".join(ch for ch in clean if ch.isalnum())

    # Very short outputs are usually hallucinations on noise - unless the decoder
    # was confident, in which case they are real short utterances.
    if len(alnum) <= 2 and not confident:
        return "too_short"

    artifact_count = sum(1 for ch in clean if ch in _KEYBOARD_ARTIFACT_CHARS)
    scored_chars = sum(
        1 for ch in clean if ch.isalnum() or ch in _KEYBOARD_ARTIFACT_CHARS
    )
    if scored_chars > 0 and artifact_count / scored_chars > 0.7:
        return "keyboard_artifacts"

    # The same one or two characters repeated is noise, not speech.
    if len(alnum) > 3 and len(set(alnum)) <= 2:
        return "character_repetition"

    return None


def _hallucination_reason(segment: Any) -> Optional[str]:
    """
    Uniform hallucination gate applied to EVERY faster-whisper segment.

    Drops a segment when:
      * the model is confident there is no speech AND the decode was low
        probability (no_speech_prob > 0.6 and avg_logprob < -1.0), or
      * the output is pathologically repetitive (compression_ratio > 2.4).
    """
    no_speech_prob = getattr(segment, "no_speech_prob", None)
    avg_logprob = getattr(segment, "avg_logprob", None)
    compression_ratio = getattr(segment, "compression_ratio", None)

    if (
        no_speech_prob is not None
        and avg_logprob is not None
        and no_speech_prob > NO_SPEECH_PROB_THRESHOLD
        and avg_logprob < AVG_LOGPROB_THRESHOLD
    ):
        return "no_speech"

    if compression_ratio is not None and compression_ratio > COMPRESSION_RATIO_THRESHOLD:
        return "repetitive_output"

    return None


def _float_or_none(value: Any) -> Optional[float]:
    """Coerce to float, preserving a legitimate 0.0 and mapping bad input to None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_words(segment: Any) -> List[Dict[str, Any]]:
    """Convert faster-whisper word objects into plain dicts with valid timings."""
    words: List[Dict[str, Any]] = []
    for word in getattr(segment, "words", None) or []:
        start = _float_or_none(getattr(word, "start", None))
        end = _float_or_none(getattr(word, "end", None))
        text = getattr(word, "word", None)
        if start is None or end is None or not text:
            continue
        if end < start:
            end = start
        words.append({
            "start": start,
            "end": end,
            "word": text,
            "probability": _float_or_none(getattr(word, "probability", None)) or 0.0,
        })
    return words


# Vocabulary file cache: (mtime, terms). Re-read only when the file changes.
_vocab_cache: tuple = (None, [])
# Speaker-name cache: (fetched_at_monotonic, names). get_all_speakers() walks every
# record, so refresh at most once per _PROMPT_NAMES_TTL_SECONDS rather than per chunk.
_prompt_names_cache: tuple = (0.0, [])


def _load_vocab_terms() -> List[str]:
    """Read the vocabulary file (one term per line, '#' comments), mtime-cached."""
    global _vocab_cache
    if not VOCAB_FILE:
        return []
    try:
        mtime = os.path.getmtime(VOCAB_FILE)
    except OSError:
        return []  # absent file simply means no custom vocabulary
    if _vocab_cache[0] == mtime:
        return _vocab_cache[1]
    try:
        with open(VOCAB_FILE, "r", encoding="utf-8") as handle:
            terms = [
                line.strip()
                for line in handle
                if line.strip() and not line.strip().startswith("#")
            ]
    except OSError as e:
        logger.warning(f"Could not read vocabulary file {VOCAB_FILE}: {e}")
        return _vocab_cache[1]
    _vocab_cache = (mtime, terms)
    logger.info(f"Loaded {len(terms)} vocabulary terms from {VOCAB_FILE}")
    return terms


_AUTO_SPEAKER_NAME = re.compile(r"^(speaker[\s_]|speaker$|auto_speaker)", re.IGNORECASE)


def _speaker_prompt_names() -> List[str]:
    """
    Display names of known speakers, most recently heard first, capped at
    _PROMPT_MAX_NAMES. Auto-generated placeholders ("Speaker 3", "Speaker
    SPEAKER_00") are skipped - they carry no spelling information. TTL-cached.
    """
    global _prompt_names_cache
    if not PROMPT_SPEAKER_NAMES or enhanced_speaker_database is None:
        return []
    now = time.monotonic()
    if now - _prompt_names_cache[0] < _PROMPT_NAMES_TTL_SECONDS:
        return _prompt_names_cache[1]
    try:
        speakers = enhanced_speaker_database.get_all_speakers()
    except Exception as e:
        logger.warning(f"Could not list speakers for prompt: {e}")
        return _prompt_names_cache[1]
    named = [
        s for s in speakers
        if s.get("display_name") and not _AUTO_SPEAKER_NAME.match(s["display_name"].strip())
    ]
    named.sort(key=lambda s: s.get("last_seen") or "", reverse=True)
    names = [s["display_name"].strip() for s in named[:_PROMPT_MAX_NAMES]]
    _prompt_names_cache = (now, names)
    return names


def _build_initial_prompt() -> Optional[str]:
    """
    Assemble the decode-biasing prompt, or None when there is nothing to bias
    with. Kept under _PROMPT_MAX_CHARS so Whisper's own 224-token truncation
    never cuts a term in half.
    """
    if INITIAL_PROMPT_OVERRIDE:
        return INITIAL_PROMPT_OVERRIDE[:_PROMPT_MAX_CHARS]

    terms = list(dict.fromkeys(_load_vocab_terms() + _speaker_prompt_names()))
    if not terms:
        return None

    prompt = "Glossary: " + ", ".join(terms) + "."
    while len(prompt) > _PROMPT_MAX_CHARS and terms:
        terms.pop()  # drop least-recent names / last vocab lines first
        prompt = "Glossary: " + ", ".join(terms) + "."
    return prompt if terms else None


def _transcribe_sync(
    audio_array: "np.ndarray", language=_LANGUAGE_DEFAULT, accuracy: bool = False
) -> Dict[str, Any]:
    """
    Blocking faster-whisper call. Runs on a worker thread and holds the ASR lock
    for its whole duration so concurrent requests queue rather than interleave.

    language: a Whisper language code pins the decode; None forces auto-detection;
    the _LANGUAGE_DEFAULT sentinel (i.e. caller said nothing) uses the server-wide
    WHISPER_LANGUAGE default.

    accuracy: spend more compute for a better transcript (offline file path):
    higher beam size, and the OREJA_FILE_MODEL model when configured.
    """
    if whisper_model is None:
        raise ValueError("Whisper model not loaded")

    effective_language = WHISPER_LANGUAGE if language is _LANGUAGE_DEFAULT else language
    beam_size = WHISPER_FILE_BEAM_SIZE if accuracy else WHISPER_BEAM_SIZE

    # Bias the decoder toward known names and domain vocabulary. Live 3-15s chunks
    # fit one 30s decode window, so the prompt covers the whole chunk; on long files
    # it applies to the first window (condition_on_previous_text=False drops it for
    # later windows, which is the accepted trade for hallucination-loop safety).
    initial_prompt = _build_initial_prompt()

    with _whisper_lock:
        # Model choice must happen under the lock: the accuracy model lazy-loads on
        # first use, and the lock is what makes that load race-free.
        model = _get_file_whisper_model() if accuracy else whisper_model
        with torch.inference_mode():
            segment_iter, info = model.transcribe(
                audio_array,
                language=effective_language,        # None => auto-detect, for ALL lengths
                task="transcribe",
                beam_size=beam_size,
                word_timestamps=True,               # required for speaker attribution
                condition_on_previous_text=False,   # stops cross-segment hallucination loops
                initial_prompt=initial_prompt,      # None when there is nothing to bias with
                vad_filter=True,                    # Silero VAD, built into faster-whisper
                vad_parameters={"min_silence_duration_ms": VAD_MIN_SILENCE_MS},
            )

            chunks: List[Dict[str, Any]] = []
            dropped: Dict[str, int] = {}

            # segment_iter is a generator: consumption is what actually decodes.
            for segment in segment_iter:
                text = (getattr(segment, "text", "") or "").strip()

                seg_logprob = _float_or_none(getattr(segment, "avg_logprob", None))
                confident = seg_logprob is not None and seg_logprob >= CONFIDENT_LOGPROB

                reason = _hallucination_reason(segment) or _artifact_reason(text, confident)
                if reason:
                    dropped[reason] = dropped.get(reason, 0) + 1
                    logger.debug(f"Dropped segment ({reason}): '{text[:60]}'")
                    continue

                words = _extract_words(segment)

                start = _float_or_none(getattr(segment, "start", None))
                end = _float_or_none(getattr(segment, "end", None))
                if start is None:
                    start = words[0]["start"] if words else 0.0
                if end is None:
                    end = words[-1]["end"] if words else start

                chunks.append({
                    # "timestamp" keeps the legacy (transformers-style) shape that
                    # merge_transcription_and_diarization and batch_transcription.py read.
                    "timestamp": [start, end],
                    "start": start,
                    "end": end,
                    "text": text,
                    "words": words,
                    "avg_logprob": _float_or_none(getattr(segment, "avg_logprob", None)),
                    "no_speech_prob": _float_or_none(getattr(segment, "no_speech_prob", None)),
                    "compression_ratio": _float_or_none(
                        getattr(segment, "compression_ratio", None)
                    ),
                })

    full_text = " ".join(chunk["text"] for chunk in chunks if chunk["text"]).strip()

    if dropped:
        logger.info(f"Hallucination gate dropped segments: {dropped}")

    return {
        "chunks": chunks,
        "text": full_text,
        "language": getattr(info, "language", None),
        "language_probability": _float_or_none(
            getattr(info, "language_probability", None)
        ),
        "dropped_segments": dropped,
    }


async def run_transcription(
    waveform: torch.Tensor, sample_rate: int, language=_LANGUAGE_DEFAULT,
    accuracy: bool = False,
) -> Dict[str, Any]:
    """
    Transcribe a waveform with faster-whisper.

    Audio of any length is handed to the model as-is: faster-whisper does its own
    windowing and VAD. The only pre-check is a cheap RMS silence fast-path.

    language: a Whisper language code pins the decode; None forces auto-detection;
    omitted (the _LANGUAGE_DEFAULT sentinel) uses the server-wide default, which
    keeps existing callers (batch_transcription.py, enhanced integrations) unchanged.

    Returns a dict with 'chunks' (each carrying 'timestamp', 'text' and word-level
    timings) and 'text'. When nothing survives, 'skipped_reason' is present.
    """
    try:
        if whisper_model is None:
            raise ValueError("Whisper model not loaded")

        # faster-whisper expects 16 kHz mono float32.
        if sample_rate and sample_rate != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(waveform)
            sample_rate = SAMPLE_RATE
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        audio_array = waveform.detach().cpu().to(torch.float32).numpy().flatten()

        duration = len(audio_array) / sample_rate if sample_rate else 0.0
        rms_energy = (
            float(np.sqrt(np.mean(np.square(audio_array)))) if audio_array.size else 0.0
        )

        # Fast path only: digital silence / near-silence. The threshold is far below
        # speech level on purpose - the old 0.015 whole-clip gate discarded real speech.
        if rms_energy < SILENCE_RMS_THRESHOLD:
            logger.info(
                f"Clip is silent (RMS {rms_energy:.6f} < {SILENCE_RMS_THRESHOLD}), skipping ASR"
            )
            return {
                "chunks": [],
                "text": "",
                "processing_time": 0.0,
                "skipped_reason": "silence",
            }

        logger.info(
            f"Transcribing {duration:.2f}s of audio with faster-whisper (RMS {rms_energy:.4f})"
        )
        if not accuracy and duration > 20.0:
            # Live clients cap chunks at ~15s; a bigger one means an old client draining
            # a backlog as a single request - decode time scales with it, so flag it.
            logger.warning(
                f"Live chunk is {duration:.1f}s (expected <= ~15s) - outdated client or backlog?"
            )

        # Off the event loop; the ASR lock inside serializes concurrent requests.
        decode_started = time.perf_counter()
        result = await asyncio.to_thread(_transcribe_sync, audio_array, language, accuracy)
        decode_elapsed = time.perf_counter() - decode_started

        # Includes time queued on the ASR lock. Logged unconditionally so a grinding decode
        # is diagnosable from the log afterwards; warn when slower than real time, which on
        # this hardware means something is badly wrong (typically VRAM oversubscribed by
        # other GPU apps, forcing WDDM to page CUDA memory through system RAM).
        logger.info(f"faster-whisper finished {duration:.2f}s of audio in {decode_elapsed:.2f}s")
        if decode_elapsed > max(10.0, duration):
            logger.warning(
                f"ASR ran slower than real time ({decode_elapsed:.1f}s for {duration:.1f}s of "
                "audio). GPU is likely paging into system RAM - check free VRAM "
                "(other GPU-heavy apps: video calls, browsers, games)."
            )

        if not result["chunks"]:
            logger.info("No speech survived VAD / hallucination gating")
            result["skipped_reason"] = "no_speech_detected"
        else:
            preview = result["text"][:60]
            logger.info(
                f"Transcription produced {len(result['chunks'])} segments "
                f"(language={result.get('language')}): '{preview}"
                f"{'...' if len(result['text']) > 60 else ''}'"
            )

        return result

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise


def _diarize_sync(
    waveform: torch.Tensor, sample_rate: int, max_speakers: Optional[int] = None
) -> Any:
    """Blocking pyannote call, serialized by the diarization lock."""
    if diarization_pipeline is None:
        raise ValueError("Diarization pipeline not loaded")

    audio_data = {"waveform": waveform, "sample_rate": sample_rate}
    # max_speakers (not num_speakers) is the right roster hint for chunked live audio:
    # a 5-second chunk usually contains a subset of the session's speakers, so forcing
    # an exact count would be wrong, but the ceiling stops pyannote from inventing
    # phantom extra speakers.
    kwargs = {"max_speakers": max_speakers} if max_speakers else {}
    with _diarization_lock:
        diarize_started = time.perf_counter()
        with torch.inference_mode():
            result = diarization_pipeline(audio_data, **kwargs)
        if torch.cuda.is_available():
            # Return pyannote's transient CUDA cache to the driver after each chunk.
            # torch's caching allocator otherwise holds its peak forever, and on a
            # 10 GB card shared with Zoom/browsers that hoarded headroom is what
            # pushes the NEXT allocation into WDDM system-RAM paging (100x slower).
            # ctranslate2 (whisper) doesn't use torch's allocator, so this only
            # releases diarization's scratch memory - a few ms, well worth it.
            torch.cuda.empty_cache()
        logger.info(
            f"Diarization finished in {time.perf_counter() - diarize_started:.2f}s"
            + (f" (max_speakers={max_speakers})" if max_speakers else "")
        )
    # pyannote.audio 4.x returns a DiarizeOutput dataclass whose Annotation
    # (the object with .itertracks) lives on .speaker_diarization; 3.x returned
    # the Annotation directly. Unwrap here so every downstream consumer keeps
    # working against the Annotation API.
    return getattr(result, "speaker_diarization", result)


async def run_diarization(
    waveform: torch.Tensor, sample_rate: int, max_speakers: Optional[int] = None
) -> Any:
    """Run speaker diarization on the full waveform, off the event loop."""
    try:
        if diarization_pipeline is None:
            raise ValueError("Diarization pipeline not loaded")

        return await asyncio.to_thread(_diarize_sync, waveform, sample_rate, max_speakers)

    except Exception as e:
        logger.error(f"Diarization error: {e}")
        raise


# ---------------------------------------------------------------------------
# Speaker attribution
# ---------------------------------------------------------------------------

def extract_diarization_turns(diarization: Any) -> List[Dict[str, Any]]:
    """
    Flatten a pyannote annotation into a time-sorted list of
    {'start', 'end', 'speaker'} turns. Speaker labels use the legacy
    "Speaker SPEAKER_00" presentation form.
    """
    turns: List[Dict[str, Any]] = []
    if diarization is None:
        return turns

    try:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = _float_or_none(getattr(turn, "start", None))
            end = _float_or_none(getattr(turn, "end", None))
            if start is None or end is None or end <= start:
                continue
            turns.append({"start": start, "end": end, "speaker": f"Speaker {speaker}"})
    except Exception as e:
        logger.warning(f"Could not read diarization turns: {e}")
        return []

    turns.sort(key=lambda t: t["start"])
    return turns


def speaker_at_time(turns: List[Dict[str, Any]], moment: float, default: str) -> str:
    """
    Speaker of the turn covering `moment`; if no turn covers it, the nearest turn.

    Nearest-turn fallback matters because Whisper word timings drift by tens of
    milliseconds against diarization boundaries, which would otherwise leave
    boundary words unattributed.
    """
    if not turns:
        return default

    best_speaker = None
    best_distance = None

    for turn in turns:
        if turn["start"] <= moment <= turn["end"]:
            return turn["speaker"]
        if moment < turn["start"]:
            distance = turn["start"] - moment
        else:
            distance = moment - turn["end"]
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_speaker = turn["speaker"]

    return best_speaker if best_speaker is not None else default


def _chunk_bounds(chunk: Dict[str, Any]) -> tuple:
    """
    Read (start, end) from a transcription chunk.

    Uses explicit None checks so a legitimate 0.0 timestamp is never treated as
    "missing" - the classic falsy-zero bug this pipeline used to have.
    """
    start = None
    end = None

    timestamp = chunk.get("timestamp")
    if isinstance(timestamp, (list, tuple)):
        if len(timestamp) > 0 and timestamp[0] is not None:
            start = _float_or_none(timestamp[0])
        if len(timestamp) > 1 and timestamp[1] is not None:
            end = _float_or_none(timestamp[1])

    if start is None and chunk.get("start") is not None:
        start = _float_or_none(chunk.get("start"))
    if end is None and chunk.get("end") is not None:
        end = _float_or_none(chunk.get("end"))

    return start, end


def _coalesce_speaker_groups(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge adjacent groups that ended up with the same speaker."""
    coalesced: List[Dict[str, Any]] = []
    for group in groups:
        if coalesced and coalesced[-1]["speaker"] == group["speaker"]:
            coalesced[-1]["words"].extend(group["words"])
        else:
            coalesced.append(group)
    return coalesced


def _smooth_speaker_groups(
    groups: List[Dict[str, Any]], min_words: int = 1, min_duration: float = 0.4
) -> List[Dict[str, Any]]:
    """
    Absorb tiny single-word speaker flips back into their neighbours.

    A one-word 200 ms "speaker change" between two runs of the same speaker is
    almost always diarization jitter, not a real turn.
    """
    if len(groups) < 3:
        return groups

    smoothed = [groups[0]]
    for index in range(1, len(groups) - 1):
        current = groups[index]
        following = groups[index + 1]

        duration = current["words"][-1]["end"] - current["words"][0]["start"]
        is_tiny = len(current["words"]) <= min_words and duration < min_duration

        if is_tiny and smoothed[-1]["speaker"] == following["speaker"]:
            # Reassign the stray word to the surrounding speaker.
            current = {"speaker": smoothed[-1]["speaker"], "words": current["words"]}

        smoothed.append(current)

    smoothed.append(groups[-1])
    return _coalesce_speaker_groups(smoothed)


def split_chunk_by_speaker(
    chunk: Dict[str, Any],
    turns: List[Dict[str, Any]],
    diarization: Any = None,
) -> List[Dict[str, Any]]:
    """
    Turn one ASR chunk into one or more per-speaker segments.

    Each word is assigned to the diarization turn covering its midpoint, then the
    chunk is split wherever the word-level speaker changes. Segment start/end come
    from the first/last word of each run, so a segment that spans a speaker change
    no longer gets mis-attributed wholesale to whoever owned its midpoint.
    """
    text = (chunk.get("text") or "").strip()
    start, end = _chunk_bounds(chunk)
    words = chunk.get("words") or []

    has_diarization = diarization is not None
    default_speaker = "SPEAKER_00"

    # No word timings (legacy transcription payloads): fall back to segment-level
    # attribution over the whole chunk.
    if not words:
        if not text:
            return []
        if turns:
            midpoint_start = start if start is not None else 0.0
            midpoint_end = end if end is not None else midpoint_start
            speaker = speaker_at_time(
                turns, (midpoint_start + midpoint_end) / 2.0, default_speaker
            )
        elif has_diarization:
            speaker = find_speaker_for_segment(
                diarization,
                start if start is not None else 0.0,
                end if end is not None else 0.0,
            )
        else:
            speaker = default_speaker

        return [{
            "start": start if start is not None else 0.0,
            "end": end if end is not None else (start if start is not None else 0.0),
            "text": text,
            "speaker": speaker,
            "embedding_confidence": 0.0,
            "embedding_speaker": None,
            "diarization_speaker": speaker,
        }]

    # Word-level attribution.
    if turns:
        groups: List[Dict[str, Any]] = []
        for word in words:
            midpoint = (word["start"] + word["end"]) / 2.0
            speaker = speaker_at_time(turns, midpoint, default_speaker)
            if groups and groups[-1]["speaker"] == speaker:
                groups[-1]["words"].append(word)
            else:
                groups.append({"speaker": speaker, "words": [word]})
        groups = _smooth_speaker_groups(groups)
    else:
        single_speaker = (
            find_speaker_for_segment(
                diarization,
                start if start is not None else words[0]["start"],
                end if end is not None else words[-1]["end"],
            )
            if has_diarization
            else default_speaker
        )
        groups = [{"speaker": single_speaker, "words": list(words)}]

    segments: List[Dict[str, Any]] = []
    for group in groups:
        group_text = "".join(word["word"] for word in group["words"]).strip()
        if not group_text:
            continue
        segments.append({
            "start": group["words"][0]["start"],
            "end": group["words"][-1]["end"],
            "text": group_text,
            "speaker": group["speaker"],
            "embedding_confidence": 0.0,
            "embedding_speaker": None,
            "diarization_speaker": group["speaker"],
            "words": group["words"],
        })

    # If word text was somehow empty but the chunk had text, keep the chunk.
    if not segments and text:
        speaker = groups[0]["speaker"] if groups else default_speaker
        segments.append({
            "start": start if start is not None else words[0]["start"],
            "end": end if end is not None else words[-1]["end"],
            "text": text,
            "speaker": speaker,
            "embedding_confidence": 0.0,
            "embedding_speaker": None,
            "diarization_speaker": speaker,
        })

    return segments


def merge_transcription_and_diarization(
    transcription: Dict[str, Any],
    diarization: Any,
    waveform: torch.Tensor = None,
    sample_rate: int = None
) -> List[Dict[str, Any]]:
    """
    Merge Whisper transcription with pyannote diarization using word-level timings.

    Every word is attributed to a diarization turn; ASR segments are split at
    speaker changes. Labels are the anonymous per-session "Speaker SPEAKER_XX";
    identify_speakers_hook() replaces them with persistent names afterwards.
    """
    try:
        if not transcription:
            return []

        turns = extract_diarization_turns(diarization)
        chunks = transcription.get("chunks")

        segments: List[Dict[str, Any]] = []

        if chunks is not None:
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                try:
                    segments.extend(split_chunk_by_speaker(chunk, turns, diarization))
                except Exception as e:
                    logger.warning(f"Skipping malformed transcription chunk: {e}")
                    continue
        else:
            # Fallback for transcription payloads with no per-segment structure.
            text = (transcription.get("text") or "").strip()
            if text:
                total_duration = 0.0
                if waveform is not None and sample_rate:
                    total_duration = waveform.shape[-1] / sample_rate
                speaker = (
                    speaker_at_time(turns, total_duration / 2.0, "SPEAKER_00")
                    if turns
                    else "SPEAKER_00"
                )
                segments.append({
                    "start": 0.0,
                    "end": total_duration,
                    "text": text,
                    "speaker": speaker,
                    "embedding_confidence": 0.0,
                    "embedding_speaker": None,
                    "diarization_speaker": speaker,
                })

        # Voiceprint naming happens once per request in identify_speakers_hook(),
        # not per segment here.
        return segments

    except Exception as e:
        logger.error(f"Error merging transcription and diarization: {e}")
        # Fallback: return basic transcription without speaker info
        text = transcription.get("text", "") if transcription else ""
        return [{
            "start": 0.0,
            "end": 0.0,
            "text": text,
            "speaker": "SPEAKER_00",
            "embedding_confidence": 0.0,
            "embedding_speaker": None,
            "diarization_speaker": "SPEAKER_00"
        }] if text else []


def _speaker_audio_groups(
    segments: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group segments by the per-session label diarization gave them.

    Identification is done per *speaker*, not per segment: a 0.8s "yeah" gives a
    useless voiceprint on its own, but pooled with the rest of that speaker's
    turns in the same request it is part of a solid one.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for segment in segments:
        label = segment.get("diarization_speaker") or segment.get("speaker")
        if not label:
            continue
        groups.setdefault(label, []).append(segment)
    return groups


def _concatenate_speaker_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    segments: List[Dict[str, Any]],
    max_seconds: float = MAX_SPEAKER_AUDIO_SECONDS,
) -> Optional[torch.Tensor]:
    """Concatenate a speaker's crops, in order, up to `max_seconds`."""
    budget = int(max_seconds * sample_rate)
    crops = []
    collected = 0

    for segment in segments:
        if collected >= budget:
            break
        start = _float_or_none(segment.get("start"))
        end = _float_or_none(segment.get("end"))
        if start is None or end is None or end <= start:
            continue

        crop = crop_waveform(waveform, sample_rate, start, end)
        if crop is None:
            continue

        remaining = budget - collected
        if crop.shape[-1] > remaining:
            crop = crop[..., :remaining]
        crops.append(crop.reshape(-1))
        collected += crop.shape[-1]

    if not crops:
        return None
    return torch.cat(crops)


def _register_unmatched_embedding(embedding, scope: Optional[str] = None) -> bool:
    """
    Record a voiceprint that matched no known speaker, and report whether it has
    now recurred often enough to deserve its own database record.

    The live path posts a 5-second chunk every 5 seconds. Minting a speaker on
    the first miss meant a 30-minute call could create hundreds of one-chunk
    "Speaker N" records - each one rewriting speaker_records.json plus every .npy
    bank, and each one landing in the WPF client's persisted speaker list. Making
    the voice prove itself across AUTO_CREATE_AFTER_MISSES separate misses filters
    out the transient noise/crosstalk crops without losing a genuine new speaker,
    who reappears in the very next chunk.

    Returns True when the caller should now create the speaker (the pending entry
    is consumed), False when it should keep the diarization label for now.
    """
    if AUTO_CREATE_AFTER_MISSES <= 1:
        return True

    with _pending_unmatched_lock:
        for entry in _pending_unmatched:
            # A pending miss only accumulates within its own source scope, so a
            # far-end voice cannot ride a mic voice's miss count into a record.
            if entry[2] != scope:
                continue
            try:
                similarity = cosine_similarity(entry[0], embedding)
            except Exception:
                similarity = 0.0
            if similarity >= SPEAKER_MATCH_THRESHOLD:
                entry[1] += 1
                if entry[1] >= AUTO_CREATE_AFTER_MISSES:
                    _pending_unmatched.remove(entry)
                    return True
                # Keep the most recent observation of this voice as the probe.
                entry[0] = embedding
                return False

        _pending_unmatched.append([embedding, 1, scope])
        # Bounded buffer: drop the oldest pending voiceprints.
        overflow = len(_pending_unmatched) - _MAX_PENDING_UNMATCHED
        if overflow > 0:
            del _pending_unmatched[:overflow]
        return False


def identify_speakers_hook(
    waveform: torch.Tensor,
    sample_rate: int,
    segments: List[Dict[str, Any]],
    diarization: Any = None,
    source_scope: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Voiceprint-based speaker naming, run once per request after diarization.

    For each diarization speaker in this request: concatenate that speaker's
    audio (up to MAX_SPEAKER_AUDIO_SECONDS), extract ONE embedding, and match it
    against the persistent v2 database. On a match the anonymous
    "Speaker SPEAKER_00" label is replaced with the stored display name; below
    threshold the database creates a new auto speaker so the same voice is
    recognised next time.

    Two audio floors apply per speaker per request:
      * below MIN_IDENTIFY_AUDIO_SECONDS: not embedded at all, diarization label kept.
      * between the two floors: a READ-ONLY match is attempted (learn=False,
        auto_create=False). Enough to recognise a known voice - which keeps the
        label stable across chunks - without polluting the bank with a short crop.
      * at or above MIN_SPEAKER_AUDIO_SECONDS: the match also reinforces the
        voiceprint, and a persistently unmatched voice can earn a new record.

    New "Speaker N" records are NOT minted on the first miss: the unmatched
    voiceprint must recur AUTO_CREATE_AFTER_MISSES times first (see
    _register_unmatched_embedding), otherwise a long call mints a throwaway
    speaker per chunk and rewrites the whole database each time.

    Degrades to a pass-through (diarization-only labels) when the embedding model
    or the database is unavailable. It must never raise: the transcript is worth
    more than the speaker names.

    Segments are mutated in place and returned in the same order, with
    "speaker_id" and "embedding_confidence" added, keeping the response schema
    backward compatible with the WPF client.
    """
    if not segments:
        return segments

    if embedding_model is None or enhanced_speaker_database is None:
        logger.debug(
            "Speaker identification unavailable "
            f"(embedding_model={embedding_model is not None}, "
            f"database={enhanced_speaker_database is not None}) - keeping diarization labels"
        )
        return segments

    if waveform is None or not sample_rate:
        return segments

    try:
        database = enhanced_speaker_database
        learn_min_samples = int(MIN_SPEAKER_AUDIO_SECONDS * sample_rate)
        identify_min_samples = int(MIN_IDENTIFY_AUDIO_SECONDS * sample_rate)
        identified = 0

        for label, label_segments in _speaker_audio_groups(segments).items():
            try:
                audio = _concatenate_speaker_audio(waveform, sample_rate, label_segments)
                if audio is None or audio.numel() < identify_min_samples:
                    logger.debug(
                        f"Speaker '{label}': "
                        f"{0 if audio is None else audio.numel() / sample_rate:.2f}s "
                        f"< {MIN_IDENTIFY_AUDIO_SECONDS}s - not identified"
                    )
                    continue

                embedding = extract_embedding_from_audio(audio, sample_rate)
                if embedding is None:
                    continue

                # Only crops at or above the learning floor may reinforce a
                # voiceprint or mint a new speaker record.
                enough_to_learn = audio.numel() >= learn_min_samples

                with _speaker_db_lock:
                    speaker_id, display_name, confidence = database.identify_speaker(
                        embedding,
                        auto_create=False,
                        learn=enough_to_learn,
                        save=enough_to_learn,
                        scope=source_scope,
                    )

                if (speaker_id is None and enough_to_learn
                        and _register_unmatched_embedding(embedding, source_scope)):
                    # This voice has now failed to match often enough to be worth
                    # a record of its own.
                    with _speaker_db_lock:
                        speaker_id, display_name, confidence = database.identify_speaker(
                            embedding, auto_create=True, learn=True, save=True,
                            scope=source_scope,
                        )

                if speaker_id is None:
                    # Unmatched (or too short to learn from): keep the per-request
                    # diarization label rather than inventing a speaker.
                    continue

                for segment in label_segments:
                    segment["speaker"] = display_name
                    segment["speaker_id"] = speaker_id
                    segment["embedding_speaker"] = display_name
                    segment["embedding_confidence"] = float(confidence)

                identified += 1
                logger.info(
                    f"Speaker '{label}' -> '{display_name}' ({speaker_id}) "
                    f"confidence {confidence:.3f}"
                )

            except Exception as e:
                # One bad speaker must not cost the whole request its labels.
                logger.warning(f"Speaker identification failed for '{label}': {e}")
                continue

        if identified:
            logger.debug(f"Identified {identified} speaker(s) against the v2 database")

        return segments

    except Exception as e:
        logger.warning(f"Speaker identification hook failed, keeping diarization labels: {e}")
        return segments


def find_speaker_for_segment(diarization: Any, start_time: float, end_time: float) -> str:
    """
    Segment-level speaker lookup (midpoint, then maximum overlap).

    Retained as a fallback for transcription payloads that carry no word timings;
    the primary path is word-level attribution in split_chunk_by_speaker().
    """
    try:
        # Validate input parameters
        if start_time is None or end_time is None:
            logger.warning("Invalid segment timing: start_time or end_time is None")
            return "Unknown Speaker"

        # Calculate the midpoint of the segment
        mid_time = (start_time + end_time) / 2

        # Find which speaker is active at the midpoint
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            # Check if segment has valid timing
            if segment.start is None or segment.end is None:
                continue

            if segment.start <= mid_time <= segment.end:
                return f"Speaker {speaker}"

        # If no speaker found, use overlap analysis
        speaker_durations = {}
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            # Check if segment has valid timing
            if segment.start is None or segment.end is None:
                continue

            overlap_start = max(segment.start, start_time)
            overlap_end = min(segment.end, end_time)

            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                speaker_label = f"Speaker {speaker}"
                speaker_durations[speaker_label] = speaker_durations.get(speaker_label, 0) + overlap_duration

        if speaker_durations:
            # Return the speaker with the most overlap
            return max(speaker_durations, key=speaker_durations.get)

        return "Unknown Speaker"

    except Exception as e:
        logger.warning(f"Error finding speaker: {e}")
        return "Unknown Speaker"


@app.get("/speakers/enhanced_stats")
async def get_enhanced_speaker_statistics():
    """
    Get comprehensive statistics from the enhanced speaker database
    
    Returns:
        Detailed speaker statistics including migration status
    """
    if enhanced_speaker_integration is None:
        raise HTTPException(status_code=503, detail="Enhanced speaker integration not available")
    
    try:
        # Synchronous body (disk + numpy) - keep it off the event loop.
        stats = await asyncio.to_thread(enhanced_speaker_integration.get_enhanced_speaker_stats)

        return {
            "status": "statistics_retrieved",
            "enhanced_database_active": True,
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get enhanced statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speakers/enhanced_feedback")
async def enhanced_speaker_feedback(corrections: Dict[str, str]):
    """
    Process speaker corrections using the enhanced database architecture
    
    This provides better handling of speaker merging, immutable IDs, and 
    separation of save vs feedback operations.
    
    Args:
        corrections: Mapping of old speaker IDs/names to new display names

    Returns:
        Enhanced feedback processing results
    """
    database = require_speaker_db()

    if not corrections:
        raise HTTPException(status_code=400, detail="No corrections supplied")

    try:
        if enhanced_speaker_integration is not None:
            # Same to_thread dispatch as the else-branch below: this writes
            # speaker_records.json and every .npy bank.
            results = await asyncio.to_thread(
                enhanced_speaker_integration.enhanced_speaker_correction_feedback, corrections
            )
        else:
            # The integration layer is a thin wrapper; go straight to the database.
            results = await asyncio.to_thread(database.send_feedback_for_learning, corrections)

        return {
            "status": "enhanced_feedback_complete",
            "results": results,
            "enhanced_processing": True,
            "message": f"Processed {len(corrections)} corrections using the v2 speaker database"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced feedback failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speakers/save_transcription_enhanced")
async def save_transcription_with_enhanced_corrections(
    transcription_data: dict,
    speaker_corrections: Dict[str, str],
    output_file: str = None
):
    """
    Save transcription with corrections using enhanced database - SEPARATE from feedback
    
    This addresses the confusion between saving files and sending learning feedback.
    This endpoint ONLY saves the file with corrections applied.
    
    Args:
        transcription_data: The transcription data to save
        speaker_corrections: Speaker name corrections to apply
        output_file: Optional output file path
        
    Returns:
        File save results WITHOUT triggering learning
    """
    if enhanced_speaker_integration is None:
        raise HTTPException(status_code=503, detail="Enhanced speaker integration not available")
    
    try:
        # Use enhanced database to save with corrections (no learning feedback).
        # Writes a JSON file, so dispatch it off the event loop.
        saved_file = await asyncio.to_thread(
            enhanced_speaker_integration.enhanced_db.save_transcription_with_corrections,
            transcription_data,
            speaker_corrections,
            output_file,
        )
        
        return {
            "status": "transcription_saved",
            "file_path": saved_file,
            "corrections_applied": len(speaker_corrections),
            "learning_feedback_sent": False,  # Explicitly false - separate operation
            "message": f"Transcription saved to {saved_file} with {len(speaker_corrections)} corrections. "
                      f"Use /speakers/enhanced_feedback to send learning feedback separately."
        }
        
    except Exception as e:
        logger.error(f"Enhanced transcription save failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/speakers/system_status")
async def get_speaker_system_status():
    """
    Status of the speaker identification system.

    There is one speaker system: the v2 database plus the pyannote embedding
    model. The legacy SpeechBrain manager has been removed, so "legacy_system"
    is reported as permanently unavailable for any old client still reading it.
    """
    try:
        database = enhanced_speaker_database

        speakers = database.get_all_speakers() if database is not None else []
        total_embeddings = sum(int(s['embedding_count']) for s in speakers)

        status = {
            "legacy_system": {
                "available": False,
                "speaker_count": 0,
                "total_embeddings": 0,
                "note": "Legacy SpeechBrain speaker manager removed - v2 database is the only store"
            },
            "enhanced_system": {
                "available": database is not None,
                "speaker_count": len(speakers),
                "total_embeddings": total_embeddings,
                "enrolled_speakers": sum(1 for s in speakers if s['is_enrolled']),
                "verified_speakers": sum(1 for s in speakers if s['is_verified']),
                "auto_speakers": sum(1 for s in speakers if s['source_type'] == 'auto'),
                "migration_completed": True,
            },
            "identification": {
                "embedding_model": EMBEDDING_MODEL,
                "embedding_model_loaded": embedding_model is not None,
                "match_threshold": SPEAKER_MATCH_THRESHOLD,
                "min_speaker_audio_seconds": MIN_SPEAKER_AUDIO_SECONDS,
                "max_speaker_audio_seconds": MAX_SPEAKER_AUDIO_SECONDS,
                "active": embedding_model is not None and database is not None,
            },
            "recommendations": []
        }

        if database is None:
            status["recommendations"].append("Speaker database failed to load - speaker naming is disabled")
        if embedding_model is None:
            status["recommendations"].append(
                "Embedding model not loaded - transcripts will carry diarization labels only"
            )
        if database is not None and embedding_model is not None:
            status["recommendations"].append("Persistent speaker identification active")
        if total_embeddings == 0 and len(speakers) > 0:
            status["recommendations"].append(
                "No voiceprints stored yet - enroll speakers or correct names on audio to start learning"
            )

        return {
            "status": "system_status_retrieved",
            "system_status": status,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        access_log=False
    ) 