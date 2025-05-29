"""
FastAPI server for Oreja audio transcription and diarization.
Processes audio in memory using local Hugging Face models with no cloud interaction.
"""

import asyncio
import io
import logging
import os
import time
from typing import List, Dict, Any, Optional
import warnings

import numpy as np
import torch
import torchaudio
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline, AutoProcessor, WhisperForConditionalGeneration
from pyannote.audio import Pipeline as DiarizationPipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import uvicorn

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
MAX_AUDIO_LENGTH = 30  # seconds
MIN_AUDIO_LENGTH = 0.1  # seconds
WHISPER_MODEL = "openai/whisper-large-v3-turbo"  # Latest model from October 2024 - faster with same accuracy as large-v3
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.0"
EMBEDDING_MODEL = "pyannote/embedding"

app = FastAPI(
    title="Oreja Audio Processing API",
    description="Local audio transcription and speaker diarization service",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
device = None
whisper_model = None
diarization_pipeline = None
embedding_model = None


def initialize_models():
    """Initialize all models and set up the device."""
    global device, whisper_model, diarization_pipeline, embedding_model
    
    # Set up device (GPU if available, otherwise CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Using device: cpu")
    
    logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
    whisper_model = pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL,
        device=device,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        return_timestamps=True,
        chunk_length_s=30,  # Optimal chunk length for quality
        stride_length_s=5,  # Overlap for better continuity
    )
    logger.info("✓ Whisper model loaded successfully")
    
    # Try to load pyannote.audio for speaker diarization
    try:
        logger.info(f"Loading diarization model: {DIARIZATION_MODEL}")
        hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        diarization_pipeline = DiarizationPipeline.from_pretrained(
            DIARIZATION_MODEL,
            use_auth_token=hf_token if hf_token else True
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
        embedding_model = PretrainedSpeakerEmbedding(
            EMBEDDING_MODEL,
            device=device
        )
        logger.info("✓ Embedding model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load embedding model: {e}")
        logger.warning("Continuing without speaker embeddings")
        embedding_model = None
    
    logger.info("Model loading completed")


@app.on_event("startup")
async def startup_event():
    """Initialize models when the server starts."""
    initialize_models()


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
        "device": device,
        "models": {
            "whisper": whisper_model is not None,
            "diarization": diarization_pipeline is not None,
            "embedding": embedding_model is not None
        },
        "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else "N/A"
    }


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Transcribe and diarize audio file.
    
    Args:
        audio: Audio file (WAV format preferred)
        
    Returns:
        Transcription result with speaker diarization
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
            logger.warning(f"Audio length {duration:.2f}s exceeds recommended {MAX_AUDIO_LENGTH}s")
        
        # Resample to 16kHz if needed
        if sample_rate != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
            waveform = resampler(waveform)
            sample_rate = SAMPLE_RATE
        
        # Ensure mono audio
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Run transcription and diarization concurrently (if available)
        transcription_task = asyncio.create_task(
            run_transcription(waveform, sample_rate)
        )
        
        if diarization_pipeline is not None:
            diarization_task = asyncio.create_task(
                run_diarization(waveform, sample_rate)
            )
            # Wait for both tasks to complete
            transcription_result, diarization_result = await asyncio.gather(
                transcription_task, diarization_task
            )
        else:
            # Only run transcription
            transcription_result = await transcription_task
            diarization_result = None
            logger.info("Diarization skipped - model not available")
        
        # Merge transcription with speaker information (if available)
        segments = merge_transcription_and_diarization(
            transcription_result, diarization_result
        )
        
        # Generate full text
        full_text = " ".join([segment["text"] for segment in segments])
        
        processing_time = time.time() - start_time
        
        result = {
            "segments": segments,
            "full_text": full_text,
            "processing_time": processing_time,
            "timestamp": time.time(),
            "audio_duration": duration,
            "sample_rate": sample_rate
        }
        
        logger.info(f"Transcription completed in {processing_time:.2f}s for {duration:.2f}s audio")
        return result
        
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
        Speaker embeddings as base64-encoded bytes
    """
    try:
        # Read and process audio
        audio_data = await audio.read()
        waveform, sample_rate = load_audio_from_bytes(audio_data)
        
        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Extract embeddings
        if embedding_model is None:
            raise HTTPException(status_code=500, detail="Embedding model not loaded")
        
        # Convert to the format expected by pyannote
        audio_array = waveform.numpy().flatten()
        embeddings = embedding_model(audio_array)
        
        # Convert to bytes for storage
        embedding_bytes = embeddings.cpu().numpy().tobytes()
        
        return {
            "embeddings": embedding_bytes.hex(),
            "embedding_size": len(embedding_bytes),
            "audio_duration": waveform.shape[1] / SAMPLE_RATE
        }
        
    except Exception as e:
        logger.error(f"Error extracting embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")


def load_audio_from_bytes(audio_data: bytes) -> tuple[torch.Tensor, int]:
    """Load audio from bytes without writing to disk."""
    try:
        # Create a BytesIO stream from the audio data
        audio_stream = io.BytesIO(audio_data)
        
        # Load with torchaudio
        waveform, sample_rate = torchaudio.load(audio_stream)
        
        return waveform, sample_rate
        
    except Exception as e:
        logger.error(f"Failed to load audio from bytes: {e}")
        raise ValueError(f"Invalid audio format: {e}")


async def run_transcription(waveform: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
    """Run Whisper transcription on audio waveform."""
    try:
        if whisper_model is None:
            raise ValueError("Whisper model not loaded")
        
        # Convert to numpy for Whisper
        audio_array = waveform.numpy().flatten()
        
        # Run transcription
        result = whisper_model(
            audio_array,
            return_timestamps=True,
            generate_kwargs={
                "language": "en", 
                "task": "transcribe",
                "temperature": 0.0,  # Deterministic output for better quality
                "compression_ratio_threshold": 2.4,  # Higher quality threshold
                "logprob_threshold": -1.0,  # Better confidence filtering
                "no_speech_threshold": 0.6,  # Stricter speech detection
                "condition_on_previous_text": True,  # Better context continuity
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise


async def run_diarization(waveform: torch.Tensor, sample_rate: int) -> Any:
    """Run speaker diarization on audio waveform."""
    try:
        if diarization_pipeline is None:
            raise ValueError("Diarization pipeline not loaded")
        
        # Prepare audio for pyannote (needs specific format)
        audio_data = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }
        
        # Run diarization
        diarization = diarization_pipeline(audio_data)
        
        return diarization
        
    except Exception as e:
        logger.error(f"Diarization error: {e}")
        raise


def merge_transcription_and_diarization(
    transcription: Dict[str, Any], 
    diarization: Any
) -> List[Dict[str, Any]]:
    """Merge Whisper transcription with pyannote diarization results."""
    try:
        segments = []
        
        # Get transcription chunks
        transcription_chunks = transcription.get("chunks", [])
        
        for chunk in transcription_chunks:
            start_time = chunk["timestamp"][0] if chunk["timestamp"][0] is not None else 0.0
            end_time = chunk["timestamp"][1] if chunk["timestamp"][1] is not None else start_time + 1.0
            text = chunk["text"].strip()
            
            if not text:
                continue
            
            # Find the speaker for this time segment (if diarization available)
            if diarization is not None:
                speaker = find_speaker_for_segment(diarization, start_time, end_time)
            else:
                speaker = "Speaker 1"  # Default when no diarization
            
            segments.append({
                "start": start_time,
                "end": end_time,
                "text": text,
                "speaker": speaker,
                "confidence": 1.0  # Whisper doesn't provide per-segment confidence
            })
        
        return segments
        
    except Exception as e:
        logger.error(f"Error merging results: {e}")
        return []


def find_speaker_for_segment(diarization: Any, start_time: float, end_time: float) -> str:
    """Find the dominant speaker for a given time segment."""
    try:
        # Calculate the midpoint of the segment
        mid_time = (start_time + end_time) / 2
        
        # Find which speaker is active at the midpoint
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if segment.start <= mid_time <= segment.end:
                return f"Speaker {speaker}"
        
        # If no speaker found, use overlap analysis
        speaker_durations = {}
        for segment, _, speaker in diarization.itertracks(yield_label=True):
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


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        access_log=False
    ) 