"""
Simplified FastAPI server for Oreja audio transcription.
This version works without requiring Hugging Face authentication.
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
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import uvicorn

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
WHISPER_MODEL = "openai/whisper-large-v3-turbo"

app = FastAPI(
    title="Oreja Audio Processing API (Simplified)",
    description="Local audio transcription service",
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


def initialize_models():
    """Initialize Whisper model and set up the device."""
    global device, whisper_model
    
    # Set up device (GPU if available, otherwise CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Using device: cpu")
    
    logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
    try:
        whisper_model = pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL,
            device=device,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            return_timestamps=True,
            chunk_length_s=30,
            stride_length_s=5,
        )
        logger.info("✓ Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        logger.info("Trying smaller model...")
        try:
            whisper_model = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",
                device=device,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                return_timestamps=True,
                chunk_length_s=30,
                stride_length_s=5,
            )
            logger.info("✓ Whisper base model loaded successfully")
        except Exception as e2:
            logger.error(f"Failed to load any Whisper model: {e2}")
            raise
    
    logger.info("Model loading completed")


@app.on_event("startup")
async def startup_event():
    """Initialize models when the server starts."""
    initialize_models()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Oreja Audio Processing API (Simplified)",
        "status": "running",
        "models_loaded": whisper_model is not None
    }


@app.get("/health")
async def health_check():
    """Detailed health check with model status."""
    return {
        "status": "healthy",
        "device": str(device) if device else "unknown",
        "models": {
            "whisper": whisper_model is not None,
            "diarization": False,  # Disabled in simplified version
            "embedding": False     # Disabled in simplified version
        },
        "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else "N/A"
    }


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Transcribe audio file (simplified version without speaker diarization).
    
    Args:
        audio: Audio file (WAV format preferred)
        
    Returns:
        Transcription result with simple speaker assignment
    """
    start_time = time.time()
    
    try:
        # Validate file
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read audio data into memory
        audio_data = await audio.read()
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        logger.info(f"Processing audio file: {audio.filename}, size: {len(audio_data)} bytes")
        
        # Load audio with torchaudio
        waveform, sample_rate = load_audio_from_bytes(audio_data)
        
        # Validate audio length
        duration = waveform.shape[1] / sample_rate
        if duration < MIN_AUDIO_LENGTH:
            raise HTTPException(status_code=400, detail=f"Audio too short: {duration:.2f}s")
        if duration > MAX_AUDIO_LENGTH:
            logger.warning(f"Audio length {duration:.2f}s exceeds recommended {MAX_AUDIO_LENGTH}s")
        
        # Resample if necessary
        if sample_rate != SAMPLE_RATE:
            logger.info(f"Resampling from {sample_rate}Hz to {SAMPLE_RATE}Hz")
            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
            sample_rate = SAMPLE_RATE
        
        # Run transcription
        transcription_result = await run_transcription(waveform, sample_rate)
        
        # Create simple segments (without real speaker diarization)
        segments = create_simple_segments(transcription_result)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        result = {
            "segments": segments,
            "full_text": transcription_result.get("text", ""),
            "processing_time": processing_time,
            "audio_duration": duration,
            "model_info": {
                "whisper": WHISPER_MODEL,
                "diarization": "disabled (simplified version)"
            }
        }
        
        logger.info(f"Transcription completed in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


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
        
        # Convert to numpy for processing
        audio_array = waveform.numpy().flatten()
        
        # Voice Activity Detection
        rms_energy = np.sqrt(np.mean(audio_array ** 2))
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_array)))) / len(audio_array)
        
        min_energy_threshold = 0.01
        min_zero_crossing_rate = 0.01
        
        logger.info(f"Audio analysis - RMS Energy: {rms_energy:.4f}, Zero Crossings: {zero_crossings:.4f}")
        
        if rms_energy < min_energy_threshold:
            logger.info("Audio energy too low - likely silence")
            return {"text": "", "chunks": []}
        
        if zero_crossings < min_zero_crossing_rate:
            logger.info("Audio variation too low - likely constant noise")
            return {"text": "", "chunks": []}
        
        # Run transcription
        result = whisper_model(audio_array, return_timestamps=True)
        
        # Check for likely hallucinations
        if result and 'text' in result:
            clean_text = result['text'].strip().replace('.', '').replace(',', '').replace('?', '').replace('!', '')
            if len(clean_text) <= 2:
                logger.info(f"Transcription too short: '{result['text']}' - likely hallucination")
                return {"text": "", "chunks": []}
        
        return result
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise


def create_simple_segments(transcription: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create segments from transcription without real speaker diarization."""
    segments = []
    
    # Get transcription chunks
    chunks = transcription.get("chunks", [])
    
    if not chunks and transcription.get("text"):
        # If no chunks but we have text, create a single segment
        segments.append({
            "start": 0.0,
            "end": 5.0,  # Default duration
            "text": transcription["text"].strip(),
            "speaker": "SPEAKER_00",  # Simple speaker naming
            "confidence": 1.0
        })
        return segments
    
    # Process chunks
    speaker_counter = 0
    current_speaker = f"SPEAKER_{speaker_counter:02d}"
    
    for i, chunk in enumerate(chunks):
        start_time = chunk["timestamp"][0] if chunk["timestamp"][0] is not None else 0.0
        end_time = chunk["timestamp"][1] if chunk["timestamp"][1] is not None else start_time + 1.0
        text = chunk["text"].strip()
        
        if not text:
            continue
        
        # Simple speaker change logic (every few segments or on silence gaps)
        if i > 0 and i % 3 == 0:  # Change speaker every 3 segments
            speaker_counter = (speaker_counter + 1) % 4  # Cycle through 4 speakers max
            current_speaker = f"SPEAKER_{speaker_counter:02d}"
        
        segments.append({
            "start": start_time,
            "end": end_time,
            "text": text,
            "speaker": current_speaker,
            "confidence": 1.0
        })
    
    return segments


if __name__ == "__main__":
    print("Starting Oreja Backend Server (Simplified Version)")
    print("This version works without Hugging Face authentication")
    print("Starting on http://127.0.0.1:8000")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        access_log=False
    ) 