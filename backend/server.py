"""
FastAPI server for Oreja audio transcription and diarization.
Processes audio in memory using local Hugging Face models with no cloud interaction.

PRIVACY GUARANTEE: NO AUDIO DATA EVER LEAVES THIS MACHINE
"""

import asyncio
import io
import logging
import os
import sys
import time
from typing import List, Dict, Any, Optional
import warnings
from pathlib import Path
from datetime import datetime

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

# Import our offline speaker embedding manager
from speaker_embeddings import OfflineSpeakerEmbeddingManager

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
MAX_AUDIO_LENGTH = 7200  # seconds (2 hours) - realistic maximum for chunked processing
MIN_AUDIO_LENGTH = 0.1  # seconds
CHUNK_THRESHOLD = 30  # seconds - audio longer than this will be chunked
WHISPER_MODEL = "openai/whisper-large-v3-turbo"  # Latest model from October 2024 - faster with same accuracy as large-v3
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.0"
EMBEDDING_MODEL = "pyannote/embedding"

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

# Global variables
device = None
whisper_model = None
diarization_pipeline = None
embedding_model = None
speaker_embedding_manager = None
enhanced_speaker_database = None
enhanced_service = None
enhanced_speaker_integration = None


def initialize_models():
    """Initialize all models and set up the device."""
    global device, whisper_model, diarization_pipeline, embedding_model, speaker_embedding_manager, enhanced_speaker_database, enhanced_service
    
    logger.info("🚀 STARTING MODEL INITIALIZATION")
    logger.info(f"🔧 Current working directory: {os.getcwd()}")
    logger.info(f"🐍 Python executable: {sys.executable}")
    logger.info(f"📦 Transformers version: {__import__('transformers').__version__}")
    logger.info(f"🔥 PyTorch version: {torch.__version__}")
    
    # Set up device (GPU if available, otherwise CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Using device: cpu")
    
    # Load Whisper model with proper error handling
    try:
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
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        logger.info("Trying fallback to smaller Whisper model...")
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
            logger.info("✓ Whisper base model loaded successfully (fallback)")
        except Exception as e2:
            logger.error(f"Failed to load any Whisper model: {e2}")
            logger.error("Transcription will not be available!")
            whisper_model = None
    
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
    
    # Initialize enhanced speaker database as primary system
    try:
        logger.info("Initializing enhanced speaker database v2...")
        from speaker_database_v2 import EnhancedSpeakerDatabase
        enhanced_speaker_database = EnhancedSpeakerDatabase("speaker_data_v2")
        logger.info("✓ Enhanced speaker database v2 initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize enhanced speaker database: {e}")
        enhanced_speaker_database = None
    
    # Legacy speaker embedding manager disabled - using enhanced database only
    speaker_embedding_manager = None
    logger.info("Legacy speaker embedding manager disabled - using enhanced database v2 only")
    
    # Initialize enhanced transcription features
    if ENHANCED_FEATURES_AVAILABLE:
        try:
            logger.info("Initializing enhanced transcription features...")
            enhanced_service = EnhancedTranscriptionService(
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
    global speaker_embedding_manager, enhanced_speaker_database, enhanced_speaker_integration
    
    initialize_models()
    
    # Initialize enhanced speaker integration
    try:
        enhanced_speaker_integration = EnhancedSpeakerServerIntegration(
            legacy_speaker_manager=speaker_embedding_manager
        )
        # Override to use enhanced database directly if available
        if enhanced_speaker_database is not None:
            enhanced_speaker_integration.enhanced_db = enhanced_speaker_database
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
            "speaker_embeddings": speaker_embedding_manager is not None,
            "enhanced_features": enhanced_service is not None
        },
        "enhanced_capabilities": {
            "sentiment_analysis": enhanced_service is not None,
            "audio_features": enhanced_service is not None,
            "conversation_analytics": enhanced_service is not None
        } if enhanced_service else None,
        "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else "N/A"
    }


@app.get("/speakers")
async def get_speaker_stats():
    """Get statistics about known speakers."""
    # Use enhanced database as primary, fallback to legacy
    if enhanced_speaker_database is not None:
        try:
            speakers = enhanced_speaker_database.get_all_speakers()
            # Convert to legacy format for compatibility
            stats = {
                'total_speakers': len(speakers),
                'speakers': [
                    {
                        'id': s['speaker_id'],
                        'name': s['display_name'],
                        'embedding_count': s['embedding_count'],
                        'last_seen': s['last_seen'],
                        'avg_confidence': s['average_confidence']
                    }
                    for s in speakers
                ]
            }
            return stats
        except Exception as e:
            logger.warning(f"Enhanced database failed, falling back to legacy: {e}")
    
    if speaker_embedding_manager is None:
        raise HTTPException(status_code=503, detail="No speaker database available")
    
    try:
        stats = speaker_embedding_manager.get_speaker_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting speaker stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speakers/enroll")
async def enroll_speaker(
    speaker_name: str,
    audio: UploadFile = File(...)
):
    """
    Enroll a new speaker with a known name using an audio sample.
    
    Args:
        speaker_name: Human-readable name for the speaker
        audio: Audio file containing the speaker's voice
        
    Returns:
        Generated speaker ID and enrollment status
    """
    if speaker_embedding_manager is None:
        raise HTTPException(status_code=503, detail="Speaker embedding manager not available")
    
    try:
        # Read and process audio
        audio_data = await audio.read()
        waveform, sample_rate = load_audio_from_bytes(audio_data)
        
        # Convert to numpy array for embedding extraction
        audio_numpy = waveform.squeeze().numpy()
        
        # Enroll speaker
        speaker_id = speaker_embedding_manager.enroll_speaker(speaker_name, audio_numpy)
        
        return {
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "status": "enrolled_successfully"
        }
        
    except Exception as e:
        logger.error(f"Error enrolling speaker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speakers/identify")
async def identify_speaker(audio: UploadFile = File(...)):
    """
    Identify a speaker from an audio sample.
    
    Args:
        audio: Audio file containing speaker's voice
        
    Returns:
        Speaker identification result
    """
    if speaker_embedding_manager is None:
        raise HTTPException(status_code=503, detail="Speaker embedding manager not available")
    
    try:
        # Read and process audio
        audio_data = await audio.read()
        waveform, sample_rate = load_audio_from_bytes(audio_data)
        
        # Convert to numpy array for embedding extraction
        audio_numpy = waveform.squeeze().numpy()
        
        # Identify speaker
        speaker_id, confidence, is_new = speaker_embedding_manager.identify_or_create_speaker(audio_numpy)
        
        # Get speaker info
        if speaker_id in speaker_embedding_manager.speaker_profiles:
            speaker_name = speaker_embedding_manager.speaker_profiles[speaker_id].name
        else:
            speaker_name = "Unknown"
        
        return {
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "confidence": confidence,
            "is_new_speaker": is_new,
            "status": "identified" if not is_new else "new_speaker_created"
        }
        
    except Exception as e:
        logger.error(f"Error identifying speaker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/speakers/{speaker_id}/name")
async def update_speaker_name(speaker_id: str, new_name: str):
    """Update the name of an existing speaker."""
    if speaker_embedding_manager is None:
        raise HTTPException(status_code=503, detail="Speaker embedding manager not available")
    
    try:
        success = speaker_embedding_manager.update_speaker_name(speaker_id, new_name)
        if success:
            return {"status": "updated", "speaker_id": speaker_id, "new_name": new_name}
        else:
            raise HTTPException(status_code=404, detail="Speaker not found")
            
    except Exception as e:
        logger.error(f"Error updating speaker name: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/speakers/{speaker_id}")
async def delete_speaker(speaker_id: str):
    """Delete a speaker profile."""
    if speaker_embedding_manager is None:
        raise HTTPException(status_code=503, detail="Speaker embedding manager not available")
    
    try:
        success = speaker_embedding_manager.delete_speaker(speaker_id)
        if success:
            return {"status": "deleted", "speaker_id": speaker_id}
        else:
            raise HTTPException(status_code=404, detail="Speaker not found")
            
    except Exception as e:
        logger.error(f"Error deleting speaker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speakers/feedback")
async def provide_speaker_feedback(
    correct_speaker_name: str,
    audio_segment_start: float,
    audio_segment_end: float,
    audio: UploadFile = File(...)
):
    """
    Provide feedback on speaker identification to improve future recognition.
    
    Args:
        correct_speaker_name: The correct speaker name for this audio segment
        audio_segment_start: Start time of the segment in seconds
        audio_segment_end: End time of the segment in seconds
        audio: The audio file containing the segment
        
    Returns:
        Status of the feedback processing
    """
    if speaker_embedding_manager is None:
        raise HTTPException(status_code=503, detail="Speaker embedding manager not available")
    
    try:
        # Read and process audio
        audio_data = await audio.read()
        waveform, sample_rate = load_audio_from_bytes(audio_data)
        
        # Extract the specific segment
        start_sample = int(audio_segment_start * sample_rate)
        end_sample = int(audio_segment_end * sample_rate)
        
        if start_sample < waveform.shape[1] and end_sample <= waveform.shape[1]:
            segment_waveform = waveform[:, start_sample:end_sample]
            segment_audio = segment_waveform.squeeze().numpy()
            
            # Provide feedback to the speaker embedding system
            success = speaker_embedding_manager.provide_correction_feedback(
                correct_speaker_name, segment_audio
            )
            
            if success:
                return {
                    "status": "feedback_processed",
                    "speaker_name": correct_speaker_name,
                    "segment_duration": audio_segment_end - audio_segment_start
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to process feedback")
        else:
            raise HTTPException(status_code=400, detail="Invalid segment timing")
            
    except Exception as e:
        logger.error(f"Error processing speaker feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speakers/batch_feedback")
async def provide_batch_speaker_feedback(
    feedback_data: dict
):
    """
    Provide batch feedback for multiple speaker corrections.
    
    Args:
        feedback_data: Dictionary containing speaker corrections
        Format: {
            "corrections": [
                {
                    "speaker_name": "John",
                    "audio_segments": [
                        {"start": 0.0, "end": 2.5, "audio_data": "base64_encoded_wav"},
                        ...
                    ]
                },
                ...
            ]
        }
        
    Returns:
        Status of the batch feedback processing
    """
    if speaker_embedding_manager is None:
        raise HTTPException(status_code=503, detail="Speaker embedding manager not available")
    
    try:
        corrections = feedback_data.get("corrections", [])
        processed_count = 0
        
        for correction in corrections:
            speaker_name = correction.get("speaker_name")
            audio_segments = correction.get("audio_segments", [])
            
            for segment in audio_segments:
                try:
                    # Decode base64 audio data
                    import base64
                    audio_bytes = base64.b64decode(segment["audio_data"])
                    
                    # Load audio
                    waveform, sample_rate = load_audio_from_bytes(audio_bytes)
                    audio_array = waveform.squeeze().numpy()
                    
                    # Provide feedback
                    speaker_embedding_manager.provide_correction_feedback(
                        speaker_name, audio_array
                    )
                    processed_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to process segment for {speaker_name}: {e}")
                    continue
        
        return {
            "status": "batch_feedback_processed",
            "processed_segments": processed_count,
            "total_corrections": len(corrections)
        }
        
    except Exception as e:
        logger.error(f"Error processing batch speaker feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speakers/merge")
async def merge_speakers(source_speaker_id: str, target_speaker_id: str, target_name: str = None):
    """
    Manually merge two speaker profiles.
    
    Args:
        source_speaker_id: The speaker ID to merge from (will be deleted)
        target_speaker_id: The speaker ID to merge into (will be kept)
        target_name: Optional new name for the target speaker
        
    Returns:
        Status of the merge operation
    """
    if speaker_embedding_manager is None:
        raise HTTPException(status_code=503, detail="Speaker embedding manager not available")
    
    try:
        # Validate inputs
        if source_speaker_id == target_speaker_id:
            raise HTTPException(status_code=400, detail="Cannot merge speaker with itself")
        
        # Check if both speakers exist
        if source_speaker_id not in speaker_embedding_manager.speaker_profiles:
            raise HTTPException(status_code=404, detail=f"Source speaker {source_speaker_id} not found")
        
        if target_speaker_id not in speaker_embedding_manager.speaker_profiles:
            raise HTTPException(status_code=404, detail=f"Target speaker {target_speaker_id} not found")
        
        # Perform the merge
        success = speaker_embedding_manager.merge_speakers(source_speaker_id, target_speaker_id)
        
        if success:
            # Update target speaker name if provided
            if target_name and target_name.strip():
                speaker_embedding_manager.update_speaker_name(target_speaker_id, target_name.strip())
            
            return {
                "status": "merged_successfully",
                "source_speaker_id": source_speaker_id,
                "target_speaker_id": target_speaker_id,
                "target_name": target_name or speaker_embedding_manager.speaker_profiles[target_speaker_id].name,
                "message": f"Successfully merged {source_speaker_id} into {target_speaker_id}"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to merge speakers")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error merging speakers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/speakers/{speaker_id}/name")
async def update_speaker_name(speaker_id: str, new_name: str):
    """
    Update a speaker's display name.
    
    Args:
        speaker_id: The speaker ID to update
        new_name: The new display name for the speaker
        
    Returns:
        Status of the name update operation
    """
    if speaker_embedding_manager is None:
        raise HTTPException(status_code=503, detail="Speaker embedding manager not available")
    
    try:
        # Check if speaker exists
        if not speaker_embedding_manager.speaker_exists(speaker_id):
            raise HTTPException(status_code=404, detail=f"Speaker {speaker_id} not found")
        
        # Update the speaker name
        success = speaker_embedding_manager.update_speaker_name(speaker_id, new_name)
        
        if success:
            logger.info(f"Updated speaker name: {speaker_id} -> {new_name}")
            return {
                "status": "success",
                "message": f"Speaker name updated to: {new_name}",
                "speaker_id": speaker_id,
                "new_name": new_name
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update speaker name")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating speaker name: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/speakers/{speaker_id}")
async def delete_speaker(speaker_id: str):
    """
    Delete a speaker profile completely.
    
    Args:
        speaker_id: The speaker ID to delete
        
    Returns:
        Status of the deletion operation
    """
    if speaker_embedding_manager is None:
        raise HTTPException(status_code=503, detail="Speaker embedding manager not available")
    
    try:
        # Check if speaker exists
        if not speaker_embedding_manager.speaker_exists(speaker_id):
            raise HTTPException(status_code=404, detail=f"Speaker {speaker_id} not found")
        
        # Delete the speaker
        success = speaker_embedding_manager.delete_speaker(speaker_id)
        
        if success:
            logger.info(f"Deleted speaker: {speaker_id}")
            return {
                "status": "success",
                "message": f"Speaker {speaker_id} deleted successfully",
                "speaker_id": speaker_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete speaker")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting speaker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speakers/name_mapping")
async def update_speaker_name_mapping(
    old_speaker_id: str,
    new_speaker_name: str
):
    """
    Update speaker name mapping and consolidate speakers.
    
    This is a simpler feedback mechanism that allows the frontend to:
    1. Rename existing speakers
    2. Merge auto-generated speakers with user-named speakers
    
    Args:
        old_speaker_id: The original speaker ID (e.g., "SPEAKER_00", "AUTO_SPEAKER_001")
        new_speaker_name: The correct speaker name (e.g., "John", "Speaker 1")
        
    Returns:
        Status of the name mapping update
    """
    if speaker_embedding_manager is None:
        raise HTTPException(status_code=503, detail="Speaker embedding manager not available")
    
    try:
        # Validate speaker name is not empty
        new_speaker_name = new_speaker_name.strip()
        if not new_speaker_name:
            raise HTTPException(status_code=400, detail="Speaker name cannot be empty")
        
        # Check if there's already a speaker with the new name
        existing_speaker_id = speaker_embedding_manager.get_speaker_by_name(new_speaker_name)
        
        if existing_speaker_id and existing_speaker_id != old_speaker_id:
            # Merge the old speaker into the existing one
            success = speaker_embedding_manager.merge_speakers(old_speaker_id, existing_speaker_id)
            if success:
                return {
                    "status": "speakers_merged",
                    "old_speaker_id": old_speaker_id,
                    "target_speaker_id": existing_speaker_id,
                    "speaker_name": new_speaker_name,
                    "message": f"Merged {old_speaker_id} into existing speaker {new_speaker_name}"
                }
            else:
                return {
                    "status": "merge_failed",
                    "message": "Could not merge speakers"
                }
        else:
            # Just rename the speaker
            success = speaker_embedding_manager.update_speaker_name(old_speaker_id, new_speaker_name)
            if success:
                return {
                    "status": "name_updated",
                    "speaker_id": old_speaker_id,
                    "new_name": new_speaker_name
                }
            else:
                # Speaker might not exist, try creating a new mapping
                logger.info(f"Speaker {old_speaker_id} not found, treating as name mapping")
                return {
                    "status": "mapping_noted",
                    "old_speaker_id": old_speaker_id,
                    "new_name": new_speaker_name
                }
        
    except Exception as e:
        logger.error(f"Error updating speaker name mapping: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Transcribe and diarize audio file with enhanced sentiment analysis and audio features.
    
    Args:
        audio: Audio file (WAV format preferred)
        
    Returns:
        Enhanced transcription result with speaker diarization, sentiment analysis, and audio features
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
        if duration > CHUNK_THRESHOLD:
            logger.info(f"Audio length {duration:.2f}s will be processed in chunks for optimal performance")
        
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
        
        # Check if transcription was skipped due to voice activity detection
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
        
        # Merge transcription with speaker information (if available)
        segments = merge_transcription_and_diarization(
            transcription_result, diarization_result, waveform, sample_rate
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
        
        # 🚀 ENHANCE WITH SENTIMENT ANALYSIS AND AUDIO FEATURES
        if enhanced_service:
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
            logger.info("Enhanced features not available - returning basic transcription")
        
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


@app.post("/speakers/real_time_feedback")
async def real_time_speaker_feedback(feedback_data: dict):
    """
    Process real-time speaker corrections and immediately update embeddings for learning.
    
    This endpoint receives speaker corrections from the transcription editor and:
    1. Updates the speaker embeddings with the corrected attribution
    2. Returns status indicating what learning occurred
    
    Args:
        feedback_data: Dictionary containing:
            - old_speaker_id: The incorrect speaker ID
            - correct_speaker_name: The correct speaker name
            - audio_segments: List of audio segment data for learning
            - transcription_file: Path to transcription file
            - audio_file: Path to audio file
    
    Returns:
        Status of the learning update
    """
    if speaker_embedding_manager is None:
        raise HTTPException(status_code=503, detail="Speaker embedding manager not available")
    
    try:
        old_speaker_id = feedback_data.get("old_speaker_id")
        correct_speaker_name = feedback_data.get("correct_speaker_name")
        audio_segments = feedback_data.get("audio_segments", [])
        audio_file = feedback_data.get("audio_file")
        
        if not old_speaker_id or not correct_speaker_name:
            raise HTTPException(status_code=400, detail="Missing required fields: old_speaker_id, correct_speaker_name")
        
        logger.info(f"Processing real-time feedback: {old_speaker_id} → {correct_speaker_name}")
        
        # Check if the correct speaker already exists
        existing_speaker_id = speaker_embedding_manager.get_speaker_by_name(correct_speaker_name)
        
        if existing_speaker_id:
            # Merge the old speaker into the existing one
            success = speaker_embedding_manager.merge_speakers(old_speaker_id, existing_speaker_id, correct_speaker_name)
            if success:
                # Learn from the audio segments
                if audio_file and audio_segments:
                    for segment in audio_segments:
                        try:
                            start_time = float(segment.get('start', 0))
                            end_time = float(segment.get('end', start_time + 1))
                            
                            # Extract and learn from this audio segment
                            speaker_embedding_manager.learn_from_segment(
                                audio_file, start_time, end_time, existing_speaker_id
                            )
                        except Exception as e:
                            logger.warning(f"Could not learn from segment: {e}")
                
                return {
                    "status": "speakers_merged",
                    "message": f"Merged {old_speaker_id} into existing speaker {correct_speaker_name}",
                    "target_speaker_id": existing_speaker_id
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to merge speakers")
        else:
            # Update the speaker name and learn from segments
            success = speaker_embedding_manager.update_speaker_name(old_speaker_id, correct_speaker_name)
            if success:
                # Learn from the audio segments to improve the model
                if audio_file and audio_segments:
                    for segment in audio_segments:
                        try:
                            start_time = float(segment.get('start', 0))
                            end_time = float(segment.get('end', start_time + 1))
                            
                            # Extract and learn from this audio segment
                            speaker_embedding_manager.learn_from_segment(
                                audio_file, start_time, end_time, old_speaker_id
                            )
                        except Exception as e:
                            logger.warning(f"Could not learn from segment: {e}")
                
                return {
                    "status": "learned",
                    "message": f"Updated speaker name and learned from {len(audio_segments)} segments",
                    "speaker_id": old_speaker_id,
                    "new_name": correct_speaker_name
                }
            else:
                return {
                    "status": "name_updated", 
                    "message": f"Updated speaker name to {correct_speaker_name}",
                    "speaker_id": old_speaker_id
                }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in real-time speaker feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reattribute_speakers")
async def reattribute_speakers(reattribution_data: dict):
    """
    Re-run speaker attribution on segments that could benefit from improved models.
    
    This endpoint takes segments with low confidence or unknown speakers and
    re-runs the speaker identification using the updated embeddings.
    
    Args:
        reattribution_data: Dictionary containing:
            - audio_file: Path to the audio file
            - segments: List of segments to re-attribute
            - transcription_context: Context about the transcription
    
    Returns:
        Updated speaker attributions with improved confidence scores
    """
    if speaker_embedding_manager is None:
        raise HTTPException(status_code=503, detail="Speaker embedding manager not available")
    
    try:
        audio_file = reattribution_data.get("audio_file")
        segments = reattribution_data.get("segments", [])
        context = reattribution_data.get("transcription_context", {})
        
        if not audio_file or not segments:
            raise HTTPException(status_code=400, detail="Missing audio_file or segments")
        
        logger.info(f"Re-attributing {len(segments)} segments from {audio_file}")
        
        updated_segments = []
        
        for segment in segments:
            try:
                start_time = float(segment.get('start', 0))
                end_time = float(segment.get('end', start_time + 1))
                current_speaker = segment.get('current_speaker', 'Unknown')
                current_confidence = float(segment.get('current_confidence', 0))
                
                # Re-run speaker identification on this segment
                new_attribution = speaker_embedding_manager.identify_speaker_in_segment(
                    audio_file, start_time, end_time
                )
                
                if new_attribution:
                    new_speaker = new_attribution.get('speaker_id', current_speaker)
                    new_confidence = float(new_attribution.get('confidence', current_confidence))
                    
                    # Only include if confidence improved significantly
                    if new_confidence > current_confidence + 0.1:
                        # Get the display name for the speaker
                        speaker_name = speaker_embedding_manager.get_speaker_name(new_speaker) or new_speaker
                        
                        updated_segments.append({
                            'segment_index': segment.get('segment_index'),
                            'new_speaker': speaker_name,
                            'new_confidence': new_confidence,
                            'old_speaker': current_speaker,
                            'old_confidence': current_confidence,
                            'improvement': new_confidence - current_confidence
                        })
                        
                        logger.info(f"Improved segment {segment.get('segment_index')}: {current_speaker} ({current_confidence:.3f}) → {speaker_name} ({new_confidence:.3f})")
            
            except Exception as e:
                logger.warning(f"Could not re-attribute segment: {e}")
                continue
        
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
    if speaker_embedding_manager is None:
        raise HTTPException(status_code=503, detail="Speaker embedding manager not available")
    
    try:
        # Validate inputs
        if not (0.0 <= split_text_position <= 1.0):
            raise HTTPException(status_code=400, detail="Split position must be between 0.0 and 1.0")
        
        if not audio_file or not Path(audio_file).exists():
            raise HTTPException(status_code=400, detail="Audio file not found")
        
        # Create the enhanced splitter
        splitter = AudioSegmentSplitter(speaker_embedding_manager)
        
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
    Reprocess embeddings for existing segments.
    
    This is useful when speaker names have been corrected and you want to 
    re-extract embeddings with the updated speaker assignments.
    
    Args:
        audio_file: Path to the audio file
        segments: List of segments to reprocess
        force_update: Whether to update even if embeddings already exist
        
    Returns:
        Results of the reprocessing operation
    """
    if speaker_embedding_manager is None:
        raise HTTPException(status_code=503, detail="Speaker embedding manager not available")
    
    try:
        if not audio_file or not Path(audio_file).exists():
            raise HTTPException(status_code=400, detail="Audio file not found")
        
        results = {
            "total_segments": len(segments),
            "processed_segments": 0,
            "updated_speakers": [],
            "failed_extractions": 0,
            "improvements": []
        }
        
        # Load audio file
        waveform, sr = torchaudio.load(audio_file)
        
        for segment in segments:
            try:
                start_time = segment.get('start', 0)
                end_time = segment.get('end', start_time + 1)
                speaker_name = segment.get('speaker', 'Unknown')
                
                # Skip if no meaningful speaker name
                if not speaker_name or speaker_name in ['Unknown', '']:
                    continue
                
                # Extract audio segment
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                
                if start_sample >= waveform.shape[1] or end_sample > waveform.shape[1]:
                    logger.warning(f"Segment bounds invalid: {start_sample}-{end_sample}")
                    continue
                
                segment_waveform = waveform[:, start_sample:end_sample]
                
                # Check if segment is long enough
                duration = (end_sample - start_sample) / sr
                if duration < 0.5:  # Minimum 0.5 seconds
                    logger.debug(f"Segment too short for embedding: {duration:.2f}s")
                    continue
                
                # Convert to numpy
                if segment_waveform.dim() > 1:
                    audio_numpy = segment_waveform.mean(dim=0).cpu().numpy()
                else:
                    audio_numpy = segment_waveform.cpu().numpy()
                
                # Extract and update embedding
                success = speaker_embedding_manager.provide_correction_feedback(
                    speaker_name, audio_numpy
                )
                
                if success:
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
                else:
                    results["failed_extractions"] += 1
                    
            except Exception as e:
                logger.warning(f"Failed to process segment {segment.get('start', 0)}: {e}")
                results["failed_extractions"] += 1
                continue
        
        # Save the updated speaker database
        if results["processed_segments"] > 0:
            speaker_embedding_manager._save_speaker_database()
        
        response = {
            "status": "reprocessing_complete",
            "results": results,
            "message": f"Reprocessed {results['processed_segments']} segments for {len(results['updated_speakers'])} speakers"
        }
        
        logger.info(f"Reprocessed embeddings for {results['processed_segments']} segments")
        return response
        
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
        
        # Load with torchaudio
        waveform, sample_rate = torchaudio.load(audio_stream)
        
        return waveform, sample_rate
        
    except Exception as e:
        logger.error(f"Failed to load audio from bytes: {e}")
        raise ValueError(f"Invalid audio format: {e}")


async def run_transcription(waveform: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
    """Run Whisper transcription on audio waveform with enhanced voice activity detection and automatic chunking."""
    try:
        if whisper_model is None:
            raise ValueError("Whisper model not loaded")
        
        # Calculate duration
        duration = waveform.shape[1] / sample_rate
        
        # If audio is longer than 30 seconds, chunk it for processing
        if duration > 30.0:
            logger.info(f"Audio is {duration:.2f}s long, chunking for optimal processing...")
            return await run_chunked_transcription(waveform, sample_rate)
        
        # Convert to numpy for analysis
        audio_array = waveform.numpy().flatten()
        
        # Enhanced Voice Activity Detection
        # Calculate RMS (Root Mean Square) energy
        rms_energy = np.sqrt(np.mean(audio_array ** 2))
        
        # Calculate zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_array)))) / len(audio_array)
        
        # Calculate spectral centroid (frequency content analysis)
        # Higher frequencies often indicate non-speech sounds like keyboard clicks
        fft = np.fft.fft(audio_array)
        freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
        magnitudes = np.abs(fft)
        
        # Only consider positive frequencies and avoid division by zero
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitudes = magnitudes[:len(magnitudes)//2]
        
        if np.sum(positive_magnitudes) > 0:
            spectral_centroid = np.sum(positive_freqs * positive_magnitudes) / np.sum(positive_magnitudes)
        else:
            spectral_centroid = 0
            
        # Calculate high frequency ratio (ratio of energy above 4kHz vs total energy)
        # Keyboard clicks and similar noises have more high-frequency content
        high_freq_mask = positive_freqs > 4000
        high_freq_energy = np.sum(positive_magnitudes[high_freq_mask] ** 2) if np.any(high_freq_mask) else 0
        total_energy = np.sum(positive_magnitudes ** 2)
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # Calculate temporal stability (variance in audio levels over time)
        # Speech has more temporal variation than constant noises
        frame_size = sample_rate // 10  # 100ms frames
        frame_energies = []
        for i in range(0, len(audio_array) - frame_size, frame_size):
            frame = audio_array[i:i + frame_size]
            frame_energy = np.sqrt(np.mean(frame ** 2))
            frame_energies.append(frame_energy)
        
        temporal_variance = np.var(frame_energies) if len(frame_energies) > 1 else 0
        
        # Enhanced thresholds for better noise filtering
        min_energy_threshold = 0.015  # Slightly higher to filter quiet keyboard clicks
        min_zero_crossing_rate = 0.015  # Minimum variation in signal
        max_spectral_centroid = 3000  # Hz - speech typically centers below 3kHz
        max_high_freq_ratio = 0.6  # Maximum ratio of high frequency content
        min_temporal_variance = 0.0001  # Minimum variation over time
        min_duration = 0.3  # Minimum duration in seconds for valid speech
        
        # Calculate actual duration
        duration = len(audio_array) / sample_rate
        
        logger.info(f"Enhanced audio analysis - RMS: {rms_energy:.4f}, ZCR: {zero_crossings:.4f}, "
                   f"Spectral Centroid: {spectral_centroid:.0f}Hz, High Freq Ratio: {high_freq_ratio:.3f}, "
                   f"Temporal Variance: {temporal_variance:.6f}, Duration: {duration:.2f}s")
        
        # Check multiple criteria for speech detection
        if rms_energy < min_energy_threshold:
            logger.info("Audio energy too low - likely silence or very quiet noise, skipping transcription")
            return {
                "segments": [],
                "full_text": "",
                "processing_time": 0.0,
                "skipped_reason": "low_energy"
            }
        
        if zero_crossings < min_zero_crossing_rate:
            logger.info("Audio variation too low - likely constant noise, skipping transcription")
            return {
                "segments": [],
                "full_text": "",
                "processing_time": 0.0,
                "skipped_reason": "low_variation"
            }
            
        if spectral_centroid > max_spectral_centroid:
            logger.info(f"Spectral centroid too high ({spectral_centroid:.0f}Hz) - likely non-speech noise (keyboard, etc.), skipping transcription")
            return {
                "segments": [],
                "full_text": "",
                "processing_time": 0.0,
                "skipped_reason": "high_frequency_noise"
            }
            
        if high_freq_ratio > max_high_freq_ratio:
            logger.info(f"High frequency content too high ({high_freq_ratio:.3f}) - likely mechanical noise, skipping transcription")
            return {
                "segments": [],
                "full_text": "",
                "processing_time": 0.0,
                "skipped_reason": "mechanical_noise"
            }
            
        if temporal_variance < min_temporal_variance:
            logger.info(f"Temporal variance too low ({temporal_variance:.6f}) - likely constant tone or noise, skipping transcription")
            return {
                "segments": [],
                "full_text": "",
                "processing_time": 0.0,
                "skipped_reason": "constant_signal"
            }
            
        if duration < min_duration:
            logger.info(f"Audio too short ({duration:.2f}s) - likely noise burst, skipping transcription")
            return {
                "segments": [],
                "full_text": "",
                "processing_time": 0.0,
                "skipped_reason": "too_short"
            }
        
        # Proceed with transcription only if audio likely contains speech
        logger.info("Audio passed voice activity detection - proceeding with transcription")
        result = whisper_model(audio_array, return_timestamps=True)
        
        # Additional post-transcription checks to catch Whisper hallucinations
        if result and 'text' in result:
            clean_text = result['text'].strip().replace('.', '').replace(',', '').replace('?', '').replace('!', '').replace(' ', '')
            
            # Check for very short transcriptions (likely hallucinations)
            if len(clean_text) <= 2:
                logger.info(f"Transcription too short/empty: '{result['text']}' - likely hallucination, skipping")
                return {
                    "segments": [],
                    "full_text": "",
                    "processing_time": 0.0,
                    "skipped_reason": "likely_hallucination"
                }
            
            # Check for common keyboard-related transcription artifacts
            keyboard_artifacts = [
                'あ', 'い', 'う', 'え', 'お',  # Common Japanese characters from key press sounds
                'な', 'に', 'ぬ', 'ね', 'の',
                'か', 'き', 'く', 'け', 'こ',
                '。', '、', 'ん', 'し', 'て',
                'ㅏ', 'ㅓ', 'ㅗ', 'ㅜ', 'ㅡ',  # Korean characters
                'ă', 'â', 'ê', 'ô', 'ơ',     # Vietnamese characters
                'ต', 'า', 'ก', 'น', 'ม',      # Thai characters
            ]
            
            # Check if transcription consists mainly of keyboard artifacts
            artifact_count = sum(1 for char in result['text'] if char in keyboard_artifacts)
            total_chars = len([c for c in result['text'] if c.isalnum() or c in keyboard_artifacts])
            
            if total_chars > 0 and artifact_count / total_chars > 0.7:  # More than 70% artifacts
                logger.info(f"Transcription contains mostly keyboard artifacts: '{result['text']}' - likely false detection, skipping")
                return {
                    "segments": [],
                    "full_text": "",
                    "processing_time": 0.0,
                    "skipped_reason": "keyboard_artifacts"
                }
            
            # Check for suspiciously uniform character repetition (often from noise)
            if len(set(clean_text)) <= 2 and len(clean_text) > 3:  # Same 1-2 characters repeated
                logger.info(f"Transcription shows character repetition: '{result['text']}' - likely noise, skipping")
                return {
                    "segments": [],
                    "full_text": "",
                    "processing_time": 0.0,
                    "skipped_reason": "character_repetition"
                }
        
        logger.info(f"Transcription successful: '{result.get('text', '')[:50]}{'...' if len(result.get('text', '')) > 50 else ''}'")
        return result
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise


def find_natural_pause_point(waveform: torch.Tensor, search_start: int, search_end: int, sample_rate: int) -> int:
    """
    Find the best natural pause point for splitting audio chunks.
    Returns sample index of the best pause, or None if no good pause found.
    """
    try:
        if search_start >= search_end or search_start >= waveform.shape[1]:
            return None
        
        # Extract the search region
        search_region = waveform[:, search_start:min(search_end, waveform.shape[1])].squeeze().numpy()
        
        # Calculate energy in sliding windows
        window_size = int(0.1 * sample_rate)  # 100ms windows
        step_size = int(0.05 * sample_rate)   # 50ms step
        
        energy_scores = []
        window_positions = []
        
        for i in range(0, len(search_region) - window_size, step_size):
            window = search_region[i:i + window_size]
            
            # Calculate RMS energy
            rms_energy = np.sqrt(np.mean(window ** 2))
            
            # Calculate zero crossing rate (indicates speech activity)
            zero_crossings = np.sum(np.diff(np.sign(window)) != 0) / len(window)
            
            # Combine metrics (lower is better for pause detection)
            pause_score = rms_energy + zero_crossings * 0.1
            
            energy_scores.append(pause_score)
            window_positions.append(search_start + i + window_size // 2)
        
        if not energy_scores:
            return None
        
        # Find windows with lowest energy (best pause candidates)
        min_energy_threshold = min(energy_scores) * 1.5  # Allow some tolerance
        
        # Find the best pause point closest to the middle of the search region
        target_position = (search_start + search_end) / 2
        best_pause = None
        best_distance = float('inf')
        
        for i, (score, position) in enumerate(zip(energy_scores, window_positions)):
            if score <= min_energy_threshold:
                distance = abs(position - target_position)
                if distance < best_distance:
                    best_distance = distance
                    best_pause = position
        
        return best_pause
        
    except Exception as e:
        logger.debug(f"Error finding natural pause point: {e}")
        return None


async def run_chunked_transcription(waveform: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
    """
    Run transcription on long audio files by splitting into chunks at natural speech pauses.
    This prevents memory issues, attention mask warnings, and improves timestamp accuracy.
    """
    try:
        # Configuration
        target_chunk_duration = 25.0  # Target seconds per chunk (slightly less than 30s to be safe)
        min_chunk_duration = 15.0     # Minimum chunk duration (don't split if chunk would be too small)
        max_chunk_duration = 30.0     # Maximum chunk duration (hard limit)
        overlap_duration = 3.0        # Increased overlap for better stitching
        
        total_samples = waveform.shape[1]
        total_duration = total_samples / sample_rate
        
        logger.info(f"Processing {total_duration:.2f}s audio with intelligent chunking at natural speech pauses")
        
        # Find natural speech pause points for chunking
        chunks = []
        chunk_timestamps = []
        
        start_sample = 0
        chunk_number = 0
        
        while start_sample < total_samples:
            # Calculate target end sample
            target_end_sample = min(start_sample + int(target_chunk_duration * sample_rate), total_samples)
            
            # Look for natural pause points within a reasonable range
            search_start = max(target_end_sample - int(5.0 * sample_rate), start_sample + int(min_chunk_duration * sample_rate))
            search_end = min(target_end_sample + int(5.0 * sample_rate), total_samples, start_sample + int(max_chunk_duration * sample_rate))
            
            # Find the best pause point in the search range
            best_pause_sample = find_natural_pause_point(waveform, search_start, search_end, sample_rate)
            
            # If no good pause found, use target end
            end_sample = best_pause_sample if best_pause_sample else target_end_sample
            
            # Extract chunk
            chunk_waveform = waveform[:, start_sample:end_sample]
            
            # Calculate timestamps for this chunk
            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate
            
            chunks.append(chunk_waveform)
            chunk_timestamps.append((start_time, end_time))
            
            chunk_number += 1
            pause_indicator = "📍 (natural pause)" if best_pause_sample else "✂️ (forced split)"
            logger.info(f"Chunk {chunk_number}: {start_time:.2f}s - {end_time:.2f}s ({chunk_waveform.shape[1]/sample_rate:.2f}s) {pause_indicator}")
            
            # Move to next chunk with overlap
            if end_sample == total_samples:
                break  # We've reached the end
            start_sample = max(0, end_sample - int(overlap_duration * sample_rate))
        
        # Process each chunk
        all_segments = []
        
        for i, (chunk_waveform, (start_time, end_time)) in enumerate(zip(chunks, chunk_timestamps)):
            logger.info(f"Transcribing chunk {i+1}/{len(chunks)} ({start_time:.2f}s - {end_time:.2f}s)")
            
            try:
                # Convert chunk to numpy
                chunk_array = chunk_waveform.numpy().flatten()
                
                # Skip very quiet chunks
                rms_energy = np.sqrt(np.mean(chunk_array ** 2))
                if rms_energy < 0.01:
                    logger.info(f"Chunk {i+1} is too quiet (RMS: {rms_energy:.4f}), skipping")
                    continue
                
                # Transcribe chunk with enhanced settings for better timestamps
                chunk_result = whisper_model(
                    chunk_array, 
                    return_timestamps=True,
                    chunk_length_s=None,  # Let Whisper handle its own chunking for short segments
                    stride_length_s=None, # Disable stride for individual chunks
                    generate_kwargs={
                        "language": "en",           # Set language to avoid language detection warnings
                        "task": "transcribe",       # Explicit task to avoid translation warnings
                        "return_timestamps": True,  # Ensure timestamps are generated
                        "word_timestamps": True     # Enable word-level timestamps for better accuracy
                    }
                )
                
                # Process chunk results
                if chunk_result and 'chunks' in chunk_result:
                    for segment in chunk_result['chunks']:
                        # Adjust timestamps to global time
                        if 'timestamp' in segment and segment['timestamp']:
                            segment_start = segment['timestamp'][0] + start_time if segment['timestamp'][0] else start_time
                            segment_end = segment['timestamp'][1] + start_time if segment['timestamp'][1] else end_time
                        else:
                            segment_start = start_time
                            segment_end = end_time
                        
                        # Add segment with adjusted timestamps
                        all_segments.append({
                            'timestamp': [segment_start, segment_end],
                            'text': segment.get('text', '').strip()
                        })
                
                elif chunk_result and 'text' in chunk_result and chunk_result['text'].strip():
                    # Single text result
                    all_segments.append({
                        'timestamp': [start_time, end_time],
                        'text': chunk_result['text'].strip()
                    })
                
            except Exception as e:
                logger.warning(f"Error processing chunk {i+1}: {e}")
                continue
        
        # Merge overlapping segments and deduplicate
        merged_segments = merge_overlapping_segments(all_segments)
        
        # Create final result
        full_text = " ".join([seg.get('text', '') for seg in merged_segments if seg.get('text', '').strip()])
        
        logger.info(f"Chunked transcription complete: {len(merged_segments)} segments, {len(full_text)} characters")
        
        return {
            'chunks': merged_segments,
            'text': full_text
        }
        
    except Exception as e:
        logger.error(f"Chunked transcription error: {e}")
        raise


def merge_overlapping_segments(segments):
    """
    Enhanced merging of overlapping segments from chunked transcription with intelligent boundary detection.
    """
    if not segments:
        return []
    
    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x['timestamp'][0])
    
    merged = []
    
    for segment in sorted_segments:
        text = segment.get('text', '').strip()
        if not text:
            continue
            
        # Check if this segment overlaps with the last merged segment
        if merged:
            last_segment = merged[-1]
            last_end = last_segment['timestamp'][1]
            current_start = segment['timestamp'][0]
            
            # Calculate overlap
            overlap_duration = max(0, last_end - current_start)
            
            if overlap_duration > 0:
                # Analyze text overlap to determine best merge strategy
                last_text = last_segment.get('text', '').strip()
                current_text = text
                
                # Check for exact duplicates or near duplicates
                if current_text == last_text:
                    # Exact duplicate, skip
                    continue
                
                # Check for partial overlaps
                last_words = last_text.lower().split()
                current_words = current_text.lower().split()
                
                # Find overlapping words at boundaries
                overlap_words = find_word_overlap(last_words, current_words)
                
                if overlap_words > len(current_words) * 0.3:  # Significant word overlap
                    # Try to merge intelligently
                    merged_text = smart_text_merge(last_text, current_text, overlap_words)
                    
                    if merged_text:
                        # Update the last segment with merged text and extended end time
                        last_segment['text'] = merged_text
                        last_segment['timestamp'][1] = segment['timestamp'][1]
                        continue
                
                # If overlap is small, adjust boundary to minimize cutoff
                if overlap_duration < 1.0:  # Less than 1 second overlap
                    # Adjust the boundary to the midpoint
                    midpoint = (last_end + current_start) / 2
                    last_segment['timestamp'][1] = midpoint
                    segment['timestamp'][0] = midpoint
        
        merged.append(segment)
    
    return merged


def find_word_overlap(words1, words2):
    """Find overlapping words between two word lists."""
    # Check for overlap at the end of words1 and beginning of words2
    max_overlap = min(len(words1), len(words2))
    
    for i in range(max_overlap, 0, -1):
        if words1[-i:] == words2[:i]:
            return i
    
    return 0


def smart_text_merge(text1, text2, overlap_words):
    """Intelligently merge two texts with overlapping words."""
    words1 = text1.split()
    words2 = text2.split()
    
    if overlap_words > 0 and overlap_words <= len(words1) and overlap_words <= len(words2):
        # Remove overlapping words from the second text
        merged_words = words1 + words2[overlap_words:]
        return ' '.join(merged_words)
    
    # Fallback: simple concatenation
    return f"{text1} {text2}"


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
    diarization: Any,
    waveform: torch.Tensor = None,
    sample_rate: int = None
) -> List[Dict[str, Any]]:
    """Merge Whisper transcription with pyannote diarization and speaker embedding identification."""
    try:
        segments = []
        
        if 'chunks' in transcription:
            # Process chunks with timestamps
            for chunk in transcription['chunks']:
                start_time = chunk.get('timestamp', [0, 0])[0]
                end_time = chunk.get('timestamp', [0, 0])[1]
                text = chunk.get('text', '').strip()
                
                if not text:
                    continue
                
                # Get speaker from diarization
                diarization_speaker = find_speaker_for_segment(diarization, start_time, end_time) if diarization else "SPEAKER_00"
                
                # Try to identify speaker using embeddings if available
                embedding_speaker = None
                embedding_confidence = 0.0
                
                if speaker_embedding_manager and waveform is not None and sample_rate is not None:
                    try:
                        # Extract audio segment for this text chunk
                        start_sample = int(start_time * sample_rate)
                        end_sample = int(end_time * sample_rate)
                        
                        if start_sample < waveform.shape[1] and end_sample <= waveform.shape[1]:
                            segment_waveform = waveform[:, start_sample:end_sample]
                            segment_audio = segment_waveform.squeeze().numpy()
                            
                            # Only try identification if segment is long enough
                            if len(segment_audio) > sample_rate * 0.5:  # At least 0.5 seconds
                                speaker_id, confidence, is_new = speaker_embedding_manager.identify_or_create_speaker(segment_audio)
                                
                                if speaker_id in speaker_embedding_manager.speaker_profiles:
                                    embedding_speaker = speaker_embedding_manager.speaker_profiles[speaker_id].name
                                    embedding_confidence = confidence
                                    
                                    logger.debug(f"Embedding identification: {embedding_speaker} (confidence: {confidence:.3f})")
                    
                    except Exception as e:
                        logger.debug(f"Error in embedding identification for segment: {e}")
                
                # Choose the best speaker identification
                final_speaker = embedding_speaker if embedding_speaker and embedding_confidence > 0.6 else diarization_speaker
                
                segments.append({
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "speaker": final_speaker,
                    "embedding_confidence": embedding_confidence,
                    "embedding_speaker": embedding_speaker,
                    "diarization_speaker": diarization_speaker
                })
        
        else:
            # Fallback for transcription without chunks
            text = transcription.get('text', '').strip()
            if text:
                # Try speaker identification on the full audio if available
                embedding_speaker = "SPEAKER_00"
                embedding_confidence = 0.0
                
                if speaker_embedding_manager and waveform is not None:
                    try:
                        audio_numpy = waveform.squeeze().numpy()
                        speaker_id, confidence, is_new = speaker_embedding_manager.identify_or_create_speaker(audio_numpy)
                        
                        if speaker_id in speaker_embedding_manager.speaker_profiles:
                            embedding_speaker = speaker_embedding_manager.speaker_profiles[speaker_id].name
                            embedding_confidence = confidence
                    
                    except Exception as e:
                        logger.debug(f"Error in full-audio embedding identification: {e}")
                
                segments.append({
                    "start": 0.0,
                    "end": waveform.shape[1] / sample_rate if waveform is not None and sample_rate else 0.0,
                    "text": text,
                    "speaker": embedding_speaker,
                    "embedding_confidence": embedding_confidence,
                    "embedding_speaker": embedding_speaker,
                    "diarization_speaker": "SPEAKER_00"
                })
        
        return segments
        
    except Exception as e:
        logger.error(f"Error merging transcription and diarization: {e}")
        # Fallback: return basic transcription without speaker info
        text = transcription.get('text', '') if transcription else ''
        return [{
            "start": 0.0,
            "end": 0.0,
            "text": text,
            "speaker": "SPEAKER_00",
            "embedding_confidence": 0.0,
            "embedding_speaker": None,
            "diarization_speaker": "SPEAKER_00"
        }] if text else []


def find_speaker_for_segment(diarization: Any, start_time: float, end_time: float) -> str:
    """Find the dominant speaker for a given time segment."""
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


@app.post("/speakers/migrate_to_enhanced")
async def migrate_to_enhanced_database(dry_run: bool = False):
    """
    Migrate from legacy speaker database to enhanced v2 architecture
    
    Args:
        dry_run: If True, only analyze what would be migrated without making changes
        
    Returns:
        Migration results and statistics
    """
    if enhanced_speaker_integration is None:
        raise HTTPException(status_code=503, detail="Enhanced speaker integration not available")
    
    try:
        results = await enhanced_speaker_integration.migrate_from_legacy_database(dry_run=dry_run)
        
        if dry_run:
            return {
                "status": "migration_analysis_complete",
                "results": results,
                "message": f"Analysis complete: {results['total_speakers_found']} speakers found, "
                          f"{results['speakers_migrated']} would be migrated"
            }
        else:
            return {
                "status": "migration_complete",
                "results": results,
                "message": f"Migration complete: {results['speakers_migrated']} speakers migrated, "
                          f"{results['speakers_merged']} duplicates merged"
            }
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        stats = await enhanced_speaker_integration.get_enhanced_speaker_stats()
        
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
    if enhanced_speaker_integration is None:
        # Fallback to legacy feedback processing
        logger.warning("Enhanced integration not available, falling back to legacy processing")
        return await provide_speaker_name_mapping_feedback(corrections)
    
    try:
        results = await enhanced_speaker_integration.enhanced_speaker_correction_feedback(corrections)
        
        return {
            "status": "enhanced_feedback_complete",
            "results": results,
            "enhanced_processing": True,
            "message": f"Processed {len(corrections)} corrections using enhanced database"
        }
        
    except Exception as e:
        logger.error(f"Enhanced feedback failed: {e}")
        # Fallback to legacy processing
        logger.info("Falling back to legacy feedback processing")
        try:
            legacy_results = await provide_speaker_name_mapping_feedback(corrections)
            legacy_results["fallback_used"] = True
            return legacy_results
        except Exception as fallback_error:
            logger.error(f"Legacy fallback also failed: {fallback_error}")
            raise HTTPException(status_code=500, detail=f"Both enhanced and legacy feedback failed: {e}")


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
        # Use enhanced database to save with corrections (no learning feedback)
        saved_file = enhanced_speaker_integration.enhanced_db.save_transcription_with_corrections(
            transcription_data=transcription_data,
            speaker_corrections=speaker_corrections,
            output_file=output_file
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
    Get comprehensive status of the speaker management system
    
    Returns:
        Status of both legacy and enhanced systems, migration recommendations
    """
    try:
        status = {
            "legacy_system": {
                "available": speaker_embedding_manager is not None,
                "speaker_count": 0,
                "total_embeddings": 0
            },
            "enhanced_system": {
                "available": enhanced_speaker_integration is not None,
                "speaker_count": 0,
                "migration_completed": False
            },
            "recommendations": []
        }
        
        # Check legacy system
        if speaker_embedding_manager:
            legacy_speakers = speaker_embedding_manager.speaker_profiles
            status["legacy_system"]["speaker_count"] = len(legacy_speakers)
            status["legacy_system"]["total_embeddings"] = sum(
                len(profile.embeddings) for profile in legacy_speakers.values()
            )
        
        # Check enhanced system
        if enhanced_speaker_integration:
            enhanced_stats = await enhanced_speaker_integration.get_enhanced_speaker_stats()
            status["enhanced_system"]["speaker_count"] = enhanced_stats["total_speakers"]
            status["enhanced_system"]["migration_completed"] = enhanced_stats["migration_completed"]
        
        # Generate recommendations
        if (status["legacy_system"]["speaker_count"] > 0 and 
            status["enhanced_system"]["speaker_count"] == 0):
            status["recommendations"].append("Migration to enhanced database recommended")
        
        if (status["legacy_system"]["speaker_count"] > 0 and 
            status["enhanced_system"]["speaker_count"] > 0 and
            not status["enhanced_system"]["migration_completed"]):
            status["recommendations"].append("Complete migration to enhanced database")
        
        if status["enhanced_system"]["speaker_count"] > 0:
            status["recommendations"].append("Enhanced speaker management active")
        
        return {
            "status": "system_status_retrieved",
            "system_status": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper function for legacy fallback
async def provide_speaker_name_mapping_feedback(corrections: Dict[str, str]):
    """Legacy speaker name mapping feedback - for fallback use"""
    if speaker_embedding_manager is None:
        raise HTTPException(status_code=503, detail="Speaker embedding manager not available")
    
    try:
        results = {
            'processed_corrections': 0,
            'speakers_created': 0,
            'speakers_merged': 0,
            'speakers_renamed': 0,
            'errors': []
        }
        
        for old_speaker_id, new_display_name in corrections.items():
            try:
                existing_speaker_id = speaker_embedding_manager.get_speaker_by_name(new_display_name)
                
                if existing_speaker_id and existing_speaker_id != old_speaker_id:
                    # Merge speakers
                    if speaker_embedding_manager.merge_speakers(old_speaker_id, existing_speaker_id):
                        results['speakers_merged'] += 1
                elif old_speaker_id in speaker_embedding_manager.speaker_profiles:
                    # Rename existing speaker
                    if speaker_embedding_manager.update_speaker_name(old_speaker_id, new_display_name):
                        results['speakers_renamed'] += 1
                
                results['processed_corrections'] += 1
                
            except Exception as e:
                results['errors'].append(f"Error processing {old_speaker_id}: {e}")
        
        return {
            "status": "legacy_feedback_processed",
            "results": results,
            "enhanced_processing": False
        }
        
    except Exception as e:
        logger.error(f"Legacy feedback processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        access_log=False
    ) 