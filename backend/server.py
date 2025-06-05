"""
FastAPI server for Oreja audio transcription and diarization.
Processes audio in memory using local Hugging Face models with no cloud interaction.

PRIVACY GUARANTEE: NO AUDIO DATA EVER LEAVES THIS MACHINE
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
enhanced_service = None


def initialize_models():
    """Initialize all models and set up the device."""
    global device, whisper_model, diarization_pipeline, embedding_model, speaker_embedding_manager, enhanced_service
    
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
    
    # Initialize offline speaker embedding manager
    try:
        logger.info("Initializing offline speaker embedding manager...")
        speaker_embedding_manager = OfflineSpeakerEmbeddingManager()
        logger.info("✓ Offline speaker embedding manager initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize speaker embedding manager: {e}")
        logger.warning("Continuing without offline speaker embeddings")
        speaker_embedding_manager = None
    
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
    if speaker_embedding_manager is None:
        raise HTTPException(status_code=503, detail="Speaker embedding manager not available")
    
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
    """Run Whisper transcription on audio waveform with enhanced voice activity detection."""
    try:
        if whisper_model is None:
            raise ValueError("Whisper model not loaded")
        
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


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        access_log=False
    ) 