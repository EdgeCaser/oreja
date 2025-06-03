#!/usr/bin/env python3
"""
Script to make enhanced transcription features the default for all transcriptions.
This modifies the existing /transcribe endpoint to include sentiment analysis and audio features.
"""

def modify_server_for_enhanced_default():
    """
    Instructions and code to make enhanced features default in server.py
    """
    
    print("🔧 MAKING ENHANCED FEATURES DEFAULT")
    print("=" * 50)
    
    modification_instructions = """
    
    1. MODIFY server.py /transcribe endpoint:
    
    Add these imports at the top:
    ```python
    from enhanced_server_integration import EnhancedTranscriptionService
    ```
    
    2. ADD enhanced service initialization after loading models:
    ```python
    # Add after model loading in load_models()
    global enhanced_service
    enhanced_service = EnhancedTranscriptionService(
        sentiment_model="vader",  # Fast default
        enable_audio_features=True
    )
    ```
    
    3. MODIFY the /transcribe endpoint to include enhancements:
    ```python
    @app.post("/transcribe")
    async def transcribe_audio(audio: UploadFile = File(...)) -> Dict[str, Any]:
        # ... existing transcription logic ...
        
        # AFTER getting the basic result, before returning:
        try:
            # Read audio data for enhancement
            audio_data = await audio.read()
            waveform, sample_rate = load_audio_from_bytes(audio_data)
            
            # Enhance the result with sentiment and audio features
            if enhanced_service:
                result = enhanced_service.enhance_transcription_result(
                    result, waveform, sample_rate
                )
        except Exception as e:
            logger.warning(f"Enhancement failed, returning basic result: {e}")
            # Continue with basic result if enhancement fails
        
        return result
    ```
    
    4. BENEFITS:
    ✅ All transcriptions automatically get sentiment analysis
    ✅ All transcriptions automatically get audio features  
    ✅ Backward compatible - old clients still work
    ✅ Graceful fallback if enhancement fails
    ✅ No breaking changes to existing API
    
    """
    
    print(modification_instructions)

def create_enhanced_default_server():
    """
    Alternative: Create a new server file with enhanced features as default
    """
    
    server_template = '''
#!/usr/bin/env python3
"""
Enhanced Oreja Server - Sentiment Analysis and Audio Features by Default
"""

import asyncio
import io
import logging
import time
from typing import Dict, Any
import warnings

import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import existing server components
from server import (
    load_models, load_audio_from_bytes, run_transcription, 
    run_diarization, merge_transcription_and_diarization
)

# Import enhanced features
from enhanced_server_integration import EnhancedTranscriptionService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Oreja Enhanced Audio Processing API",
    description="Local audio transcription with sentiment analysis and audio features",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global enhanced service
enhanced_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and enhanced features on startup"""
    global enhanced_service
    
    # Load basic transcription models
    await load_models()
    
    # Initialize enhanced features
    try:
        enhanced_service = EnhancedTranscriptionService(
            sentiment_model="vader",  # Fast and reliable
            enable_audio_features=True
        )
        logger.info("✅ Enhanced features initialized successfully")
    except Exception as e:
        logger.warning(f"⚠️ Enhanced features failed to initialize: {e}")
        logger.warning("Continuing with basic transcription only")

@app.post("/transcribe")
async def transcribe_audio_enhanced(audio: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Enhanced transcription with sentiment analysis and audio features BY DEFAULT
    """
    start_time = time.time()
    
    try:
        # Validate file
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read audio data
        audio_data = await audio.read()
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        logger.info(f"Processing audio file: {audio.filename}")
        
        # Load and process audio
        waveform, sample_rate = load_audio_from_bytes(audio_data)
        
        # Validate audio length
        duration = waveform.shape[1] / sample_rate
        if duration < 0.1:
            raise HTTPException(status_code=400, detail=f"Audio too short: {duration:.2f}s")
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Run basic transcription and diarization
        transcription_task = asyncio.create_task(run_transcription(waveform, sample_rate))
        
        # Run diarization if available
        from server import diarization_pipeline
        if diarization_pipeline is not None:
            diarization_task = asyncio.create_task(run_diarization(waveform, sample_rate))
            transcription_result, diarization_result = await asyncio.gather(
                transcription_task, diarization_task
            )
        else:
            transcription_result = await transcription_task
            diarization_result = None
        
        # Check for skipped transcription
        if transcription_result and "skipped_reason" in transcription_result:
            return {
                "segments": [],
                "full_text": "",
                "processing_time": time.time() - start_time,
                "timestamp": time.time(),
                "audio_duration": duration,
                "sample_rate": sample_rate,
                "skipped_reason": transcription_result["skipped_reason"]
            }
        
        # Merge transcription with speakers
        segments = merge_transcription_and_diarization(
            transcription_result, diarization_result, waveform, sample_rate
        )
        
        # Generate basic result
        basic_result = {
            "segments": segments,
            "full_text": " ".join([segment["text"] for segment in segments]),
            "processing_time": time.time() - start_time,
            "timestamp": time.time(),
            "audio_duration": duration,
            "sample_rate": sample_rate
        }
        
        # 🚀 ENHANCE WITH SENTIMENT AND AUDIO FEATURES
        if enhanced_service:
            try:
                enhanced_result = enhanced_service.enhance_transcription_result(
                    basic_result, waveform, sample_rate
                )
                logger.info("✅ Enhanced features applied successfully")
                return enhanced_result
            except Exception as e:
                logger.warning(f"⚠️ Enhancement failed, returning basic result: {e}")
        
        # Return basic result if enhancement not available
        return basic_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "enhanced_features": enhanced_service is not None,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    print("📝 ENHANCED DEFAULT SERVER TEMPLATE")
    print("=" * 50)
    print("Save this as 'server_enhanced_default.py':")
    print(server_template)
    
    return server_template

def show_configuration_options():
    """Show different configuration options for enhanced features"""
    
    print("\n🎛️ CONFIGURATION OPTIONS")
    print("=" * 50)
    
    configurations = {
        "Fast & Lightweight": {
            "sentiment_model": "vader",
            "enable_audio_features": False,
            "description": "Only sentiment analysis, very fast processing"
        },
        "Balanced (Recommended)": {
            "sentiment_model": "vader", 
            "enable_audio_features": True,
            "description": "Sentiment + basic audio features, good performance"
        },
        "High Accuracy": {
            "sentiment_model": "transformer",
            "enable_audio_features": True, 
            "description": "Best accuracy, slower processing"
        },
        "Audio Focus": {
            "sentiment_model": "vader",
            "enable_audio_features": True,
            "description": "Emphasis on audio characteristics and voice analysis"
        }
    }
    
    for name, config in configurations.items():
        print(f"\n{name}:")
        print(f"  Sentiment Model: {config['sentiment_model']}")
        print(f"  Audio Features: {config['enable_audio_features']}")
        print(f"  Use Case: {config['description']}")
        
        code = f"""
enhanced_service = EnhancedTranscriptionService(
    sentiment_model="{config['sentiment_model']}",
    enable_audio_features={config['enable_audio_features']}
)"""
        print(f"  Code: {code}")

def main():
    """Main function to show all options"""
    print("🎙️ MAKING ENHANCED FEATURES DEFAULT")
    print("=" * 60)
    print("Choose how to integrate enhanced features as default:")
    print()
    
    modify_server_for_enhanced_default()
    
    print("\n" + "=" * 60)
    create_enhanced_default_server()
    
    print("\n" + "=" * 60)
    show_configuration_options()
    
    print("\n🎯 RECOMMENDATIONS")
    print("=" * 50)
    print("For LIVE transcription: Use 'Fast & Lightweight' config")
    print("For RECORDED transcription: Use 'Balanced' config") 
    print("For ANALYSIS purposes: Use 'High Accuracy' config")

if __name__ == "__main__":
    main() 