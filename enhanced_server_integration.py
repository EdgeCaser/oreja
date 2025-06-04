#!/usr/bin/env python3
"""
Enhanced Server Integration
Adds sentiment analysis and audio features to the existing transcription server
"""

import logging
import json
import os
from typing import Dict, Any, Optional
import torch
import time
import asyncio
import torchaudio
from fastapi import HTTPException, File, UploadFile, Query, Body
from fastapi.responses import JSONResponse
from datetime import datetime

try:
    from enhanced_transcription_processor import EnhancedTranscriptionProcessor
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    logging.warning("Enhanced features not available. Install requirements: pip install -r requirements_enhanced.txt")

logger = logging.getLogger(__name__)

class EnhancedTranscriptionService:
    """
    Service wrapper that adds enhanced features to transcription results
    """
    
    def __init__(self, sentiment_model: str = "vader", enable_audio_features: bool = True):
        self.sentiment_model = sentiment_model
        self.enable_audio_features = enable_audio_features
        self.enhanced_processor = None
        
        if ENHANCED_FEATURES_AVAILABLE:
            try:
                self.enhanced_processor = EnhancedTranscriptionProcessor(sentiment_model=sentiment_model)
                logger.info(f"Enhanced transcription features enabled with {sentiment_model} sentiment model")
            except Exception as e:
                logger.error(f"Failed to initialize enhanced features: {e}")
                self.enhanced_processor = None
        else:
            logger.info("Enhanced features disabled - missing dependencies")
    
    def enhance_transcription_result(self, transcription_result: Dict[str, Any], 
                                   waveform: Optional[torch.Tensor] = None, 
                                   sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Add sentiment analysis and audio features to transcription result
        """
        if not self.enhanced_processor:
            # Return original result if enhanced features not available
            return transcription_result
        
        try:
            if waveform is not None and self.enable_audio_features:
                # Full enhancement with audio features
                enhanced_result = self.enhanced_processor.process_enhanced_transcription(
                    transcription_result, waveform, sample_rate
                )
            else:
                # Sentiment analysis only
                enhanced_result = self._add_sentiment_only(transcription_result)
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error during transcription enhancement: {e}")
            # Return original result on error
            return transcription_result
    
    def _add_sentiment_only(self, transcription_result: Dict[str, Any]) -> Dict[str, Any]:
        """Add sentiment analysis without audio features"""
        enhanced_segments = []
        
        for segment in transcription_result.get("segments", []):
            text = segment.get("text", "")
            enhanced_segment = segment.copy()
            
            # Add sentiment analysis
            if text.strip():
                sentiment_analysis = self.enhanced_processor.analyze_sentiment(text)
                enhanced_segment["sentiment_analysis"] = sentiment_analysis
            
            enhanced_segments.append(enhanced_segment)
        
        # Analyze conversation dynamics
        conversation_analysis = self.enhanced_processor.analyze_conversation_dynamics(enhanced_segments)
        
        # Create enhanced result
        enhanced_result = transcription_result.copy()
        enhanced_result["segments"] = enhanced_segments
        enhanced_result["conversation_analysis"] = conversation_analysis
        enhanced_result["enhancement_info"] = {
            "sentiment_model": self.sentiment_model,
            "features_extracted": ["sentiment", "conversation_dynamics"],
            "processing_timestamp": self.enhanced_processor.process_enhanced_transcription.__defaults__[0] if hasattr(self.enhanced_processor, 'process_enhanced_transcription') else None
        }
        
        return enhanced_result


def add_enhanced_endpoints(app, enhanced_service: Optional[EnhancedTranscriptionService] = None):
    """
    Add enhanced transcription endpoints to FastAPI app
    """
    # Import required functions from server
    from server import (
        load_audio_from_bytes, run_transcription, run_diarization, 
        merge_transcription_and_diarization, MIN_AUDIO_LENGTH, 
        MAX_AUDIO_LENGTH, SAMPLE_RATE, diarization_pipeline, transcribe_audio
    )
    
    if enhanced_service is None:
        enhanced_service = EnhancedTranscriptionService()
    
    @app.post("/transcribe_enhanced")
    async def transcribe_audio_enhanced(
        audio: UploadFile = File(...),
        sentiment_model: str = Query("vader", description="Sentiment model: vader, textblob, or transformer"),
        include_audio_features: bool = Query(True, description="Include audio feature analysis")
    ) -> Dict[str, Any]:
        """
        Enhanced transcription with sentiment analysis and audio features
        """
        if not ENHANCED_FEATURES_AVAILABLE:
            raise HTTPException(
                status_code=501, 
                detail="Enhanced features not available. Install requirements: pip install -r requirements_enhanced.txt"
            )
        
        try:
            # Get basic transcription first
            basic_result = await transcribe_audio(audio)
            
            # Read audio data for enhancement
            audio_data = await audio.read()
            waveform, sample_rate = load_audio_from_bytes(audio_data)
            
            # Create temporary enhanced service with requested settings
            temp_service = EnhancedTranscriptionService(
                sentiment_model=sentiment_model,
                enable_audio_features=include_audio_features
            )
            
            # Enhance the result
            enhanced_result = temp_service.enhance_transcription_result(
                basic_result, waveform, sample_rate
            )
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Enhanced transcription failed: {e}")
            raise HTTPException(status_code=500, detail=f"Enhanced transcription failed: {e}")
    
    @app.post("/analyze_sentiment")
    async def analyze_text_sentiment(
        text: str = Query(..., description="Text to analyze"),
        sentiment_model: str = Query("vader", description="Sentiment model: vader, textblob, or transformer")
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of provided text
        """
        if not ENHANCED_FEATURES_AVAILABLE:
            raise HTTPException(
                status_code=501,
                detail="Enhanced features not available. Install requirements: pip install -r requirements_enhanced.txt"
            )
        
        try:
            temp_service = EnhancedTranscriptionService(sentiment_model=sentiment_model)
            
            if temp_service.enhanced_processor:
                sentiment_result = temp_service.enhanced_processor.analyze_sentiment(text)
                return {
                    "text": text,
                    "sentiment_analysis": sentiment_result,
                    "model_used": sentiment_model
                }
            else:
                raise HTTPException(status_code=500, detail="Sentiment analyzer not available")
                
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {e}")
    
    @app.get("/enhanced_features_status")
    async def get_enhanced_features_status():
        """
        Check status of enhanced features
        """
        return {
            "enhanced_features_available": ENHANCED_FEATURES_AVAILABLE,
            "sentiment_models_available": {
                "vader": True,  # Always available
                "textblob": enhanced_service.enhanced_processor.textblob_available if enhanced_service.enhanced_processor else False,
                "transformer": enhanced_service.enhanced_processor.transformers_available if enhanced_service.enhanced_processor else False
            },
            "audio_features_available": ENHANCED_FEATURES_AVAILABLE,
            "message": "Enhanced features provide sentiment analysis and audio characteristics" if ENHANCED_FEATURES_AVAILABLE else "Install enhanced requirements to enable advanced features"
        }

    @app.post("/transcribe_with_summary")
    async def transcribe_with_summary_and_annotations(
        audio: UploadFile = File(...),
        include_summary: bool = Query(True, description="Include conversation summary"),
        include_annotations: bool = Query(True, description="Include conversation annotations"),
        summary_types: str = Query("overall,by_speaker,key_points", description="Comma-separated summary types"),
        annotation_types: str = Query("topics,action_items,questions_answers,decisions", description="Comma-separated annotation types"),
        sentiment_model: str = Query("vader", description="Sentiment model: vader, textblob, or transformer"),
        include_audio_features: bool = Query(True, description="Include audio feature analysis")
    ) -> Dict[str, Any]:
        """
        Transcribe audio with enhanced summarization and annotation features
        
        This endpoint provides everything the regular transcription does PLUS:
        - Conversation summaries (overall, by speaker, by time, key points)
        - Conversation annotations (topics, action items, Q&A pairs, decisions, emotional moments)
        - Enhanced sentiment analysis and audio features
        """
        if not ENHANCED_FEATURES_AVAILABLE:
            raise HTTPException(
                status_code=501,
                detail="Enhanced features not available. Install requirements: pip install -r requirements_enhanced.txt"
            )
        
        start_time = time.time()
        
        try:
            # Validate file
            if not audio.filename:
                raise HTTPException(status_code=400, detail="No file provided")
            
            # Read audio data
            audio_data = await audio.read()
            if len(audio_data) == 0:
                raise HTTPException(status_code=400, detail="Empty audio file")
            
            logger.info(f"Processing audio with summarization: {audio.filename}, size: {len(audio_data)} bytes")
            
            # Load audio with torchaudio
            waveform, sample_rate = load_audio_from_bytes(audio_data)
            
            # Validate and process audio (same as regular transcription)
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
            
            # Run transcription and diarization
            transcription_task = asyncio.create_task(
                run_transcription(waveform, sample_rate)
            )
            
            if diarization_pipeline is not None:
                diarization_task = asyncio.create_task(
                    run_diarization(waveform, sample_rate)
                )
                transcription_result, diarization_result = await asyncio.gather(
                    transcription_task, diarization_task
                )
            else:
                transcription_result = await transcription_task
                diarization_result = None
            
            # Check if transcription was skipped
            if transcription_result and "skipped_reason" in transcription_result:
                processing_time = time.time() - start_time
                result = {
                    "segments": [],
                    "full_text": "",
                    "processing_time": processing_time,
                    "timestamp": time.time(),
                    "audio_duration": duration,
                    "sample_rate": sample_rate,
                    "skipped_reason": transcription_result["skipped_reason"],
                    "enhanced_features": False
                }
                return result
            
            # Merge transcription with speaker information
            segments = merge_transcription_and_diarization(
                transcription_result, diarization_result, waveform, sample_rate
            )
            
            # Generate full text
            full_text = " ".join([segment["text"] for segment in segments])
            
            # Create basic transcription result
            basic_result = {
                "segments": segments,
                "full_text": full_text,
                "processing_time": 0,  # Will update later
                "timestamp": time.time(),
                "audio_duration": duration,
                "sample_rate": sample_rate
            }
            
            # Parse summary and annotation types
            summary_type_list = [s.strip() for s in summary_types.split(",") if s.strip()]
            annotation_type_list = [a.strip() for a in annotation_types.split(",") if a.strip()]
            
            # Create enhanced processor
            enhanced_service = EnhancedTranscriptionService(
                sentiment_model=sentiment_model,
                include_audio_features=include_audio_features
            )
            
            if enhanced_service.enhanced_processor:
                # Process with enhanced features including summarization and annotations
                enhanced_result = enhanced_service.enhanced_processor.process_with_summary_and_annotations(
                    basic_result, 
                    waveform, 
                    sample_rate,
                    include_summary=include_summary,
                    include_annotations=include_annotations,
                    summary_types=summary_type_list,
                    annotation_types=annotation_type_list
                )
                
                processing_time = time.time() - start_time
                enhanced_result["processing_time"] = processing_time
                
                logger.info(f"Enhanced transcription with summarization completed in {processing_time:.2f}s")
                return enhanced_result
            else:
                # Fallback to basic result if enhanced processor not available
                processing_time = time.time() - start_time
                basic_result["processing_time"] = processing_time
                basic_result["enhanced_features"] = False
                basic_result["error"] = "Enhanced processor not available"
                
                return basic_result
                
        except Exception as e:
            logger.error(f"Enhanced transcription with summarization failed: {e}")
            raise HTTPException(status_code=500, detail=f"Enhanced transcription failed: {e}")

    @app.post("/summarize_transcription")
    async def summarize_existing_transcription(
        transcription: Dict[str, Any] = Body(..., description="Transcription result to summarize"),
        summary_types: str = Query("overall,by_speaker,key_points", description="Comma-separated summary types"),
        annotation_types: str = Query("topics,action_items,questions_answers", description="Comma-separated annotation types"),
        include_annotations: bool = Query(True, description="Include conversation annotations")
    ) -> Dict[str, Any]:
        """
        Generate summary and annotations for an existing transcription
        
        This endpoint takes an existing transcription result and adds:
        - Conversation summaries
        - Conversation annotations
        
        Useful for post-processing transcriptions that were created without these features.
        """
        if not ENHANCED_FEATURES_AVAILABLE:
            raise HTTPException(
                status_code=501,
                detail="Enhanced features not available. Install requirements: pip install -r requirements_enhanced.txt"
            )
        
        try:
            # Validate transcription input
            if not transcription.get("segments"):
                raise HTTPException(status_code=400, detail="Transcription must contain segments")
            
            # Parse types
            summary_type_list = [s.strip() for s in summary_types.split(",") if s.strip()]
            annotation_type_list = [a.strip() for a in annotation_types.split(",") if a.strip()]
            
            # Create enhanced processor
            enhanced_service = EnhancedTranscriptionService()
            
            if enhanced_service.enhanced_processor:
                # Generate summaries
                summaries = enhanced_service.enhanced_processor.generate_conversation_summary(
                    transcription, summary_type_list
                )
                
                # Generate annotations
                annotations = {}
                if include_annotations:
                    annotations = enhanced_service.enhanced_processor.generate_conversation_annotations(
                        transcription, annotation_type_list
                    )
                
                result = {
                    "original_transcription": transcription,
                    "conversation_summary": summaries,
                    "conversation_annotations": annotations if include_annotations else {},
                    "processing_info": {
                        "summary_types": summary_type_list,
                        "annotation_types": annotation_type_list if include_annotations else [],
                        "processing_timestamp": datetime.now().isoformat()
                    }
                }
                
                return result
            else:
                raise HTTPException(status_code=500, detail="Enhanced processor not available")
                
        except Exception as e:
            logger.error(f"Transcription summarization failed: {e}")
            raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")

    @app.get("/summary_options")
    async def get_summary_options() -> Dict[str, Any]:
        """
        Get available summary and annotation options
        """
        return {
            "summary_types": {
                "overall": "High-level conversation summary with key metrics",
                "by_speaker": "Individual summaries for each speaker",
                "by_time": "Time-based interval summaries",
                "key_points": "Most important points and statements"
            },
            "annotation_types": {
                "topics": "Main topics and themes discussed",
                "action_items": "Tasks, assignments, and action items",
                "questions_answers": "Question and answer pairs",
                "decisions": "Decisions and conclusions reached",
                "emotional_moments": "Emotionally significant moments"
            },
            "sentiment_models": {
                "vader": "Rule-based sentiment analysis (fast, good for social media text)",
                "textblob": "Pattern-based sentiment analysis (balanced)",
                "transformer": "Neural network sentiment analysis (most accurate, slower)"
            }
        }


# Integration example for existing server.py
def integrate_with_existing_server():
    """
    Example of how to integrate enhanced features with the existing server
    """
    print("Enhanced Transcription Integration Guide:")
    print("=" * 50)
    print()
    print("1. Install enhanced dependencies:")
    print("   pip install -r requirements_enhanced.txt")
    print()
    print("2. Add to your server.py:")
    print("   from enhanced_server_integration import EnhancedTranscriptionService, add_enhanced_endpoints")
    print("   enhanced_service = EnhancedTranscriptionService()")
    print("   add_enhanced_endpoints(app, enhanced_service)")
    print()
    print("3. Modify your existing transcribe endpoint:")
    print("   # After getting basic transcription result:")
    print("   enhanced_result = enhanced_service.enhance_transcription_result(result, waveform, sample_rate)")
    print()
    print("4. New endpoints available:")
    print("   POST /transcribe_enhanced - Full enhanced transcription")
    print("   POST /analyze_sentiment - Sentiment analysis only")
    print("   GET /enhanced_features_status - Check feature availability")
    print()
    print("5. Features added to transcription:")
    print("   • Sentiment analysis per segment (positive/negative/neutral)")
    print("   • Speaker emotion tracking")
    print("   • Audio characteristics (volume, speaking rate, stress)")
    print("   • Conversation dynamics (interruptions, participation)")
    print("   • Voice stress indicators")
    print("   • Speaking style analysis")


if __name__ == "__main__":
    integrate_with_existing_server() 