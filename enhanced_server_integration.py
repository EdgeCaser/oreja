#!/usr/bin/env python3
"""
Enhanced Server Integration
Adds sentiment analysis and audio features to the existing transcription server
"""

import logging
from typing import Dict, Any, Optional
import torch

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
    from fastapi import HTTPException, File, UploadFile, Query
    from fastapi.responses import JSONResponse
    
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
            # Import the existing transcription function
            from server import transcribe_audio, load_audio_from_bytes
            
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