#!/usr/bin/env python3
"""
Enhanced Server Integration
Adds sentiment analysis, audio features, conversation summarization, and
annotation endpoints to the transcription server.

Consolidated file: this used to be a small (233 line) stub in backend/ that
only exposed /transcribe_enhanced, /analyze_sentiment and
/enhanced_features_status, while the full version - with the
/transcribe_with_summary, /summarize_transcription and /summary_options
routes that CONVERSATION_SUMMARIZATION_README.md documents - sat unreachable
at the repo root. This file is now the single, full version.

Summarization engine selection (new): /transcribe_with_summary and
/summarize_transcription accept `use_llm` (default True). When true they try
backend/llm_summarizer.py first, which asks a local Ollama server to write an
abstractive summary (meeting minutes / action items / decisions) from the
speaker-labeled transcript. If Ollama is not running, not reachable, or
returns anything unexpected, that raises cleanly and callers here catch it
and fall back to the extractive EnhancedTranscriptionProcessor summarizer
that was already here. Every response says which engine actually ran via an
"engine" field ("ollama-<model>" or "extractive") so a client can tell the
two apart.
"""

import logging
import threading
import time
import asyncio
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, File, UploadFile, Query, Body
from datetime import datetime

try:
    from enhanced_transcription_processor import EnhancedTranscriptionProcessor
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    EnhancedTranscriptionProcessor = None  # type: ignore[assignment]
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
                                   waveform: Optional[Any] = None,
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
            "processing_timestamp": datetime.now().isoformat()
        }

        return enhanced_result


def _summarize_with_engine(
    segments: List[Dict[str, Any]],
    extractive_source: Dict[str, Any],
    summary_type_list: List[str],
    enhanced_processor: "EnhancedTranscriptionProcessor",
    use_llm: bool,
    llm_style: str,
) -> Dict[str, Any]:
    """
    Produce the "conversation_summary" block, preferring a local Ollama LLM
    and always falling back to the extractive processor.

    Returns a dict that always carries an "engine" key:
      - {"engine": "ollama-<model>", "style": ..., "text": ..., "generated_at": ...}
      - {"engine": "extractive", "overall": {...}, "by_speaker": {...}, ...}
    """
    if use_llm:
        try:
            from llm_summarizer import summarize_with_llm
            llm_result = summarize_with_llm(segments, style=llm_style)
            return {
                "engine": llm_result["engine"],
                "style": llm_result["style"],
                "text": llm_result["text"],
                "generated_at": llm_result["generated_at"],
            }
        except Exception as e:
            # Covers OllamaError, ImportError (llm_summarizer missing), and
            # any unexpected failure talking to Ollama - the transcript is
            # always worth more than the fancier summary.
            logger.info(f"LLM summarization unavailable, falling back to extractive summary: {e}")

    extractive = enhanced_processor.generate_conversation_summary(extractive_source, summary_type_list)
    return {"engine": "extractive", **extractive}


# Building an EnhancedTranscriptionService is expensive: with
# sentiment_model="transformer" it constructs the cardiffnlp RoBERTa pipeline.
# The per-request endpoints below used to build a fresh one every call, so a
# stream of requests reloaded the model each time. Cache one service per
# (sentiment_model, enable_audio_features) combination instead.
_service_cache: Dict[tuple, EnhancedTranscriptionService] = {}
_service_cache_lock = threading.Lock()


def get_enhanced_service(sentiment_model: str = "vader",
                         enable_audio_features: bool = True) -> EnhancedTranscriptionService:
    """Return a cached EnhancedTranscriptionService for these settings."""
    key = (sentiment_model, bool(enable_audio_features))
    with _service_cache_lock:
        service = _service_cache.get(key)
        if service is None:
            service = EnhancedTranscriptionService(
                sentiment_model=sentiment_model,
                enable_audio_features=enable_audio_features
            )
            _service_cache[key] = service
        return service


def add_enhanced_endpoints(app, enhanced_service: Optional[EnhancedTranscriptionService] = None):
    """
    Add enhanced transcription endpoints to FastAPI app.

    `enhanced_service` is optional. When omitted, no service is constructed
    here - registering routes must stay free of side effects because server.py
    calls this at module scope, and building a service used to reach out to
    the network (NLTK corpus download). Handlers resolve a cached service
    lazily instead.
    """
    if enhanced_service is not None:
        # Make the caller-supplied service the cached default so the
        # per-request endpoints reuse it instead of building a second one.
        with _service_cache_lock:
            _service_cache.setdefault(
                (enhanced_service.sentiment_model, bool(enhanced_service.enable_audio_features)),
                enhanced_service
            )

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
            # Import lazily so this module can be imported without server.py
            # already being fully initialized.
            from server import transcribe_audio, load_audio_from_bytes

            # Get basic transcription first
            basic_result = await transcribe_audio(audio)

            # transcribe_audio() above already consumed the whole UploadFile
            # stream via audio.read() - without rewinding, this second read
            # would return b"" and load_audio_from_bytes() would fail on an
            # empty buffer.
            await audio.seek(0)

            # Read audio data for enhancement
            audio_data = await audio.read()
            waveform, sample_rate = load_audio_from_bytes(audio_data)

            # Cached enhanced service for the requested settings (building one
            # per request reloads the sentiment model every time).
            temp_service = get_enhanced_service(
                sentiment_model=sentiment_model,
                enable_audio_features=include_audio_features
            )

            # Enhance the result. This is CPU/disk bound (librosa feature
            # extraction), so keep it off the event loop.
            enhanced_result = await asyncio.to_thread(
                temp_service.enhance_transcription_result,
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
            temp_service = get_enhanced_service(sentiment_model=sentiment_model)

            if temp_service.enhanced_processor:
                sentiment_result = await asyncio.to_thread(
                    temp_service.enhanced_processor.analyze_sentiment, text
                )
                return {
                    "text": text,
                    "sentiment_analysis": sentiment_result,
                    "model_used": sentiment_model
                }
            else:
                raise HTTPException(status_code=500, detail="Sentiment analyzer not available")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {e}")

    @app.get("/enhanced_features_status")
    async def get_enhanced_features_status():
        """
        Check status of enhanced features
        """
        try:
            from llm_summarizer import get_status as get_llm_status
            # get_status() is a synchronous urllib probe of the local Ollama
            # server with a multi-second timeout - never run it on the loop.
            llm_status = await asyncio.to_thread(get_llm_status)
        except Exception as e:
            llm_status = {"available": False, "error": str(e)}

        status_service = enhanced_service if enhanced_service is not None else get_enhanced_service()
        processor = status_service.enhanced_processor

        return {
            "enhanced_features_available": ENHANCED_FEATURES_AVAILABLE,
            "sentiment_models_available": {
                "vader": True,  # Always reported as the default; degrades to neutral if vaderSentiment isn't installed
                "textblob": processor.textblob_available if processor else False,
                "transformer": processor.transformers_available if processor else False
            },
            "audio_features_available": ENHANCED_FEATURES_AVAILABLE,
            "llm_summarization": llm_status,
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
        include_audio_features: bool = Query(True, description="Include audio feature analysis"),
        use_llm: bool = Query(True, description="Prefer a local Ollama LLM for the summary, falling back to extractive summarization"),
        llm_style: str = Query("meeting_minutes", description="LLM summary style: meeting_minutes, action_items, decisions, or summary")
    ) -> Dict[str, Any]:
        """
        Transcribe audio with enhanced summarization and annotation features

        This endpoint provides everything the regular transcription does PLUS:
        - A conversation summary (LLM-generated when Ollama is reachable, else extractive)
        - Conversation annotations (topics, action items, Q&A pairs, decisions, emotional moments)
        - Enhanced sentiment analysis and audio features

        This is a batch/file endpoint, not the live 5-second-chunk path, so
        analysis is on by default (unlike POST /transcribe).
        """
        if not ENHANCED_FEATURES_AVAILABLE:
            raise HTTPException(
                status_code=501,
                detail="Enhanced features not available. Install requirements: pip install -r requirements_enhanced.txt"
            )

        start_time = time.time()

        try:
            # Lazy import: server.py imports this module, so importing server
            # back at module load time would be circular. By request time
            # server.py has finished initializing.
            import torch
            import torchaudio
            from server import (
                load_audio_from_bytes, run_transcription, run_diarization,
                merge_transcription_and_diarization, identify_speakers_hook,
                MIN_AUDIO_LENGTH, MAX_AUDIO_LENGTH, SAMPLE_RATE, diarization_pipeline
            )

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
                # return_exceptions=True so a failure on one side does not
                # abandon the other task still running in a worker thread
                # (which would hold the whisper lock and log an
                # "exception was never retrieved" warning).
                transcription_result, diarization_result = await asyncio.gather(
                    transcription_task, diarization_task, return_exceptions=True
                )
                if isinstance(transcription_result, BaseException):
                    raise transcription_result
                if isinstance(diarization_result, BaseException):
                    logger.error(f"Diarization failed, continuing without speakers: {diarization_result}")
                    diarization_result = None
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

            # Voiceprint-based speaker naming, same hook the live /transcribe
            # path uses, so speaker names here match the rest of the app.
            segments = await asyncio.to_thread(
                identify_speakers_hook, waveform, sample_rate, segments, diarization_result
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

            # Cached enhanced processor for these settings
            request_service = get_enhanced_service(
                sentiment_model=sentiment_model,
                enable_audio_features=include_audio_features
            )

            if request_service.enhanced_processor:
                # Sentiment + audio features per segment, same as process_enhanced_transcription.
                # Everything below is synchronous and slow (librosa, NLTK, and a
                # blocking HTTP call to Ollama in the summarizer), so each step
                # runs in a worker thread - otherwise the whole uvicorn loop
                # freezes and the live 5-second /transcribe chunks stall behind it.
                enhanced_result = await asyncio.to_thread(
                    request_service.enhanced_processor.process_enhanced_transcription,
                    basic_result, waveform, sample_rate
                )

                if include_summary:
                    enhanced_result["conversation_summary"] = await asyncio.to_thread(
                        _summarize_with_engine,
                        enhanced_result.get("segments", segments),
                        enhanced_result,
                        summary_type_list,
                        request_service.enhanced_processor,
                        use_llm,
                        llm_style,
                    )

                if include_annotations:
                    enhanced_result["conversation_annotations"] = await asyncio.to_thread(
                        request_service.enhanced_processor.generate_conversation_annotations,
                        enhanced_result, annotation_type_list
                    )

                enhanced_result["processing_info"] = {
                    "enhanced_features": True,
                    "summarization_enabled": include_summary,
                    "annotations_enabled": include_annotations,
                    "processing_timestamp": datetime.now().isoformat(),
                    "summary_types": summary_type_list if include_summary else [],
                    "annotation_types": annotation_type_list if include_annotations else []
                }

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

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Enhanced transcription with summarization failed: {e}")
            raise HTTPException(status_code=500, detail=f"Enhanced transcription failed: {e}")

    @app.post("/summarize_transcription")
    async def summarize_existing_transcription(
        transcription: Dict[str, Any] = Body(..., description="Transcription result to summarize"),
        summary_types: str = Query("overall,by_speaker,key_points", description="Comma-separated summary types"),
        annotation_types: str = Query("topics,action_items,questions_answers", description="Comma-separated annotation types"),
        include_annotations: bool = Query(True, description="Include conversation annotations"),
        use_llm: bool = Query(True, description="Prefer a local Ollama LLM for the summary, falling back to extractive summarization"),
        llm_style: str = Query("meeting_minutes", description="LLM summary style: meeting_minutes, action_items, decisions, or summary")
    ) -> Dict[str, Any]:
        """
        Generate summary and annotations for an existing transcription

        This endpoint takes an existing transcription result (as returned by
        POST /transcribe) and adds:
        - A conversation summary (LLM-generated when Ollama is reachable, else extractive)
        - Conversation annotations

        Useful for post-processing transcriptions that were created without
        these features (e.g. every live 5-second /transcribe chunk, which
        skips analysis by default for latency).
        """
        if not ENHANCED_FEATURES_AVAILABLE:
            raise HTTPException(
                status_code=501,
                detail="Enhanced features not available. Install requirements: pip install -r requirements_enhanced.txt"
            )

        try:
            # Validate transcription input
            segments = transcription.get("segments")
            if not segments:
                raise HTTPException(status_code=400, detail="Transcription must contain segments")

            # Parse types
            summary_type_list = [s.strip() for s in summary_types.split(",") if s.strip()]
            annotation_type_list = [a.strip() for a in annotation_types.split(",") if a.strip()]

            # Cached enhanced processor (text-only path - no waveform available here)
            request_service = get_enhanced_service()

            if request_service.enhanced_processor:
                # Off the event loop: the LLM path is a blocking urllib call
                # with a 60s budget, the extractive fallback is NLTK-heavy.
                summaries = await asyncio.to_thread(
                    _summarize_with_engine,
                    segments,
                    transcription,
                    summary_type_list,
                    request_service.enhanced_processor,
                    use_llm,
                    llm_style,
                )

                # Generate annotations
                annotations = {}
                if include_annotations:
                    annotations = await asyncio.to_thread(
                        request_service.enhanced_processor.generate_conversation_annotations,
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

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Transcription summarization failed: {e}")
            raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")

    @app.get("/summary_options")
    async def get_summary_options() -> Dict[str, Any]:
        """
        Get available summary and annotation options
        """
        try:
            from llm_summarizer import get_status as get_llm_status, STYLE_DESCRIPTIONS
            llm_status = await asyncio.to_thread(get_llm_status)
        except Exception as e:
            llm_status = {"available": False, "error": str(e)}
            STYLE_DESCRIPTIONS = {}

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
            },
            "llm_summary_styles": STYLE_DESCRIPTIONS,
            "llm_summarization": llm_status
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
    print("   POST /transcribe_with_summary - Transcription + summary + annotations")
    print("   POST /summarize_transcription - Summarize an existing transcription")
    print("   GET /summary_options - Available summary/annotation/LLM options")
    print()
    print("5. Features added to transcription:")
    print("   • Sentiment analysis per segment (positive/negative/neutral)")
    print("   • Speaker emotion tracking")
    print("   • Audio characteristics (volume, speaking rate, stress)")
    print("   • Conversation dynamics (interruptions, participation)")
    print("   • Voice stress indicators")
    print("   • Speaking style analysis")
    print("   • Conversation summaries (LLM via Ollama, or extractive fallback)")
    print("   • Conversation annotations (topics, action items, Q&A, decisions)")


if __name__ == "__main__":
    integrate_with_existing_server()
