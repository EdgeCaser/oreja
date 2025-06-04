#!/usr/bin/env python3
"""
Enhanced Transcription Processor with Sentiment Analysis and Audio Features
Extends the basic transcription with emotional intelligence and audio analytics
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import re

import torch
import torchaudio
from scipy import signal
from scipy.stats import pearsonr
import librosa

# Sentiment Analysis Libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTranscriptionProcessor:
    """
    Enhanced transcription processor with sentiment analysis, conversation summarization,
    and annotation capabilities.
    """
    
    def __init__(self, sentiment_model: str = "vader"):
        self.sentiment_model = sentiment_model
        self.sample_rate = 16000
        
        # Initialize sentiment analyzers
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.textblob_available = TEXTBLOB_AVAILABLE
        self.transformers_available = TRANSFORMERS_AVAILABLE
        
        # Initialize transformer-based sentiment analyzer if available
        self.transformer_sentiment = None
        if TRANSFORMERS_AVAILABLE and sentiment_model == "transformer":
            try:
                self.transformer_sentiment = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
                logger.info("Loaded RoBERTa sentiment model for enhanced accuracy")
            except Exception as e:
                logger.warning(f"Failed to load transformer model, falling back to VADER: {e}")
                self.sentiment_model = "vader"
        
        # Add summarization and annotation settings
        self.enable_summarization = True
        self.enable_annotations = True
        self.summary_types = ["overall", "by_speaker", "by_time", "key_points"]
        self.annotation_types = ["topics", "action_items", "questions_answers", "decisions", "emotional_moments"]
        
        # Import additional libraries for advanced text processing
        try:
            import nltk
            from nltk.tokenize import sent_tokenize, word_tokenize
            from nltk.corpus import stopwords
            from nltk.tag import pos_tag
            self.nltk_available = True
            
            # Download required NLTK data if not present
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                
            self.stopwords = set(stopwords.words('english'))
            
        except ImportError:
            logger.warning("NLTK not available - advanced text processing will be limited")
            self.nltk_available = False
            self.stopwords = set()
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using the selected model
        """
        results = {}
        
        # Clean text for analysis
        clean_text = self._clean_text_for_sentiment(text)
        
        if not clean_text.strip():
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                "method": "empty_text"
            }
        
        # VADER Analysis (always available)
        vader_scores = self.vader_analyzer.polarity_scores(clean_text)
        results["vader"] = {
            "compound": vader_scores["compound"],
            "positive": vader_scores["pos"],
            "negative": vader_scores["neg"],
            "neutral": vader_scores["neu"]
        }
        
        # TextBlob Analysis
        if self.textblob_available:
            try:
                blob = TextBlob(clean_text)
                results["textblob"] = {
                    "polarity": float(blob.sentiment.polarity),  # -1 to 1
                    "subjectivity": float(blob.sentiment.subjectivity)  # 0 to 1
                }
            except Exception as e:
                logger.debug(f"TextBlob analysis failed: {e}")
        
        # Transformer Analysis
        if self.transformer_sentiment and self.sentiment_model == "transformer":
            try:
                # Handle long text by truncating
                truncated_text = clean_text[:512] if len(clean_text) > 512 else clean_text
                transformer_result = self.transformer_sentiment(truncated_text)[0]
                
                results["transformer"] = {
                    "label": transformer_result["label"].lower(),
                    "confidence": transformer_result["score"]
                }
            except Exception as e:
                logger.debug(f"Transformer analysis failed: {e}")
        
        # Determine primary sentiment
        primary_sentiment = self._determine_primary_sentiment(results)
        
        return {
            "sentiment": primary_sentiment["label"],
            "confidence": primary_sentiment["confidence"],
            "scores": primary_sentiment["scores"],
            "method": self.sentiment_model,
            "all_results": results
        }
    
    def _clean_text_for_sentiment(self, text: str) -> str:
        """Clean text for better sentiment analysis"""
        if not text:
            return ""
        
        # Remove timestamps and speaker labels
        text = re.sub(r'\[\d+:\d+\]', '', text)
        text = re.sub(r'SPEAKER_\d+:', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _determine_primary_sentiment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the primary sentiment from multiple analyzer results"""
        
        if self.sentiment_model == "transformer" and "transformer" in results:
            # Use transformer results as primary
            transformer = results["transformer"]
            label_map = {"positive": "positive", "negative": "negative", "neutral": "neutral"}
            
            return {
                "label": label_map.get(transformer["label"], "neutral"),
                "confidence": transformer["confidence"],
                "scores": {transformer["label"]: transformer["confidence"]}
            }
        
        elif self.sentiment_model == "textblob" and "textblob" in results:
            # Use TextBlob results
            polarity = results["textblob"]["polarity"]
            if polarity > 0.1:
                sentiment = "positive"
            elif polarity < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            confidence = abs(polarity) if sentiment != "neutral" else 1.0 - abs(polarity)
            
            return {
                "label": sentiment,
                "confidence": confidence,
                "scores": {"polarity": polarity}
            }
        
        else:
            # Use VADER results (default)
            vader = results["vader"]
            compound = vader["compound"]
            
            if compound >= 0.05:
                sentiment = "positive"
                confidence = compound
            elif compound <= -0.05:
                sentiment = "negative"
                confidence = abs(compound)
            else:
                sentiment = "neutral"
                confidence = 1.0 - abs(compound)
            
            return {
                "label": sentiment,
                "confidence": confidence,
                "scores": vader
            }
    
    def extract_audio_features(self, waveform: torch.Tensor, sample_rate: int, 
                             start_time: float, end_time: float) -> Dict[str, Any]:
        """
        Extract various audio features from a segment
        """
        features = {}
        
        # Convert to numpy for analysis
        audio_segment = waveform.squeeze().numpy()
        duration = len(audio_segment) / sample_rate
        
        # 1. Energy/Volume Analysis
        rms_energy = np.sqrt(np.mean(audio_segment ** 2))
        peak_amplitude = np.max(np.abs(audio_segment))
        
        features["energy"] = {
            "rms_energy": float(rms_energy),
            "peak_amplitude": float(peak_amplitude),
            "volume_level": self._categorize_volume(rms_energy)
        }
        
        # 2. Speaking Rate Analysis
        speaking_stats = self._analyze_speaking_rate(audio_segment, sample_rate)
        features["speaking_rate"] = speaking_stats
        
        # 3. Spectral Features
        spectral_features = self._extract_spectral_features(audio_segment, sample_rate)
        features["spectral"] = spectral_features
        
        # 4. Voice Stress Indicators
        stress_indicators = self._analyze_voice_stress(audio_segment, sample_rate)
        features["stress"] = stress_indicators
        
        # 5. Pause Analysis
        pause_analysis = self._analyze_pauses(audio_segment, sample_rate)
        features["pauses"] = pause_analysis
        
        return features
    
    def _categorize_volume(self, rms_energy: float) -> str:
        """Categorize volume level"""
        if rms_energy > 0.1:
            return "loud"
        elif rms_energy > 0.05:
            return "normal"
        elif rms_energy > 0.01:
            return "quiet"
        else:
            return "very_quiet"
    
    def _analyze_speaking_rate(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze speaking rate and patterns"""
        duration = len(audio) / sample_rate
        
        # Simple voice activity detection based on energy
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.01 * sample_rate)     # 10ms hops
        
        frames = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy = np.sum(frame ** 2)
            frames.append(energy)
        
        # Threshold for voice activity (adaptive)
        energy_threshold = np.mean(frames) * 0.5
        active_frames = np.sum(np.array(frames) > energy_threshold)
        
        # Estimate speech activity ratio
        speech_ratio = active_frames / len(frames) if frames else 0
        
        # Categorize speaking rate
        if speech_ratio > 0.8:
            rate_category = "fast"
        elif speech_ratio > 0.5:
            rate_category = "normal"
        elif speech_ratio > 0.2:
            rate_category = "slow"
        else:
            rate_category = "very_slow"
        
        return {
            "speech_ratio": float(speech_ratio),
            "rate_category": rate_category,
            "estimated_speech_duration": float(speech_ratio * duration)
        }
    
    def _extract_spectral_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract spectral features using librosa if available"""
        try:
            # Compute spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
            spectral_centroid_mean = np.mean(spectral_centroids)
            
            # Compute zero crossing rate (related to pitch and timbre)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            zcr_mean = np.mean(zcr)
            
            # Compute spectral rolloff (energy distribution)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
            rolloff_mean = np.mean(spectral_rolloff)
            
            return {
                "spectral_centroid": float(spectral_centroid_mean),
                "zero_crossing_rate": float(zcr_mean),
                "spectral_rolloff": float(rolloff_mean),
                "voice_characteristics": self._interpret_spectral_features(
                    spectral_centroid_mean, zcr_mean, rolloff_mean
                )
            }
            
        except Exception as e:
            logger.debug(f"Librosa not available or spectral analysis failed: {e}")
            
            # Fallback to basic spectral analysis
            fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
            magnitude = np.abs(fft)
            
            # Find dominant frequency
            dominant_freq_idx = np.argmax(magnitude[:len(magnitude)//2])
            dominant_freq = abs(freqs[dominant_freq_idx])
            
            return {
                "dominant_frequency": float(dominant_freq),
                "voice_characteristics": self._categorize_voice_basic(dominant_freq)
            }
    
    def _interpret_spectral_features(self, centroid: float, zcr: float, rolloff: float) -> str:
        """Interpret spectral features to describe voice characteristics"""
        if centroid > 2000 and zcr > 0.1:
            return "energetic_clear"
        elif centroid < 1000 and zcr < 0.05:
            return "calm_deep"
        elif centroid > 1500:
            return "bright_expressive"
        else:
            return "normal"
    
    def _categorize_voice_basic(self, dominant_freq: float) -> str:
        """Basic voice categorization based on dominant frequency"""
        if dominant_freq > 200:
            return "high_pitched"
        elif dominant_freq > 100:
            return "normal_pitch"
        else:
            return "low_pitched"
    
    def _analyze_voice_stress(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze voice stress indicators"""
        
        # Jitter analysis (frequency variation)
        # Simple approximation using energy variance
        frame_size = int(0.02 * sample_rate)  # 20ms frames
        frame_energies = []
        
        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i + frame_size]
            energy = np.sum(frame ** 2)
            frame_energies.append(energy)
        
        if len(frame_energies) > 1:
            energy_variance = np.var(frame_energies)
            energy_mean = np.mean(frame_energies)
            jitter_estimate = energy_variance / (energy_mean + 1e-8)
        else:
            jitter_estimate = 0.0
        
        # Stress level categorization
        if jitter_estimate > 0.5:
            stress_level = "high"
        elif jitter_estimate > 0.2:
            stress_level = "moderate"
        else:
            stress_level = "low"
        
        return {
            "jitter_estimate": float(jitter_estimate),
            "stress_level": stress_level,
            "energy_variance": float(energy_variance) if 'energy_variance' in locals() else 0.0
        }
    
    def _analyze_pauses(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze pauses and silence patterns"""
        
        # Detect silence/pauses
        frame_length = int(0.01 * sample_rate)  # 10ms frames
        silence_threshold = np.max(np.abs(audio)) * 0.01  # 1% of peak amplitude
        
        silence_frames = []
        for i in range(0, len(audio) - frame_length, frame_length):
            frame = audio[i:i + frame_length]
            is_silent = np.max(np.abs(frame)) < silence_threshold
            silence_frames.append(is_silent)
        
        # Count pause segments
        pause_segments = []
        in_pause = False
        pause_start = 0
        
        for i, is_silent in enumerate(silence_frames):
            if is_silent and not in_pause:
                in_pause = True
                pause_start = i
            elif not is_silent and in_pause:
                in_pause = False
                pause_duration = (i - pause_start) * frame_length / sample_rate
                if pause_duration > 0.1:  # Only count pauses > 100ms
                    pause_segments.append(pause_duration)
        
        # Analyze pause patterns
        total_pause_time = sum(pause_segments)
        avg_pause_duration = np.mean(pause_segments) if pause_segments else 0
        pause_count = len(pause_segments)
        
        # Categorize speaking style based on pauses
        if pause_count == 0:
            speaking_style = "continuous"
        elif avg_pause_duration > 1.0:
            speaking_style = "deliberate"
        elif avg_pause_duration > 0.5:
            speaking_style = "measured"
        else:
            speaking_style = "fluent"
        
        return {
            "total_pause_time": float(total_pause_time),
            "average_pause_duration": float(avg_pause_duration),
            "pause_count": int(pause_count),
            "speaking_style": speaking_style
        }
    
    def analyze_conversation_dynamics(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall conversation dynamics"""
        
        if not segments:
            return {}
        
        # Speaker participation analysis
        speaker_stats = {}
        speaker_sentiments = {}
        speaker_interruptions = {}
        
        for i, segment in enumerate(segments):
            speaker = segment.get("speaker", "Unknown")
            sentiment_info = segment.get("sentiment_analysis", {})
            duration = segment.get("end", 0) - segment.get("start", 0)
            
            # Initialize speaker stats
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    "total_time": 0,
                    "segment_count": 0,
                    "avg_sentiment": [],
                    "interruption_count": 0
                }
                speaker_sentiments[speaker] = []
                speaker_interruptions[speaker] = 0
            
            # Update stats
            speaker_stats[speaker]["total_time"] += duration
            speaker_stats[speaker]["segment_count"] += 1
            
            # Collect sentiment scores
            if sentiment_info.get("confidence", 0) > 0.3:
                sentiment_score = self._sentiment_to_score(sentiment_info.get("sentiment", "neutral"))
                speaker_sentiments[speaker].append(sentiment_score)
        
        # Calculate average sentiments
        for speaker in speaker_sentiments:
            if speaker_sentiments[speaker]:
                speaker_stats[speaker]["avg_sentiment"] = np.mean(speaker_sentiments[speaker])
            else:
                speaker_stats[speaker]["avg_sentiment"] = 0.0
        
        # Analyze interruptions (simplified)
        for i in range(1, len(segments)):
            current_segment = segments[i]
            prev_segment = segments[i-1]
            
            # Check for quick speaker changes (potential interruptions)
            time_gap = current_segment.get("start", 0) - prev_segment.get("end", 0)
            if time_gap < 0.5 and current_segment.get("speaker") != prev_segment.get("speaker"):
                current_speaker = current_segment.get("speaker", "Unknown")
                if current_speaker in speaker_interruptions:
                    speaker_interruptions[current_speaker] += 1
        
        # Update interruption counts
        for speaker in speaker_interruptions:
            if speaker in speaker_stats:
                speaker_stats[speaker]["interruption_count"] = speaker_interruptions[speaker]
        
        # Calculate conversation metrics
        total_duration = max([seg.get("end", 0) for seg in segments], default=0)
        overall_sentiment = np.mean([
            self._sentiment_to_score(seg.get("sentiment_analysis", {}).get("sentiment", "neutral"))
            for seg in segments
            if seg.get("sentiment_analysis", {}).get("confidence", 0) > 0.3
        ]) if segments else 0.0
        
        return {
            "speaker_statistics": speaker_stats,
            "conversation_metrics": {
                "total_duration": total_duration,
                "overall_sentiment": float(overall_sentiment),
                "overall_sentiment_label": self._score_to_sentiment(overall_sentiment),
                "total_speakers": len(speaker_stats),
                "total_segments": len(segments)
            }
        }
    
    def _sentiment_to_score(self, sentiment: str) -> float:
        """Convert sentiment label to numerical score"""
        mapping = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        return mapping.get(sentiment, 0.0)
    
    def _score_to_sentiment(self, score: float) -> str:
        """Convert numerical score to sentiment label"""
        if score > 0.1:
            return "positive"
        elif score < -0.1:
            return "negative"
        else:
            return "neutral"
    
    def process_enhanced_transcription(self, transcription_result: Dict[str, Any], 
                                     waveform: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
        """
        Process transcription with enhanced sentiment and audio analysis
        """
        enhanced_segments = []
        
        for segment in transcription_result.get("segments", []):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "")
            speaker = segment.get("speaker", "Unknown")
            
            # Extract audio segment
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Create enhanced segment
            enhanced_segment = segment.copy()
            
            # Add sentiment analysis
            if text.strip():
                sentiment_analysis = self.analyze_sentiment(text)
                enhanced_segment["sentiment_analysis"] = sentiment_analysis
            
            # Add audio features if segment is long enough
            if end_sample > start_sample and end_sample <= waveform.shape[1]:
                segment_duration = (end_sample - start_sample) / sample_rate
                if segment_duration >= 0.5:  # Only analyze segments >= 0.5 seconds
                    segment_waveform = waveform[:, start_sample:end_sample]
                    audio_features = self.extract_audio_features(
                        segment_waveform, sample_rate, start_time, end_time
                    )
                    enhanced_segment["audio_features"] = audio_features
            
            enhanced_segments.append(enhanced_segment)
        
        # Analyze overall conversation dynamics
        conversation_analysis = self.analyze_conversation_dynamics(enhanced_segments)
        
        # Create enhanced result
        enhanced_result = transcription_result.copy()
        enhanced_result["segments"] = enhanced_segments
        enhanced_result["conversation_analysis"] = conversation_analysis
        enhanced_result["enhancement_info"] = {
            "sentiment_model": self.sentiment_model,
            "features_extracted": ["sentiment", "audio_features", "conversation_dynamics"],
            "processing_timestamp": datetime.now().isoformat()
        }
        
        return enhanced_result
    
    def process_with_summary_and_annotations(self, transcription_result: Dict[str, Any], 
                                           waveform: Any, sample_rate: int,
                                           include_summary: bool = True,
                                           include_annotations: bool = True,
                                           summary_types: Optional[List[str]] = None,
                                           annotation_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process transcription with enhanced features including summarization and annotations
        
        Args:
            transcription_result: Original transcription result
            waveform: Audio waveform tensor
            sample_rate: Audio sample rate
            include_summary: Whether to generate summaries
            include_annotations: Whether to generate annotations
            summary_types: Types of summaries to generate
            annotation_types: Types of annotations to generate
            
        Returns:
            Enhanced transcription with summaries and annotations
        """
        # First, run the existing enhanced processing
        enhanced_result = self.process_enhanced_transcription(transcription_result, waveform, sample_rate)
        
        # Add summarization
        if include_summary:
            summary_types = summary_types or self.summary_types
            enhanced_result["conversation_summary"] = self.generate_conversation_summary(
                enhanced_result, summary_types
            )
        
        # Add annotations
        if include_annotations:
            annotation_types = annotation_types or self.annotation_types
            enhanced_result["conversation_annotations"] = self.generate_conversation_annotations(
                enhanced_result, annotation_types
            )
        
        # Add processing metadata
        enhanced_result["processing_info"] = {
            "enhanced_features": True,
            "summarization_enabled": include_summary,
            "annotations_enabled": include_annotations,
            "processing_timestamp": datetime.now().isoformat(),
            "summary_types": summary_types if include_summary else [],
            "annotation_types": annotation_types if include_annotations else []
        }
        
        return enhanced_result
    
    def generate_conversation_summary(self, transcription_result: Dict[str, Any], 
                                    summary_types: List[str]) -> Dict[str, Any]:
        """Generate various types of conversation summaries"""
        segments = transcription_result.get("segments", [])
        full_text = transcription_result.get("full_text", "")
        
        summaries = {}
        
        if "overall" in summary_types:
            summaries["overall"] = self._generate_overall_summary(segments, full_text)
        
        if "by_speaker" in summary_types:
            summaries["by_speaker"] = self._generate_speaker_summaries(segments)
        
        if "by_time" in summary_types:
            summaries["by_time"] = self._generate_time_based_summaries(segments)
        
        if "key_points" in summary_types:
            summaries["key_points"] = self._extract_key_points(segments)
        
        return summaries
    
    def _generate_overall_summary(self, segments: List[Dict[str, Any]], full_text: str) -> Dict[str, Any]:
        """Generate an overall conversation summary"""
        if not segments:
            return {"summary": "No content to summarize", "method": "empty"}
        
        # Basic extractive summarization
        important_sentences = self._extract_important_sentences(segments)
        
        # Calculate conversation metrics
        total_duration = segments[-1].get("end", 0) - segments[0].get("start", 0) if segments else 0
        speaker_count = len(set(seg.get("speaker", "Unknown") for seg in segments))
        
        # Identify dominant themes/topics
        topics = self._identify_main_topics(segments)
        
        # Overall sentiment
        overall_sentiment = self._calculate_overall_sentiment(segments)
        
        return {
            "summary": " ".join(important_sentences[:3]) if important_sentences else "Conversation recorded.",
            "key_metrics": {
                "duration_minutes": round(total_duration / 60, 1),
                "speaker_count": speaker_count,
                "segment_count": len(segments),
                "dominant_sentiment": overall_sentiment
            },
            "main_topics": topics[:5],  # Top 5 topics
            "method": "extractive_with_metrics"
        }
    
    def _generate_speaker_summaries(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summaries for each speaker"""
        speaker_data = {}
        
        for segment in segments:
            speaker = segment.get("speaker", "Unknown")
            text = segment.get("text", "").strip()
            duration = segment.get("end", 0) - segment.get("start", 0)
            sentiment = segment.get("sentiment_analysis", {})
            
            if speaker not in speaker_data:
                speaker_data[speaker] = {
                    "texts": [],
                    "total_time": 0,
                    "sentiments": [],
                    "segment_count": 0
                }
            
            speaker_data[speaker]["texts"].append(text)
            speaker_data[speaker]["total_time"] += duration
            speaker_data[speaker]["segment_count"] += 1
            
            if sentiment.get("confidence", 0) > 0.3:
                speaker_data[speaker]["sentiments"].append(sentiment.get("sentiment", "neutral"))
        
        # Generate summaries for each speaker
        summaries = {}
        for speaker, data in speaker_data.items():
            combined_text = " ".join(data["texts"])
            key_sentences = self._extract_key_sentences_from_text(combined_text)
            
            # Calculate speaker stats
            avg_sentiment = self._calculate_dominant_sentiment(data["sentiments"])
            participation_percentage = 0
            
            summaries[speaker] = {
                "summary": " ".join(key_sentences[:2]) if key_sentences else f"{speaker} participated in conversation.",
                "participation_stats": {
                    "total_time_seconds": round(data["total_time"], 1),
                    "segment_count": data["segment_count"],
                    "dominant_sentiment": avg_sentiment,
                    "participation_percentage": participation_percentage  # Will calculate after all speakers
                }
            }
        
        # Calculate participation percentages
        total_time = sum(data["total_time"] for data in speaker_data.values())
        if total_time > 0:
            for speaker in summaries:
                summaries[speaker]["participation_stats"]["participation_percentage"] = round(
                    (speaker_data[speaker]["total_time"] / total_time) * 100, 1
                )
        
        return summaries
    
    def _generate_time_based_summaries(self, segments: List[Dict[str, Any]], 
                                     interval_minutes: int = 5) -> List[Dict[str, Any]]:
        """Generate summaries for time-based intervals"""
        if not segments:
            return []
        
        summaries = []
        start_time = segments[0].get("start", 0)
        end_time = segments[-1].get("end", 0)
        interval_seconds = interval_minutes * 60
        
        current_time = start_time
        while current_time < end_time:
            interval_end = min(current_time + interval_seconds, end_time)
            
            # Get segments in this interval
            interval_segments = [
                seg for seg in segments 
                if seg.get("start", 0) >= current_time and seg.get("start", 0) < interval_end
            ]
            
            if interval_segments:
                interval_text = " ".join(seg.get("text", "") for seg in interval_segments)
                key_sentences = self._extract_key_sentences_from_text(interval_text)
                speakers = list(set(seg.get("speaker", "Unknown") for seg in interval_segments))
                
                summaries.append({
                    "time_range": {
                        "start_minutes": round(current_time / 60, 1),
                        "end_minutes": round(interval_end / 60, 1)
                    },
                    "summary": " ".join(key_sentences[:2]) if key_sentences else "Continued conversation.",
                    "active_speakers": speakers,
                    "segment_count": len(interval_segments)
                })
            
            current_time = interval_end
        
        return summaries
    
    def _extract_key_points(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract key points from the conversation"""
        key_points = []
        
        for segment in segments:
            text = segment.get("text", "").strip()
            sentiment = segment.get("sentiment_analysis", {})
            speaker = segment.get("speaker", "Unknown")
            
            # Look for key indicators
            if self._is_key_point(text):
                key_points.append({
                    "text": text,
                    "speaker": speaker,
                    "timestamp": segment.get("start", 0),
                    "type": self._classify_key_point_type(text),
                    "sentiment": sentiment.get("sentiment", "neutral"),
                    "confidence": sentiment.get("confidence", 0)
                })
        
        # Sort by importance/confidence
        key_points.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return key_points[:10]  # Return top 10 key points
    
    def generate_conversation_annotations(self, transcription_result: Dict[str, Any],
                                        annotation_types: List[str]) -> Dict[str, Any]:
        """Generate conversation annotations"""
        segments = transcription_result.get("segments", [])
        annotations = {}
        
        if "topics" in annotation_types:
            annotations["topics"] = self._identify_conversation_topics(segments)
        
        if "action_items" in annotation_types:
            annotations["action_items"] = self._extract_action_items(segments)
        
        if "questions_answers" in annotation_types:
            annotations["questions_answers"] = self._identify_question_answer_pairs(segments)
        
        if "decisions" in annotation_types:
            annotations["decisions"] = self._extract_decisions(segments)
        
        if "emotional_moments" in annotation_types:
            annotations["emotional_moments"] = self._identify_emotional_moments(segments)
        
        return annotations
    
    def _identify_conversation_topics(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify main topics discussed in the conversation"""
        # Combine all text
        all_text = " ".join(seg.get("text", "") for seg in segments)
        
        # Extract keywords and phrases
        topics = self._extract_topics_from_text(all_text)
        
        # Add timing information
        topic_segments = []
        for topic in topics:
            # Find segments that mention this topic
            relevant_segments = []
            for seg in segments:
                if any(keyword.lower() in seg.get("text", "").lower() for keyword in topic["keywords"]):
                    relevant_segments.append({
                        "timestamp": seg.get("start", 0),
                        "speaker": seg.get("speaker", "Unknown"),
                        "text": seg.get("text", "")
                    })
            
            if relevant_segments:
                topic_segments.append({
                    "topic": topic["topic"],
                    "keywords": topic["keywords"],
                    "mentions": len(relevant_segments),
                    "first_mentioned": relevant_segments[0]["timestamp"],
                    "last_mentioned": relevant_segments[-1]["timestamp"],
                    "relevant_segments": relevant_segments[:3]  # Keep first 3 mentions
                })
        
        return sorted(topic_segments, key=lambda x: x["mentions"], reverse=True)[:10]
    
    def _extract_action_items(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract action items and tasks from the conversation"""
        action_patterns = [
            r'\b(?:need to|have to|must|should|will)\s+([^.!?]+)',
            r'\b(?:action item|to-?do|task|assignment):\s*([^.!?]+)',
            r'\b(?:I\'ll|we\'ll|you\'ll|they\'ll)\s+([^.!?]+)',
            r'\b(?:let\'s|let us)\s+([^.!?]+)',
            r'\b(?:follow up|next step|action):\s*([^.!?]+)'
        ]
        
        action_items = []
        
        for segment in segments:
            text = segment.get("text", "")
            speaker = segment.get("speaker", "Unknown")
            timestamp = segment.get("start", 0)
            
            for pattern in action_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    action_items.append({
                        "action": match.strip(),
                        "speaker": speaker,
                        "timestamp": timestamp,
                        "context": text,
                        "confidence": 0.7  # Basic confidence score
                    })
        
        # Remove duplicates and sort by timestamp
        unique_actions = []
        for item in action_items:
            if not any(item["action"].lower() in existing["action"].lower() 
                      for existing in unique_actions):
                unique_actions.append(item)
        
        return sorted(unique_actions, key=lambda x: x["timestamp"])[:10]
    
    def _identify_question_answer_pairs(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify question and answer pairs in the conversation"""
        qa_pairs = []
        questions = []
        
        # First pass: identify questions
        for i, segment in enumerate(segments):
            text = segment.get("text", "").strip()
            if text.endswith('?') or any(text.lower().startswith(q) for q in ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'could', 'would', 'should', 'do', 'does', 'did', 'is', 'are', 'was', 'were']):
                questions.append({
                    "question": text,
                    "speaker": segment.get("speaker", "Unknown"),
                    "timestamp": segment.get("start", 0),
                    "segment_index": i
                })
        
        # Second pass: find potential answers
        for question in questions:
            potential_answers = []
            
            # Look for answers in the next few segments
            start_idx = question["segment_index"] + 1
            for i in range(start_idx, min(start_idx + 5, len(segments))):
                answer_segment = segments[i]
                
                # Skip if same speaker (unless significant time gap)
                if (answer_segment.get("speaker") == question["speaker"] and 
                    answer_segment.get("start", 0) - question["timestamp"] < 10):
                    continue
                
                potential_answers.append({
                    "answer": answer_segment.get("text", ""),
                    "speaker": answer_segment.get("speaker", "Unknown"),
                    "timestamp": answer_segment.get("start", 0),
                    "confidence": self._calculate_answer_confidence(question["question"], answer_segment.get("text", ""))
                })
            
            # Keep best answer
            if potential_answers:
                best_answer = max(potential_answers, key=lambda x: x["confidence"])
                if best_answer["confidence"] > 0.3:  # Minimum confidence threshold
                    qa_pairs.append({
                        "question": question["question"],
                        "question_speaker": question["speaker"],
                        "question_timestamp": question["timestamp"],
                        "answer": best_answer["answer"],
                        "answer_speaker": best_answer["speaker"],
                        "answer_timestamp": best_answer["timestamp"],
                        "confidence": best_answer["confidence"]
                    })
        
        return sorted(qa_pairs, key=lambda x: x["question_timestamp"])[:10]
    
    def _extract_decisions(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract decisions and conclusions from the conversation"""
        decision_patterns = [
            r'\b(?:decided|conclude|resolution|final|agreed|settled)\b.*?([^.!?]+)',
            r'\b(?:decision|conclusion|agreement):\s*([^.!?]+)',
            r'\b(?:so we\'ll|therefore|thus|hence)\s+([^.!?]+)',
            r'\b(?:final|ultimate|definitive)\s+([^.!?]+)'
        ]
        
        decisions = []
        
        for segment in segments:
            text = segment.get("text", "")
            speaker = segment.get("speaker", "Unknown")
            timestamp = segment.get("start", 0)
            sentiment = segment.get("sentiment_analysis", {})
            
            for pattern in decision_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    decisions.append({
                        "decision": match.strip(),
                        "speaker": speaker,
                        "timestamp": timestamp,
                        "context": text,
                        "sentiment": sentiment.get("sentiment", "neutral"),
                        "confidence": 0.6
                    })
        
        # Remove duplicates
        unique_decisions = []
        for decision in decisions:
            if not any(decision["decision"].lower() in existing["decision"].lower() 
                      for existing in unique_decisions):
                unique_decisions.append(decision)
        
        return sorted(unique_decisions, key=lambda x: x["timestamp"])[:10]
    
    def _identify_emotional_moments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify emotionally significant moments in the conversation"""
        emotional_moments = []
        
        for segment in segments:
            sentiment = segment.get("sentiment_analysis", {})
            confidence = sentiment.get("confidence", 0)
            sentiment_label = sentiment.get("sentiment", "neutral")
            
            # High confidence non-neutral sentiment indicates emotional moment
            if confidence > 0.7 and sentiment_label != "neutral":
                audio_features = segment.get("audio_features", {})
                energy = audio_features.get("energy", {})
                
                emotional_moments.append({
                    "text": segment.get("text", ""),
                    "speaker": segment.get("speaker", "Unknown"),
                    "timestamp": segment.get("start", 0),
                    "sentiment": sentiment_label,
                    "sentiment_confidence": confidence,
                    "audio_intensity": energy.get("volume_level", "unknown"),
                    "emotional_indicator": f"{sentiment_label.capitalize()} (confidence: {confidence:.2f})"
                })
        
        return sorted(emotional_moments, key=lambda x: x["sentiment_confidence"], reverse=True)[:10]
    
    # Helper methods for text processing
    def _extract_important_sentences(self, segments: List[Dict[str, Any]]) -> List[str]:
        """Extract the most important sentences from segments"""
        sentences = []
        
        for segment in segments:
            text = segment.get("text", "").strip()
            sentiment = segment.get("sentiment_analysis", {})
            
            if len(text) > 20:  # Minimum length
                # Score based on sentiment confidence and length
                score = sentiment.get("confidence", 0) + min(len(text) / 100, 1.0)
                sentences.append((text, score))
        
        # Sort by score and return top sentences
        sentences.sort(key=lambda x: x[1], reverse=True)
        return [sent[0] for sent in sentences[:5]]
    
    def _extract_key_sentences_from_text(self, text: str) -> List[str]:
        """Extract key sentences from a text block"""
        if not self.nltk_available or not text.strip():
            return [text[:200] + "..." if len(text) > 200 else text]
        
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
            
            # Score sentences by length and keyword presence
            scored_sentences = []
            for sent in sentences:
                if len(sent) > 20:  # Minimum length
                    score = len(sent) / 100  # Base score on length
                    
                    # Boost score for important keywords
                    important_words = ['important', 'key', 'main', 'significant', 'crucial', 'decision', 'agree', 'disagree']
                    for word in important_words:
                        if word in sent.lower():
                            score += 0.5
                    
                    scored_sentences.append((sent, score))
            
            # Sort and return top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            return [sent[0] for sent in scored_sentences[:3]]
            
        except Exception as e:
            logger.warning(f"Error in sentence extraction: {e}")
            return [text[:200] + "..." if len(text) > 200 else text]
    
    def _identify_main_topics(self, segments: List[Dict[str, Any]]) -> List[str]:
        """Identify main topics from conversation segments"""
        # Combine all text
        all_text = " ".join(seg.get("text", "") for seg in segments)
        
        if not self.nltk_available:
            # Simple keyword extraction without NLTK
            words = all_text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3 and word.isalpha():
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Return most frequent words as topics
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word[0] for word in sorted_words[:5]]
        
        try:
            from nltk.tokenize import word_tokenize
            from nltk.tag import pos_tag
            
            # Tokenize and get part-of-speech tags
            words = word_tokenize(all_text.lower())
            pos_tags = pos_tag(words)
            
            # Extract nouns as potential topics
            nouns = [word for word, pos in pos_tags if pos.startswith('NN') and len(word) > 3]
            
            # Remove stopwords
            filtered_nouns = [word for word in nouns if word not in self.stopwords]
            
            # Count frequency
            noun_freq = {}
            for noun in filtered_nouns:
                noun_freq[noun] = noun_freq.get(noun, 0) + 1
            
            # Return most frequent nouns as topics
            sorted_nouns = sorted(noun_freq.items(), key=lambda x: x[1], reverse=True)
            return [noun[0] for noun in sorted_nouns[:5]]
            
        except Exception as e:
            logger.warning(f"Error in topic identification: {e}")
            return ["conversation", "discussion"]
    
    def _extract_topics_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract topics with keywords from text"""
        topics = []
        
        # Simple topic extraction - in real implementation, you might use more sophisticated methods
        topic_keywords = {
            "business": ["business", "company", "revenue", "profit", "market", "customer", "client"],
            "technology": ["technology", "software", "system", "technical", "development", "programming"],
            "project": ["project", "task", "deadline", "milestone", "timeline", "deliverable"],
            "meeting": ["meeting", "agenda", "discussion", "presentation", "schedule"],
            "finance": ["budget", "cost", "expense", "financial", "money", "price", "payment"]
        }
        
        text_lower = text.lower()
        
        for topic, keywords in topic_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                topics.append({
                    "topic": topic,
                    "keywords": [kw for kw in keywords if kw in text_lower],
                    "relevance_score": matches / len(keywords)
                })
        
        return sorted(topics, key=lambda x: x["relevance_score"], reverse=True)
    
    def _is_key_point(self, text: str) -> bool:
        """Determine if a text segment represents a key point"""
        key_indicators = [
            'important', 'key', 'main', 'significant', 'crucial', 'critical',
            'decision', 'conclusion', 'summary', 'result', 'outcome',
            'agree', 'disagree', 'decided', 'resolved', 'final'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in key_indicators) or len(text) > 50
    
    def _classify_key_point_type(self, text: str) -> str:
        """Classify the type of key point"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['decision', 'decided', 'final', 'concluded']):
            return "decision"
        elif any(word in text_lower for word in ['action', 'task', 'todo', 'need to', 'will']):
            return "action_item"
        elif any(word in text_lower for word in ['important', 'key', 'significant', 'crucial']):
            return "important_point"
        elif text.endswith('?'):
            return "question"
        else:
            return "general"
    
    def _calculate_overall_sentiment(self, segments: List[Dict[str, Any]]) -> str:
        """Calculate overall sentiment of the conversation"""
        sentiments = []
        
        for segment in segments:
            sentiment_info = segment.get("sentiment_analysis", {})
            if sentiment_info.get("confidence", 0) > 0.3:
                sentiments.append(sentiment_info.get("sentiment", "neutral"))
        
        if not sentiments:
            return "neutral"
        
        # Count sentiment occurrences
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for sentiment in sentiments:
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        # Return dominant sentiment
        return max(sentiment_counts.items(), key=lambda x: x[1])[0]
    
    def _calculate_dominant_sentiment(self, sentiments: List[str]) -> str:
        """Calculate dominant sentiment from a list"""
        if not sentiments:
            return "neutral"
        
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for sentiment in sentiments:
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        return max(sentiment_counts.items(), key=lambda x: x[1])[0]
    
    def _calculate_answer_confidence(self, question: str, answer: str) -> float:
        """Calculate confidence that an answer relates to a question"""
        # Simple heuristic - in practice, you might use more sophisticated NLP
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        question_words -= common_words
        answer_words -= common_words
        
        if not question_words:
            return 0.0
        
        # Calculate word overlap
        overlap = len(question_words.intersection(answer_words))
        confidence = overlap / len(question_words)
        
        # Boost confidence if answer is reasonably long
        if len(answer) > 20:
            confidence += 0.2
        
        return min(confidence, 1.0)


def main():
    """Example usage of the enhanced transcription processor"""
    
    # Initialize processor
    processor = EnhancedTranscriptionProcessor(sentiment_model="vader")
    
    # Example: Process a mock transcription result
    mock_transcription = {
        "segments": [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "Hello, I'm really excited about this project!",
                "speaker": "SPEAKER_00"
            },
            {
                "start": 3.5,
                "end": 6.0,
                "text": "That's great! I'm a bit worried about the timeline though.",
                "speaker": "SPEAKER_01"
            },
            {
                "start": 6.5,
                "end": 9.0,
                "text": "Don't worry, we'll figure it out together.",
                "speaker": "SPEAKER_00"
            }
        ],
        "full_text": "Hello, I'm really excited about this project! That's great! I'm a bit worried about the timeline though. Don't worry, we'll figure it out together."
    }
    
    # Create mock audio (sine wave)
    duration = 10  # seconds
    sample_rate = 16000
    t = np.linspace(0, duration, duration * sample_rate)
    mock_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    mock_waveform = torch.tensor(mock_audio).unsqueeze(0)
    
    # Process with enhancements
    enhanced_result = processor.process_with_summary_and_annotations(
        mock_transcription, mock_waveform, sample_rate
    )
    
    # Print results
    print("Enhanced Transcription Results:")
    print("=" * 50)
    
    for i, segment in enumerate(enhanced_result["segments"]):
        print(f"\nSegment {i+1}:")
        print(f"Speaker: {segment['speaker']}")
        print(f"Text: {segment['text']}")
        
        if "sentiment_analysis" in segment:
            sentiment = segment["sentiment_analysis"]
            print(f"Sentiment: {sentiment['sentiment']} (confidence: {sentiment['confidence']:.2f})")
        
        if "audio_features" in segment:
            features = segment["audio_features"]
            print(f"Volume: {features['energy']['volume_level']}")
            print(f"Speaking Rate: {features['speaking_rate']['rate_category']}")
            print(f"Speaking Style: {features['pauses']['speaking_style']}")
    
    # Print conversation analysis
    if "conversation_analysis" in enhanced_result:
        conv_analysis = enhanced_result["conversation_analysis"]
        print(f"\nConversation Analysis:")
        print(f"Overall Sentiment: {conv_analysis['conversation_metrics']['overall_sentiment_label']}")
        print(f"Total Speakers: {conv_analysis['conversation_metrics']['total_speakers']}")
        print(f"Total Duration: {conv_analysis['conversation_metrics']['total_duration']:.1f}s")


if __name__ == "__main__":
    main() 