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
from datetime import datetime
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
    Enhanced transcription processor with sentiment analysis and audio feature extraction
    """
    
    def __init__(self, sentiment_model="vader"):
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
    enhanced_result = processor.process_enhanced_transcription(
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