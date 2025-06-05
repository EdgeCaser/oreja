"""
Enhanced Segment Splitting with Audio Re-analysis
Addresses the critical flaw where split segments don't get proper embedding extraction
"""

import torch
import torchaudio
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

class AudioSegmentSplitter:
    """
    Handles intelligent splitting of audio segments with proper embedding extraction
    """
    
    def __init__(self, speaker_embedding_manager, sample_rate: int = 16000):
        self.speaker_embedding_manager = speaker_embedding_manager
        self.sample_rate = sample_rate
        self.min_segment_length = 0.5  # Minimum 0.5 seconds for reliable embedding
        
    def split_segment_with_audio_analysis(self, 
                                        audio_file: str,
                                        original_segment: Dict,
                                        split_text_position: float,
                                        first_speaker: str,
                                        second_speaker: str) -> Tuple[Dict, Dict, bool]:
        """
        Split a segment with proper audio re-analysis and embedding extraction.
        
        Args:
            audio_file: Path to the audio file
            original_segment: The original segment to split
            split_text_position: Position in text (0.0-1.0) where to split
            first_speaker: Speaker name for first part
            second_speaker: Speaker name for second part
            
        Returns:
            Tuple of (first_segment, second_segment, success)
        """
        try:
            # Load original audio segment
            waveform, sr = torchaudio.load(audio_file)
            
            start_time = original_segment.get('start', 0)
            end_time = original_segment.get('end', start_time + 1)
            
            # Extract original audio segment
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            if start_sample >= waveform.shape[1] or end_sample > waveform.shape[1]:
                logger.error(f"Audio segment bounds invalid: {start_sample}-{end_sample} vs {waveform.shape[1]}")
                return None, None, False
            
            segment_waveform = waveform[:, start_sample:end_sample]
            
            # Find the optimal split point using audio analysis
            optimal_split_time = self._find_optimal_split_point(
                segment_waveform, sr, split_text_position, start_time, end_time
            )
            
            # Calculate split sample
            split_sample = int((optimal_split_time - start_time) * sr)
            
            # Ensure minimum segment lengths
            min_samples = int(self.min_segment_length * sr)
            if split_sample < min_samples or (segment_waveform.shape[1] - split_sample) < min_samples:
                logger.warning(f"Split would create segments too short for reliable embedding")
                # Adjust split to ensure minimum lengths
                if split_sample < min_samples:
                    split_sample = min_samples
                elif (segment_waveform.shape[1] - split_sample) < min_samples:
                    split_sample = segment_waveform.shape[1] - min_samples
            
            # Extract the two audio parts
            first_audio = segment_waveform[:, :split_sample]
            second_audio = segment_waveform[:, split_sample:]
            
            # Extract embeddings for each part
            first_embedding = self._extract_embedding_from_waveform(first_audio, sr)
            second_embedding = self._extract_embedding_from_waveform(second_audio, sr)
            
            # Create the split segments
            first_segment = {
                **original_segment,
                'speaker': first_speaker,
                'end': optimal_split_time,
                'split_confidence': self._calculate_split_confidence(first_audio, second_audio, sr),
                'embedding_extracted': first_embedding is not None,
                'audio_duration': split_sample / sr
            }
            
            second_segment = {
                **original_segment,
                'speaker': second_speaker,
                'start': optimal_split_time,
                'split_confidence': self._calculate_split_confidence(second_audio, first_audio, sr),
                'embedding_extracted': second_embedding is not None,
                'audio_duration': (segment_waveform.shape[1] - split_sample) / sr
            }
            
            # Send embeddings to speaker database for learning
            success_first = self._update_speaker_with_embedding(
                first_speaker, first_audio.squeeze().cpu().numpy() if first_embedding is not None else None
            )
            success_second = self._update_speaker_with_embedding(
                second_speaker, second_audio.squeeze().cpu().numpy() if second_embedding is not None else None
            )
            
            logger.info(f"Split segment at {optimal_split_time:.2f}s: "
                       f"{first_speaker} ({split_sample/sr:.2f}s) | "
                       f"{second_speaker} ({(segment_waveform.shape[1]-split_sample)/sr:.2f}s)")
            
            return first_segment, second_segment, (success_first and success_second)
            
        except Exception as e:
            logger.error(f"Error splitting segment with audio analysis: {e}")
            return None, None, False
    
    def _find_optimal_split_point(self, 
                                 segment_waveform: torch.Tensor, 
                                 sr: int,
                                 text_position: float,
                                 start_time: float,
                                 end_time: float) -> float:
        """
        Find the optimal audio split point using multiple techniques:
        1. Voice Activity Detection (VAD)
        2. Energy-based silence detection
        3. Text position as fallback
        """
        try:
            # Convert text position to initial audio time estimate
            duration = end_time - start_time
            estimated_split_time = start_time + (duration * text_position)
            estimated_split_sample = int((estimated_split_time - start_time) * sr)
            
            # Define search window around the estimated position (±1 second)
            search_window = int(1.0 * sr)  # 1 second window
            window_start = max(0, estimated_split_sample - search_window // 2)
            window_end = min(segment_waveform.shape[1], estimated_split_sample + search_window // 2)
            
            # Find the best split point within the window
            optimal_sample = self._find_silence_or_speaker_change(
                segment_waveform, window_start, window_end, sr
            )
            
            # Convert back to absolute time
            if optimal_sample is not None:
                optimal_time = start_time + (optimal_sample / sr)
                logger.debug(f"Found optimal split at {optimal_time:.2f}s (estimated: {estimated_split_time:.2f}s)")
                return optimal_time
            else:
                # Fallback to text-based estimate
                logger.debug(f"Using text-based split estimate: {estimated_split_time:.2f}s")
                return estimated_split_time
                
        except Exception as e:
            logger.warning(f"Error finding optimal split point: {e}")
            # Fallback to text-based estimate
            duration = end_time - start_time
            return start_time + (duration * text_position)
    
    def _find_silence_or_speaker_change(self, 
                                       waveform: torch.Tensor, 
                                       start_idx: int, 
                                       end_idx: int, 
                                       sr: int) -> Optional[int]:
        """
        Find the best split point by detecting silence or speaker changes
        """
        try:
            # Convert to mono if needed
            if waveform.dim() > 1:
                audio = waveform.mean(dim=0)
            else:
                audio = waveform
            
            search_audio = audio[start_idx:end_idx]
            
            # Calculate energy in small windows
            window_size = int(0.1 * sr)  # 100ms windows
            hop_size = int(0.05 * sr)    # 50ms hop
            
            energies = []
            positions = []
            
            for i in range(0, len(search_audio) - window_size, hop_size):
                window = search_audio[i:i + window_size]
                energy = torch.mean(window ** 2).item()
                energies.append(energy)
                positions.append(start_idx + i + window_size // 2)
            
            if not energies:
                return None
            
            # Find the minimum energy position (likely silence or speaker transition)
            min_energy_idx = np.argmin(energies)
            optimal_position = positions[min_energy_idx]
            
            # Validate the position is reasonable
            if start_idx <= optimal_position <= end_idx:
                return optimal_position
            
            return None
            
        except Exception as e:
            logger.warning(f"Error in silence detection: {e}")
            return None
    
    def _extract_embedding_from_waveform(self, waveform: torch.Tensor, sr: int) -> Optional[np.ndarray]:
        """Extract speaker embedding from audio waveform"""
        try:
            if self.speaker_embedding_manager is None:
                return None
            
            # Convert to numpy
            if waveform.dim() > 1:
                audio_numpy = waveform.mean(dim=0).cpu().numpy()
            else:
                audio_numpy = waveform.cpu().numpy()
            
            # Resample if needed
            if sr != self.sample_rate:
                # Simple resampling - could be improved with proper resampling
                target_length = int(len(audio_numpy) * self.sample_rate / sr)
                audio_numpy = np.interp(
                    np.linspace(0, len(audio_numpy), target_length),
                    np.arange(len(audio_numpy)),
                    audio_numpy
                )
            
            # Extract embedding
            return self.speaker_embedding_manager.extract_embedding(audio_numpy)
            
        except Exception as e:
            logger.error(f"Error extracting embedding from waveform: {e}")
            return None
    
    def _calculate_split_confidence(self, 
                                   segment_audio: torch.Tensor, 
                                   other_audio: torch.Tensor, 
                                   sr: int) -> float:
        """
        Calculate confidence in the split based on audio characteristics
        """
        try:
            # Extract embeddings for both segments
            embedding1 = self._extract_embedding_from_waveform(segment_audio, sr)
            embedding2 = self._extract_embedding_from_waveform(other_audio, sr)
            
            if embedding1 is None or embedding2 is None:
                return 0.5  # Default confidence
            
            # Calculate similarity between the two segments
            similarity = 1 - cosine(embedding1, embedding2)
            
            # Lower similarity = higher confidence in the split being meaningful
            confidence = 1 - similarity
            
            # Clamp between 0.1 and 0.9
            return max(0.1, min(0.9, confidence))
            
        except Exception as e:
            logger.warning(f"Error calculating split confidence: {e}")
            return 0.5
    
    def _update_speaker_with_embedding(self, speaker_name: str, audio_data: Optional[np.ndarray]) -> bool:
        """Update speaker profile with new embedding from split segment"""
        try:
            if audio_data is None or self.speaker_embedding_manager is None:
                return False
            
            # Provide feedback to improve speaker recognition
            success = self.speaker_embedding_manager.provide_correction_feedback(
                speaker_name, audio_data
            )
            
            if success:
                logger.debug(f"Updated speaker model for {speaker_name} with split segment audio")
            else:
                logger.warning(f"Failed to update speaker model for {speaker_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating speaker with embedding: {e}")
            return False

class SegmentSplitValidator:
    """
    Validates and improves segment splits using audio analysis
    """
    
    def __init__(self, splitter: AudioSegmentSplitter):
        self.splitter = splitter
    
    def validate_split(self, 
                      first_segment: Dict, 
                      second_segment: Dict, 
                      audio_file: str) -> Dict:
        """
        Validate a split and suggest improvements
        
        Returns:
            Dictionary with validation results and suggestions
        """
        try:
            validation_result = {
                'is_valid': True,
                'confidence': 0.0,
                'issues': [],
                'suggestions': []
            }
            
            # Check minimum segment lengths
            first_duration = first_segment.get('audio_duration', 0)
            second_duration = second_segment.get('audio_duration', 0)
            
            if first_duration < self.splitter.min_segment_length:
                validation_result['is_valid'] = False
                validation_result['issues'].append(f"First segment too short: {first_duration:.2f}s")
                validation_result['suggestions'].append("Consider merging with adjacent segment or adjusting split point")
            
            if second_duration < self.splitter.min_segment_length:
                validation_result['is_valid'] = False
                validation_result['issues'].append(f"Second segment too short: {second_duration:.2f}s")
                validation_result['suggestions'].append("Consider merging with adjacent segment or adjusting split point")
            
            # Check embedding extraction success
            if not first_segment.get('embedding_extracted', False):
                validation_result['issues'].append("Failed to extract embedding for first segment")
                validation_result['suggestions'].append("Audio quality may be poor - consider manual verification")
            
            if not second_segment.get('embedding_extracted', False):
                validation_result['issues'].append("Failed to extract embedding for second segment")
                validation_result['suggestions'].append("Audio quality may be poor - consider manual verification")
            
            # Calculate overall confidence
            first_conf = first_segment.get('split_confidence', 0.5)
            second_conf = second_segment.get('split_confidence', 0.5)
            validation_result['confidence'] = (first_conf + second_conf) / 2
            
            # Add confidence-based suggestions
            if validation_result['confidence'] < 0.3:
                validation_result['suggestions'].append("Low confidence split - consider adjusting split point")
            elif validation_result['confidence'] > 0.7:
                validation_result['suggestions'].append("High confidence split - good speaker separation detected")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating split: {e}")
            return {
                'is_valid': False,
                'confidence': 0.0,
                'issues': [f"Validation error: {e}"],
                'suggestions': ["Manual review recommended"]
            } 