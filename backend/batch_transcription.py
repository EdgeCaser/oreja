#!/usr/bin/env python3
"""
Batch Transcription Module for Recorded Calls
Leverages existing speaker embeddings to transcribe recordings and improve speaker recognition.
"""

import os
import json
import logging
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import numpy as np

import torch
import torchaudio
from torchaudio.transforms import Resample
import requests
from scipy.spatial.distance import cosine

from speaker_embeddings import OfflineSpeakerEmbeddingManager
from server import load_audio_from_bytes, run_transcription

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchTranscriptionProcessor:
    """
    Batch processor for recorded calls that leverages existing speaker embeddings
    """
    
    def __init__(self, backend_url: str = "http://127.0.0.1:8000"):
        self.backend_url = backend_url
        self.speaker_manager = OfflineSpeakerEmbeddingManager()
        self.results: List[Dict[str, Any]] = []
        
        # Configuration
        self.SAMPLE_RATE = 16000
        self.MIN_SEGMENT_LENGTH = 0.5  # seconds
        self.CONFIDENCE_THRESHOLD = 0.7
        self.SIMILARITY_THRESHOLD = 0.75
        
    def process_recording(self, 
                         audio_path: Path,
                         output_dir: Optional[Path] = None,
                         improve_speakers: bool = True,
                         speaker_name_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process a single recorded call
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save results (optional)
            improve_speakers: Whether to use this recording to improve speaker models
            speaker_name_mapping: Manual mapping of auto-detected speakers to known names
            
        Returns:
            Transcription result with speaker identification
        """
        logger.info(f"Processing recording: {audio_path}")
        
        try:
            # Load and preprocess audio
            waveform, sample_rate = self._load_audio(audio_path)
            
            # Get transcription from backend
            transcription_result = self._transcribe_audio(waveform, sample_rate)
            
            # Enhance speaker identification using existing embeddings
            enhanced_result = self._enhance_speaker_identification(
                transcription_result, waveform, sample_rate, speaker_name_mapping
            )
            
            # Improve speaker models if requested
            if improve_speakers:
                self._improve_speaker_models(enhanced_result, waveform, sample_rate)
            
            # Save results
            if output_dir:
                self._save_results(enhanced_result, audio_path, output_dir)
            
            # Add to batch results
            self.results.append({
                'file': str(audio_path),
                'result': enhanced_result,
                'processed_at': datetime.now().isoformat(),
                'speakers_improved': improve_speakers
            })
            
            logger.info(f"Successfully processed {audio_path}")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            error_result = {
                'error': str(e),
                'file': str(audio_path),
                'processed_at': datetime.now().isoformat()
            }
            self.results.append(error_result)
            return error_result
    
    def process_batch(self,
                     audio_files: List[Path],
                     output_dir: Path,
                     improve_speakers: bool = True,
                     speaker_name_mapping: Optional[Dict[str, str]] = None,
                     parallel: bool = False) -> List[Dict[str, Any]]:
        """
        Process multiple recordings in batch
        
        Args:
            audio_files: List of audio file paths
            output_dir: Directory to save all results
            improve_speakers: Whether to use recordings to improve speaker models
            speaker_name_mapping: Manual mapping of speakers
            parallel: Whether to process files in parallel
            
        Returns:
            List of transcription results
        """
        logger.info(f"Starting batch processing of {len(audio_files)} files")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if parallel and len(audio_files) > 1:
            # Process in parallel (be careful with resource usage)
            return self._process_parallel(audio_files, output_dir, improve_speakers, speaker_name_mapping)
        else:
            # Process sequentially
            results = []
            for i, audio_file in enumerate(audio_files, 1):
                logger.info(f"Processing file {i}/{len(audio_files)}: {audio_file.name}")
                result = self.process_recording(
                    audio_file, output_dir, improve_speakers, speaker_name_mapping
                )
                results.append(result)
        
        # Save batch summary
        self._save_batch_summary(output_dir)
        
        logger.info(f"Batch processing complete. Processed {len(audio_files)} files.")
        return results
    
    def _load_audio(self, audio_path: Path) -> Tuple[torch.Tensor, int]:
        """Load and preprocess audio file"""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if needed
            if sample_rate != self.SAMPLE_RATE:
                resampler = Resample(sample_rate, self.SAMPLE_RATE)
                waveform = resampler(waveform)
                sample_rate = self.SAMPLE_RATE
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            logger.debug(f"Loaded audio: {waveform.shape}, {sample_rate} Hz")
            return waveform, sample_rate
            
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            raise
    
    def _transcribe_audio(self, waveform: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
        """Get transcription from the backend"""
        try:
            # Use the existing transcription function
            result = asyncio.run(run_transcription(waveform, sample_rate))
            return result
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            # Fallback: try backend API
            return self._transcribe_via_api(waveform, sample_rate)
    
    def _transcribe_via_api(self, waveform: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
        """Fallback transcription via API"""
        try:
            # Convert tensor to wav bytes
            wav_data = self._tensor_to_wav_bytes(waveform, sample_rate)
            
            # Send to backend
            files = {'audio': ('recording.wav', wav_data, 'audio/wav')}
            response = requests.post(f"{self.backend_url}/transcribe", files=files)
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"API transcription failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"API transcription error: {e}")
            raise
    
    def _enhance_speaker_identification(self,
                                      transcription_result: Dict[str, Any],
                                      waveform: torch.Tensor,
                                      sample_rate: int,
                                      speaker_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Enhance speaker identification using existing embeddings
        """
        enhanced_segments = []
        
        for segment in transcription_result.get('segments', []):
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            text = segment.get('text', '')
            original_speaker = segment.get('speaker', 'Unknown')
            
            # Extract audio segment
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            if end_sample > start_sample and end_sample <= waveform.shape[1]:
                segment_waveform = waveform[:, start_sample:end_sample]
                segment_duration = (end_sample - start_sample) / sample_rate
                
                # Only process segments long enough for reliable identification
                if segment_duration >= self.MIN_SEGMENT_LENGTH:
                    enhanced_speaker, confidence = self._identify_speaker_from_segment(
                        segment_waveform, sample_rate
                    )
                    
                    # Apply manual mapping if provided
                    if speaker_mapping and enhanced_speaker in speaker_mapping:
                        enhanced_speaker = speaker_mapping[enhanced_speaker]
                    
                    segment['enhanced_speaker'] = enhanced_speaker
                    segment['speaker_confidence'] = confidence
                    segment['segment_duration'] = segment_duration
                    
                    # Use enhanced speaker if confidence is high enough
                    if confidence >= self.CONFIDENCE_THRESHOLD:
                        segment['speaker'] = enhanced_speaker
                        segment['identification_method'] = 'embedding_enhanced'
                    else:
                        segment['identification_method'] = 'fallback_diarization'
                else:
                    segment['enhanced_speaker'] = original_speaker
                    segment['speaker_confidence'] = 0.0
                    segment['identification_method'] = 'too_short'
            else:
                segment['enhanced_speaker'] = original_speaker
                segment['speaker_confidence'] = 0.0
                segment['identification_method'] = 'invalid_timing'
            
            enhanced_segments.append(segment)
        
        # Update result
        enhanced_result = transcription_result.copy()
        enhanced_result['segments'] = enhanced_segments
        enhanced_result['enhancement_info'] = {
            'total_segments': len(enhanced_segments),
            'enhanced_segments': len([s for s in enhanced_segments if s.get('identification_method') == 'embedding_enhanced']),
            'confidence_threshold': self.CONFIDENCE_THRESHOLD,
            'processing_time': datetime.now().isoformat()
        }
        
        return enhanced_result
    
    def _identify_speaker_from_segment(self, segment_waveform: torch.Tensor, sample_rate: int) -> Tuple[str, float]:
        """
        Identify speaker from audio segment using existing embeddings
        """
        try:
            # Convert to numpy for embedding extraction
            audio_numpy = segment_waveform.squeeze().numpy()
            
            # Use speaker manager to identify
            speaker_id, confidence, is_new = self.speaker_manager.identify_or_create_speaker(
                audio_numpy, min_confidence=self.SIMILARITY_THRESHOLD
            )
            
            # Get speaker name
            if speaker_id in self.speaker_manager.speaker_profiles:
                speaker_name = self.speaker_manager.speaker_profiles[speaker_id].name
            else:
                speaker_name = speaker_id
            
            logger.debug(f"Speaker identification: {speaker_name} (confidence: {confidence:.3f})")
            return speaker_name, confidence
            
        except Exception as e:
            logger.warning(f"Error in speaker identification: {e}")
            return "Unknown", 0.0
    
    def _improve_speaker_models(self,
                               transcription_result: Dict[str, Any],
                               waveform: torch.Tensor,
                               sample_rate: int):
        """
        Use the recording to improve existing speaker models
        """
        improvement_count = 0
        
        for segment in transcription_result.get('segments', []):
            if segment.get('identification_method') == 'embedding_enhanced':
                speaker_name = segment.get('enhanced_speaker')
                confidence = segment.get('speaker_confidence', 0)
                
                # Only use high-confidence segments for improvement
                if confidence >= 0.8 and speaker_name and speaker_name != 'Unknown':
                    start_time = segment.get('start', 0)
                    end_time = segment.get('end', 0)
                    
                    # Extract segment audio
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    
                    if end_sample > start_sample and end_sample <= waveform.shape[1]:
                        segment_waveform = waveform[:, start_sample:end_sample]
                        audio_numpy = segment_waveform.squeeze().numpy()
                        
                        # Provide feedback to improve the model
                        success = self.speaker_manager.provide_correction_feedback(
                            speaker_name, audio_numpy
                        )
                        
                        if success:
                            improvement_count += 1
                            logger.debug(f"Improved model for {speaker_name}")
        
        if improvement_count > 0:
            logger.info(f"Improved speaker models with {improvement_count} segments")
    
    def _save_results(self, result: Dict[str, Any], audio_path: Path, output_dir: Path):
        """Save transcription results to files"""
        base_name = audio_path.stem
        
        # Save JSON result
        json_path = output_dir / f"{base_name}_transcription.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save readable transcript
        txt_path = output_dir / f"{base_name}_transcript.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Transcription for: {audio_path.name}\n")
            f.write(f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            for segment in result.get('segments', []):
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)
                speaker = segment.get('speaker', 'Unknown')
                text = segment.get('text', '')
                confidence = segment.get('speaker_confidence', 0)
                method = segment.get('identification_method', 'unknown')
                
                f.write(f"[{start_time:.1f}s - {end_time:.1f}s] {speaker}")
                if confidence > 0:
                    f.write(f" (conf: {confidence:.2f}, {method})")
                f.write(f": {text}\n")
            
            # Add summary
            enhancement_info = result.get('enhancement_info', {})
            f.write("\n" + "=" * 60 + "\n")
            f.write("Enhancement Summary:\n")
            f.write(f"Total segments: {enhancement_info.get('total_segments', 0)}\n")
            f.write(f"Enhanced segments: {enhancement_info.get('enhanced_segments', 0)}\n")
            f.write(f"Confidence threshold: {enhancement_info.get('confidence_threshold', 0)}\n")
    
    def _save_batch_summary(self, output_dir: Path):
        """Save summary of batch processing"""
        summary_path = output_dir / "batch_summary.json"
        
        summary = {
            'batch_info': {
                'total_files': len(self.results),
                'successful_files': len([r for r in self.results if 'error' not in r]),
                'failed_files': len([r for r in self.results if 'error' in r]),
                'processed_at': datetime.now().isoformat()
            },
            'results': self.results
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch summary saved to {summary_path}")
    
    def _process_parallel(self, audio_files: List[Path], output_dir: Path, 
                         improve_speakers: bool, speaker_mapping: Optional[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process files in parallel (simplified version)"""
        # For now, process sequentially to avoid resource conflicts
        # In a production environment, you'd want proper parallel processing with resource management
        return [self.process_recording(f, output_dir, improve_speakers, speaker_mapping) for f in audio_files]
    
    def _tensor_to_wav_bytes(self, waveform: torch.Tensor, sample_rate: int) -> bytes:
        """Convert tensor to WAV bytes for API calls"""
        import io
        buffer = io.BytesIO()
        torchaudio.save(buffer, waveform, sample_rate, format='wav')
        return buffer.getvalue()


def main():
    """Command-line interface for batch transcription"""
    parser = argparse.ArgumentParser(description="Batch transcription with speaker enhancement")
    parser.add_argument("input", help="Audio file or directory containing audio files")
    parser.add_argument("-o", "--output", help="Output directory for results")
    parser.add_argument("--improve-speakers", action="store_true", default=True,
                       help="Use recordings to improve speaker models")
    parser.add_argument("--speaker-mapping", help="JSON file with speaker name mappings")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000",
                       help="Backend server URL")
    parser.add_argument("--parallel", action="store_true",
                       help="Process files in parallel")
    parser.add_argument("--extensions", nargs="+", 
                       default=[".wav", ".mp3", ".flac", ".m4a", ".ogg"],
                       help="Audio file extensions to process")
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else Path("transcription_results")
    
    # Load speaker mapping if provided
    speaker_mapping = None
    if args.speaker_mapping:
        with open(args.speaker_mapping, 'r') as f:
            speaker_mapping = json.load(f)
    
    # Get audio files
    if input_path.is_file():
        audio_files = [input_path]
    elif input_path.is_dir():
        audio_files = []
        for ext in args.extensions:
            audio_files.extend(input_path.glob(f"*{ext}"))
            audio_files.extend(input_path.glob(f"*{ext.upper()}"))
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return
    
    if not audio_files:
        logger.error("No audio files found")
        return
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Process files
    processor = BatchTranscriptionProcessor(args.backend_url)
    
    try:
        results = processor.process_batch(
            audio_files,
            output_dir,
            improve_speakers=args.improve_speakers,
            speaker_name_mapping=speaker_mapping,
            parallel=args.parallel
        )
        
        # Print summary
        successful = len([r for r in results if 'error' not in r])
        failed = len(results) - successful
        
        print(f"\nBatch processing complete!")
        print(f"Successfully processed: {successful}/{len(results)} files")
        if failed > 0:
            print(f"Failed: {failed} files")
        print(f"Results saved to: {output_dir}")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise


def process_audio_file(audio_path: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Wrapper function to process a single audio file.
    Expected by tests - delegates to BatchTranscriptionProcessor.
    """
    processor = BatchTranscriptionProcessor()
    audio_path_obj = Path(audio_path)
    output_dir_obj = Path(output_dir) if output_dir else None
    
    result = processor.process_recording(
        audio_path_obj, 
        output_dir_obj, 
        improve_speakers=False
    )
    
    return result


def save_transcription_result(transcription_result: Dict[str, Any], output_path: str):
    """
    Save transcription result to a JSON file.
    Expected by tests.
    """
    output_path_obj = Path(output_path)
    
    # Ensure directory exists
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Save JSON file
    with open(output_path_obj, 'w', encoding='utf-8') as f:
        json.dump(transcription_result, f, indent=2, ensure_ascii=False)


def batch_process_directory(input_dir: str, output_dir: str, 
                          audio_extensions: List[str] = None) -> List[Dict[str, Any]]:
    """
    Process all audio files in a directory.
    Expected by tests - delegates to BatchTranscriptionProcessor.
    """
    if audio_extensions is None:
        audio_extensions = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    # Find all audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_path.glob(f"*{ext}"))
        audio_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not audio_files:
        return []
    
    # Process using BatchTranscriptionProcessor
    processor = BatchTranscriptionProcessor()
    results = processor.process_batch(
        audio_files,
        output_path,
        improve_speakers=False,
        parallel=False
    )
    
    return results


if __name__ == "__main__":
    main() 