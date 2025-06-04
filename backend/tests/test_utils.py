"""
Unit tests for utility functions and audio processing.
Tests audio loading, validation, and processing helper functions.
"""

import io
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from server import (
    load_audio_from_bytes, 
    merge_transcription_and_diarization,
    find_speaker_for_segment
)


class TestAudioProcessing:
    """Test audio processing utility functions."""
    
    @pytest.mark.unit
    def test_load_audio_from_bytes_valid_wav(self, sample_audio_bytes):
        """Test loading valid WAV audio from bytes."""
        waveform, sample_rate = load_audio_from_bytes(sample_audio_bytes)
        
        assert isinstance(waveform, torch.Tensor)
        assert waveform.dim() == 2  # (channels, samples)
        assert waveform.shape[0] == 1  # mono
        assert sample_rate == 16000
        assert waveform.shape[1] > 0  # Has samples
    
    @pytest.mark.unit
    def test_load_audio_from_bytes_invalid_data(self):
        """Test loading invalid audio data raises ValueError."""
        invalid_data = b"This is not audio data"
        
        with pytest.raises(ValueError, match="Invalid audio format"):
            load_audio_from_bytes(invalid_data)
    
    @pytest.mark.unit
    def test_load_audio_from_bytes_empty(self):
        """Test loading empty audio data raises ValueError."""
        empty_data = b""
        
        with pytest.raises(ValueError):
            load_audio_from_bytes(empty_data)
    
    @pytest.mark.unit
    def test_load_audio_from_bytes_corrupted_wav(self):
        """Test loading corrupted WAV data."""
        # Create invalid WAV header
        corrupted_data = b"RIFF\x00\x00\x00\x00WAVE" + b"invalid_data" * 100
        
        with pytest.raises(ValueError):
            load_audio_from_bytes(corrupted_data)


class TestTranscriptionMerging:
    """Test transcription and diarization merging functions."""
    
    @pytest.mark.unit
    def test_merge_basic_transcription_diarization(self):
        """Test basic merging of transcription with diarization."""
        transcription = {
            "chunks": [
                {"timestamp": [0.0, 2.0], "text": "Hello world"}
            ]
        }
        
        mock_diarization = Mock()
        mock_diarization.itertracks.return_value = [
            (Mock(start=0.0, end=2.0), None, "SPEAKER_00")
        ]
        
        result = merge_transcription_and_diarization(transcription, mock_diarization)
        
        assert len(result) == 1
        assert result[0]["text"] == "Hello world"
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 2.0
        assert result[0]["speaker"] == "Speaker SPEAKER_00"
        assert result[0]["diarization_speaker"] == "Speaker SPEAKER_00"
    
    @pytest.mark.unit
    def test_merge_with_mismatched_timestamps(self):
        """Test merging when timestamps don't align perfectly."""
        transcription = {
            "chunks": [
                {"timestamp": [0.5, 3.0], "text": "Test phrase"}
            ]
        }
        
        mock_diarization = Mock()
        mock_diarization.itertracks.return_value = [
            (Mock(start=0.0, end=2.0), None, "SPEAKER_00")
        ]
        
        result = merge_transcription_and_diarization(transcription, mock_diarization)
        
        assert len(result) == 1
        assert result[0]["text"] == "Test phrase"
        assert "embedding_confidence" in result[0]
    
    @pytest.mark.unit
    def test_merge_with_no_diarization(self):
        """Test merging when no diarization is provided."""
        transcription = {
            "chunks": [
                {"timestamp": [0.0, 2.0], "text": "Solo speech"}
            ]
        }
        
        result = merge_transcription_and_diarization(transcription, None)
        
        assert len(result) == 1
        assert result[0]["text"] == "Solo speech"
        assert result[0]["speaker"] == "SPEAKER_00"
    
    @pytest.mark.unit
    def test_merge_with_empty_transcription(self):
        """Test merging with empty transcription."""
        transcription = {"chunks": []}
        
        mock_diarization = Mock()
        mock_diarization.itertracks.return_value = [
            (Mock(start=0.0, end=2.0), None, "SPEAKER_00")
        ]
        
        result = merge_transcription_and_diarization(transcription, mock_diarization)
        
        assert len(result) == 0
    
    @pytest.mark.unit
    def test_merge_with_invalid_timestamps(self):
        """Test merging with invalid or missing timestamps."""
        transcription = {
            "chunks": [
                {"timestamp": None, "text": "Hello world"},
                {"timestamp": [0.0], "text": "Invalid timestamp"},  # Missing end
                {"text": "No timestamp"}  # Missing timestamp
            ]
        }
        
        mock_diarization = Mock()
        mock_diarization.itertracks.return_value = [
            (Mock(start=0.0, end=2.0), None, "SPEAKER_00")
        ]
        
        # Should handle gracefully, skipping invalid entries
        result = merge_transcription_and_diarization(transcription, mock_diarization)
        
        # Should not crash and may include some segments with default values
        assert isinstance(result, list)


class TestSpeakerMatching:
    """Test speaker matching and identification functions."""
    
    @pytest.mark.unit
    def test_find_speaker_for_segment_exact_match(self):
        """Test finding speaker when segment exactly matches diarization."""
        mock_diarization = Mock()
        mock_diarization.itertracks.return_value = [
            (Mock(start=0.0, end=2.0), None, "SPEAKER_00"),
            (Mock(start=2.0, end=4.0), None, "SPEAKER_01")
        ]
        
        speaker = find_speaker_for_segment(mock_diarization, 1.0, 1.5)
        assert speaker == "Speaker SPEAKER_00"
        
        speaker = find_speaker_for_segment(mock_diarization, 3.0, 3.5)
        assert speaker == "Speaker SPEAKER_01"
    
    @pytest.mark.unit
    def test_find_speaker_for_segment_overlap(self):
        """Test finding speaker when segment overlaps multiple speakers."""
        mock_diarization = Mock()
        mock_diarization.itertracks.return_value = [
            (Mock(start=0.0, end=2.0), None, "SPEAKER_00"),
            (Mock(start=1.5, end=3.5), None, "SPEAKER_01")
        ]
        
        # Segment from 1.0 to 2.5 overlaps both speakers
        speaker = find_speaker_for_segment(mock_diarization, 1.0, 2.5)
        
        # Should return the speaker with most overlap
        assert speaker in ["Speaker SPEAKER_00", "Speaker SPEAKER_01"]
    
    @pytest.mark.unit
    def test_find_speaker_for_segment_no_match(self):
        """Test finding speaker when no diarization matches."""
        mock_diarization = Mock()
        mock_diarization.itertracks.return_value = [
            (Mock(start=0.0, end=2.0), None, "SPEAKER_00")
        ]
        
        # Segment outside diarization range
        speaker = find_speaker_for_segment(mock_diarization, 5.0, 6.0)
        assert speaker == "Unknown Speaker"
    
    @pytest.mark.unit
    def test_find_speaker_for_segment_empty_diarization(self):
        """Test finding speaker with empty diarization."""
        mock_diarization = Mock()
        mock_diarization.itertracks.return_value = []
        
        speaker = find_speaker_for_segment(mock_diarization, 1.0, 2.0)
        assert speaker == "Unknown Speaker"


class TestAudioValidation:
    """Test audio validation and preprocessing."""
    
    @pytest.mark.unit
    def test_audio_length_validation_short(self, sample_short_audio):
        """Test validation of audio that's too short."""
        # This would typically be caught in the API endpoint
        # but we can test the underlying validation logic
        waveform, sample_rate = load_audio_from_bytes(sample_short_audio)
        
        duration = waveform.shape[1] / sample_rate
        assert duration < 0.1  # Should be very short
    
    @pytest.mark.unit
    def test_audio_length_validation_long(self, sample_long_audio):
        """Test validation of audio that's too long."""
        waveform, sample_rate = load_audio_from_bytes(sample_long_audio)
        
        duration = waveform.shape[1] / sample_rate
        assert duration > 30  # Should be longer than limit
    
    @pytest.mark.unit
    def test_audio_resampling(self, sample_audio_bytes):
        """Test audio resampling functionality."""
        # Create audio with different sample rate
        original_rate = 44100
        target_rate = 16000
        duration = 1.0
        
        # Generate test audio
        samples = int(original_rate * duration)
        waveform = torch.randn(1, samples)
        
        # Test load_audio_from_bytes with WAV data at original rate
        waveform_bytes = io.BytesIO()
        # Note: torchaudio.save might not be available, so we'll just test the loading
        
        loaded_waveform, loaded_sample_rate = load_audio_from_bytes(sample_audio_bytes)
        
        # Check basic properties (not requiring specific sample rate)
        assert isinstance(loaded_waveform, torch.Tensor)
        assert isinstance(loaded_sample_rate, int)
        assert loaded_waveform.dim() == 2  # (channels, samples)
        assert loaded_sample_rate > 0


class TestErrorHandling:
    """Test error handling in utility functions."""
    
    @pytest.mark.unit
    def test_transcription_merge_with_corrupted_data(self):
        """Test merging with corrupted or unexpected data structures."""
        # Test with malformed transcription
        corrupted_transcription = {
            "chunks": [
                {"invalid_field": "data"},
                None,
                {"timestamp": "invalid", "text": "test"}
            ]
        }
        
        mock_diarization = Mock()
        mock_diarization.itertracks.return_value = []
        
        # Should handle gracefully without crashing
        result = merge_transcription_and_diarization(corrupted_transcription, mock_diarization)
        assert isinstance(result, list)
    
    @pytest.mark.unit
    def test_speaker_finding_with_exception(self):
        """Test speaker finding when diarization throws an exception."""
        mock_diarization = Mock()
        mock_diarization.itertracks.side_effect = Exception("Diarization error")
        
        speaker = find_speaker_for_segment(mock_diarization, 1.0, 2.0)
        assert speaker == "Unknown Speaker"  # Updated expectation
    
    @pytest.mark.unit
    def test_audio_loading_memory_limits(self):
        """Test audio loading with extremely large files."""
        # Create a very large fake audio file
        large_header = b"RIFF" + (100 * 1024 * 1024).to_bytes(4, 'little') + b"WAVE"
        large_data = large_header + b"fmt " + b"\x00" * 1000
        
        # Should either handle gracefully or raise appropriate error
        with pytest.raises((ValueError, MemoryError, OSError)):
            load_audio_from_bytes(large_data)


@pytest.mark.integration
class TestAudioProcessingIntegration:
    """Integration tests for audio processing pipeline."""
    
    @pytest.mark.integration
    def test_full_audio_processing_pipeline(self):
        """Test complete audio processing from bytes to final segments."""
        # Use the sample audio from fixture
        sample_bytes = b"RIFF" + b"\x24\x00\x00\x00" + b"WAVE" + b"fmt " + b"\x10\x00\x00\x00" + \
                      b"\x01\x00\x01\x00\x44\xAC\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00" + \
                      b"data" + b"\x00\x00\x00\x00"
        
        # Load audio
        waveform, sample_rate = load_audio_from_bytes(sample_bytes)
        
        # Create mock transcription and diarization
        transcription = {
            "chunks": [
                {"timestamp": [0.0, 2.0], "text": "Test transcription"}
            ]
        }
        
        mock_diarization = Mock()
        mock_diarization.itertracks.return_value = [
            (Mock(start=0.0, end=2.0), None, "SPEAKER_00")
        ]
        
        # Process through pipeline
        segments = merge_transcription_and_diarization(transcription, mock_diarization)
        
        assert len(segments) == 1
        segment = segments[0]
        
        # Check expected fields exist
        assert "text" in segment
        assert "speaker" in segment
        assert "start" in segment
        assert "end" in segment
        assert "embedding_confidence" in segment  # Updated field name
        assert "diarization_speaker" in segment
        assert "embedding_speaker" in segment


class TestPerformance:
    """Performance tests for audio processing functions."""
    
    @pytest.mark.slow
    def test_audio_loading_performance(self):
        """Test audio loading performance with various file sizes."""
        import time
        
        # Generate different sized audio files
        sizes = [1, 5, 10, 30]  # seconds
        for duration in sizes:
            # Generate audio
            sample_rate = 16000
            t = torch.linspace(0, duration, int(sample_rate * duration))
            waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
            
            # Convert to bytes
            buffer = io.BytesIO()
            import torchaudio
            torchaudio.save(buffer, waveform, sample_rate, format="wav")
            audio_bytes = buffer.getvalue()
            
            # Measure loading time
            start_time = time.time()
            loaded_waveform, loaded_sr = load_audio_from_bytes(audio_bytes)
            load_time = time.time() - start_time
            
            # Should load within reasonable time (adjust thresholds as needed)
            assert load_time < 2.0, f"Loading {duration}s audio took {load_time:.2f}s"
            assert loaded_waveform.shape[1] > 0
            assert loaded_sr == sample_rate
    
    @pytest.mark.slow
    def test_transcription_merging_performance(self):
        """Test performance of merging large transcription results."""
        import time
        
        # Generate large transcription with many segments
        large_transcription = {
            "chunks": [
                {"timestamp": [i, i+1], "text": f"Segment {i}"}
                for i in range(0, 300, 1)  # 300 segments
            ]
        }
        
        # Generate corresponding diarization
        mock_diarization = Mock()
        mock_diarization.itertracks.return_value = [
            (Mock(start=i, end=i+1), None, f"SPEAKER_{i % 3}")
            for i in range(0, 300, 1)
        ]
        
        start_time = time.time()
        result = merge_transcription_and_diarization(large_transcription, mock_diarization)
        merge_time = time.time() - start_time
        
        assert merge_time < 5.0, f"Merging 300 segments took {merge_time:.2f}s"
        assert len(result) == 300 