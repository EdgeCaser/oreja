"""
Unit tests for batch transcription functionality.
Tests batch processing, file I/O, and transcription result management.
"""

import json
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open

# Import the batch transcription module
# Note: This assumes batch_transcription.py exists in the backend directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import batch_transcription
from batch_transcription import BatchTranscriptionProcessor, process_audio_file, save_transcription_result


class TestBatchTranscriptionModule:
    """Test batch transcription module structure and imports."""
    
    @pytest.mark.unit
    def test_module_imports(self):
        """Test that the module imports correctly."""
        assert hasattr(batch_transcription, 'BatchTranscriptionProcessor')
        assert hasattr(batch_transcription, 'process_audio_file')
        assert hasattr(batch_transcription, 'save_transcription_result')
    
    @pytest.mark.unit
    def test_batch_processor_initialization(self):
        """Test BatchTranscriptionProcessor initialization."""
        processor = BatchTranscriptionProcessor()
        
        assert processor.backend_url == "http://127.0.0.1:8000"
        assert processor.SAMPLE_RATE == 16000
        assert hasattr(processor, 'speaker_manager')
        assert hasattr(processor, 'results')


class TestBatchTranscriptionProcessor:
    """Test the BatchTranscriptionProcessor class."""
    
    @pytest.mark.unit
    def test_processor_creation(self):
        """Test creating a batch transcription processor."""
        processor = BatchTranscriptionProcessor(backend_url="http://localhost:8000")
        
        assert processor.backend_url == "http://localhost:8000"
        assert processor.SAMPLE_RATE == 16000
        assert processor.MIN_SEGMENT_LENGTH == 0.5
        assert isinstance(processor.results, list)
    
    @pytest.mark.unit
    @patch('batch_transcription.torchaudio.load')
    def test_load_audio(self, mock_load):
        """Test audio loading functionality."""
        import torch
        
        # Mock torchaudio.load
        mock_waveform = torch.randn(1, 16000)  # 1 second of audio
        mock_load.return_value = (mock_waveform, 16000)
        
        processor = BatchTranscriptionProcessor()
        
        # Test loading
        waveform, sample_rate = processor._load_audio(Path("test.wav"))
        
        assert waveform.shape == (1, 16000)
        assert sample_rate == 16000
        mock_load.assert_called_once()
    
    @pytest.mark.unit
    @patch('batch_transcription.asyncio.run')
    def test_transcribe_audio(self, mock_asyncio_run):
        """Test transcription functionality."""
        import torch
        
        # Mock the transcription result
        mock_result = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Test transcription", "speaker": "SPEAKER_00"}
            ],
            "full_text": "Test transcription"
        }
        mock_asyncio_run.return_value = mock_result
        
        processor = BatchTranscriptionProcessor()
        waveform = torch.randn(1, 16000)
        
        result = processor._transcribe_audio(waveform, 16000)
        
        assert result == mock_result
        mock_asyncio_run.assert_called_once()


class TestWrapperFunctions:
    """Test wrapper functions expected by tests."""
    
    @pytest.mark.unit
    @patch('batch_transcription.BatchTranscriptionProcessor')
    def test_process_audio_file(self, mock_processor_class):
        """Test process_audio_file wrapper function."""
        mock_processor = Mock()
        mock_processor.process_recording.return_value = {"success": True}
        mock_processor_class.return_value = mock_processor
        
        result = process_audio_file("test.wav", "output_dir")
        
        assert result == {"success": True}
        mock_processor_class.assert_called_once()
        mock_processor.process_recording.assert_called_once()
    
    @pytest.mark.unit
    def test_save_transcription_result(self, tmp_path):
        """Test save_transcription_result function."""
        test_result = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Test", "speaker": "SPEAKER_00"}
            ],
            "full_text": "Test"
        }
        
        output_path = tmp_path / "test_result.json"
        
        save_transcription_result(test_result, str(output_path))
        
        assert output_path.exists()
        
        # Verify content
        with open(output_path, 'r') as f:
            loaded_result = json.load(f)
        
        assert loaded_result == test_result


class TestBatchProcessing:
    """Test batch processing functionality."""
    
    @pytest.mark.unit
    @patch('batch_transcription.BatchTranscriptionProcessor.process_recording')
    def test_batch_processing_multiple_files(self, mock_process):
        """Test processing multiple files in batch."""
        # Mock successful processing result
        mock_process.return_value = {
            "segments": [{"start": 0.0, "end": 2.0, "text": "Test", "speaker": "SPEAKER_00"}],
            "success": True
        }
        
        processor = BatchTranscriptionProcessor()
        
        # Create mock audio files
        audio_files = [Path(f"test_{i}.wav") for i in range(3)]
        output_dir = Path("output")
        
        results = processor.process_batch(
            audio_files,
            output_dir,
            improve_speakers=False,
            parallel=False
        )
        
        assert len(results) == 3
        assert mock_process.call_count == 3
        # Each result should have the mocked structure
        for result in results:
            assert "success" in result
    
    @pytest.mark.unit 
    def test_tensor_to_wav_bytes(self):
        """Test converting tensor to WAV bytes."""
        import torch
        
        processor = BatchTranscriptionProcessor()
        waveform = torch.randn(1, 16000)
        
        with patch('batch_transcription.torchaudio.save') as mock_save:
            mock_save.return_value = None
            
            wav_bytes = processor._tensor_to_wav_bytes(waveform, 16000)
            
            mock_save.assert_called_once()
            # Should return bytes from the buffer
            assert isinstance(wav_bytes, bytes)


@pytest.mark.integration
class TestBatchTranscriptionIntegration:
    """Integration tests for batch transcription."""
    
    @pytest.mark.integration
    @patch('batch_transcription.torchaudio.load')
    @patch('batch_transcription.asyncio.run')
    def test_full_processing_pipeline(self, mock_asyncio, mock_load):
        """Test complete processing pipeline."""
        import torch
        
        # Setup mocks
        mock_waveform = torch.randn(1, 32000)  # 2 seconds
        mock_load.return_value = (mock_waveform, 16000)
        
        mock_transcription = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Hello world", "speaker": "SPEAKER_00"}
            ],
            "full_text": "Hello world"
        }
        mock_asyncio.return_value = mock_transcription
        
        processor = BatchTranscriptionProcessor()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "test.wav"
            output_dir = Path(tmp_dir) / "output"
            
            # Create a dummy audio file
            audio_path.touch()
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Mock the file save operation to avoid file path issues
            with patch('json.dump') as mock_json_dump, \
                 patch('builtins.open', mock_open()) as mock_file_open:
                
                result = processor.process_recording(
                    audio_path,
                    output_dir,
                    improve_speakers=False
                )
                
                assert "segments" in result
                assert result["segments"][0]["text"] == "Hello world"


# Performance and stress tests
@pytest.mark.performance
class TestBatchPerformance:
    """Test performance characteristics of batch processing."""
    
    @pytest.mark.performance
    def test_large_batch_processing(self):
        """Test processing large batches efficiently."""
        processor = BatchTranscriptionProcessor()
        
        # Test with many files (mocked)
        with patch.object(processor, 'process_recording') as mock_process:
            # Mock successful processing result with proper structure
            mock_process.return_value = {
                "segments": [{"start": 0.0, "end": 2.0, "text": "Test", "speaker": "SPEAKER_00"}],
                "success": True,
                "file": "mocked_file.wav"
            }
            
            # Simulate 50 files
            audio_files = [Path(f"file_{i}.wav") for i in range(50)]
            output_dir = Path("output")
            
            import time
            start_time = time.time()
            
            results = processor.process_batch(
                audio_files,
                output_dir,
                improve_speakers=False,
                parallel=False
            )
            
            end_time = time.time()
            
            assert len(results) == 50
            assert mock_process.call_count == 50
            
            # Should complete in reasonable time (with mocking)
            assert end_time - start_time < 5.0  # seconds 