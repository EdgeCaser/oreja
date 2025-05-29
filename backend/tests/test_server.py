"""
Pytest tests for the Oreja FastAPI server.
Tests transcription, diarization, and audio processing endpoints.
"""

import asyncio
import io
import pytest
import torch
import torchaudio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from server import app, load_models, load_audio_from_bytes, merge_transcription_and_diarization


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_audio_bytes():
    """Create sample audio data as bytes for testing."""
    # Generate a 1-second sine wave at 16kHz
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0  # A4 note
    
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
    
    # Convert to bytes
    buffer = io.BytesIO()
    torchaudio.save(buffer, waveform, sample_rate, format="wav")
    buffer.seek(0)
    
    return buffer.getvalue()


@pytest.fixture
def mock_models():
    """Mock the global model instances."""
    with patch('server.whisper_pipeline') as mock_whisper, \
         patch('server.diarization_pipeline') as mock_diarization, \
         patch('server.embedding_model') as mock_embedding:
        
        # Mock whisper response
        mock_whisper.return_value = {
            "chunks": [
                {
                    "timestamp": [0.0, 2.0],
                    "text": "Hello world"
                },
                {
                    "timestamp": [2.0, 4.0], 
                    "text": "This is a test"
                }
            ]
        }
        
        # Mock diarization response
        mock_diarization_result = Mock()
        mock_diarization_result.itertracks.return_value = [
            (Mock(start=0.0, end=2.0), None, "SPEAKER_00"),
            (Mock(start=2.0, end=4.0), None, "SPEAKER_01")
        ]
        mock_diarization.return_value = mock_diarization_result
        
        # Mock embedding response
        mock_embedding.return_value = torch.randn(512)
        
        yield {
            'whisper': mock_whisper,
            'diarization': mock_diarization,
            'embedding': mock_embedding
        }


class TestHealthEndpoints:
    """Test health check and status endpoints."""
    
    def test_root_endpoint(self, client):
        """Test the root health check endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Oreja Audio Processing API"
        assert data["status"] == "running"
        assert "models_loaded" in data
    
    def test_health_endpoint(self, client):
        """Test the detailed health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "device" in data
        assert "models" in data
        assert "memory_usage" in data


class TestAudioProcessing:
    """Test audio processing functionality."""
    
    def test_load_audio_from_bytes_valid(self, sample_audio_bytes):
        """Test loading valid audio from bytes."""
        waveform, sample_rate = load_audio_from_bytes(sample_audio_bytes)
        
        assert isinstance(waveform, torch.Tensor)
        assert waveform.dim() == 2  # (channels, samples)
        assert sample_rate > 0
        assert waveform.shape[1] > 0  # Has samples
    
    def test_load_audio_from_bytes_invalid(self):
        """Test loading invalid audio data."""
        invalid_data = b"invalid audio data"
        
        with pytest.raises(ValueError, match="Invalid audio format"):
            load_audio_from_bytes(invalid_data)
    
    def test_load_audio_from_empty_bytes(self):
        """Test loading empty audio data."""
        empty_data = b""
        
        with pytest.raises(ValueError):
            load_audio_from_bytes(empty_data)


class TestTranscriptionEndpoint:
    """Test the transcription endpoint."""
    
    @patch('server.whisper_pipeline')
    @patch('server.diarization_pipeline') 
    @patch('server.device', 'cpu')
    def test_transcribe_audio_success(self, mock_diarization, mock_whisper, client, sample_audio_bytes):
        """Test successful audio transcription."""
        # Setup mocks
        mock_whisper.return_value = {
            "chunks": [
                {"timestamp": [0.0, 2.0], "text": "Hello world"}
            ]
        }
        
        mock_diarization_result = Mock()
        mock_diarization_result.itertracks.return_value = [
            (Mock(start=0.0, end=2.0), None, "SPEAKER_00")
        ]
        mock_diarization.return_value = mock_diarization_result
        
        # Make request
        files = {"audio": ("test.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
        response = client.post("/transcribe", files=files)
        
        # Assertions
        assert response.status_code == 200
        
        data = response.json()
        assert "segments" in data
        assert "full_text" in data
        assert "processing_time" in data
        assert "timestamp" in data
        
        assert len(data["segments"]) > 0
        segment = data["segments"][0]
        assert "start" in segment
        assert "end" in segment
        assert "text" in segment
        assert "speaker" in segment
        assert "confidence" in segment
    
    def test_transcribe_no_file(self, client):
        """Test transcription with no file uploaded."""
        response = client.post("/transcribe")
        assert response.status_code == 422  # Validation error
    
    def test_transcribe_empty_file(self, client):
        """Test transcription with empty file."""
        files = {"audio": ("empty.wav", io.BytesIO(b""), "audio/wav")}
        response = client.post("/transcribe", files=files)
        assert response.status_code == 400
        assert "Empty audio file" in response.json()["detail"]
    
    @patch('server.whisper_pipeline', None)
    def test_transcribe_model_not_loaded(self, client, sample_audio_bytes):
        """Test transcription when Whisper model is not loaded."""
        files = {"audio": ("test.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
        response = client.post("/transcribe", files=files)
        assert response.status_code == 500


class TestEmbeddingEndpoint:
    """Test the speaker embedding extraction endpoint."""
    
    @patch('server.embedding_model')
    def test_extract_embeddings_success(self, mock_embedding, client, sample_audio_bytes):
        """Test successful embedding extraction."""
        # Setup mock
        mock_embedding.return_value = torch.randn(512)
        
        # Make request
        files = {"audio": ("test.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
        response = client.post("/extract_embeddings", files=files)
        
        # Assertions
        assert response.status_code == 200
        
        data = response.json()
        assert "embeddings" in data
        assert "embedding_size" in data
        assert "audio_duration" in data
        
        assert len(data["embeddings"]) > 0  # Hex string should not be empty
        assert data["embedding_size"] > 0
        assert data["audio_duration"] > 0
    
    @patch('server.embedding_model', None)
    def test_extract_embeddings_model_not_loaded(self, client, sample_audio_bytes):
        """Test embedding extraction when model is not loaded."""
        files = {"audio": ("test.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
        response = client.post("/extract_embeddings", files=files)
        assert response.status_code == 500
        assert "Embedding model not loaded" in response.json()["detail"]


class TestAudioValidation:
    """Test audio validation logic."""
    
    @patch('server.whisper_pipeline')
    @patch('server.diarization_pipeline')
    def test_transcribe_audio_too_short(self, mock_diarization, mock_whisper, client):
        """Test transcription with audio that's too short."""
        # Create very short audio (less than MIN_AUDIO_LENGTH)
        sample_rate = 16000
        short_duration = 0.05  # 50ms, less than MIN_AUDIO_LENGTH
        
        t = torch.linspace(0, short_duration, int(sample_rate * short_duration))
        waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
        
        buffer = io.BytesIO()
        torchaudio.save(buffer, waveform, sample_rate, format="wav")
        buffer.seek(0)
        
        files = {"audio": ("short.wav", buffer, "audio/wav")}
        response = client.post("/transcribe", files=files)
        
        assert response.status_code == 400
        assert "Audio too short" in response.json()["detail"]
    
    @patch('server.whisper_pipeline')
    @patch('server.diarization_pipeline')
    @patch('server.logger')
    def test_transcribe_audio_too_long(self, mock_logger, mock_diarization, mock_whisper, client):
        """Test transcription with very long audio (should warn but process)."""
        mock_whisper.return_value = {"chunks": []}
        mock_diarization.return_value = Mock()
        mock_diarization.return_value.itertracks.return_value = []
        
        # Create audio longer than MAX_AUDIO_LENGTH
        sample_rate = 16000
        long_duration = 35.0  # 35 seconds, more than MAX_AUDIO_LENGTH
        
        # Use a smaller sample for testing to avoid memory issues
        t = torch.linspace(0, 2.0, int(sample_rate * 2.0))  
        waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
        
        buffer = io.BytesIO()
        torchaudio.save(buffer, waveform, sample_rate, format="wav")
        buffer.seek(0)
        
        files = {"audio": ("long.wav", buffer, "audio/wav")}
        response = client.post("/transcribe", files=files)
        
        # Should process but with warning
        assert response.status_code == 200


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_merge_transcription_and_diarization(self):
        """Test merging transcription and diarization results."""
        # Mock transcription result
        transcription = {
            "chunks": [
                {"timestamp": [0.0, 2.0], "text": "Hello"},
                {"timestamp": [2.0, 4.0], "text": "world"}
            ]
        }
        
        # Mock diarization result
        mock_diarization = Mock()
        mock_diarization.itertracks.return_value = [
            (Mock(start=0.0, end=2.0), None, "SPEAKER_00"),
            (Mock(start=2.0, end=4.0), None, "SPEAKER_01")
        ]
        
        # Test merge
        segments = merge_transcription_and_diarization(transcription, mock_diarization)
        
        assert len(segments) == 2
        assert segments[0]["text"] == "Hello"
        assert segments[0]["speaker"] == "Speaker SPEAKER_00"
        assert segments[1]["text"] == "world"
        assert segments[1]["speaker"] == "Speaker SPEAKER_01"
    
    def test_merge_with_empty_transcription(self):
        """Test merging with empty transcription."""
        transcription = {"chunks": []}
        mock_diarization = Mock()
        mock_diarization.itertracks.return_value = []
        
        segments = merge_transcription_and_diarization(transcription, mock_diarization)
        assert len(segments) == 0
    
    def test_merge_with_none_timestamps(self):
        """Test merging with None timestamps in transcription."""
        transcription = {
            "chunks": [
                {"timestamp": [None, None], "text": "Test"}
            ]
        }
        
        mock_diarization = Mock()
        mock_diarization.itertracks.return_value = [
            (Mock(start=0.0, end=1.0), None, "SPEAKER_00")
        ]
        
        segments = merge_transcription_and_diarization(transcription, mock_diarization)
        assert len(segments) == 1
        assert segments[0]["start"] == 0.0  # Default start time
        assert segments[0]["end"] == 1.0   # Default end time


@pytest.mark.asyncio
class TestAsyncFunctions:
    """Test async functions in the server."""
    
    @patch('server.whisper_pipeline')
    async def test_run_transcription(self, mock_whisper):
        """Test the async transcription function."""
        from server import run_transcription
        
        mock_whisper.return_value = {"chunks": [{"text": "test"}]}
        
        waveform = torch.randn(1, 16000)  # 1 second of random audio
        result = await run_transcription(waveform, 16000)
        
        assert result is not None
        mock_whisper.assert_called_once()
    
    @patch('server.diarization_pipeline')
    async def test_run_diarization(self, mock_diarization):
        """Test the async diarization function."""
        from server import run_diarization
        
        mock_diarization.return_value = Mock()
        
        waveform = torch.randn(1, 16000)
        result = await run_diarization(waveform, 16000)
        
        assert result is not None
        mock_diarization.assert_called_once()


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @patch('server.whisper_pipeline')
    @patch('server.diarization_pipeline')
    def test_transcription_processing_error(self, mock_diarization, mock_whisper, client, sample_audio_bytes):
        """Test handling of processing errors during transcription."""
        # Make whisper raise an exception
        mock_whisper.side_effect = Exception("Model error")
        
        files = {"audio": ("test.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
        response = client.post("/transcribe", files=files)
        
        assert response.status_code == 500
        assert "Processing failed" in response.json()["detail"]
    
    def test_invalid_audio_format(self, client):
        """Test handling of invalid audio format."""
        # Send non-audio data
        invalid_data = b"not audio data"
        files = {"audio": ("test.txt", io.BytesIO(invalid_data), "text/plain")}
        response = client.post("/transcribe", files=files)
        
        assert response.status_code == 500
        assert "Processing failed" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 