"""
Pytest configuration and shared fixtures for Oreja tests.
"""

import asyncio
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock

import pytest
import torch
import torchaudio
import numpy as np
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Import the main application components
from server import app, initialize_models
from speaker_embeddings import OfflineSpeakerEmbeddingManager, SpeakerProfile


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create an async test client for the FastAPI app."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing."""
    sample_rate = 16000
    duration = 2.0  # seconds
    frequency = 440.0  # A4 note
    
    # Generate sine wave
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
    
    return waveform, sample_rate


@pytest.fixture
def sample_audio_bytes(sample_audio_data):
    """Convert sample audio to bytes format."""
    waveform, sample_rate = sample_audio_data
    
    buffer = io.BytesIO()
    torchaudio.save(buffer, waveform, sample_rate, format="wav")
    buffer.seek(0)
    
    return buffer.getvalue()


@pytest.fixture
def sample_short_audio():
    """Generate short audio sample (under minimum length)."""
    sample_rate = 16000
    duration = 0.05  # 50ms - very short
    frequency = 440.0
    
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
    
    buffer = io.BytesIO()
    torchaudio.save(buffer, waveform, sample_rate, format="wav")
    buffer.seek(0)
    
    return buffer.getvalue()


@pytest.fixture
def sample_long_audio():
    """Generate long audio sample (over maximum length)."""
    sample_rate = 16000
    duration = 35.0  # 35 seconds - over limit
    frequency = 440.0
    
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
    
    buffer = io.BytesIO()
    torchaudio.save(buffer, waveform, sample_rate, format="wav")
    buffer.seek(0)
    
    return buffer.getvalue()


@pytest.fixture
def mock_whisper_response():
    """Mock Whisper model response."""
    return {
        "chunks": [
            {
                "timestamp": [0.0, 2.0],
                "text": "Hello world, this is a test."
            },
            {
                "timestamp": [2.0, 4.0],
                "text": "How are you doing today?"
            }
        ]
    }


@pytest.fixture
def mock_diarization_response():
    """Mock diarization pipeline response."""
    mock_diarization = Mock()
    mock_diarization.itertracks.return_value = [
        (Mock(start=0.0, end=2.0), None, "SPEAKER_00"),
        (Mock(start=2.0, end=4.0), None, "SPEAKER_01")
    ]
    return mock_diarization


@pytest.fixture
def mock_embedding():
    """Mock speaker embedding."""
    return torch.randn(512).numpy()


@pytest.fixture
def temporary_speaker_data():
    """Create temporary directory for speaker data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_speaker_profile():
    """Create a sample speaker profile for testing."""
    profile = SpeakerProfile("test_speaker_001", "Test Speaker")
    profile.add_embedding(np.random.randn(512), confidence=0.9)
    profile.add_embedding(np.random.randn(512), confidence=0.8)
    return profile


@pytest.fixture
def mock_models():
    """Mock all AI models for faster testing."""
    with patch('server.whisper_model') as mock_whisper, \
         patch('server.diarization_pipeline') as mock_diarization, \
         patch('server.embedding_model') as mock_embedding, \
         patch('server.device', 'cpu'):
        
        # Configure mock responses
        mock_whisper.return_value = {
            "chunks": [
                {"timestamp": [0.0, 2.0], "text": "Test transcription"}
            ]
        }
        
        mock_diarization_result = Mock()
        mock_diarization_result.itertracks.return_value = [
            (Mock(start=0.0, end=2.0), None, "SPEAKER_00")
        ]
        mock_diarization.return_value = mock_diarization_result
        
        mock_embedding.return_value = torch.randn(512)
        
        yield {
            'whisper': mock_whisper,
            'diarization': mock_diarization,
            'embedding': mock_embedding
        }


@pytest.fixture
def speaker_embedding_manager(temporary_speaker_data):
    """Create a speaker embedding manager with temporary data directory."""
    with patch('speaker_embeddings.SPEECHBRAIN_AVAILABLE', True):
        manager = OfflineSpeakerEmbeddingManager(data_dir=temporary_speaker_data)
        # Mock the embedding model to avoid loading actual model in tests
        manager.embedding_model = Mock()
        manager.embedding_model.encode_batch.return_value = torch.randn(1, 512)
        yield manager


@pytest.fixture
def api_response_schemas():
    """Expected API response schemas for validation."""
    return {
        "transcribe": {
            "segments": list,
            "full_text": str,
            "processing_time": float,
            "timestamp": str
        },
        "health": {
            "status": str,
            "device": str,
            "models": dict,
            "memory_usage": dict
        },
        "speakers": {
            "total_speakers": int,
            "speakers": list
        }
    }


class MockAudioFile:
    """Mock audio file for testing file uploads."""
    
    def __init__(self, filename: str, content: bytes, content_type: str = "audio/wav"):
        self.filename = filename
        self.content = content
        self.content_type = content_type
        self.file = io.BytesIO(content)
    
    def read(self):
        return self.content
    
    def seek(self, position):
        self.file.seek(position)


@pytest.fixture
def mock_audio_file(sample_audio_bytes):
    """Create a mock audio file for upload testing."""
    return MockAudioFile("test_audio.wav", sample_audio_bytes)


# Utility functions for tests
def assert_valid_transcription_response(response_data: Dict[str, Any]):
    """Assert that a transcription response has the expected structure."""
    required_fields = ["segments", "full_text", "processing_time", "timestamp"]
    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"
    
    assert isinstance(response_data["segments"], list)
    assert isinstance(response_data["full_text"], str)
    assert isinstance(response_data["processing_time"], (int, float))
    assert isinstance(response_data["timestamp"], str)
    
    # Validate segment structure
    for segment in response_data["segments"]:
        segment_fields = ["start", "end", "text", "speaker", "confidence"]
        for field in segment_fields:
            assert field in segment, f"Missing segment field: {field}"


def assert_valid_speaker_response(response_data: Dict[str, Any]):
    """Assert that a speaker response has the expected structure."""
    required_fields = ["total_speakers", "speakers"]
    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"
    
    assert isinstance(response_data["total_speakers"], int)
    assert isinstance(response_data["speakers"], list) 