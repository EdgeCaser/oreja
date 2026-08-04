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
import numpy as np
from fastapi.testclient import TestClient
from httpx import AsyncClient

# torch/torchaudio and the "server" module (which imports torch at module
# scope for its waveform handling) are real hard dependencies of this whole
# suite - on a fully provisioned dev/CI machine (see requirements.txt) both
# imports succeed exactly as before. Guarded here only so that dependency-light
# tests elsewhere in this directory (pure-Python/numpy - no torch, no models)
# can still be collected and run on a machine where the heavy ML wheels are not
# installed; any test that actually needs torch/app is skipped via
# pytest.importorskip inside the fixture that builds it, not silently faked.
try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    torchaudio = None
    TORCH_AVAILABLE = False

try:
    from server import app, initialize_models
    SERVER_IMPORTABLE = True
except Exception:
    app = None
    initialize_models = None
    SERVER_IMPORTABLE = False

# NOTE: speaker_embeddings.py (the legacy SpeechBrain OfflineSpeakerEmbeddingManager)
# was removed - there is one embedding system now (server.py's pyannote
# PretrainedSpeakerEmbedding + speaker_database_v2.EnhancedSpeakerDatabase).
# The fixtures that used to build on OfflineSpeakerEmbeddingManager/SpeakerProfile
# were removed below along with this import.


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    if not SERVER_IMPORTABLE:
        pytest.skip("server module could not be imported (torch/heavy ML deps not installed)")
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create an async test client for the FastAPI app."""
    if not SERVER_IMPORTABLE:
        pytest.skip("server module could not be imported (torch/heavy ML deps not installed)")
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing."""
    pytest.importorskip("torch")
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
    pytest.importorskip("torch")
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
    """
    Generate a long audio sample.

    NOTE: this is no longer "over the maximum". MAX_AUDIO_LENGTH is 7200s
    (2 hours) now that faster-whisper streams long audio internally, so
    generating a genuinely over-limit buffer here would cost gigabytes. This
    fixture just exercises the multi-segment / long-buffer path.
    """
    pytest.importorskip("torch")
    sample_rate = 16000
    duration = 35.0  # 35 seconds - long, but well within MAX_AUDIO_LENGTH
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
    """Mock speaker embedding (plain numpy - no torch needed to build this one)."""
    return np.random.randn(512).astype(np.float32)


@pytest.fixture
def temporary_speaker_data():
    """Create temporary directory for speaker data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


# NOTE: the old `sample_speaker_profile` fixture built a legacy
# speaker_embeddings.SpeakerProfile, which no longer exists - speaker identity
# is now speaker_database_v2.SpeakerRecord + raw numpy embedding vectors
# (see test_speaker_database_v2.py for coverage of the current API).


@pytest.fixture
def mock_models():
    """Mock all AI models for faster testing."""
    if not SERVER_IMPORTABLE:
        pytest.skip("server module could not be imported (torch/heavy ML deps not installed)")
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
def api_response_schemas():
    """Expected API response schemas for validation."""
    return {
        "transcribe": {
            "segments": list,
            "full_text": str,
            "processing_time": float,
            # server.py sets "timestamp": time.time() - a float, not a string.
            "timestamp": float
        },
        "health": {
            "status": str,
            "device": str,
            "models": dict,
            # torch.cuda.memory_allocated() (int) on CUDA, the string "N/A" otherwise.
            "memory_usage": (int, str)
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
    """
    Assert that a POST /transcribe response has the expected structure.

    This is the contract the shipped WPF client reads: segments[] each with
    start/end/text/speaker, plus full_text. `timestamp` is a float
    (time.time()), and segments carry `embedding_confidence` - there is no
    plain `confidence` key.
    """
    required_fields = ["segments", "full_text", "processing_time", "timestamp"]
    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"

    assert isinstance(response_data["segments"], list)
    assert isinstance(response_data["full_text"], str)
    assert isinstance(response_data["processing_time"], (int, float))
    assert isinstance(response_data["timestamp"], (int, float))

    # Validate segment structure
    for segment in response_data["segments"]:
        segment_fields = ["start", "end", "text", "speaker"]
        for field in segment_fields:
            assert field in segment, f"Missing segment field: {field}"
        assert isinstance(segment["start"], (int, float))
        assert isinstance(segment["end"], (int, float))
        assert isinstance(segment["text"], str)
        assert isinstance(segment["speaker"], str)


def assert_valid_speaker_response(response_data: Dict[str, Any]):
    """
    Assert that a GET /speakers response has the expected structure.

    The shipped WPF client reads speakers[].id / .name / .embedding_count
    (as an int), so those are asserted per entry.
    """
    required_fields = ["total_speakers", "speakers"]
    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"

    assert isinstance(response_data["total_speakers"], int)
    assert isinstance(response_data["speakers"], list)

    for speaker in response_data["speakers"]:
        for field in ("id", "name", "embedding_count"):
            assert field in speaker, f"Missing speaker field: {field}"
        assert isinstance(speaker["id"], str)
        assert isinstance(speaker["name"], str)
        assert isinstance(speaker["embedding_count"], int)
