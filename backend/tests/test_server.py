"""
Pytest tests for the Oreja FastAPI server.
Tests transcription, diarization, and audio processing endpoints.
"""

import _stub_heavy_deps  # noqa: F401 - installs sys.modules stubs before `import server`

import asyncio
import io
from types import SimpleNamespace
import pytest
import torch
import torchaudio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from server import app, initialize_models, load_audio_from_bytes, merge_transcription_and_diarization

# Real audio fixtures below are built with real torch/torchaudio calls
# (torch.linspace, torch.sin, torchaudio.save) - under the stub these are
# MagicMock calls that produce empty/garbage bytes, not decodable audio, so
# any test that uploads them needs the genuine packages. Skipped when heavy ML
# deps are stubbed; runs for real on a fully provisioned machine.
requires_real_torch = pytest.mark.skipif(
    _stub_heavy_deps.HEAVY_DEPS_STUBBED,
    reason="requires real torch/torchaudio audio (heavy ML deps are stubbed)",
)


def _fw_word(start, end, word, probability=0.9):
    """Stand-in for a faster_whisper word object (only attribute access is used)."""
    return SimpleNamespace(start=start, end=end, word=word, probability=probability)


def _fw_segment(start, end, text, words=None,
                avg_logprob=-0.2, no_speech_prob=0.01, compression_ratio=1.4):
    """Stand-in for a faster_whisper.transcribe.Segment."""
    return SimpleNamespace(
        start=start, end=end, text=text, words=words or [],
        avg_logprob=avg_logprob, no_speech_prob=no_speech_prob,
        compression_ratio=compression_ratio,
    )


def _fw_info(language="en", language_probability=0.99):
    """Stand-in for the faster_whisper TranscriptionInfo returned alongside segments."""
    return SimpleNamespace(language=language, language_probability=language_probability)


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
    
    @requires_real_torch
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
    
    @requires_real_torch
    @patch('server.whisper_model')
    @patch('server.diarization_pipeline')
    @patch('server.device', 'cpu')
    def test_transcribe_audio_success(self, mock_diarization, mock_whisper, client, sample_audio_bytes):
        """Test successful audio transcription."""
        # Setup mocks. whisper_model.transcribe() returns (segment_iterator, info) -
        # the faster-whisper contract _transcribe_sync actually calls (the old
        # `whisper_pipeline(...) -> {"chunks": [...]}` transformers-pipeline shape
        # this used to mock no longer exists).
        mock_whisper.transcribe.return_value = (
            iter([
                _fw_segment(0.0, 2.0, "Hello world", words=[
                    _fw_word(0.0, 1.0, "Hello"),
                    _fw_word(1.0, 2.0, " world"),
                ])
            ]),
            _fw_info(),
        )

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
        # server.py's segments carry "embedding_confidence" (float, 0.0 when no
        # voiceprint identification ran) - there is no plain "confidence" key.
        assert "embedding_confidence" in segment
    
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
    
    @requires_real_torch
    @patch('server.whisper_model', None)
    def test_transcribe_model_not_loaded(self, client, sample_audio_bytes):
        """Test transcription when Whisper model is not loaded."""
        files = {"audio": ("test.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
        response = client.post("/transcribe", files=files)
        assert response.status_code == 500


class TestEmbeddingEndpoint:
    """Test the speaker embedding extraction endpoint."""

    @requires_real_torch
    @patch('server.embedding_model')
    def test_extract_embeddings_success(self, mock_embedding, client, sample_audio_bytes):
        """Test successful embedding extraction."""
        # embedding_model is called directly as `embedding_model(batch)` in
        # extract_embedding_from_audio(), so the patched mock's return_value
        # (not the mock itself) stands in for that call's result.
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
        # require_embedding_model() now fails the request fast, before the
        # upload is even read - so this doesn't need real decodable audio, and
        # the response is 503 (service unavailable) rather than a generic 500.
        files = {"audio": ("test.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
        response = client.post("/extract_embeddings", files=files)
        assert response.status_code == 503
        assert "embedding model not loaded" in response.json()["detail"]


class TestAudioValidation:
    """Test audio validation logic."""
    
    @requires_real_torch
    @patch('server.whisper_model')
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

    @requires_real_torch
    @patch('server.whisper_model')
    @patch('server.diarization_pipeline')
    @patch('server.logger')
    def test_transcribe_audio_too_long(self, mock_logger, mock_diarization, mock_whisper, client):
        """
        Test transcription with a longer clip (should process normally).

        NOTE: MAX_AUDIO_LENGTH is 7200s (2 hours) now that faster-whisper
        streams long audio internally - the 2s clip generated below is nowhere
        near that limit. This exercises the normal (non-empty-chunks) success
        path rather than the max-length rejection.
        """
        # whisper_model.transcribe() returns (segment_iterator, info); an empty
        # iterator means "no speech survived", which is still a 200 with empty
        # segments - not a validation error.
        mock_whisper.transcribe.return_value = (iter([]), _fw_info())
        mock_diarization.return_value = Mock()
        mock_diarization.return_value.itertracks.return_value = []

        sample_rate = 16000
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
        """
        Test merging with None timestamps in transcription.

        This is the falsy-zero regression check: server.py's _chunk_bounds()
        and split_chunk_by_speaker() must tell a genuinely missing timestamp
        (None) apart from a legitimate 0.0 one using explicit `is not None`
        checks, never a bare truthiness check (which would treat 0.0 the same
        as None). With BOTH start and end missing, there is no timing
        information to recover at all - not even from the diarization turn,
        since a chunk is attributed to a turn by its own midpoint, not the
        other way around - so the current implementation defaults both to
        0.0 (a zero-length segment at t=0) rather than fabricating an
        arbitrary duration. See test_word_speaker_attribution.py for direct
        coverage of _chunk_bounds() preserving a real 0.0.
        """
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
        assert segments[0]["end"] == 0.0    # Default end time: no timestamp survives, not diarization's 1.0


@pytest.mark.asyncio
class TestAsyncFunctions:
    """Test async functions in the server."""
    
    @requires_real_torch
    @patch('server.whisper_model')
    async def test_run_transcription(self, mock_whisper):
        """Test the async transcription function."""
        from server import run_transcription

        # run_transcription() -> _transcribe_sync() calls whisper_model.transcribe(),
        # which returns (segment_iterator, info) - not the old transformers-style
        # `whisper_pipeline(waveform) -> {"chunks": [...]}` callable.
        mock_whisper.transcribe.return_value = (
            iter([_fw_segment(0.0, 1.0, "test", words=[_fw_word(0.0, 1.0, "test")])]),
            _fw_info(),
        )

        waveform = torch.randn(1, 16000)  # 1 second of random audio
        result = await run_transcription(waveform, 16000)

        assert result is not None
        mock_whisper.transcribe.assert_called_once()
    
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
    
    @requires_real_torch
    @patch('server.whisper_model')
    @patch('server.diarization_pipeline')
    def test_transcription_processing_error(self, mock_diarization, mock_whisper, client, sample_audio_bytes):
        """Test handling of processing errors during transcription."""
        # Make whisper raise an exception - whisper_model.transcribe() is what
        # _transcribe_sync() actually calls.
        mock_whisper.transcribe.side_effect = Exception("Model error")

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