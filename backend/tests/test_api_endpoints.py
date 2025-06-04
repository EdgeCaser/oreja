"""
Unit tests for FastAPI endpoints in server.py.
Tests all API routes, error handling, and integration points.
"""

import json
import pytest
import io
import tempfile
import torch
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Import server app with proper mocking
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock all heavy dependencies before importing server
def create_mock_models():
    """Create comprehensive mocks for all models and dependencies."""
    mock_whisper = Mock()
    mock_whisper.transcribe.return_value = {
        "text": "Test transcription",
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Test transcription"
            }
        ]
    }
    
    mock_diarization = Mock()
    mock_diarization.itertracks.return_value = [
        (Mock(start=0.0, end=2.0), None, "SPEAKER_00")
    ]
    
    mock_speaker_manager = Mock()
    mock_speaker_manager.get_speaker_stats.return_value = {
        "total_speakers": 0,
        "speakers": []
    }
    mock_speaker_manager.identify_or_create_speaker.return_value = ("sp_001", 0.9, False)
    mock_speaker_manager.speaker_profiles = {}
    
    mock_embedding_model = Mock()
    mock_embedding_model.return_value = torch.randn(512)
    
    return mock_whisper, mock_diarization, mock_speaker_manager, mock_embedding_model

# Create mocks
mock_whisper, mock_diarization, mock_speaker_manager, mock_embedding_model = create_mock_models()

# Mock imports before server import
with patch.dict('sys.modules', {
    'whisper': Mock(),
    'pyannote.audio': Mock(),
    'pyannote.audio.pipelines': Mock(),
    'pyannote.audio.pipelines.speaker_diarization': Mock(),
    'transformers': Mock(),
    'sentence_transformers': Mock(),
}):
    with patch('server.whisper_model', mock_whisper), \
         patch('server.diarization_pipeline', mock_diarization), \
         patch('server.speaker_embedding_manager', mock_speaker_manager), \
         patch('server.embedding_model', mock_embedding_model):
        
        from server import app

# Initialize test client
test_client = TestClient(app)

from conftest import assert_valid_transcription_response, assert_valid_speaker_response


@pytest.fixture
def client():
    """Test client fixture."""
    return test_client


@pytest.fixture
def mock_models():
    """Mock models fixture."""
    return {
        'whisper': mock_whisper,
        'diarization': mock_diarization,
        'speaker_manager': mock_speaker_manager,
        'embedding': mock_embedding_model
    }


class TestHealthEndpoints:
    """Test health check and status endpoints."""
    
    @pytest.mark.api
    def test_root_endpoint(self, client):
        """Test the root health check endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Oreja Audio Processing API"
        assert data["status"] == "running"
        assert "models_loaded" in data
    
    @pytest.mark.api
    def test_health_endpoint_detailed(self, client):
        """Test the detailed health check endpoint."""
        with patch('server.whisper_model', mock_whisper):
            with patch('server.diarization_pipeline', mock_diarization):
                with patch('server.embedding_model', mock_embedding_model):
                    response = client.get("/health")
                    assert response.status_code == 200
                    
                    data = response.json()
                    assert data["status"] == "healthy"
                    assert "device" in data
                    assert "models" in data
                    assert "memory_usage" in data


class TestTranscriptionEndpoint:
    """Test the transcription endpoint with various scenarios."""
    
    @pytest.mark.api
    def test_transcribe_success(self, client, sample_audio_bytes, mock_models):
        """Test successful transcription with both whisper and diarization."""
        files = {"audio": ("test.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
        
        with patch('server.load_audio_from_bytes') as mock_load_audio:
            mock_load_audio.return_value = (torch.randn(1, 16000), 16000)
            
            response = client.post("/transcribe", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert_valid_transcription_response(data)
            
            # Check specific content
            assert len(data["segments"]) > 0
            assert data["full_text"] == "Test transcription"
            assert data["processing_time"] >= 0
    
    @pytest.mark.api
    def test_transcribe_no_file(self, client):
        """Test transcription endpoint with no file."""
        response = client.post("/transcribe")
        
        assert response.status_code == 422  # Validation error
        assert "detail" in response.json()
    
    @pytest.mark.api
    def test_transcribe_empty_file(self, client):
        """Test transcription with empty file."""
        files = {"audio": ("empty.wav", io.BytesIO(b""), "audio/wav")}
        
        response = client.post("/transcribe", files=files)
        
        assert response.status_code == 400
        assert "Empty audio file" in response.json()["detail"]
    
    @pytest.mark.api
    def test_transcribe_invalid_audio_format(self, client):
        """Test transcription with invalid audio format."""
        invalid_data = b"This is not audio data"
        files = {"audio": ("invalid.wav", io.BytesIO(invalid_data), "audio/wav")}
        
        with patch('server.load_audio_from_bytes') as mock_load_audio:
            mock_load_audio.side_effect = ValueError("Invalid audio format")
            
            response = client.post("/transcribe", files=files)
            
            assert response.status_code == 400
            assert "Invalid audio format" in response.json()["detail"]
    
    @pytest.mark.api
    def test_transcribe_audio_too_short(self, client, sample_short_audio, mock_models):
        """Test transcription with audio that's too short."""
        files = {"audio": ("short.wav", io.BytesIO(sample_short_audio), "audio/wav")}
        
        with patch('server.load_audio_from_bytes') as mock_load_audio:
            mock_load_audio.return_value = (torch.randn(1, 1000), 16000)  # Very short audio
            
            response = client.post("/transcribe", files=files)
            
            assert response.status_code == 400
            assert "too short" in response.json()["detail"].lower()
    
    @pytest.mark.api
    def test_transcribe_audio_too_long(self, client, sample_long_audio, mock_models):
        """Test transcription with audio that's too long."""
        files = {"audio": ("long.wav", io.BytesIO(sample_long_audio), "audio/wav")}
        
        response = client.post("/transcribe", files=files)
        
        # Should either succeed with truncation or return error
        assert response.status_code in [200, 400]
        if response.status_code == 400:
            assert "too long" in response.json()["detail"].lower()
    
    @pytest.mark.api
    def test_transcribe_whisper_model_not_loaded(self, client, sample_audio_bytes):
        """Test transcription when Whisper model is not loaded."""
        with patch('server.whisper_model', None):
            files = {"audio": ("test.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
            
            response = client.post("/transcribe", files=files)
            
            assert response.status_code == 500
            assert "not loaded" in response.json()["detail"].lower()
    
    @pytest.mark.api
    def test_transcribe_model_processing_error(self, client, sample_audio_bytes):
        """Test transcription when model processing fails."""
        with patch('server.whisper_model') as mock_whisper:
            mock_whisper.side_effect = Exception("Model processing failed")
            
            files = {"audio": ("test.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
            
            response = client.post("/transcribe", files=files)
            
            assert response.status_code == 500
            assert "processing" in response.json()["detail"].lower()


class TestSpeakerEndpoints:
    """Test speaker-related endpoints."""
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_get_speakers_empty(self, client):
        """Test getting speakers when none exist."""
        with patch('server.speaker_embedding_manager', mock_speaker_manager):
            response = client.get("/speakers")
            
            assert response.status_code == 200
            data = response.json()
            assert_valid_speaker_response(data)
            assert data["total_speakers"] == 0
            assert len(data["speakers"]) == 0
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_get_speakers_with_data(self, client):
        """Test getting speakers when some exist."""
        mock_speakers = [
            {
                "speaker_id": "sp_001",
                "name": "Alice",
                "created_date": "2024-01-01T00:00:00",
                "embedding_count": 5
            },
            {
                "speaker_id": "sp_002", 
                "name": "Bob",
                "created_date": "2024-01-02T00:00:00",
                "embedding_count": 3
            }
        ]
        
        with patch('server.speaker_embedding_manager') as mock_manager:
            mock_manager.get_speaker_stats.return_value = {
                "total_speakers": 2,
                "speakers": mock_speakers
            }
            
            response = client.get("/speakers")
            
            assert response.status_code == 200
            data = response.json()
            assert_valid_speaker_response(data)
            assert data["total_speakers"] == 2
            assert len(data["speakers"]) == 2
            assert data["speakers"][0]["name"] == "Alice"
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_enroll_speaker_success(self, client, sample_audio_bytes):
        """Test successful speaker enrollment."""
        with patch('server.speaker_embedding_manager') as mock_manager:
            mock_manager.enroll_speaker.return_value = "new_speaker_001"
            
            files = {"audio": ("enrollment.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
            params = {"speaker_name": "New Speaker"}
            
            response = client.post("/speakers/enroll", files=files, params=params)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["speaker_id"] == "new_speaker_001"
            assert data["message"] == "Speaker enrolled successfully"
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_enroll_speaker_no_name(self, client, sample_audio_bytes):
        """Test speaker enrollment without providing name."""
        files = {"audio": ("enrollment.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
        
        response = client.post("/speakers/enroll", files=files)
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_enroll_speaker_no_audio(self, client):
        """Test speaker enrollment without audio file."""
        params = {"speaker_name": "Test Speaker"}
        
        response = client.post("/speakers/enroll", params=params)
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_enroll_speaker_enrollment_failed(self, client, sample_audio_bytes):
        """Test speaker enrollment when enrollment fails."""
        with patch('server.speaker_embedding_manager') as mock_manager:
            mock_manager.enroll_speaker.return_value = None  # Enrollment failed
            
            files = {"audio": ("enrollment.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
            params = {"speaker_name": "Failed Speaker"}
            
            response = client.post("/speakers/enroll", files=files, params=params)
            
            assert response.status_code == 400
            assert "failed" in response.json()["detail"].lower()
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_identify_speaker_success(self, client, sample_audio_bytes):
        """Test successful speaker identification."""
        with patch('server.speaker_embedding_manager') as mock_manager:
            mock_manager.identify_or_create_speaker.return_value = ("sp_001", 0.95, False)
            mock_manager.speaker_profiles = {
                "sp_001": Mock(speaker_id="sp_001", name="Alice")
            }
            
            files = {"audio": ("identify.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
            
            response = client.post("/speakers/identify", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert data["speaker_id"] == "sp_001"
            assert data["speaker_name"] == "Alice"
            assert data["confidence"] == 0.95
            assert data["is_new_speaker"] is False
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_identify_speaker_new_speaker(self, client, sample_audio_bytes):
        """Test speaker identification that creates new speaker."""
        with patch('server.speaker_embedding_manager') as mock_manager:
            mock_manager.identify_or_create_speaker.return_value = ("sp_new", 0.8, True)
            mock_manager.speaker_profiles = {
                "sp_new": Mock(speaker_id="sp_new", name="Unknown Speaker")
            }
            
            files = {"audio": ("identify.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
            
            response = client.post("/speakers/identify", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert data["speaker_id"] == "sp_new"
            assert data["is_new_speaker"] is True
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_update_speaker_name_success(self, client):
        """Test successful speaker name update."""
        with patch('server.speaker_embedding_manager') as mock_manager:
            mock_manager.update_speaker_name.return_value = True
            
            response = client.put("/speakers/sp_001/name?new_name=Updated Name")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["message"] == "Speaker name updated successfully"
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_update_speaker_name_not_found(self, client):
        """Test updating name of non-existent speaker."""
        with patch('server.speaker_embedding_manager') as mock_manager:
            mock_manager.update_speaker_name.return_value = False
            
            response = client.put("/speakers/nonexistent/name?new_name=New Name")
            
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_delete_speaker_success(self, client):
        """Test successful speaker deletion."""
        with patch('server.speaker_embedding_manager') as mock_manager:
            mock_manager.delete_speaker.return_value = True
            
            response = client.delete("/speakers/sp_001")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["message"] == "Speaker deleted successfully"
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_delete_speaker_not_found(self, client):
        """Test deleting non-existent speaker."""
        with patch('server.speaker_embedding_manager') as mock_manager:
            mock_manager.delete_speaker.return_value = False
            
            response = client.delete("/speakers/nonexistent")
            
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()


class TestSpeakerFeedbackEndpoints:
    """Test speaker feedback and correction endpoints."""
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_provide_speaker_feedback_success(self, client, sample_audio_bytes):
        """Test providing speaker correction feedback."""
        with patch('server.speaker_embedding_manager') as mock_manager:
            mock_manager.provide_correction_feedback.return_value = True
            
            files = {"audio": ("feedback.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
            data = {
                "correct_speaker_name": "Alice",
                "audio_segment_start": "0.0",
                "audio_segment_end": "2.0"
            }
            
            response = client.post("/speakers/feedback", files=files, data=data)
            
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["success"] is True
            assert "feedback" in response_data["message"].lower()
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_provide_speaker_feedback_missing_params(self, client, sample_audio_bytes):
        """Test speaker feedback with missing parameters."""
        files = {"audio": ("feedback.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
        # Missing required parameters
        
        response = client.post("/speakers/feedback", files=files)
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_provide_speaker_feedback_failed(self, client, sample_audio_bytes):
        """Test speaker feedback when processing fails."""
        with patch('server.speaker_embedding_manager') as mock_manager:
            mock_manager.provide_correction_feedback.return_value = False
            
            files = {"audio": ("feedback.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
            data = {
                "correct_speaker_name": "Alice",
                "audio_segment_start": "0.0",
                "audio_segment_end": "2.0"
            }
            
            response = client.post("/speakers/feedback", files=files, data=data)
            
            assert response.status_code == 400
            assert "failed" in response.json()["detail"].lower()
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_batch_speaker_feedback_success(self, client):
        """Test batch speaker feedback processing."""
        with patch('server.speaker_embedding_manager') as mock_manager:
            mock_manager.provide_correction_feedback.return_value = True
            
            feedback_data = {
                "feedback_items": [
                    {
                        "correct_speaker_name": "Alice",
                        "audio_segment_start": 0.0,
                        "audio_segment_end": 2.0,
                        "audio_base64": "base64encodedaudio1"
                    },
                    {
                        "correct_speaker_name": "Bob", 
                        "audio_segment_start": 2.0,
                        "audio_segment_end": 4.0,
                        "audio_base64": "base64encodedaudio2"
                    }
                ]
            }
            
            response = client.post("/speakers/batch_feedback", json=feedback_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["processed_count"] == 2
            assert data["failed_count"] == 0
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_update_speaker_name_mapping_success(self, client):
        """Test updating speaker name mapping."""
        with patch('server.speaker_embedding_manager') as mock_manager:
            mock_manager.get_speaker_by_name.return_value = "sp_002"
            mock_manager.update_speaker_name.return_value = True
            
            data = {
                "old_speaker_id": "sp_001",
                "new_speaker_name": "Updated Name"
            }
            
            response = client.post("/speakers/name_mapping", json=data)
            
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["success"] is True
            assert "mapping updated" in response_data["message"].lower()


class TestEmbeddingEndpoint:
    """Test speaker embedding extraction endpoint."""
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_extract_embeddings_success(self, client, sample_audio_bytes):
        """Test successful embedding extraction."""
        with patch('server.embedding_model', mock_embedding_model):
            files = {"audio": ("extract.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
            
            response = client.post("/extract_embeddings", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert "embedding" in data
            assert len(data["embedding"]) == 512
            assert data["success"] is True
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_extract_embeddings_model_not_loaded(self, client, sample_audio_bytes):
        """Test embedding extraction when model is not loaded."""
        with patch('server.embedding_model', None):
            files = {"audio": ("extract.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
            
            response = client.post("/extract_embeddings", files=files)
            
            assert response.status_code == 500
            assert "not loaded" in response.json()["detail"].lower()
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_extract_embeddings_processing_error(self, client, sample_audio_bytes):
        """Test embedding extraction when processing fails."""
        with patch('server.embedding_model') as mock_embedding:
            mock_embedding.side_effect = Exception("Embedding extraction failed")
            
            files = {"audio": ("extract.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
            
            response = client.post("/extract_embeddings", files=files)
            
            assert response.status_code == 500
            assert "error" in response.json()["detail"].lower()


class TestErrorHandling:
    """Test error handling across different scenarios."""
    
    @pytest.mark.api
    def test_invalid_audio_file_format(self, client):
        """Test handling of invalid audio file formats."""
        # Text file instead of audio
        invalid_file = io.BytesIO(b"This is not an audio file")
        files = {"audio": ("invalid.txt", invalid_file, "text/plain")}
        
        with patch('server.load_audio_from_bytes') as mock_load_audio:
            mock_load_audio.side_effect = ValueError("Invalid audio format")
            
            response = client.post("/transcribe", files=files)
            
            assert response.status_code == 400
            assert "invalid" in response.json()["detail"].lower()
    
    @pytest.mark.api
    def test_large_file_handling(self, client):
        """Test handling of very large files."""
        # Create a large fake audio file (>100MB)
        large_data = b"x" * (100 * 1024 * 1024)  # 100MB of data
        files = {"audio": ("large.wav", io.BytesIO(large_data), "audio/wav")}
        
        response = client.post("/transcribe", files=files)
        
        # Should either reject or handle gracefully
        assert response.status_code in [400, 413, 500]
    
    @pytest.mark.api
    def test_concurrent_requests_handling(self, client, sample_audio_bytes, mock_models):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            files = {"audio": ("test.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
            response = client.post("/transcribe", files=files)
            results.append(response.status_code)
        
        # Make 5 concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should complete successfully
        assert len(results) == 5
        for status_code in results:
            assert status_code == 200
    
    @pytest.mark.api
    def test_malformed_json_requests(self, client):
        """Test handling of malformed JSON in POST requests."""
        # Invalid JSON for batch feedback
        response = client.post(
            "/speakers/batch_feedback",
            data="{ invalid json }",
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.api
    def test_missing_required_headers(self, client, sample_audio_bytes):
        """Test handling requests with missing required headers."""
        # Send audio without proper content-type
        response = client.post(
            "/transcribe",
            data=sample_audio_bytes,
            headers={"content-type": "text/plain"}
        )
        
        assert response.status_code in [400, 422]


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests that combine multiple endpoints."""
    
    def test_complete_speaker_workflow(self, client, sample_audio_bytes):
        """Test complete workflow: enroll -> identify -> update -> delete."""
        with patch('server.speaker_embedding_manager') as mock_manager:
            # Setup mock responses for each step
            mock_manager.enroll_speaker.return_value = "sp_test"
            mock_manager.identify_or_create_speaker.return_value = ("sp_test", 0.95, False)
            mock_manager.speaker_profiles = {
                "sp_test": Mock(speaker_id="sp_test", name="Test Speaker")
            }
            mock_manager.update_speaker_name.return_value = True
            mock_manager.delete_speaker.return_value = True
            
            # 1. Enroll speaker
            files = {"audio": ("enroll.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
            params = {"speaker_name": "Test Speaker"}
            response = client.post("/speakers/enroll", files=files, params=params)
            assert response.status_code == 200
            
            # 2. Identify speaker
            files = {"audio": ("identify.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
            response = client.post("/speakers/identify", files=files)
            assert response.status_code == 200
            assert response.json()["speaker_id"] == "sp_test"
            
            # 3. Update speaker name
            response = client.put("/speakers/sp_test/name?new_name=Updated Speaker")
            assert response.status_code == 200
            
            # 4. Delete speaker
            response = client.delete("/speakers/sp_test")
            assert response.status_code == 200
    
    def test_transcription_with_speaker_identification(self, client, sample_audio_bytes, mock_models):
        """Test transcription that includes speaker identification."""
        with patch('server.speaker_embedding_manager') as mock_manager:
            mock_manager.identify_or_create_speaker.return_value = ("sp_001", 0.9, False)
            mock_manager.speaker_profiles = {
                "sp_001": Mock(speaker_id="sp_001", name="Alice")
            }
            
            with patch('server.load_audio_from_bytes') as mock_load_audio:
                mock_load_audio.return_value = (torch.randn(1, 16000), 16000)
                
                files = {"audio": ("transcribe.wav", io.BytesIO(sample_audio_bytes), "audio/wav")}
                response = client.post("/transcribe", files=files)
                
                assert response.status_code == 200
                data = response.json()
                assert_valid_transcription_response(data)
                
                # Should include speaker information in segments
                assert len(data["segments"]) > 0
                segment = data["segments"][0]
                assert "speaker" in segment 