"""
Unit tests for Enhanced Speaker API endpoints in server.py.
Tests the new enhanced speaker database integration endpoints.
"""

import json
import pytest
import io
import tempfile
import shutil
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock dependencies before importing server
def create_enhanced_mocks():
    """Create enhanced mocks for the new speaker system."""
    mock_enhanced_db = Mock()
    
    # Mock speaker records
    mock_enhanced_db.get_all_speakers.return_value = [
        {
            'speaker_id': 'spk_123abc456def',
            'display_name': 'Test Speaker',
            'embedding_count': 5,
            'last_seen': '2024-01-01T00:00:00',
            'average_confidence': 0.85,
            'is_enrolled': True,
            'is_verified': False,
            'source_type': 'enrolled',
            'created_date': '2024-01-01T00:00:00'
        }
    ]
    
    mock_enhanced_db.create_speaker.return_value = 'spk_new123abc456'
    mock_enhanced_db.find_speaker_by_name.return_value = 'spk_123abc456def'
    mock_enhanced_db.update_display_name.return_value = True
    mock_enhanced_db.delete_speaker.return_value = True
    mock_enhanced_db.merge_speakers.return_value = True
    mock_enhanced_db.send_feedback_for_learning.return_value = {
        "status": "success",
        "corrections_processed": 2,
        "speakers_updated": 1
    }
    mock_enhanced_db.save_transcription_with_corrections.return_value = "/path/to/output.json"
    
    return mock_enhanced_db

# Create mock
mock_enhanced_db = create_enhanced_mocks()

# Mock the server imports
with patch.dict('sys.modules', {
    'whisper': Mock(),
    'pyannote.audio': Mock(),
    'transformers': Mock(),
}):
    with patch('server.enhanced_speaker_db', mock_enhanced_db):
        from server import app

# Initialize test client
test_client = TestClient(app)


@pytest.fixture
def client():
    """Test client fixture."""
    return test_client


@pytest.fixture
def sample_audio_bytes():
    """Sample audio data for testing."""
    # Create a simple sine wave WAV file in memory
    import wave
    import struct
    
    sample_rate = 16000
    duration = 2.0  # seconds
    frequency = 440  # Hz
    
    frames = int(sample_rate * duration)
    audio_data = []
    
    for i in range(frames):
        value = int(32767 * np.sin(2 * np.pi * frequency * i / sample_rate))
        audio_data.append(struct.pack('<h', value))
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b''.join(audio_data))
    
    wav_buffer.seek(0)
    return wav_buffer.read()


class TestEnhancedSpeakerEndpoints:
    """Test enhanced speaker management endpoints."""
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_get_enhanced_speakers_stats(self, client):
        """Test getting enhanced speaker statistics."""
        with patch('server.enhanced_speaker_db', mock_enhanced_db):
            response = client.get("/speakers/enhanced_stats")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "total_speakers" in data
            assert "speakers" in data
            assert isinstance(data["speakers"], list)
            
            # Verify speaker data structure
            if data["speakers"]:
                speaker = data["speakers"][0]
                required_fields = [
                    'speaker_id', 'display_name', 'embedding_count',
                    'last_seen', 'average_confidence', 'is_enrolled',
                    'is_verified', 'source_type', 'created_date'
                ]
                for field in required_fields:
                    assert field in speaker
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_enhanced_speaker_feedback_success(self, client):
        """Test providing enhanced speaker feedback."""
        feedback_data = {
            "speaker_corrections": {
                "segment_1": "Alice Johnson",
                "segment_2": "Bob Smith"
            },
            "audio_segments": [
                {"segment_id": "segment_1", "start_time": 0.0, "end_time": 2.0},
                {"segment_id": "segment_2", "start_time": 2.0, "end_time": 4.0}
            ]
        }
        
        with patch('server.enhanced_speaker_db', mock_enhanced_db):
            response = client.post(
                "/speakers/enhanced_feedback",
                json=feedback_data
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "success"
            assert "corrections_processed" in data
            assert "speakers_updated" in data
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_enhanced_speaker_feedback_missing_data(self, client):
        """Test enhanced feedback with missing required data."""
        incomplete_data = {
            "speaker_corrections": {
                "segment_1": "Alice Johnson"
            }
            # Missing audio_segments
        }
        
        response = client.post(
            "/speakers/enhanced_feedback",
            json=incomplete_data
        )
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_save_transcription_enhanced_success(self, client):
        """Test saving transcription with enhanced speaker corrections."""
        transcription_data = {
            "transcription_data": {
                "segments": [
                    {
                        "id": "seg_1",
                        "text": "Hello world",
                        "start": 0.0,
                        "end": 2.0,
                        "speaker": "SPEAKER_00"
                    }
                ],
                "full_text": "Hello world"
            },
            "speaker_corrections": {
                "SPEAKER_00": "Alice Johnson"
            },
            "output_file": "test_output.json"
        }
        
        with patch('server.enhanced_speaker_db', mock_enhanced_db):
            response = client.post(
                "/speakers/save_transcription_enhanced",
                json=transcription_data
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "success"
            assert "output_file" in data
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_save_transcription_enhanced_missing_data(self, client):
        """Test saving transcription with missing required data."""
        incomplete_data = {
            "transcription_data": {
                "segments": [],
                "full_text": ""
            }
            # Missing speaker_corrections
        }
        
        response = client.post(
            "/speakers/save_transcription_enhanced",
            json=incomplete_data
        )
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_speaker_system_status(self, client):
        """Test getting speaker system status."""
        with patch('server.enhanced_speaker_db', mock_enhanced_db):
            with patch('server.speaker_embedding_manager') as mock_legacy:
                mock_legacy.get_speaker_stats.return_value = {
                    "total_speakers": 3,
                    "speakers": []
                }
                
                response = client.get("/speakers/system_status")
                
                assert response.status_code == 200
                data = response.json()
                
                assert "enhanced_system" in data
                assert "legacy_system" in data
                assert "migration_status" in data
                
                # Check enhanced system data
                enhanced = data["enhanced_system"]
                assert "available" in enhanced
                assert "speaker_count" in enhanced
                
                # Check legacy system data
                legacy = data["legacy_system"]
                assert "available" in legacy
                assert "speaker_count" in legacy


class TestEnhancedSpeakerMerging:
    """Test enhanced speaker merging functionality."""
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_merge_speakers_enhanced_success(self, client):
        """Test successful speaker merging."""
        merge_data = {
            "source_speaker_id": "spk_source123",
            "target_speaker_id": "spk_target456",
            "final_name": "Merged Speaker"
        }
        
        with patch('server.enhanced_speaker_db', mock_enhanced_db):
            response = client.post("/speakers/merge", json=merge_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "success"
            assert "target_speaker_id" in data
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_merge_speakers_enhanced_missing_data(self, client):
        """Test speaker merging with missing data."""
        incomplete_data = {
            "source_speaker_id": "spk_source123"
            # Missing target_speaker_id and final_name
        }
        
        response = client.post("/speakers/merge", json=incomplete_data)
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_merge_speakers_enhanced_failure(self, client):
        """Test speaker merging failure."""
        merge_data = {
            "source_speaker_id": "nonexistent_source",
            "target_speaker_id": "nonexistent_target",
            "final_name": "Merged Speaker"
        }
        
        enhanced_db_mock = Mock()
        enhanced_db_mock.merge_speakers.return_value = False
        
        with patch('server.enhanced_speaker_db', enhanced_db_mock):
            response = client.post("/speakers/merge", json=merge_data)
            
            assert response.status_code == 400
            assert "Failed to merge speakers" in response.json()["detail"]


class TestEnhancedSpeakerMigration:
    """Test migration between legacy and enhanced systems."""
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_migrate_to_enhanced_success(self, client):
        """Test successful migration to enhanced system."""
        migration_data = {
            "preserve_legacy": True,
            "backup_legacy": True
        }
        
        # Mock the migration process
        with patch('server.enhanced_speaker_db', mock_enhanced_db):
            with patch('server.speaker_embedding_manager') as mock_legacy:
                mock_legacy.get_speaker_stats.return_value = {
                    "total_speakers": 2,
                    "speakers": [
                        {"speaker_id": "legacy_1", "name": "Legacy Speaker 1"},
                        {"speaker_id": "legacy_2", "name": "Legacy Speaker 2"}
                    ]
                }
                
                with patch('server.migrate_legacy_to_enhanced') as mock_migrate:
                    mock_migrate.return_value = {
                        "status": "success",
                        "migrated_speakers": 2,
                        "enhanced_speaker_ids": ["spk_new1", "spk_new2"]
                    }
                    
                    response = client.post(
                        "/speakers/migrate_to_enhanced",
                        json=migration_data
                    )
                    
                    assert response.status_code == 200
                    data = response.json()
                    
                    assert data["status"] == "success"
                    assert "migrated_speakers" in data
                    assert "enhanced_speaker_ids" in data
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_migrate_to_enhanced_no_legacy_data(self, client):
        """Test migration when no legacy data exists."""
        migration_data = {
            "preserve_legacy": False,
            "backup_legacy": True
        }
        
        with patch('server.enhanced_speaker_db', mock_enhanced_db):
            with patch('server.speaker_embedding_manager') as mock_legacy:
                mock_legacy.get_speaker_stats.return_value = {
                    "total_speakers": 0,
                    "speakers": []
                }
                
                response = client.post(
                    "/speakers/migrate_to_enhanced",
                    json=migration_data
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["status"] == "success"
                assert data["migrated_speakers"] == 0


@pytest.mark.integration
class TestEnhancedSpeakerIntegration:
    """Integration tests for enhanced speaker functionality."""
    
    def test_complete_enhanced_workflow(self, client, sample_audio_bytes):
        """Test complete enhanced speaker workflow."""
        with patch('server.enhanced_speaker_db', mock_enhanced_db):
            # 1. Get initial system status
            status_response = client.get("/speakers/system_status")
            assert status_response.status_code == 200
            
            # 2. Get enhanced speaker stats
            stats_response = client.get("/speakers/enhanced_stats")
            assert stats_response.status_code == 200
            
            # 3. Provide speaker feedback
            feedback_data = {
                "speaker_corrections": {
                    "segment_1": "Alice Johnson"
                },
                "audio_segments": [
                    {"segment_id": "segment_1", "start_time": 0.0, "end_time": 2.0}
                ]
            }
            
            feedback_response = client.post(
                "/speakers/enhanced_feedback",
                json=feedback_data
            )
            assert feedback_response.status_code == 200
            
            # 4. Save transcription with corrections
            transcription_data = {
                "transcription_data": {
                    "segments": [
                        {
                            "id": "seg_1",
                            "text": "Hello world",
                            "start": 0.0,
                            "end": 2.0,
                            "speaker": "SPEAKER_00"
                        }
                    ],
                    "full_text": "Hello world"
                },
                "speaker_corrections": {
                    "SPEAKER_00": "Alice Johnson"
                }
            }
            
            save_response = client.post(
                "/speakers/save_transcription_enhanced",
                json=transcription_data
            )
            assert save_response.status_code == 200
    
    def test_enhanced_error_handling(self, client):
        """Test enhanced speaker system error handling."""
        # Test with database error
        enhanced_db_mock = Mock()
        enhanced_db_mock.get_all_speakers.side_effect = Exception("Database error")
        
        with patch('server.enhanced_speaker_db', enhanced_db_mock):
            response = client.get("/speakers/enhanced_stats")
            assert response.status_code == 500
            assert "Database error" in response.json()["detail"]
    
    def test_enhanced_concurrent_operations(self, client):
        """Test concurrent operations on enhanced speaker system."""
        import threading
        import time
        
        results = []
        
        def make_request():
            with patch('server.enhanced_speaker_db', mock_enhanced_db):
                response = client.get("/speakers/enhanced_stats")
                results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5


class TestEnhancedSpeakerValidation:
    """Test input validation for enhanced speaker endpoints."""
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_enhanced_feedback_validation(self, client):
        """Test input validation for enhanced feedback."""
        # Test with invalid speaker corrections format
        invalid_data = {
            "speaker_corrections": "invalid_format",  # Should be dict
            "audio_segments": []
        }
        
        response = client.post(
            "/speakers/enhanced_feedback",
            json=invalid_data
        )
        assert response.status_code == 422
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_save_transcription_validation(self, client):
        """Test input validation for save transcription."""
        # Test with invalid transcription data
        invalid_data = {
            "transcription_data": "invalid_format",  # Should be dict
            "speaker_corrections": {}
        }
        
        response = client.post(
            "/speakers/save_transcription_enhanced",
            json=invalid_data
        )
        assert response.status_code == 422
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_merge_speakers_validation(self, client):
        """Test input validation for speaker merging."""
        # Test with missing required fields
        invalid_data = {
            "source_speaker_id": "spk_123"
            # Missing target_speaker_id and final_name
        }
        
        response = client.post("/speakers/merge", json=invalid_data)
        assert response.status_code == 422
    
    @pytest.mark.api
    @pytest.mark.speaker
    def test_migration_validation(self, client):
        """Test input validation for migration."""
        # Test with invalid boolean values
        invalid_data = {
            "preserve_legacy": "not_a_boolean",
            "backup_legacy": True
        }
        
        response = client.post(
            "/speakers/migrate_to_enhanced",
            json=invalid_data
        )
        assert response.status_code == 422 