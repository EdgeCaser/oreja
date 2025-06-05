"""
Unit tests for the EnhancedSpeakerDatabase class.
Tests enhanced speaker management with immutable IDs, proper merging, and confidence tracking.
"""

import json
import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from speaker_database_v2 import EnhancedSpeakerDatabase, SpeakerRecord


class TestSpeakerRecord:
    """Test the SpeakerRecord dataclass."""
    
    @pytest.mark.unit
    def test_speaker_record_creation(self):
        """Test basic speaker record creation."""
        record = SpeakerRecord(
            speaker_id="spk_123abc456def",
            created_date="2024-01-01T00:00:00",
            display_name="John Doe",
            last_seen="2024-01-01T00:00:00"
        )
        
        assert record.speaker_id == "spk_123abc456def"
        assert record.display_name == "John Doe"
        assert record.created_date == "2024-01-01T00:00:00"
        assert record.last_seen == "2024-01-01T00:00:00"
        assert record.session_count == 0
        assert record.total_audio_seconds == 0.0
        assert record.embedding_count == 0
        assert record.average_confidence == 0.0
        assert record.is_enrolled == False
        assert record.is_verified == False
        assert record.source_type == "auto"
    
    @pytest.mark.unit
    def test_speaker_record_update_stats(self):
        """Test updating speaker statistics."""
        record = SpeakerRecord(
            speaker_id="spk_123abc456def",
            created_date="2024-01-01T00:00:00",
            display_name="John Doe",
            last_seen="2024-01-01T00:00:00"
        )
        
        embeddings = [np.random.rand(512) for _ in range(3)]
        confidences = [0.85, 0.92, 0.78]
        
        record.update_stats(embeddings, confidences)
        
        assert record.embedding_count == 3
        assert record.average_confidence == np.mean(confidences)
        assert record.last_seen is not None
        # Verify last_seen is a valid ISO timestamp
        datetime.fromisoformat(record.last_seen)


class TestEnhancedSpeakerDatabase:
    """Test the EnhancedSpeakerDatabase class."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def enhanced_db(self, temp_data_dir):
        """Create enhanced speaker database for testing."""
        return EnhancedSpeakerDatabase(data_dir=temp_data_dir)
    
    @pytest.mark.unit
    def test_database_initialization(self, temp_data_dir):
        """Test database initialization."""
        db = EnhancedSpeakerDatabase(data_dir=temp_data_dir)
        
        assert db.data_dir == Path(temp_data_dir)
        assert db.data_dir.exists()
        assert db.embeddings_dir.exists()
        assert len(db.speaker_records) == 0
        assert len(db.speaker_embeddings) == 0
        assert len(db.confidence_scores) == 0
        assert len(db.name_to_id_index) == 0
    
    @pytest.mark.unit
    def test_generate_speaker_id(self, enhanced_db):
        """Test speaker ID generation."""
        speaker_id = enhanced_db._generate_speaker_id()
        
        assert speaker_id.startswith("spk_")
        assert len(speaker_id) == 16  # "spk_" + 12 hex chars
        
        # Generate multiple IDs to ensure uniqueness
        ids = [enhanced_db._generate_speaker_id() for _ in range(10)]
        assert len(set(ids)) == 10  # All unique
    
    @pytest.mark.unit
    def test_create_speaker(self, enhanced_db):
        """Test creating a new speaker."""
        speaker_id = enhanced_db.create_speaker(
            display_name="Alice Johnson",
            source_type="enrolled",
            is_enrolled=True
        )
        
        assert speaker_id.startswith("spk_")
        assert speaker_id in enhanced_db.speaker_records
        
        record = enhanced_db.speaker_records[speaker_id]
        assert record.speaker_id == speaker_id
        assert record.display_name == "Alice Johnson"
        assert record.source_type == "enrolled"
        assert record.is_enrolled == True
        assert record.is_verified == False
        
        # Check indexing
        assert enhanced_db.name_to_id_index["alice johnson"] == speaker_id
    
    @pytest.mark.unit
    def test_update_display_name(self, enhanced_db):
        """Test updating speaker display name."""
        speaker_id = enhanced_db.create_speaker("John Doe")
        
        # Update name
        success = enhanced_db.update_display_name(speaker_id, "John Smith")
        
        assert success == True
        record = enhanced_db.speaker_records[speaker_id]
        assert record.display_name == "John Smith"
        
        # Check index update
        assert "john doe" not in enhanced_db.name_to_id_index
        assert enhanced_db.name_to_id_index["john smith"] == speaker_id
    
    @pytest.mark.unit
    def test_update_display_name_nonexistent(self, enhanced_db):
        """Test updating name for nonexistent speaker."""
        success = enhanced_db.update_display_name("nonexistent_id", "New Name")
        assert success == False
    
    @pytest.mark.unit
    def test_find_speaker_by_name(self, enhanced_db):
        """Test finding speaker by display name."""
        speaker_id = enhanced_db.create_speaker("Bob Wilson")
        
        # Test exact match (case insensitive)
        found_id = enhanced_db.find_speaker_by_name("Bob Wilson")
        assert found_id == speaker_id
        
        found_id = enhanced_db.find_speaker_by_name("bob wilson")
        assert found_id == speaker_id
        
        found_id = enhanced_db.find_speaker_by_name("BOB WILSON")
        assert found_id == speaker_id
        
        # Test not found
        found_id = enhanced_db.find_speaker_by_name("Nonexistent Speaker")
        assert found_id is None
    
    @pytest.mark.unit
    def test_add_embedding(self, enhanced_db):
        """Test adding embeddings to a speaker."""
        speaker_id = enhanced_db.create_speaker("Test Speaker")
        
        embedding1 = np.random.rand(512)
        embedding2 = np.random.rand(512)
        
        enhanced_db.add_embedding(speaker_id, embedding1, confidence=0.85)
        enhanced_db.add_embedding(speaker_id, embedding2, confidence=0.92)
        
        assert len(enhanced_db.speaker_embeddings[speaker_id]) == 2
        assert len(enhanced_db.confidence_scores[speaker_id]) == 2
        assert np.array_equal(enhanced_db.speaker_embeddings[speaker_id][0], embedding1)
        assert enhanced_db.confidence_scores[speaker_id][0] == 0.85
        assert enhanced_db.confidence_scores[speaker_id][1] == 0.92
    
    @pytest.mark.unit
    def test_add_embedding_limit(self, enhanced_db):
        """Test embedding limit enforcement."""
        speaker_id = enhanced_db.create_speaker("Test Speaker")
        
        # Add more embeddings than the limit
        for i in range(55):  # max is 50
            embedding = np.random.rand(512)
            enhanced_db.add_embedding(speaker_id, embedding, confidence=0.8)
        
        # Should be limited to max
        assert len(enhanced_db.speaker_embeddings[speaker_id]) == 50
        assert len(enhanced_db.confidence_scores[speaker_id]) == 50
    
    @pytest.mark.unit
    def test_merge_speakers_basic(self, enhanced_db):
        """Test basic speaker merging."""
        # Create two speakers
        source_id = enhanced_db.create_speaker("Source Speaker")
        target_id = enhanced_db.create_speaker("Target Speaker")
        
        # Add embeddings to both
        for i in range(3):
            enhanced_db.add_embedding(source_id, np.random.rand(512), confidence=0.8)
        
        for i in range(5):
            enhanced_db.add_embedding(target_id, np.random.rand(512), confidence=0.9)
        
        # Merge source into target
        success = enhanced_db.merge_speakers(source_id, target_id)
        
        assert success == True
        assert source_id not in enhanced_db.speaker_records
        assert target_id in enhanced_db.speaker_records
        
        # Target should have all embeddings
        assert len(enhanced_db.speaker_embeddings[target_id]) == 8
        assert len(enhanced_db.confidence_scores[target_id]) == 8
    
    @pytest.mark.unit
    def test_merge_speakers_smart_target_selection(self, enhanced_db):
        """Test that merging selects the speaker with more embeddings as target."""
        # Create speakers with different embedding counts
        sparse_id = enhanced_db.create_speaker("Sparse Speaker")
        rich_id = enhanced_db.create_speaker("Rich Speaker")
        
        # Add few embeddings to sparse
        for i in range(2):
            enhanced_db.add_embedding(sparse_id, np.random.rand(512), confidence=0.7)
        
        # Add many embeddings to rich
        for i in range(8):
            enhanced_db.add_embedding(rich_id, np.random.rand(512), confidence=0.85)
        
        # Try to merge sparse into rich - should swap and merge sparse into rich
        success = enhanced_db.merge_speakers(sparse_id, rich_id)
        
        assert success == True
        assert sparse_id not in enhanced_db.speaker_records  # Sparse was merged out
        assert rich_id in enhanced_db.speaker_records  # Rich became the target
        
        # Rich should now have all embeddings
        assert len(enhanced_db.speaker_embeddings[rich_id]) == 10
    
    @pytest.mark.unit
    def test_merge_speakers_nonexistent(self, enhanced_db):
        """Test merging with nonexistent speakers."""
        valid_id = enhanced_db.create_speaker("Valid Speaker")
        
        # Try to merge nonexistent speakers
        success = enhanced_db.merge_speakers("nonexistent1", "nonexistent2")
        assert success == False
        
        success = enhanced_db.merge_speakers(valid_id, "nonexistent")
        assert success == False
        
        success = enhanced_db.merge_speakers("nonexistent", valid_id)
        assert success == False
    
    @pytest.mark.unit
    def test_delete_speaker(self, enhanced_db):
        """Test deleting a speaker."""
        speaker_id = enhanced_db.create_speaker("To Be Deleted")
        enhanced_db.add_embedding(speaker_id, np.random.rand(512), confidence=0.8)
        
        success = enhanced_db.delete_speaker(speaker_id)
        
        assert success == True
        assert speaker_id not in enhanced_db.speaker_records
        assert speaker_id not in enhanced_db.speaker_embeddings
        assert speaker_id not in enhanced_db.confidence_scores
        assert "to be deleted" not in enhanced_db.name_to_id_index
    
    @pytest.mark.unit
    def test_delete_speaker_nonexistent(self, enhanced_db):
        """Test deleting nonexistent speaker."""
        success = enhanced_db.delete_speaker("nonexistent_id")
        assert success == False
    
    @pytest.mark.unit
    def test_get_all_speakers(self, enhanced_db):
        """Test getting all speakers."""
        # Initially empty
        speakers = enhanced_db.get_all_speakers()
        assert len(speakers) == 0
        
        # Add some speakers
        id1 = enhanced_db.create_speaker("Alice", source_type="enrolled", is_enrolled=True)
        id2 = enhanced_db.create_speaker("Bob", source_type="auto")
        
        enhanced_db.add_embedding(id1, np.random.rand(512), confidence=0.9)
        enhanced_db.add_embedding(id2, np.random.rand(512), confidence=0.8)
        
        speakers = enhanced_db.get_all_speakers()
        assert len(speakers) == 2
        
        # Check format compatibility
        for speaker in speakers:
            required_fields = [
                'speaker_id', 'display_name', 'embedding_count', 'last_seen',
                'average_confidence', 'is_enrolled', 'is_verified', 'source_type', 'created_date'
            ]
            for field in required_fields:
                assert field in speaker
    
    @pytest.mark.unit
    def test_save_and_load_database(self, temp_data_dir):
        """Test saving and loading the database."""
        # Create database and add data
        db1 = EnhancedSpeakerDatabase(data_dir=temp_data_dir)
        
        speaker_id = db1.create_speaker("Test Speaker", source_type="enrolled", is_enrolled=True)
        embedding = np.random.rand(512)
        db1.add_embedding(speaker_id, embedding, confidence=0.87)
        
        # Save database
        db1._save_database()
        
        # Create new database instance (should load existing data)
        db2 = EnhancedSpeakerDatabase(data_dir=temp_data_dir)
        
        assert len(db2.speaker_records) == 1
        assert speaker_id in db2.speaker_records
        
        record = db2.speaker_records[speaker_id]
        assert record.display_name == "Test Speaker"
        assert record.source_type == "enrolled"
        assert record.is_enrolled == True
        assert record.embedding_count == 1
        assert record.average_confidence == 0.87
        
        # Check embeddings loaded
        assert len(db2.speaker_embeddings[speaker_id]) == 1
        assert len(db2.confidence_scores[speaker_id]) == 1
        assert np.array_equal(db2.speaker_embeddings[speaker_id][0], embedding)
        assert db2.confidence_scores[speaker_id][0] == 0.87
    
    @pytest.mark.unit
    def test_database_corruption_handling(self, temp_data_dir):
        """Test handling of corrupted database files."""
        db = EnhancedSpeakerDatabase(data_dir=temp_data_dir)
        
        # Create corrupted records file
        with open(db.records_file, 'w') as f:
            f.write("invalid json content")
        
        # Should handle gracefully and start fresh
        db2 = EnhancedSpeakerDatabase(data_dir=temp_data_dir)
        assert len(db2.speaker_records) == 0
    
    @pytest.mark.integration
    def test_feedback_operations(self, enhanced_db):
        """Test feedback operations with speaker corrections."""
        # Create initial speaker
        speaker_id = enhanced_db.create_speaker("Initial Name")
        
        # Simulate feedback with corrections
        speaker_corrections = {
            speaker_id: "Corrected Name"
        }
        
        audio_segments = [
            {"segment_id": "segment_1", "audio_data": np.random.rand(16000)}
        ]
        
        result = enhanced_db.send_feedback_for_learning(
            speaker_corrections, 
            audio_segments
        )
        
        assert "processed_corrections" in result
        assert "speakers_renamed" in result
        assert result["processed_corrections"] >= 1


@pytest.mark.integration
class TestEnhancedDatabaseIntegration:
    """Integration tests for enhanced database functionality."""
    
    @pytest.fixture
    def populated_enhanced_db(self, temp_data_dir):
        """Create enhanced database with test data."""
        db = EnhancedSpeakerDatabase(data_dir=temp_data_dir)
        
        # Add multiple speakers with embeddings
        alice_id = db.create_speaker("Alice Johnson", source_type="enrolled", is_enrolled=True)
        bob_id = db.create_speaker("Bob Smith", source_type="auto")
        charlie_id = db.create_speaker("Charlie Brown", source_type="corrected")
        
        # Add embeddings
        for i in range(5):
            db.add_embedding(alice_id, np.random.rand(512), confidence=0.85 + i*0.02)
        
        for i in range(3):
            db.add_embedding(bob_id, np.random.rand(512), confidence=0.75 + i*0.03)
        
        for i in range(7):
            db.add_embedding(charlie_id, np.random.rand(512), confidence=0.8 + i*0.01)
        
        return db
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_complete_speaker_workflow(self, populated_enhanced_db):
        """Test complete speaker management workflow."""
        db = populated_enhanced_db
        
        # 1. List all speakers
        speakers = db.get_all_speakers()
        assert len(speakers) == 3
        
        # 2. Find speaker by name
        alice_id = db.find_speaker_by_name("Alice Johnson")
        assert alice_id is not None
        
        # 3. Update speaker name
        success = db.update_display_name(alice_id, "Alice Williams")
        assert success == True
        
        # 4. Verify name change
        updated_speakers = db.get_all_speakers()
        alice_record = next(s for s in updated_speakers if s['speaker_id'] == alice_id)
        assert alice_record['display_name'] == "Alice Williams"
        
        # 5. Merge speakers
        bob_id = db.find_speaker_by_name("Bob Smith")
        charlie_id = db.find_speaker_by_name("Charlie Brown")
        
        # Charlie should win (has more embeddings: 7 vs 3)
        success = db.merge_speakers(bob_id, charlie_id)
        assert success == True
        
        # 6. Verify merge
        final_speakers = db.get_all_speakers()
        assert len(final_speakers) == 2  # One less after merge
        
        charlie_record = next(s for s in final_speakers if s['speaker_id'] == charlie_id)
        assert charlie_record['embedding_count'] == 10  # 7 + 3
    
    def test_persistence_across_sessions(self, temp_data_dir):
        """Test that data persists across database sessions."""
        # Session 1: Create and populate database
        db1 = EnhancedSpeakerDatabase(data_dir=temp_data_dir)
        
        speaker_id = db1.create_speaker("Persistent Speaker", is_enrolled=True)
        db1.add_embedding(speaker_id, np.random.rand(512), confidence=0.9)
        
        db1._save_database()
        del db1
        
        # Session 2: Load database and verify data
        db2 = EnhancedSpeakerDatabase(data_dir=temp_data_dir)
        
        speakers = db2.get_all_speakers()
        assert len(speakers) == 1
        
        speaker = speakers[0]
        assert speaker['display_name'] == "Persistent Speaker"
        assert speaker['is_enrolled'] == True
        assert speaker['embedding_count'] == 1
        assert speaker['average_confidence'] == 0.9
        
        # Session 3: Modify and verify changes persist
        db2.update_display_name(speaker_id, "Updated Name")
        db2._save_database()
        del db2
        
        db3 = EnhancedSpeakerDatabase(data_dir=temp_data_dir)
        speakers = db3.get_all_speakers()
        assert speakers[0]['display_name'] == "Updated Name" 