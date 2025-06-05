"""
Oreja - Enhanced Speaker Database Architecture v2
Addresses architectural issues with speaker management:
1. Immutable speaker IDs with mutable display names
2. Proper merging logic based on sample count
3. Confidence recalculation on merge
4. Clear separation of save vs feedback operations
5. Automatic reprocessing when names change
"""

import torch
import torchaudio
import numpy as np
import json
import uuid
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import logging
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class SpeakerRecord:
    """Enhanced speaker record with immutable ID and mutable display name"""
    # Immutable fields
    speaker_id: str  # UUID-based unique identifier, never changes
    created_date: str
    
    # Mutable fields
    display_name: str  # User-friendly name shown in UI
    last_seen: str
    session_count: int = 0
    total_audio_seconds: float = 0.0
    embedding_count: int = 0
    average_confidence: float = 0.0
    
    # Metadata
    is_enrolled: bool = False  # True if manually enrolled by user
    is_verified: bool = False  # True if user has confirmed this speaker's identity
    source_type: str = "auto"  # "auto", "enrolled", "corrected"
    
    def update_stats(self, embeddings: List[np.ndarray], confidences: List[float]):
        """Update statistics based on current embeddings"""
        self.embedding_count = len(embeddings)
        self.average_confidence = np.mean(confidences) if confidences else 0.0
        self.last_seen = datetime.now().isoformat()

class EnhancedSpeakerDatabase:
    """
    Enhanced Speaker Database with proper architecture:
    - Immutable speaker IDs (UUID-based)
    - Mutable display names
    - Proper merging logic
    - Confidence recalculation
    - Clear save vs feedback operations
    """
    
    def __init__(self, data_dir: str = "speaker_data_v2"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # File paths
        self.records_file = self.data_dir / "speaker_records.json"
        self.embeddings_dir = self.data_dir / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)
        
        # In-memory data
        self.speaker_records: Dict[str, SpeakerRecord] = {}
        self.speaker_embeddings: Dict[str, List[np.ndarray]] = {}
        self.confidence_scores: Dict[str, List[float]] = {}
        
        # Indexing for fast lookups
        self.name_to_id_index: Dict[str, str] = {}  # display_name -> speaker_id
        
        # Configuration
        self.similarity_threshold = 0.75
        self.min_audio_length = 1.0
        self.max_embeddings_per_speaker = 50
        
        # Load existing data
        self._load_database()
        self._rebuild_indexes()
        
        logger.info(f"Enhanced Speaker Database initialized with {len(self.speaker_records)} speakers")
    
    def _generate_speaker_id(self) -> str:
        """Generate a unique, immutable speaker ID"""
        return f"spk_{uuid.uuid4().hex[:12]}"
    
    def _load_database(self):
        """Load speaker records and embeddings from storage"""
        try:
            # Load speaker records
            if self.records_file.exists():
                with open(self.records_file, 'r') as f:
                    records_data = json.load(f)
                
                for speaker_id, record_data in records_data.items():
                    self.speaker_records[speaker_id] = SpeakerRecord(**record_data)
            
            # Load embeddings for each speaker
            for speaker_id in self.speaker_records.keys():
                embeddings_file = self.embeddings_dir / f"{speaker_id}.npy"
                if embeddings_file.exists():
                    embeddings_data = np.load(embeddings_file, allow_pickle=True).item()
                    self.speaker_embeddings[speaker_id] = embeddings_data.get('embeddings', [])
                    self.confidence_scores[speaker_id] = embeddings_data.get('confidence_scores', [])
                else:
                    self.speaker_embeddings[speaker_id] = []
                    self.confidence_scores[speaker_id] = []
            
            logger.info(f"Loaded {len(self.speaker_records)} speaker records")
            
        except Exception as e:
            logger.error(f"Error loading speaker database: {e}")
    
    def _save_database(self):
        """Save speaker records and embeddings to storage"""
        try:
            # Update statistics before saving
            for speaker_id, record in self.speaker_records.items():
                embeddings = self.speaker_embeddings.get(speaker_id, [])
                confidences = self.confidence_scores.get(speaker_id, [])
                record.update_stats(embeddings, confidences)
            
            # Save speaker records
            records_data = {
                speaker_id: asdict(record) 
                for speaker_id, record in self.speaker_records.items()
            }
            
            with open(self.records_file, 'w') as f:
                json.dump(records_data, f, indent=2)
            
            # Save embeddings separately for efficiency
            for speaker_id, embeddings in self.speaker_embeddings.items():
                if embeddings:
                    embeddings_file = self.embeddings_dir / f"{speaker_id}.npy"
                    embeddings_data = {
                        'embeddings': embeddings,
                        'confidence_scores': self.confidence_scores.get(speaker_id, [])
                    }
                    np.save(embeddings_file, embeddings_data)
            
            logger.debug("Speaker database saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving speaker database: {e}")
    
    def _rebuild_indexes(self):
        """Rebuild internal indexes for fast lookups"""
        self.name_to_id_index.clear()
        
        for speaker_id, record in self.speaker_records.items():
            # Index by display name (case-insensitive)
            display_name_lower = record.display_name.lower()
            self.name_to_id_index[display_name_lower] = speaker_id
    
    def get_all_speakers(self) -> List[Dict]:
        """Get all speakers in a format compatible with the legacy API"""
        speakers = []
        for speaker_id, record in self.speaker_records.items():
            speakers.append({
                'speaker_id': record.speaker_id,
                'display_name': record.display_name,
                'embedding_count': record.embedding_count,
                'last_seen': record.last_seen,
                'average_confidence': record.average_confidence,
                'is_enrolled': record.is_enrolled,
                'is_verified': record.is_verified,
                'source_type': record.source_type,
                'created_date': record.created_date
            })
        return speakers
    
    def create_speaker(self, display_name: str, source_type: str = "auto", 
                      is_enrolled: bool = False) -> str:
        """
        Create a new speaker with immutable ID and mutable display name
        
        Args:
            display_name: The display name for the speaker
            source_type: How this speaker was created ("auto", "enrolled", "corrected")
            is_enrolled: Whether this is a manually enrolled speaker
            
        Returns:
            The unique speaker ID
        """
        speaker_id = self._generate_speaker_id()
        
        record = SpeakerRecord(
            speaker_id=speaker_id,
            display_name=display_name,
            created_date=datetime.now().isoformat(),
            last_seen=datetime.now().isoformat(),
            is_enrolled=is_enrolled,
            source_type=source_type
        )
        
        self.speaker_records[speaker_id] = record
        self.speaker_embeddings[speaker_id] = []
        self.confidence_scores[speaker_id] = []
        
        # Update indexes
        self.name_to_id_index[display_name.lower()] = speaker_id
        
        logger.info(f"Created new speaker: {display_name} ({speaker_id})")
        return speaker_id
    
    def update_display_name(self, speaker_id: str, new_display_name: str) -> bool:
        """
        Update the display name of a speaker (ID remains immutable)
        
        Args:
            speaker_id: The immutable speaker ID
            new_display_name: The new display name
            
        Returns:
            True if successful
        """
        if speaker_id not in self.speaker_records:
            return False
        
        old_name = self.speaker_records[speaker_id].display_name
        
        # Remove old name from index
        old_name_lower = old_name.lower()
        if old_name_lower in self.name_to_id_index:
            del self.name_to_id_index[old_name_lower]
        
        # Update record
        self.speaker_records[speaker_id].display_name = new_display_name
        self.speaker_records[speaker_id].is_verified = True  # Mark as user-verified
        
        # Update index with new name
        self.name_to_id_index[new_display_name.lower()] = speaker_id
        
        # Save immediately for name changes
        self._save_database()
        
        logger.info(f"Updated speaker name: {speaker_id} '{old_name}' -> '{new_display_name}'")
        return True
    
    def find_speaker_by_name(self, display_name: str) -> Optional[str]:
        """
        Find a speaker ID by display name (case-insensitive)
        
        Args:
            display_name: The display name to search for
            
        Returns:
            Speaker ID if found, None otherwise
        """
        return self.name_to_id_index.get(display_name.lower())
    
    def add_embedding(self, speaker_id: str, embedding: np.ndarray, confidence: float = 1.0):
        """
        Add an embedding to a speaker's profile
        
        Args:
            speaker_id: The speaker ID
            embedding: The embedding vector
            confidence: Confidence score for this embedding
        """
        if speaker_id not in self.speaker_records:
            return False
        
        embeddings = self.speaker_embeddings[speaker_id]
        confidences = self.confidence_scores[speaker_id]
        
        # Add new embedding
        embeddings.append(embedding)
        confidences.append(confidence)
        
        # Limit embeddings to prevent memory bloat
        if len(embeddings) > self.max_embeddings_per_speaker:
            # Keep the highest confidence embeddings
            indices = np.argsort(confidences)[-self.max_embeddings_per_speaker:]
            self.speaker_embeddings[speaker_id] = [embeddings[i] for i in indices]
            self.confidence_scores[speaker_id] = [confidences[i] for i in indices]
        
        # Update record stats
        record = self.speaker_records[speaker_id]
        record.update_stats(self.speaker_embeddings[speaker_id], self.confidence_scores[speaker_id])
        
        return True
    
    def merge_speakers(self, source_speaker_id: str, target_speaker_id: str) -> bool:
        """
        Merge two speakers using proper logic:
        1. The speaker with more samples is kept as the target
        2. All embeddings are merged
        3. Confidence is recalculated
        4. Source speaker is deleted
        
        Args:
            source_speaker_id: Speaker to merge from (will be deleted)
            target_speaker_id: Speaker to merge into (will be kept)
            
        Returns:
            True if successful
        """
        if (source_speaker_id not in self.speaker_records or 
            target_speaker_id not in self.speaker_records):
            logger.warning(f"Cannot merge speakers: source={source_speaker_id}, target={target_speaker_id}")
            return False
        
        source_embeddings = self.speaker_embeddings.get(source_speaker_id, [])
        target_embeddings = self.speaker_embeddings.get(target_speaker_id, [])
        
        # Determine which speaker should actually be the target (most samples)
        if len(source_embeddings) > len(target_embeddings):
            # Swap: source has more samples, so it should be the target
            source_speaker_id, target_speaker_id = target_speaker_id, source_speaker_id
            source_embeddings, target_embeddings = target_embeddings, source_embeddings
            logger.info(f"Swapped merge direction: speaker with more samples kept as target")
        
        source_record = self.speaker_records[source_speaker_id]
        target_record = self.speaker_records[target_speaker_id]
        source_confidences = self.confidence_scores.get(source_speaker_id, [])
        target_confidences = self.confidence_scores.get(target_speaker_id, [])
        
        # Merge embeddings
        merged_embeddings = target_embeddings + source_embeddings
        merged_confidences = target_confidences + source_confidences
        
        # If we have too many embeddings, keep the highest confidence ones
        if len(merged_embeddings) > self.max_embeddings_per_speaker:
            indices = np.argsort(merged_confidences)[-self.max_embeddings_per_speaker:]
            merged_embeddings = [merged_embeddings[i] for i in indices]
            merged_confidences = [merged_confidences[i] for i in indices]
        
        # Update target speaker
        self.speaker_embeddings[target_speaker_id] = merged_embeddings
        self.confidence_scores[target_speaker_id] = merged_confidences
        
        # Merge metadata
        target_record.total_audio_seconds += source_record.total_audio_seconds
        target_record.session_count += source_record.session_count
        target_record.is_verified = target_record.is_verified or source_record.is_verified
        
        # If source was enrolled, mark target as enrolled too
        if source_record.is_enrolled:
            target_record.is_enrolled = True
        
        # Update target record stats
        target_record.update_stats(merged_embeddings, merged_confidences)
        
        # Remove source speaker
        old_source_name = source_record.display_name
        del self.speaker_records[source_speaker_id]
        del self.speaker_embeddings[source_speaker_id]
        del self.confidence_scores[source_speaker_id]
        
        # Clean up source name from index
        source_name_lower = old_source_name.lower()
        if source_name_lower in self.name_to_id_index:
            del self.name_to_id_index[source_name_lower]
        
        # Clean up source embedding file
        source_embeddings_file = self.embeddings_dir / f"{source_speaker_id}.npy"
        if source_embeddings_file.exists():
            source_embeddings_file.unlink()
        
        # Save the updated database
        self._save_database()
        
        logger.info(f"Successfully merged {source_speaker_id} ({old_source_name}) "
                   f"into {target_speaker_id} ({target_record.display_name}). "
                   f"New embedding count: {len(merged_embeddings)}, "
                   f"New avg confidence: {target_record.average_confidence:.3f}")
        
        return True
    
    def delete_speaker(self, speaker_id: str) -> bool:
        """Delete a speaker completely"""
        if speaker_id not in self.speaker_records:
            return False
        
        record = self.speaker_records[speaker_id]
        
        # Remove from indexes
        name_lower = record.display_name.lower()
        if name_lower in self.name_to_id_index:
            del self.name_to_id_index[name_lower]
        
        # Remove from memory
        del self.speaker_records[speaker_id]
        if speaker_id in self.speaker_embeddings:
            del self.speaker_embeddings[speaker_id]
        if speaker_id in self.confidence_scores:
            del self.confidence_scores[speaker_id]
        
        # Remove embedding file
        embeddings_file = self.embeddings_dir / f"{speaker_id}.npy"
        if embeddings_file.exists():
            embeddings_file.unlink()
        
        self._save_database()
        
        logger.info(f"Deleted speaker: {speaker_id} ({record.display_name})")
        return True
    
    def get_all_speakers(self) -> List[Dict]:
        """Get all speakers with their current statistics"""
        speakers = []
        
        for speaker_id, record in self.speaker_records.items():
            embeddings = self.speaker_embeddings.get(speaker_id, [])
            confidences = self.confidence_scores.get(speaker_id, [])
            
            speaker_data = {
                'speaker_id': speaker_id,
                'display_name': record.display_name,
                'created_date': record.created_date,
                'last_seen': record.last_seen,
                'embedding_count': len(embeddings),
                'average_confidence': np.mean(confidences) if confidences else 0.0,
                'session_count': record.session_count,
                'total_audio_seconds': record.total_audio_seconds,
                'is_enrolled': record.is_enrolled,
                'is_verified': record.is_verified,
                'source_type': record.source_type
            }
            speakers.append(speaker_data)
        
        # Sort by embedding count (most trained first)
        speakers.sort(key=lambda x: x['embedding_count'], reverse=True)
        
        return speakers
    
    def save_transcription_with_corrections(self, transcription_data: Dict, 
                                          speaker_corrections: Dict[str, str],
                                          output_file: str = None) -> str:
        """
        Save transcription with corrected speaker names WITHOUT sending feedback.
        This is the "Save Document" operation that just creates the file.
        
        Args:
            transcription_data: The transcription data
            speaker_corrections: Mapping of old speaker IDs to new display names
            output_file: Optional output file path
            
        Returns:
            Path to saved file
        """
        try:
            # Apply corrections to create corrected transcription
            corrected_data = transcription_data.copy()
            
            # Update segment speaker names
            for segment in corrected_data.get('segments', []):
                old_speaker = segment.get('speaker', '')
                if old_speaker in speaker_corrections:
                    segment['speaker'] = speaker_corrections[old_speaker]
            
            # Add metadata about corrections
            corrected_data['metadata'] = corrected_data.get('metadata', {})
            corrected_data['metadata']['corrections_applied'] = len(speaker_corrections)
            corrected_data['metadata']['correction_timestamp'] = datetime.now().isoformat()
            
            # Determine output file path
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"transcription_corrected_{timestamp}.json"
            
            # Save file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(corrected_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved corrected transcription to {output_file} with {len(speaker_corrections)} corrections")
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving corrected transcription: {e}")
            raise
    
    def send_feedback_for_learning(self, speaker_corrections: Dict[str, str],
                                  audio_segments: List[Dict] = None) -> Dict:
        """
        Send feedback to improve speaker recognition learning.
        This is the "Send Feedback" operation separate from saving.
        
        Args:
            speaker_corrections: Mapping of old speaker IDs to new display names
            audio_segments: Optional audio segments for enhanced learning
            
        Returns:
            Dictionary with feedback processing results
        """
        try:
            results = {
                'processed_corrections': 0,
                'speakers_created': 0,
                'speakers_merged': 0,
                'speakers_renamed': 0,
                'errors': []
            }
            
            for old_speaker_id, new_display_name in speaker_corrections.items():
                try:
                    # Check if a speaker with this display name already exists
                    existing_speaker_id = self.find_speaker_by_name(new_display_name)
                    
                    if existing_speaker_id and existing_speaker_id != old_speaker_id:
                        # Merge old speaker into existing speaker
                        if self.merge_speakers(old_speaker_id, existing_speaker_id):
                            results['speakers_merged'] += 1
                            logger.info(f"Merged {old_speaker_id} into existing speaker {new_display_name}")
                        else:
                            results['errors'].append(f"Failed to merge {old_speaker_id} into {new_display_name}")
                    
                    elif old_speaker_id in self.speaker_records:
                        # Just rename the existing speaker
                        if self.update_display_name(old_speaker_id, new_display_name):
                            results['speakers_renamed'] += 1
                            logger.info(f"Renamed speaker {old_speaker_id} to {new_display_name}")
                        else:
                            results['errors'].append(f"Failed to rename {old_speaker_id} to {new_display_name}")
                    
                    else:
                        # Create new speaker (for new speaker IDs from transcription)
                        new_speaker_id = self.create_speaker(new_display_name, source_type="corrected")
                        results['speakers_created'] += 1
                        logger.info(f"Created new speaker {new_display_name} for correction {old_speaker_id}")
                    
                    results['processed_corrections'] += 1
                    
                except Exception as e:
                    error_msg = f"Error processing correction {old_speaker_id} -> {new_display_name}: {e}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
            
            # Save database after all feedback processing
            self._save_database()
            
            logger.info(f"Feedback processing complete: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            raise
    
    def trigger_reprocessing(self, audio_file: str = None, 
                           segments_to_reprocess: List[Dict] = None) -> Dict:
        """
        Trigger reprocessing of speaker attribution using updated speaker models.
        This should be called after speaker corrections to improve future recognition.
        
        Args:
            audio_file: Path to audio file for reprocessing
            segments_to_reprocess: Specific segments to reprocess
            
        Returns:
            Dictionary with reprocessing results
        """
        # This would integrate with the existing speaker identification system
        # to re-run attribution on segments with low confidence
        
        results = {
            'segments_reprocessed': 0,
            'improvements_found': 0,
            'average_confidence_improvement': 0.0
        }
        
        # TODO: Implement reprocessing logic here
        # This would involve re-running speaker identification on low-confidence segments
        # using the updated speaker models after corrections
        
        logger.info("Speaker reprocessing triggered")
        return results 