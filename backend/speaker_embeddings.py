"""
Oreja - Offline Speaker Embedding System
100% Local, Privacy-First Speaker Recognition

NO AUDIO DATA EVER LEAVES THIS MACHINE
"""

import torch
import torchaudio
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN

# SpeechBrain for offline speaker embeddings
try:
    from speechbrain.pretrained import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    logging.warning("SpeechBrain not available. Speaker embeddings will be disabled.")

logger = logging.getLogger(__name__)

class SpeakerProfile:
    """Represents a speaker with their embeddings and metadata"""
    def __init__(self, speaker_id: str, name: str = None):
        self.speaker_id = speaker_id
        self.name = name or f"Speaker_{speaker_id}"
        self.embeddings: List[np.ndarray] = []
        self.created_date = datetime.now().isoformat()
        self.last_seen = datetime.now().isoformat()
        self.session_count = 0
        self.total_audio_seconds = 0.0
        self.confidence_scores: List[float] = []
    
    def add_embedding(self, embedding: np.ndarray, confidence: float = 1.0):
        """Add a new embedding for this speaker"""
        self.embeddings.append(embedding)
        self.confidence_scores.append(confidence)
        self.last_seen = datetime.now().isoformat()
        
        # Keep only the most recent N embeddings to prevent memory bloat
        max_embeddings = 50
        if len(self.embeddings) > max_embeddings:
            self.embeddings = self.embeddings[-max_embeddings:]
            self.confidence_scores = self.confidence_scores[-max_embeddings:]
    
    def get_average_embedding(self) -> np.ndarray:
        """Get the average embedding for this speaker"""
        if not self.embeddings:
            return None
        
        # Weight by confidence scores
        weights = np.array(self.confidence_scores)
        weights = weights / weights.sum()
        
        weighted_embeddings = np.array([emb * w for emb, w in zip(self.embeddings, weights)])
        return np.mean(weighted_embeddings, axis=0)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'speaker_id': self.speaker_id,
            'name': self.name,
            'created_date': self.created_date,
            'last_seen': self.last_seen,
            'session_count': self.session_count,
            'total_audio_seconds': self.total_audio_seconds,
            'embedding_count': len(self.embeddings),
            'average_confidence': np.mean(self.confidence_scores) if self.confidence_scores else 0.0,
            # Note: embeddings stored separately for efficiency
        }

class OfflineSpeakerEmbeddingManager:
    """
    100% Offline Speaker Embedding System
    
    PRIVACY GUARANTEE: No audio data ever leaves this machine
    """
    
    def __init__(self, data_dir: str = "speaker_data_v2_legacy_compatibility"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # File paths - all local
        self.profiles_file = self.data_dir / "speaker_profiles.json"
        self.embeddings_dir = self.data_dir / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)
        
        # Speaker database
        self.speaker_profiles: Dict[str, SpeakerProfile] = {}
        
        # Model settings
        self.similarity_threshold = 0.75  # Adjust for sensitivity
        self.min_audio_length = 1.0  # Minimum seconds of audio for embedding
        self.sample_rate = 16000
        
        # Load pretrained model (completely offline)
        self.embedding_model = None
        self._load_embedding_model()
        
        # Load existing speaker data
        self._load_speaker_database()
        
        logger.info(f"OfflineSpeakerEmbeddingManager initialized with {len(self.speaker_profiles)} speakers")
    
    def _load_embedding_model(self):
        """Load the offline speaker embedding model"""
        if not SPEECHBRAIN_AVAILABLE:
            logger.warning("SpeechBrain not available - speaker embeddings disabled")
            return
        
        try:
            # Load ECAPA-TDNN model for speaker embeddings (downloads once, then cached locally)
            model_source = "speechbrain/spkrec-ecapa-voxceleb"
            save_dir = self.data_dir / "pretrained_models" / "spk-ecapa-voxceleb"
            
            logger.info("Loading offline speaker embedding model...")
            self.embedding_model = EncoderClassifier.from_hparams(
                source=model_source,
                savedir=str(save_dir),
                run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )
            logger.info("Speaker embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load speaker embedding model: {e}")
            self.embedding_model = None
    
    def _load_speaker_database(self):
        """Load speaker profiles from local storage"""
        try:
            if self.profiles_file.exists():
                with open(self.profiles_file, 'r') as f:
                    profiles_data = json.load(f)
                
                for speaker_id, profile_data in profiles_data.items():
                    profile = SpeakerProfile(speaker_id, profile_data.get('name'))
                    profile.created_date = profile_data.get('created_date', profile.created_date)
                    profile.last_seen = profile_data.get('last_seen', profile.last_seen)
                    profile.session_count = profile_data.get('session_count', 0)
                    profile.total_audio_seconds = profile_data.get('total_audio_seconds', 0.0)
                    
                    # Load embeddings from separate file
                    embeddings_file = self.embeddings_dir / f"{speaker_id}.npy"
                    if embeddings_file.exists():
                        embeddings_data = np.load(embeddings_file, allow_pickle=True).item()
                        profile.embeddings = embeddings_data.get('embeddings', [])
                        profile.confidence_scores = embeddings_data.get('confidence_scores', [])
                    
                    self.speaker_profiles[speaker_id] = profile
                
                logger.info(f"Loaded {len(self.speaker_profiles)} speaker profiles")
        
        except Exception as e:
            logger.error(f"Error loading speaker database: {e}")
    
    def _save_speaker_database(self):
        """Save speaker profiles to local storage"""
        try:
            # Save profile metadata
            profiles_data = {
                speaker_id: profile.to_dict() 
                for speaker_id, profile in self.speaker_profiles.items()
            }
            
            with open(self.profiles_file, 'w') as f:
                json.dump(profiles_data, f, indent=2)
            
            # Save embeddings separately (more efficient)
            for speaker_id, profile in self.speaker_profiles.items():
                if profile.embeddings:
                    embeddings_file = self.embeddings_dir / f"{speaker_id}.npy"
                    embeddings_data = {
                        'embeddings': profile.embeddings,
                        'confidence_scores': profile.confidence_scores
                    }
                    np.save(embeddings_file, embeddings_data)
            
            logger.debug("Speaker database saved successfully")
        
        except Exception as e:
            logger.error(f"Error saving speaker database: {e}")
    
    def extract_embedding(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract speaker embedding from audio data
        
        Args:
            audio_data: Raw audio samples (numpy array)
            
        Returns:
            Speaker embedding vector or None if extraction fails
        """
        if self.embedding_model is None:
            return None
        
        try:
            # Ensure audio is the right format
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)  # Convert to mono
            
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.embedding_model.encode_batch(audio_tensor)
            
            return embedding.squeeze().cpu().numpy()
        
        except Exception as e:
            logger.error(f"Error extracting speaker embedding: {e}")
            return None
    
    def identify_or_create_speaker(self, audio_data: np.ndarray, min_confidence: float = 0.7) -> Tuple[str, float, bool]:
        """
        Identify speaker from audio or create new speaker profile
        
        Args:
            audio_data: Raw audio samples
            min_confidence: Minimum confidence for speaker identification
            
        Returns:
            Tuple of (speaker_id, confidence, is_new_speaker)
        """
        # Extract embedding
        embedding = self.extract_embedding(audio_data)
        if embedding is None:
            # Fallback: create unknown speaker
            return self._create_new_speaker("UNKNOWN", 0.0), 0.0, True
        
        # Compare with existing speakers
        best_match_id = None
        best_similarity = 0.0
        
        for speaker_id, profile in self.speaker_profiles.items():
            avg_embedding = profile.get_average_embedding()
            if avg_embedding is not None:
                # Calculate cosine similarity
                similarity = 1 - cosine(embedding, avg_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = speaker_id
        
        # Decide if match is good enough
        if best_match_id and best_similarity >= self.similarity_threshold:
            # Add embedding to existing speaker
            self.speaker_profiles[best_match_id].add_embedding(embedding, best_similarity)
            self._save_speaker_database()
            
            logger.info(f"Identified existing speaker {best_match_id} with confidence {best_similarity:.3f}")
            return best_match_id, best_similarity, False
        
        else:
            # Create new speaker
            new_speaker_id = self._create_new_speaker("AUTO", best_similarity)
            self.speaker_profiles[new_speaker_id].add_embedding(embedding, 1.0)
            self._save_speaker_database()
            
            logger.info(f"Created new speaker {new_speaker_id}")
            return new_speaker_id, 1.0, True
    
    def _create_new_speaker(self, prefix: str = "AUTO", confidence: float = 1.0) -> str:
        """Create a new speaker profile"""
        # Generate unique ID
        speaker_count = len(self.speaker_profiles) + 1
        speaker_id = f"{prefix}_SPEAKER_{speaker_count:03d}"
        
        # Ensure uniqueness
        while speaker_id in self.speaker_profiles:
            speaker_count += 1
            speaker_id = f"{prefix}_SPEAKER_{speaker_count:03d}"
        
        # Create profile
        profile = SpeakerProfile(speaker_id)
        self.speaker_profiles[speaker_id] = profile
        
        return speaker_id
    
    def enroll_speaker(self, speaker_name: str, audio_data: np.ndarray) -> str:
        """
        Manually enroll a speaker with known name
        
        Args:
            speaker_name: Human-readable name
            audio_data: Audio sample for enrollment
            
        Returns:
            Generated speaker_id
        """
        embedding = self.extract_embedding(audio_data)
        if embedding is None:
            raise ValueError("Could not extract speaker embedding from audio")
        
        # Create new speaker profile
        speaker_id = self._create_new_speaker("ENROLLED")
        self.speaker_profiles[speaker_id].name = speaker_name
        self.speaker_profiles[speaker_id].add_embedding(embedding, 1.0)
        
        self._save_speaker_database()
        
        logger.info(f"Enrolled new speaker: {speaker_name} as {speaker_id}")
        return speaker_id
    
    def get_speaker_stats(self) -> Dict:
        """Get statistics about the speaker database"""
        return {
            'total_speakers': len(self.speaker_profiles),
            'speakers': [
                {
                    'id': profile.speaker_id,
                    'name': profile.name,
                    'embedding_count': len(profile.embeddings),
                    'last_seen': profile.last_seen,
                    'avg_confidence': np.mean(profile.confidence_scores) if profile.confidence_scores else 0.0
                }
                for profile in self.speaker_profiles.values()
            ]
        }
    
    def delete_speaker(self, speaker_id: str) -> bool:
        """Delete a speaker profile"""
        if speaker_id in self.speaker_profiles:
            del self.speaker_profiles[speaker_id]
            
            # Delete embedding file
            embeddings_file = self.embeddings_dir / f"{speaker_id}.npy"
            if embeddings_file.exists():
                embeddings_file.unlink()
            
            self._save_speaker_database()
            logger.info(f"Deleted speaker: {speaker_id}")
            return True
        
        return False
    
    def update_speaker_name(self, speaker_id: str, new_name: str) -> bool:
        """Update speaker name"""
        if speaker_id in self.speaker_profiles:
            self.speaker_profiles[speaker_id].name = new_name
            self._save_speaker_database()
            logger.info(f"Updated speaker {speaker_id} name to: {new_name}")
            return True
        
        return False
    
    def provide_correction_feedback(self, correct_speaker_name: str, audio_data: np.ndarray) -> bool:
        """
        Provide correction feedback to improve speaker recognition.
        
        This method allows the system to learn from user corrections by:
        1. Finding or creating a speaker profile for the correct name
        2. Adding the audio embedding to that speaker's profile
        3. Optionally removing similar embeddings from incorrect speakers
        
        Args:
            correct_speaker_name: The correct speaker name for this audio
            audio_data: The audio data that was incorrectly identified
            
        Returns:
            True if feedback was processed successfully
        """
        try:
            # Extract embedding from the corrected audio
            embedding = self.extract_embedding(audio_data)
            if embedding is None:
                logger.warning("Could not extract embedding for correction feedback")
                return False
            
            # Find existing speaker with this name or create a new one
            target_speaker_id = None
            for speaker_id, profile in self.speaker_profiles.items():
                if profile.name.lower() == correct_speaker_name.lower():
                    target_speaker_id = speaker_id
                    break
            
            # If speaker doesn't exist, create a new profile
            if target_speaker_id is None:
                target_speaker_id = self._create_new_speaker("CORRECTED")
                self.speaker_profiles[target_speaker_id].name = correct_speaker_name
                logger.info(f"Created new speaker profile for correction: {correct_speaker_name}")
            
            # Add the embedding to the correct speaker with high confidence
            # Use high confidence since this is user-corrected
            self.speaker_profiles[target_speaker_id].add_embedding(embedding, confidence=0.95)
            
            # Optional: Remove similar embeddings from other speakers to prevent confusion
            # This helps when the system initially assigned the audio to the wrong speaker
            self._remove_similar_embeddings_from_other_speakers(target_speaker_id, embedding)
            
            # Save the updated database
            self._save_speaker_database()
            
            logger.info(f"Successfully processed correction feedback for {correct_speaker_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing correction feedback: {e}")
            return False
    
    def _remove_similar_embeddings_from_other_speakers(self, target_speaker_id: str, target_embedding: np.ndarray, similarity_threshold: float = 0.85):
        """
        Remove similar embeddings from other speakers to prevent confusion.
        
        This helps when an audio segment was initially assigned to the wrong speaker
        and we want to clean up those incorrect associations.
        """
        try:
            for speaker_id, profile in self.speaker_profiles.items():
                if speaker_id == target_speaker_id:
                    continue  # Skip the target speaker
                
                # Check for similar embeddings in other speakers
                embeddings_to_remove = []
                for i, embedding in enumerate(profile.embeddings):
                    similarity = 1 - cosine(target_embedding, embedding)
                    if similarity > similarity_threshold:
                        embeddings_to_remove.append(i)
                        logger.debug(f"Removing similar embedding from {speaker_id} (similarity: {similarity:.3f})")
                
                # Remove embeddings in reverse order to maintain indices
                for i in reversed(embeddings_to_remove):
                    profile.embeddings.pop(i)
                    profile.confidence_scores.pop(i)
                
                # If a speaker has no embeddings left, consider removing them
                # (but keep speakers with names to preserve user-created speakers)
                if len(profile.embeddings) == 0 and profile.name.startswith("AUTO_SPEAKER"):
                    logger.info(f"Removing empty auto-generated speaker: {speaker_id}")
                    # Mark for deletion
                    # We'll do this in a separate pass to avoid modifying dict during iteration
        
        except Exception as e:
            logger.warning(f"Error removing similar embeddings: {e}")
    
    def optimize_speaker_profiles(self):
        """
        Optimize speaker profiles by removing duplicates and consolidating similar speakers.
        This can be called periodically to clean up the speaker database.
        """
        try:
            # Remove empty auto-generated speakers
            speakers_to_remove = []
            for speaker_id, profile in self.speaker_profiles.items():
                if len(profile.embeddings) == 0 and profile.name.startswith("AUTO_SPEAKER"):
                    speakers_to_remove.append(speaker_id)
            
            for speaker_id in speakers_to_remove:
                logger.info(f"Removing empty auto-generated speaker: {speaker_id}")
                del self.speaker_profiles[speaker_id]
                # Clean up embedding file
                embeddings_file = self.embeddings_dir / f"{speaker_id}.npy"
                if embeddings_file.exists():
                    embeddings_file.unlink()
            
            # Save optimized database
            if speakers_to_remove:
                self._save_speaker_database()
                logger.info(f"Optimized speaker database: removed {len(speakers_to_remove)} empty speakers")
        
        except Exception as e:
            logger.error(f"Error optimizing speaker profiles: {e}")
    
    def get_speaker_by_name(self, speaker_name: str) -> Optional[str]:
        """
        Find a speaker ID by name (case-insensitive).
        
        Args:
            speaker_name: The speaker name to search for
            
        Returns:
            Speaker ID if found, None otherwise
        """
        for speaker_id, profile in self.speaker_profiles.items():
            if profile.name.lower() == speaker_name.lower():
                return speaker_id
        return None
    
    def merge_speakers(self, source_speaker_id: str, target_speaker_id: str) -> bool:
        """
        Merge one speaker profile into another.
        
        This is useful when the system creates multiple speaker IDs for the same person
        and the user corrects them to be the same speaker.
        
        Args:
            source_speaker_id: The speaker ID to merge from (will be deleted)
            target_speaker_id: The speaker ID to merge into (will be kept)
            
        Returns:
            True if merge was successful
        """
        try:
            if source_speaker_id not in self.speaker_profiles or target_speaker_id not in self.speaker_profiles:
                logger.warning(f"Cannot merge speakers: source={source_speaker_id}, target={target_speaker_id}")
                return False
            
            source_profile = self.speaker_profiles[source_speaker_id]
            target_profile = self.speaker_profiles[target_speaker_id]
            
            # Merge embeddings from source to target
            for i, embedding in enumerate(source_profile.embeddings):
                confidence = source_profile.confidence_scores[i] if i < len(source_profile.confidence_scores) else 0.8
                target_profile.add_embedding(embedding, confidence)
            
            # Update metadata
            target_profile.total_audio_seconds += source_profile.total_audio_seconds
            target_profile.session_count += source_profile.session_count
            
            # Remove the source speaker
            del self.speaker_profiles[source_speaker_id]
            
            # Clean up source embedding file
            source_embeddings_file = self.embeddings_dir / f"{source_speaker_id}.npy"
            if source_embeddings_file.exists():
                source_embeddings_file.unlink()
            
            # Save the updated database
            self._save_speaker_database()
            
            logger.info(f"Successfully merged speaker {source_speaker_id} into {target_speaker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error merging speakers: {e}")
            return False 