"""
Enhanced Speaker Server Integration
Bridges the new speaker database v2 architecture with existing server infrastructure
Provides migration utilities and enhanced endpoints
"""

import logging
import json
import asyncio
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import uuid

from fastapi import HTTPException
from speaker_database_v2 import EnhancedSpeakerDatabase, SpeakerRecord
from speaker_embeddings import OfflineSpeakerEmbeddingManager
from enhanced_segment_splitting import AudioSegmentSplitter, SegmentSplitValidator

logger = logging.getLogger(__name__)

class EnhancedSpeakerServerIntegration:
    """
    Integration layer between the enhanced speaker database v2 and existing server infrastructure
    """
    
    def __init__(self, 
                 legacy_speaker_manager: OfflineSpeakerEmbeddingManager = None,
                 enhanced_db_path: str = "speaker_data_v2"):
        self.legacy_manager = legacy_speaker_manager
        self.enhanced_db = EnhancedSpeakerDatabase(enhanced_db_path)
        self.audio_splitter = None
        
        # Initialize audio splitter if we have a legacy manager
        if self.legacy_manager:
            self.audio_splitter = AudioSegmentSplitter(self.legacy_manager)
        
        # Migration tracking
        self.migration_completed = False
        self.migration_log = []
        
        logger.info("Enhanced Speaker Server Integration initialized")
    
    async def migrate_from_legacy_database(self, 
                                         backup_legacy: bool = True,
                                         dry_run: bool = False) -> Dict:
        """
        Migrate from the legacy speaker database to the enhanced v2 architecture
        
        Args:
            backup_legacy: Whether to backup the legacy database before migration
            dry_run: If True, only analyze what would be migrated without making changes
            
        Returns:
            Migration results and statistics
        """
        try:
            migration_results = {
                'total_speakers_found': 0,
                'speakers_migrated': 0,
                'speakers_merged': 0,
                'migration_errors': [],
                'dry_run': dry_run,
                'timestamp': datetime.now().isoformat()
            }
            
            if not self.legacy_manager:
                raise ValueError("No legacy speaker manager provided for migration")
            
            # Create backup if requested
            if backup_legacy and not dry_run:
                backup_path = self._create_legacy_backup()
                migration_results['backup_path'] = str(backup_path)
                logger.info(f"Created legacy database backup at {backup_path}")
            
            # Get all speakers from legacy database
            legacy_speakers = self.legacy_manager.speaker_profiles
            migration_results['total_speakers_found'] = len(legacy_speakers)
            
            logger.info(f"Found {len(legacy_speakers)} speakers in legacy database")
            
            # Group speakers by display name to detect duplicates
            speakers_by_name = {}
            for speaker_id, profile in legacy_speakers.items():
                display_name = profile.name
                if display_name not in speakers_by_name:
                    speakers_by_name[display_name] = []
                speakers_by_name[display_name].append((speaker_id, profile))
            
            # Migrate speakers, merging duplicates
            for display_name, speaker_list in speakers_by_name.items():
                try:
                    if len(speaker_list) == 1:
                        # Single speaker - direct migration
                        speaker_id, profile = speaker_list[0]
                        success = await self._migrate_single_speaker(speaker_id, profile, dry_run)
                        if success:
                            migration_results['speakers_migrated'] += 1
                    else:
                        # Multiple speakers with same name - merge them
                        success = await self._migrate_and_merge_speakers(display_name, speaker_list, dry_run)
                        if success:
                            migration_results['speakers_migrated'] += 1
                            migration_results['speakers_merged'] += len(speaker_list) - 1
                            
                except Exception as e:
                    error_msg = f"Failed to migrate speaker(s) with name '{display_name}': {e}"
                    migration_results['migration_errors'].append(error_msg)
                    logger.error(error_msg)
            
            if not dry_run:
                self.migration_completed = True
                self._save_migration_log(migration_results)
            
            logger.info(f"Migration {'analysis' if dry_run else 'completed'}: "
                       f"{migration_results['speakers_migrated']} speakers migrated, "
                       f"{migration_results['speakers_merged']} duplicates merged")
            
            return migration_results
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise HTTPException(status_code=500, detail=f"Migration failed: {e}")
    
    async def _migrate_single_speaker(self, legacy_id: str, legacy_profile, dry_run: bool) -> bool:
        """Migrate a single speaker from legacy to enhanced database"""
        try:
            if dry_run:
                logger.debug(f"Would migrate speaker {legacy_id} -> {legacy_profile.name}")
                return True
            
            # Determine source type based on legacy speaker ID
            if legacy_id.startswith("ENROLLED_"):
                source_type = "enrolled"
                is_enrolled = True
            elif legacy_id.startswith("CORRECTED_"):
                source_type = "corrected"
                is_enrolled = False
            else:
                source_type = "auto"
                is_enrolled = False
            
            # Create new speaker in enhanced database
            new_speaker_id = self.enhanced_db.create_speaker(
                display_name=legacy_profile.name,
                source_type=source_type,
                is_enrolled=is_enrolled
            )
            
            # Transfer embeddings
            legacy_embeddings = self.legacy_manager.speaker_embeddings.get(legacy_id, [])
            legacy_confidences = self.legacy_manager.confidence_scores.get(legacy_id, [])
            
            for i, embedding in enumerate(legacy_embeddings):
                confidence = legacy_confidences[i] if i < len(legacy_confidences) else 0.8
                self.enhanced_db.add_embedding(new_speaker_id, embedding, confidence)
            
            # Update metadata
            if new_speaker_id in self.enhanced_db.speaker_records:
                record = self.enhanced_db.speaker_records[new_speaker_id]
                record.session_count = legacy_profile.session_count
                record.total_audio_seconds = legacy_profile.total_audio_seconds
                record.created_date = legacy_profile.created_date
                record.last_seen = legacy_profile.last_seen
            
            logger.debug(f"Migrated speaker {legacy_id} -> {new_speaker_id} ({legacy_profile.name})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate speaker {legacy_id}: {e}")
            return False
    
    async def _migrate_and_merge_speakers(self, display_name: str, speaker_list: List, dry_run: bool) -> bool:
        """Migrate multiple speakers with the same name by merging them"""
        try:
            if dry_run:
                logger.debug(f"Would merge {len(speaker_list)} speakers with name '{display_name}'")
                return True
            
            # Sort by embedding count (most embeddings first)
            speaker_list.sort(key=lambda x: len(x[1].embeddings), reverse=True)
            
            # Migrate the primary speaker (one with most embeddings)
            primary_id, primary_profile = speaker_list[0]
            success = await self._migrate_single_speaker(primary_id, primary_profile, dry_run=False)
            
            if not success:
                return False
            
            # Find the new speaker ID
            new_speaker_id = self.enhanced_db.find_speaker_by_name(display_name)
            if not new_speaker_id:
                logger.error(f"Could not find migrated speaker {display_name}")
                return False
            
            # Merge the remaining speakers into the primary one
            for secondary_id, secondary_profile in speaker_list[1:]:
                try:
                    # Add embeddings from secondary speaker
                    secondary_embeddings = self.legacy_manager.speaker_embeddings.get(secondary_id, [])
                    secondary_confidences = self.legacy_manager.confidence_scores.get(secondary_id, [])
                    
                    for i, embedding in enumerate(secondary_embeddings):
                        confidence = secondary_confidences[i] if i < len(secondary_confidences) else 0.8
                        self.enhanced_db.add_embedding(new_speaker_id, embedding, confidence)
                    
                    # Update metadata
                    if new_speaker_id in self.enhanced_db.speaker_records:
                        record = self.enhanced_db.speaker_records[new_speaker_id]
                        record.session_count += secondary_profile.session_count
                        record.total_audio_seconds += secondary_profile.total_audio_seconds
                        
                        # Use earliest creation date
                        if secondary_profile.created_date < record.created_date:
                            record.created_date = secondary_profile.created_date
                    
                    logger.debug(f"Merged speaker {secondary_id} into {new_speaker_id}")
                    
                except Exception as e:
                    logger.warning(f"Failed to merge speaker {secondary_id}: {e}")
                    continue
            
            logger.info(f"Successfully merged {len(speaker_list)} speakers with name '{display_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate and merge speakers '{display_name}': {e}")
            return False
    
    def _create_legacy_backup(self) -> Path:
        """Create a backup of the legacy speaker database"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path("speaker_data_backup_legacy") / timestamp
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy legacy database files
            if self.legacy_manager:
                # Backup profiles file
                if self.legacy_manager.profiles_file.exists():
                    import shutil
                    shutil.copy2(self.legacy_manager.profiles_file, backup_dir / "speaker_profiles.json")
                
                # Backup embeddings directory
                if self.legacy_manager.embeddings_dir.exists():
                    shutil.copytree(self.legacy_manager.embeddings_dir, backup_dir / "embeddings")
            
            logger.info(f"Created legacy database backup at {backup_dir}")
            return backup_dir
            
        except Exception as e:
            logger.error(f"Failed to create legacy backup: {e}")
            raise
    
    def _save_migration_log(self, migration_results: Dict):
        """Save migration log for future reference"""
        try:
            log_file = Path("migration_log.json")
            with open(log_file, 'w') as f:
                json.dump(migration_results, f, indent=2)
            logger.info(f"Migration log saved to {log_file}")
        except Exception as e:
            logger.warning(f"Failed to save migration log: {e}")
    
    async def get_enhanced_speaker_stats(self) -> Dict:
        """Get comprehensive speaker statistics from enhanced database"""
        try:
            speakers = self.enhanced_db.get_all_speakers()
            
            stats = {
                'total_speakers': len(speakers),
                'enrolled_speakers': sum(1 for s in speakers if s['is_enrolled']),
                'verified_speakers': sum(1 for s in speakers if s['is_verified']),
                'auto_speakers': sum(1 for s in speakers if s['source_type'] == 'auto'),
                'corrected_speakers': sum(1 for s in speakers if s['source_type'] == 'corrected'),
                'total_embeddings': sum(s['embedding_count'] for s in speakers),
                'avg_embeddings_per_speaker': 0,
                'speakers_by_confidence': {
                    'high_confidence': 0,  # > 0.8
                    'medium_confidence': 0,  # 0.5 - 0.8
                    'low_confidence': 0   # < 0.5
                },
                'migration_status': {
                    'migration_completed': self.migration_completed,
                    'migration_errors': len(self.migration_log)
                }
            }
            
            if len(speakers) > 0:
                stats['avg_embeddings_per_speaker'] = stats['total_embeddings'] / len(speakers)
                
                # Confidence distribution
                for speaker in speakers:
                    confidence = speaker['average_confidence']
                    if confidence > 0.8:
                        stats['speakers_by_confidence']['high_confidence'] += 1
                    elif confidence > 0.5:
                        stats['speakers_by_confidence']['medium_confidence'] += 1
                    else:
                        stats['speakers_by_confidence']['low_confidence'] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get enhanced speaker stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def enhanced_speaker_correction_feedback(self, 
                                                 corrections: Dict[str, str],
                                                 audio_file: str = None,
                                                 segments: List[Dict] = None) -> Dict:
        """
        Process speaker corrections using the enhanced database architecture
        
        Args:
            corrections: Mapping of old speaker IDs to new display names
            audio_file: Optional audio file for enhanced learning
            segments: Optional segment data for audio-based learning
            
        Returns:
            Detailed feedback processing results
        """
        try:
            # Process corrections using enhanced database
            feedback_results = self.enhanced_db.send_feedback_for_learning(corrections)
            
            # If audio file and segments provided, enhance with audio-based learning
            if audio_file and segments and self.legacy_manager:
                try:
                    # Use the reprocessing endpoint functionality
                    from server import reprocess_segment_embeddings
                    
                    # Filter segments that were corrected
                    corrected_segments = [
                        seg for seg in segments 
                        if seg.get('speaker') in corrections.values()
                    ]
                    
                    if corrected_segments:
                        audio_results = await self._reprocess_embeddings_for_segments(
                            audio_file, corrected_segments
                        )
                        feedback_results['audio_learning'] = audio_results
                        
                except Exception as e:
                    logger.warning(f"Audio-based learning failed: {e}")
                    feedback_results['audio_learning_error'] = str(e)
            
            return {
                'status': 'enhanced_feedback_processed',
                'enhanced_database_used': True,
                'feedback_results': feedback_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Enhanced speaker correction feedback failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _reprocess_embeddings_for_segments(self, audio_file: str, segments: List[Dict]) -> Dict:
        """Reprocess embeddings for segments using enhanced audio analysis"""
        try:
            if not self.legacy_manager:
                return {'error': 'No legacy manager available for audio processing'}
            
            import torchaudio
            waveform, sr = torchaudio.load(audio_file)
            
            results = {
                'processed_segments': 0,
                'failed_segments': 0,
                'updated_speakers': set()
            }
            
            for segment in segments:
                try:
                    start_time = segment.get('start', 0)
                    end_time = segment.get('end', start_time + 1)
                    speaker_name = segment.get('speaker', '')
                    
                    if not speaker_name or speaker_name == 'Unknown':
                        continue
                    
                    # Extract audio segment
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    
                    if start_sample >= waveform.shape[1] or end_sample > waveform.shape[1]:
                        continue
                    
                    segment_waveform = waveform[:, start_sample:end_sample]
                    duration = (end_sample - start_sample) / sr
                    
                    if duration < 0.5:  # Minimum duration
                        continue
                    
                    # Convert to numpy
                    if segment_waveform.dim() > 1:
                        audio_numpy = segment_waveform.mean(dim=0).cpu().numpy()
                    else:
                        audio_numpy = segment_waveform.cpu().numpy()
                    
                    # Update speaker model
                    success = self.legacy_manager.provide_correction_feedback(speaker_name, audio_numpy)
                    
                    if success:
                        results['processed_segments'] += 1
                        results['updated_speakers'].add(speaker_name)
                    else:
                        results['failed_segments'] += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to process segment: {e}")
                    results['failed_segments'] += 1
                    continue
            
            results['updated_speakers'] = list(results['updated_speakers'])
            return results
            
        except Exception as e:
            logger.error(f"Error reprocessing embeddings: {e}")
            return {'error': str(e)}
    
    async def validate_system_consistency(self) -> Dict:
        """
        Validate consistency between enhanced database and legacy system
        """
        try:
            validation_results = {
                'consistent': True,
                'issues': [],
                'recommendations': [],
                'enhanced_db_speakers': 0,
                'legacy_db_speakers': 0,
                'migration_needed': False
            }
            
            # Check enhanced database
            enhanced_speakers = self.enhanced_db.get_all_speakers()
            validation_results['enhanced_db_speakers'] = len(enhanced_speakers)
            
            # Check legacy database if available
            if self.legacy_manager:
                legacy_speakers = self.legacy_manager.speaker_profiles
                validation_results['legacy_db_speakers'] = len(legacy_speakers)
                
                # Check if migration is needed
                if len(legacy_speakers) > 0 and len(enhanced_speakers) == 0:
                    validation_results['migration_needed'] = True
                    validation_results['recommendations'].append("Migration from legacy database recommended")
                
                # Check for inconsistencies
                if len(legacy_speakers) > 0 and len(enhanced_speakers) > 0:
                    validation_results['issues'].append("Both legacy and enhanced databases contain speakers")
                    validation_results['recommendations'].append("Consider completing migration to enhanced database")
            
            # Check for common issues
            low_confidence_speakers = [
                s for s in enhanced_speakers 
                if s['average_confidence'] < 0.3
            ]
            
            if low_confidence_speakers:
                validation_results['issues'].append(f"{len(low_confidence_speakers)} speakers have low confidence")
                validation_results['recommendations'].append("Consider reprocessing low-confidence speakers")
            
            # Check for speakers with few embeddings
            sparse_speakers = [
                s for s in enhanced_speakers 
                if s['embedding_count'] < 3
            ]
            
            if sparse_speakers:
                validation_results['issues'].append(f"{len(sparse_speakers)} speakers have few embeddings")
                validation_results['recommendations'].append("Encourage more audio samples for sparse speakers")
            
            validation_results['consistent'] = len(validation_results['issues']) == 0
            
            return validation_results
            
        except Exception as e:
            logger.error(f"System validation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def cleanup_and_optimize(self) -> Dict:
        """
        Cleanup and optimize the enhanced speaker database
        """
        try:
            cleanup_results = {
                'speakers_before': len(self.enhanced_db.speaker_records),
                'speakers_removed': 0,
                'embeddings_pruned': 0,
                'optimizations_applied': []
            }
            
            # Remove speakers with no embeddings and no verified status
            speakers_to_remove = []
            for speaker_id, record in self.enhanced_db.speaker_records.items():
                embeddings = self.enhanced_db.speaker_embeddings.get(speaker_id, [])
                if (len(embeddings) == 0 and 
                    not record.is_verified and 
                    not record.is_enrolled and
                    record.source_type == "auto"):
                    speakers_to_remove.append(speaker_id)
            
            for speaker_id in speakers_to_remove:
                if self.enhanced_db.delete_speaker(speaker_id):
                    cleanup_results['speakers_removed'] += 1
            
            if cleanup_results['speakers_removed'] > 0:
                cleanup_results['optimizations_applied'].append("Removed empty auto-generated speakers")
            
            # TODO: Add more optimization strategies
            # - Merge very similar speakers
            # - Prune old/low-confidence embeddings
            # - Consolidate speaker names with minor variations
            
            cleanup_results['speakers_after'] = len(self.enhanced_db.speaker_records)
            
            logger.info(f"Cleanup completed: removed {cleanup_results['speakers_removed']} speakers")
            return cleanup_results
            
        except Exception as e:
            logger.error(f"Cleanup and optimization failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) 