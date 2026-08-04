"""
Enhanced Speaker Server Integration

Thin service layer over the v2 speaker database (speaker_database_v2). It exists
so the FastAPI handlers stay small: statistics, correction feedback, consistency
validation and maintenance live here.

There is no legacy database any more - the SpeechBrain OfflineSpeakerEmbeddingManager
and its migration path were removed along with speaker_embeddings.py. Voiceprint
extraction belongs to server.py (pyannote PretrainedSpeakerEmbedding); this module
only ever handles vectors that have already been extracted, so it imports nothing
heavier than the database itself.

Every method here is a plain (synchronous) ``def``: the bodies are disk I/O and
numpy, with nothing to await. They used to be ``async def``, which meant an
``await`` from a FastAPI handler ran the whole thing - including
send_feedback_for_learning()'s rewrite of speaker_records.json and every .npy -
directly on the asyncio event loop, stalling the live 5-second /transcribe
chunks. Call them with ``await asyncio.to_thread(...)`` from async handlers.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from fastapi import HTTPException

from speaker_database_v2 import EnhancedSpeakerDatabase, SpeakerRecord  # noqa: F401 (re-exported)

logger = logging.getLogger(__name__)


class EnhancedSpeakerServerIntegration:
    """Service layer around a single EnhancedSpeakerDatabase instance."""

    def __init__(self,
                 enhanced_db: Optional[EnhancedSpeakerDatabase] = None,
                 enhanced_db_path: str = "speaker_data_v2"):
        # Reuse the caller's database when given one. Two instances over the same
        # directory would each hold their own in-memory copy and clobber each
        # other's saves.
        self.enhanced_db = enhanced_db if enhanced_db is not None else EnhancedSpeakerDatabase(enhanced_db_path)

        logger.info(
            f"Enhanced Speaker Server Integration initialized "
            f"({len(self.enhanced_db.speaker_records)} speakers)"
        )

    def get_enhanced_speaker_stats(self) -> Dict:
        """Get comprehensive speaker statistics from the v2 database."""
        try:
            speakers = self.enhanced_db.get_all_speakers()

            stats = {
                'total_speakers': len(speakers),
                'enrolled_speakers': sum(1 for s in speakers if s['is_enrolled']),
                'verified_speakers': sum(1 for s in speakers if s['is_verified']),
                'auto_speakers': sum(1 for s in speakers if s['source_type'] == 'auto'),
                'corrected_speakers': sum(1 for s in speakers if s['source_type'] == 'corrected'),
                'total_embeddings': sum(int(s['embedding_count']) for s in speakers),
                'avg_embeddings_per_speaker': 0.0,
                'match_threshold': float(self.enhanced_db.similarity_threshold),
                'speakers_by_confidence': {
                    'high_confidence': 0,    # > 0.8
                    'medium_confidence': 0,  # 0.5 - 0.8
                    'low_confidence': 0      # < 0.5
                },
                # Retained for wire compatibility with older callers that read it.
                'migration_status': {
                    'migration_completed': True,
                    'migration_errors': 0
                }
            }

            if speakers:
                stats['avg_embeddings_per_speaker'] = stats['total_embeddings'] / len(speakers)

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

    def enhanced_speaker_correction_feedback(self,
                                                   corrections: Dict[str, str],
                                                   audio_file: str = None,
                                                   segments: List[Dict] = None,
                                                   embeddings: Dict[str, Sequence] = None) -> Dict:
        """
        Apply speaker corrections, learning voiceprints when they are supplied.

        Args:
            corrections: {old speaker id or session label -> corrected display name}
            audio_file: informational only; kept in the response for traceability.
                Extracting voiceprints from a file is server.py's job (it owns the
                embedding model) - pass the resulting vectors in `embeddings`.
            segments: optional segment dicts; entries carrying an "embedding" are
                learned from
            embeddings: optional {old speaker id/label -> [embedding, ...]}

        Returns:
            Feedback processing results
        """
        try:
            feedback_results = self.enhanced_db.send_feedback_for_learning(
                corrections,
                audio_segments=segments,
                embeddings=embeddings,
            )

            return {
                'status': 'enhanced_feedback_processed',
                'enhanced_database_used': True,
                'audio_file': audio_file,
                'feedback_results': feedback_results,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Enhanced speaker correction feedback failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def reprocess_stored_embeddings(self, segments: List[Dict] = None,
                                          audio_file: str = None) -> Dict:
        """
        Re-run identification and report what would change. Mutates nothing.
        """
        try:
            return self.enhanced_db.trigger_reprocessing(
                audio_file=audio_file, segments_to_reprocess=segments
            )
        except Exception as e:
            logger.error(f"Reprocessing audit failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def validate_system_consistency(self) -> Dict:
        """Report data-quality problems in the speaker database."""
        try:
            speakers = self.enhanced_db.get_all_speakers()

            validation_results = {
                'consistent': True,
                'issues': [],
                'recommendations': [],
                'enhanced_db_speakers': len(speakers),
                'legacy_db_speakers': 0,
                'migration_needed': False
            }

            duplicate_names = {}
            for speaker in speakers:
                key = speaker['display_name'].strip().lower()
                duplicate_names[key] = duplicate_names.get(key, 0) + 1
            duplicates = [name for name, count in duplicate_names.items() if count > 1]
            if duplicates:
                validation_results['issues'].append(
                    f"{len(duplicates)} display name(s) are used by more than one speaker"
                )
                validation_results['recommendations'].append(
                    "Run POST /speakers/cleanup to merge duplicate-name speakers"
                )

            empty_speakers = [s for s in speakers if s['embedding_count'] == 0]
            if empty_speakers:
                validation_results['issues'].append(
                    f"{len(empty_speakers)} speakers have no voiceprints and cannot be recognised"
                )
                validation_results['recommendations'].append(
                    "Enroll audio for name-only speakers, or clean them up"
                )

            low_confidence_speakers = [s for s in speakers if 0 < s['embedding_count'] and s['average_confidence'] < 0.3]
            if low_confidence_speakers:
                validation_results['issues'].append(
                    f"{len(low_confidence_speakers)} speakers have low average confidence"
                )
                validation_results['recommendations'].append("Consider reprocessing low-confidence speakers")

            sparse_speakers = [s for s in speakers if 0 < s['embedding_count'] < 3]
            if sparse_speakers:
                validation_results['issues'].append(
                    f"{len(sparse_speakers)} speakers have fewer than 3 voiceprints"
                )
                validation_results['recommendations'].append("Encourage more audio samples for sparse speakers")

            validation_results['consistent'] = len(validation_results['issues']) == 0

            return validation_results

        except Exception as e:
            logger.error(f"System validation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def cleanup_and_optimize(self, min_embeddings: int = 1) -> Dict:
        """Merge duplicate-name speakers and drop under-sampled auto speakers."""
        try:
            results = self.enhanced_db.cleanup(
                min_embeddings=min_embeddings, merge_duplicate_names=True
            )

            optimizations = []
            if results['duplicates_merged']:
                optimizations.append("Merged speakers sharing a display name")
            if results['speakers_removed']:
                optimizations.append("Removed under-sampled auto-generated speakers")

            results['optimizations_applied'] = optimizations
            results['embeddings_pruned'] = 0

            logger.info(f"Cleanup completed: removed {results['speakers_removed']} speakers")
            return results

        except Exception as e:
            logger.error(f"Cleanup and optimization failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
