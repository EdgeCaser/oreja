"""
Oreja - Enhanced Speaker Database v2

Single source of truth for persistent speaker identity:
  * immutable ``spk_<uuid>`` IDs with mutable display names
  * per-speaker embedding banks persisted as ``<data_dir>/embeddings/<id>.npy``
    (top-N by confidence retention)
  * cosine-similarity identification against BOTH the per-speaker mean embedding
    and the best single stored embedding - the higher of the two wins
  * rename / merge / enroll / correction-learning / cleanup operations

Embeddings are produced *outside* this module: server.py owns the pyannote
``PretrainedSpeakerEmbedding`` model and hands raw vectors in. This module
therefore imports nothing heavier than numpy, so it stays importable (and
testable) on machines without torch / pyannote installed.

Vectors of differing dimensionality never compare: a probe is only scored
against stored embeddings with the same dimension. That is what lets a database
carrying legacy 192-d ECAPA vectors coexist with new 512-d pyannote ones
instead of raising or producing nonsense similarities.
"""

import json
import logging
import os
import uuid
from collections import defaultdict
from dataclasses import dataclass, asdict, fields as dataclass_fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Cosine similarity at or above this counts as "the same person". Tuned for
# pyannote/embedding; override with OREJA_SPEAKER_THRESHOLD.
DEFAULT_MATCH_THRESHOLD = float(os.getenv("OREJA_SPEAKER_THRESHOLD", "0.72"))

# Retention cap per speaker (highest-confidence embeddings are kept).
DEFAULT_MAX_EMBEDDINGS = int(os.getenv("OREJA_MAX_EMBEDDINGS_PER_SPEAKER", "50"))

# Confidence tiers. Retention keeps the highest-confidence embeddings, so these
# values encode a precedence: what the USER told us always outranks what the
# machine confirmed to itself. Without this, a long call would flood a speaker's
# bank with 0.99 self-matches and evict the enrollment samples that made those
# matches possible in the first place.
ENROLLED_CONFIDENCE = 0.95      # user enrolled this audio under this name
CORRECTED_CONFIDENCE = 0.90     # user corrected a label on this audio
AUTO_SEED_CONFIDENCE = 0.70     # first sample of a newly auto-created speaker
AUTO_LEARN_CONFIDENCE_CAP = 0.85  # ceiling for self-confirmed reinforcement
# A match this strong on an already-full bank teaches nothing new, so it is not
# stored - which also means the steady state of a long call writes nothing.
REINFORCE_SKIP_ABOVE = 0.95


def as_vector(embedding) -> Optional[np.ndarray]:
    """
    Coerce anything embedding-shaped into a finite 1-D float32 vector.

    Returns None for values that cannot be compared (empty, ragged, NaN/inf),
    so callers never have to guard against exploding dot products.
    """
    if embedding is None:
        return None
    try:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        return None
    return vector


def cosine_similarity(first, second) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 for unusable/mismatched vectors."""
    left = as_vector(first)
    right = as_vector(second)
    if left is None or right is None or left.shape != right.shape:
        return 0.0

    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return float(np.clip(float(np.dot(left, right)) / (left_norm * right_norm), -1.0, 1.0))


def _unit(vector: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return None
    return vector / norm


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
    source_type: str = "auto"  # "auto", "enrolled", "corrected", "imported"

    def update_stats(self, embeddings: Sequence, confidences: Sequence):
        """Update statistics based on current embeddings"""
        self.embedding_count = len(embeddings)
        self.average_confidence = float(np.mean(confidences)) if len(confidences) else 0.0
        self.last_seen = datetime.now().isoformat()


class EnhancedSpeakerDatabase:
    """
    Persistent speaker registry + voiceprint bank.

    Identification contract (``identify_speaker``): given one embedding, return
    ``(speaker_id, display_name, confidence)``. When nothing clears the match
    threshold a new auto speaker is created holding that embedding, so the next
    time the same voice appears it is recognised.
    """

    def __init__(self, data_dir: str = "speaker_data_v2"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.records_file = self.data_dir / "speaker_records.json"
        self.embeddings_dir = self.data_dir / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        # In-memory data
        self.speaker_records: Dict[str, SpeakerRecord] = {}
        self.speaker_embeddings: Dict[str, List[np.ndarray]] = {}
        self.confidence_scores: Dict[str, List[float]] = {}

        # Indexing for fast lookups
        self.name_to_id_index: Dict[str, str] = {}  # lowercased display_name -> speaker_id

        # Derived-vector cache: speaker_id -> {dim: (mean_unit_vector, [unit_vectors])}
        self._vector_cache: Dict[str, Dict[int, Tuple[Optional[np.ndarray], List[np.ndarray]]]] = {}

        # Configuration
        self.similarity_threshold = DEFAULT_MATCH_THRESHOLD
        self.min_audio_length = 1.0
        self.max_embeddings_per_speaker = DEFAULT_MAX_EMBEDDINGS

        # Load existing data
        self._load_database()
        self._rebuild_indexes()

        logger.info(
            f"Enhanced Speaker Database initialized with {len(self.speaker_records)} speakers "
            f"(match threshold {self.similarity_threshold:.2f})"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _generate_speaker_id(self) -> str:
        """Generate a unique, immutable speaker ID"""
        return f"spk_{uuid.uuid4().hex[:12]}"

    def _load_database(self):
        """Load speaker records and embeddings from storage"""
        try:
            if self.records_file.exists():
                with open(self.records_file, 'r') as f:
                    records_data = json.load(f)

                known_fields = {field.name for field in dataclass_fields(SpeakerRecord)}
                for speaker_id, record_data in records_data.items():
                    # Ignore unknown keys so an older/newer on-disk schema cannot
                    # make the whole database fail to load.
                    filtered = {k: v for k, v in record_data.items() if k in known_fields}
                    filtered.setdefault("speaker_id", speaker_id)
                    self.speaker_records[speaker_id] = SpeakerRecord(**filtered)

            for speaker_id in self.speaker_records.keys():
                embeddings_file = self.embeddings_dir / f"{speaker_id}.npy"
                embeddings: List[np.ndarray] = []
                confidences: List[float] = []
                if embeddings_file.exists():
                    try:
                        payload = np.load(embeddings_file, allow_pickle=True).item()
                        embeddings = list(payload.get('embeddings', []))
                        confidences = [float(c) for c in payload.get('confidence_scores', [])]
                    except Exception as e:
                        logger.warning(f"Could not read embeddings for {speaker_id}: {e}")
                        embeddings, confidences = [], []

                # Keep the two lists the same length - a truncated confidence list
                # would otherwise silently misalign retention decisions.
                if len(confidences) < len(embeddings):
                    confidences.extend([0.8] * (len(embeddings) - len(confidences)))
                elif len(confidences) > len(embeddings):
                    confidences = confidences[:len(embeddings)]

                self.speaker_embeddings[speaker_id] = embeddings
                self.confidence_scores[speaker_id] = confidences

            logger.info(f"Loaded {len(self.speaker_records)} speaker records")

        except Exception as e:
            logger.error(f"Error loading speaker database: {e}")

    def _save_database(self):
        """Save speaker records and embeddings to storage"""
        try:
            for speaker_id, record in self.speaker_records.items():
                embeddings = self.speaker_embeddings.get(speaker_id, [])
                confidences = self.confidence_scores.get(speaker_id, [])
                record.update_stats(embeddings, confidences)

            records_data = {
                speaker_id: asdict(record)
                for speaker_id, record in self.speaker_records.items()
            }

            with open(self.records_file, 'w') as f:
                json.dump(records_data, f, indent=2)

            for speaker_id in list(self.speaker_records.keys()):
                embeddings = self.speaker_embeddings.get(speaker_id, [])
                embeddings_file = self.embeddings_dir / f"{speaker_id}.npy"
                if embeddings:
                    payload = {
                        'embeddings': embeddings,
                        'confidence_scores': self.confidence_scores.get(speaker_id, [])
                    }
                    np.save(embeddings_file, payload)
                elif embeddings_file.exists():
                    # A speaker whose bank was emptied must not keep a stale file
                    # that would be re-loaded on the next start.
                    embeddings_file.unlink()

            logger.debug("Speaker database saved successfully")

        except Exception as e:
            logger.error(f"Error saving speaker database: {e}")

    def save(self):
        """Public flush of records + embeddings to disk."""
        self._save_database()

    def _rebuild_indexes(self):
        """Rebuild internal indexes for fast lookups"""
        self.name_to_id_index.clear()
        for speaker_id, record in self.speaker_records.items():
            self.name_to_id_index[record.display_name.lower()] = speaker_id

    def _invalidate_cache(self, speaker_id: Optional[str] = None):
        if speaker_id is None:
            self._vector_cache.clear()
        else:
            self._vector_cache.pop(speaker_id, None)

    # ------------------------------------------------------------------
    # Read accessors
    # ------------------------------------------------------------------

    def get_speaker(self, speaker_id: str) -> Optional[SpeakerRecord]:
        """Return the record for a speaker ID, or None."""
        return self.speaker_records.get(speaker_id)

    def get_display_name(self, speaker_id: str) -> Optional[str]:
        record = self.speaker_records.get(speaker_id)
        return record.display_name if record else None

    def get_all_speakers(self) -> List[Dict]:
        """
        Get all speakers with their current statistics.

        Every numeric value is a plain Python scalar: numpy scalars are not JSON
        serializable and would 500 the /speakers endpoint the WPF client polls.
        """
        speakers = []

        for speaker_id, record in self.speaker_records.items():
            embeddings = self.speaker_embeddings.get(speaker_id, [])
            confidences = self.confidence_scores.get(speaker_id, [])

            speakers.append({
                'speaker_id': speaker_id,
                'display_name': record.display_name,
                'created_date': record.created_date,
                'last_seen': record.last_seen,
                'embedding_count': int(len(embeddings)),
                'average_confidence': float(np.mean(confidences)) if confidences else 0.0,
                'session_count': int(record.session_count),
                'total_audio_seconds': float(record.total_audio_seconds),
                'is_enrolled': bool(record.is_enrolled),
                'is_verified': bool(record.is_verified),
                'source_type': record.source_type
            })

        speakers.sort(key=lambda x: x['embedding_count'], reverse=True)
        return speakers

    def find_speaker_by_name(self, display_name: str) -> Optional[str]:
        """Find a speaker ID by display name (case-insensitive)."""
        if not display_name:
            return None
        return self.name_to_id_index.get(display_name.strip().lower())

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def create_speaker(self, display_name: str, source_type: str = "auto",
                       is_enrolled: bool = False, is_verified: bool = False) -> str:
        """
        Create a new speaker with immutable ID and mutable display name.

        Returns the unique speaker ID.
        """
        speaker_id = self._generate_speaker_id()

        record = SpeakerRecord(
            speaker_id=speaker_id,
            display_name=display_name,
            created_date=datetime.now().isoformat(),
            last_seen=datetime.now().isoformat(),
            is_enrolled=is_enrolled,
            is_verified=is_verified,
            source_type=source_type
        )

        self.speaker_records[speaker_id] = record
        self.speaker_embeddings[speaker_id] = []
        self.confidence_scores[speaker_id] = []
        self.name_to_id_index[display_name.lower()] = speaker_id
        self._invalidate_cache(speaker_id)

        logger.info(f"Created new speaker: {display_name} ({speaker_id})")
        return speaker_id

    def update_display_name(self, speaker_id: str, new_display_name: str) -> bool:
        """Update the display name of a speaker (ID remains immutable)."""
        if speaker_id not in self.speaker_records:
            return False

        new_display_name = (new_display_name or "").strip()
        if not new_display_name:
            return False

        old_name = self.speaker_records[speaker_id].display_name
        old_name_lower = old_name.lower()
        if self.name_to_id_index.get(old_name_lower) == speaker_id:
            del self.name_to_id_index[old_name_lower]

        self.speaker_records[speaker_id].display_name = new_display_name
        self.speaker_records[speaker_id].is_verified = True  # user-confirmed
        self.name_to_id_index[new_display_name.lower()] = speaker_id

        self._save_database()

        logger.info(f"Updated speaker name: {speaker_id} '{old_name}' -> '{new_display_name}'")
        return True

    def add_embedding(self, speaker_id: str, embedding, confidence: float = 1.0,
                      save: bool = False) -> bool:
        """
        Add one embedding to a speaker's bank.

        Unusable vectors are rejected rather than stored, so the bank never holds
        anything that would poison a later similarity computation.
        """
        if speaker_id not in self.speaker_records:
            return False

        vector = as_vector(embedding)
        if vector is None:
            logger.debug(f"Rejected unusable embedding for {speaker_id}")
            return False

        embeddings = self.speaker_embeddings.setdefault(speaker_id, [])
        confidences = self.confidence_scores.setdefault(speaker_id, [])

        embeddings.append(vector)
        confidences.append(float(confidence))

        if len(embeddings) > self.max_embeddings_per_speaker:
            # Keep the highest-confidence embeddings.
            indices = np.argsort(confidences)[-self.max_embeddings_per_speaker:]
            self.speaker_embeddings[speaker_id] = [embeddings[i] for i in indices]
            self.confidence_scores[speaker_id] = [confidences[i] for i in indices]

        record = self.speaker_records[speaker_id]
        record.update_stats(self.speaker_embeddings[speaker_id], self.confidence_scores[speaker_id])
        self._invalidate_cache(speaker_id)

        if save:
            self._save_database()

        return True

    def add_embeddings(self, speaker_id: str, embeddings: Sequence, confidence: float = 1.0,
                       save: bool = True) -> int:
        """Add several embeddings at once; returns how many were accepted."""
        added = 0
        for embedding in embeddings or []:
            if self.add_embedding(speaker_id, embedding, confidence=confidence, save=False):
                added += 1
        if added and save:
            self._save_database()
        return added

    def merge_speakers(self, source_speaker_id: str, target_speaker_id: str) -> bool:
        """
        Merge two speakers. The speaker with more samples always survives, so the
        richer voiceprint is never thrown away; the other record is deleted.
        """
        if (source_speaker_id not in self.speaker_records or
                target_speaker_id not in self.speaker_records):
            logger.warning(f"Cannot merge speakers: source={source_speaker_id}, target={target_speaker_id}")
            return False

        if source_speaker_id == target_speaker_id:
            return False

        source_embeddings = self.speaker_embeddings.get(source_speaker_id, [])
        target_embeddings = self.speaker_embeddings.get(target_speaker_id, [])

        if len(source_embeddings) > len(target_embeddings):
            source_speaker_id, target_speaker_id = target_speaker_id, source_speaker_id
            source_embeddings, target_embeddings = target_embeddings, source_embeddings
            logger.info("Swapped merge direction: speaker with more samples kept as target")

        source_record = self.speaker_records[source_speaker_id]
        target_record = self.speaker_records[target_speaker_id]
        source_confidences = self.confidence_scores.get(source_speaker_id, [])
        target_confidences = self.confidence_scores.get(target_speaker_id, [])

        merged_embeddings = list(target_embeddings) + list(source_embeddings)
        merged_confidences = list(target_confidences) + list(source_confidences)

        if len(merged_embeddings) > self.max_embeddings_per_speaker:
            indices = np.argsort(merged_confidences)[-self.max_embeddings_per_speaker:]
            merged_embeddings = [merged_embeddings[i] for i in indices]
            merged_confidences = [merged_confidences[i] for i in indices]

        self.speaker_embeddings[target_speaker_id] = merged_embeddings
        self.confidence_scores[target_speaker_id] = merged_confidences

        target_record.total_audio_seconds += source_record.total_audio_seconds
        target_record.session_count += source_record.session_count
        target_record.is_verified = target_record.is_verified or source_record.is_verified
        if source_record.is_enrolled:
            target_record.is_enrolled = True
            if target_record.source_type == "auto":
                target_record.source_type = "enrolled"

        target_record.update_stats(merged_embeddings, merged_confidences)

        old_source_name = source_record.display_name
        del self.speaker_records[source_speaker_id]
        self.speaker_embeddings.pop(source_speaker_id, None)
        self.confidence_scores.pop(source_speaker_id, None)

        source_name_lower = old_source_name.lower()
        if self.name_to_id_index.get(source_name_lower) == source_speaker_id:
            del self.name_to_id_index[source_name_lower]

        source_embeddings_file = self.embeddings_dir / f"{source_speaker_id}.npy"
        if source_embeddings_file.exists():
            source_embeddings_file.unlink()

        self._invalidate_cache(source_speaker_id)
        self._invalidate_cache(target_speaker_id)
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

        name_lower = record.display_name.lower()
        if self.name_to_id_index.get(name_lower) == speaker_id:
            del self.name_to_id_index[name_lower]

        del self.speaker_records[speaker_id]
        self.speaker_embeddings.pop(speaker_id, None)
        self.confidence_scores.pop(speaker_id, None)
        self._invalidate_cache(speaker_id)

        embeddings_file = self.embeddings_dir / f"{speaker_id}.npy"
        if embeddings_file.exists():
            embeddings_file.unlink()

        self._save_database()

        logger.info(f"Deleted speaker: {speaker_id} ({record.display_name})")
        return True

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------

    def _speaker_vectors(self, speaker_id: str, dim: int) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        """
        Unit-normalised (mean, singles) for one speaker restricted to `dim`.

        The mean is confidence-weighted over unit vectors, which keeps a single
        loud/long sample from dominating the centroid.
        """
        cached = self._vector_cache.get(speaker_id)
        if cached is not None and dim in cached:
            return cached[dim]

        units: List[np.ndarray] = []
        weights: List[float] = []
        embeddings = self.speaker_embeddings.get(speaker_id, [])
        confidences = self.confidence_scores.get(speaker_id, [])

        for index, raw in enumerate(embeddings):
            vector = as_vector(raw)
            if vector is None or vector.shape[0] != dim:
                continue
            unit = _unit(vector)
            if unit is None:
                continue
            units.append(unit)
            weight = confidences[index] if index < len(confidences) else 1.0
            weights.append(max(float(weight), 1e-3))

        mean_unit: Optional[np.ndarray] = None
        if units:
            weight_array = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
            mean_vector = np.sum(np.asarray(units, dtype=np.float32) * weight_array, axis=0)
            mean_unit = _unit(mean_vector)

        result = (mean_unit, units)
        self._vector_cache.setdefault(speaker_id, {})[dim] = result
        return result

    def score_speaker(self, speaker_id: str, embedding) -> float:
        """
        Similarity of one probe embedding to a stored speaker.

        Scored against the speaker's mean embedding AND its best single
        embedding; the maximum wins. The mean is robust for well-sampled
        speakers, the best-single rescues speakers whose bank spans several
        recording conditions and whose centroid is therefore washed out.
        """
        probe = as_vector(embedding)
        if probe is None:
            return 0.0
        probe_unit = _unit(probe)
        if probe_unit is None:
            return 0.0

        mean_unit, units = self._speaker_vectors(speaker_id, probe_unit.shape[0])
        if not units:
            return 0.0

        best = 0.0
        if mean_unit is not None:
            best = float(np.dot(probe_unit, mean_unit))
        for unit in units:
            score = float(np.dot(probe_unit, unit))
            if score > best:
                best = score

        return float(np.clip(best, -1.0, 1.0))

    def rank_speakers(self, embedding, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Return the best-matching speakers as (speaker_id, display_name, score)."""
        probe = as_vector(embedding)
        if probe is None:
            return []

        scored = []
        for speaker_id, record in self.speaker_records.items():
            score = self.score_speaker(speaker_id, probe)
            if score > 0.0:
                scored.append((speaker_id, record.display_name, score))

        scored.sort(key=lambda item: item[2], reverse=True)
        return scored[:top_k] if top_k else scored

    def _should_reinforce(self, speaker_id: str, score: float) -> bool:
        """
        Decide whether a matched observation is worth storing.

        While the bank has room, everything is worth keeping. Once it is full,
        a near-perfect match adds no information - it is the same voice under the
        same conditions we already have - so it is dropped. Only mid-confidence
        matches (a new room, a cold, a different mic) still earn a slot.
        """
        bank = self.speaker_embeddings.get(speaker_id, [])
        if len(bank) < self.max_embeddings_per_speaker:
            return True
        return score < REINFORCE_SKIP_ABOVE

    def _next_auto_name(self) -> str:
        index = 1
        while f"speaker {index}" in self.name_to_id_index:
            index += 1
        return f"Speaker {index}"

    def identify_speaker(self, embedding, threshold: Optional[float] = None,
                         auto_create: bool = True, learn: bool = True,
                         save: bool = True) -> Tuple[Optional[str], str, float]:
        """
        Identify one embedding against the database.

        Args:
            embedding: probe vector (any array-like; coerced to 1-D float32)
            threshold: cosine cut-off; defaults to OREJA_SPEAKER_THRESHOLD (0.72)
            auto_create: create a new auto speaker when nothing matches
            learn: store the probe on the matched/created speaker
            save: flush to disk when anything changed

        Returns:
            (speaker_id, display_name, confidence). speaker_id is None and the
            name is "Unknown" only when the embedding itself is unusable or when
            nothing matched and auto_create is False.
        """
        probe = as_vector(embedding)
        if probe is None:
            return None, "Unknown", 0.0

        cutoff = self.similarity_threshold if threshold is None else float(threshold)

        best_id: Optional[str] = None
        best_score = 0.0
        for speaker_id in self.speaker_records:
            score = self.score_speaker(speaker_id, probe)
            if score > best_score:
                best_score = score
                best_id = speaker_id

        if best_id is not None and best_score >= cutoff:
            record = self.speaker_records[best_id]
            record.last_seen = datetime.now().isoformat()

            if learn and self._should_reinforce(best_id, best_score):
                # Store the observation that produced the match, capped so it can
                # never outrank enrolled/corrected samples during retention.
                added = self.add_embedding(
                    best_id, probe,
                    confidence=min(float(best_score), AUTO_LEARN_CONFIDENCE_CAP),
                    save=False,
                )
                if added and save:
                    self._save_database()

            logger.debug(f"Matched speaker {record.display_name} ({best_id}) at {best_score:.3f}")
            return best_id, record.display_name, float(best_score)

        if not auto_create:
            return None, "Unknown", float(best_score)

        new_id = self.create_speaker(self._next_auto_name(), source_type="auto")
        self.add_embedding(new_id, probe, confidence=AUTO_SEED_CONFIDENCE, save=False)
        self.speaker_records[new_id].session_count = 1
        if save:
            self._save_database()

        display_name = self.speaker_records[new_id].display_name
        logger.info(
            f"No speaker above {cutoff:.2f} (best {best_score:.3f}); "
            f"created {display_name} ({new_id})"
        )
        return new_id, display_name, float(best_score)

    # ------------------------------------------------------------------
    # Enrollment and correction learning
    # ------------------------------------------------------------------

    def enroll_speaker(self, display_name: str, embeddings: Optional[Sequence] = None,
                       confidence: float = ENROLLED_CONFIDENCE,
                       source_type: str = "enrolled") -> str:
        """
        Enroll (or top up) a named speaker with known-good embeddings.

        An existing speaker with the same display name is reused and promoted to
        enrolled rather than duplicated - duplicate name records are exactly the
        problem this database exists to prevent.
        """
        display_name = (display_name or "").strip()
        if not display_name:
            raise ValueError("Speaker name cannot be empty")

        speaker_id = self.find_speaker_by_name(display_name)
        if speaker_id is None:
            speaker_id = self.create_speaker(
                display_name, source_type=source_type, is_enrolled=True, is_verified=True
            )
        else:
            record = self.speaker_records[speaker_id]
            record.is_enrolled = True
            record.is_verified = True
            if record.source_type == "auto":
                record.source_type = source_type

        added = self.add_embeddings(speaker_id, embeddings or [], confidence=confidence, save=False)
        self._save_database()

        logger.info(f"Enrolled '{display_name}' ({speaker_id}) with {added} embedding(s)")
        return speaker_id

    def apply_name_correction(self, old_speaker_id: str, new_display_name: str,
                              embeddings: Optional[Sequence] = None,
                              confidence: float = CORRECTED_CONFIDENCE) -> Dict:
        """
        Apply a user speaker-name correction, learning from it when audio is given.

        Handles the three cases the UI can produce:
          * ``old_speaker_id`` is a real v2 ID -> rename it, or merge it into an
            existing speaker that already owns the target name
          * ``old_speaker_id`` is an anonymous per-session label
            ("Speaker SPEAKER_00", "AUTO_SPEAKER_001", ...) -> attach to the
            existing speaker with that name, or create one
          * the target name is already taken -> merge instead of duplicating

        Returns a dict describing what happened (action, speaker_id,
        display_name, embeddings_added).
        """
        new_display_name = (new_display_name or "").strip()
        if not new_display_name:
            raise ValueError("Speaker name cannot be empty")

        existing_id = self.find_speaker_by_name(new_display_name)
        action = "noop"
        target_id: Optional[str] = None

        if old_speaker_id in self.speaker_records:
            if existing_id and existing_id != old_speaker_id:
                if self.merge_speakers(old_speaker_id, existing_id):
                    # merge_speakers keeps whichever record had more samples.
                    target_id = existing_id if existing_id in self.speaker_records else old_speaker_id
                    action = "merged"
                    if self.speaker_records[target_id].display_name != new_display_name:
                        self.update_display_name(target_id, new_display_name)
                else:
                    target_id = existing_id
                    action = "merge_failed"
            else:
                target_id = old_speaker_id
                if self.update_display_name(old_speaker_id, new_display_name):
                    action = "renamed"
        elif existing_id:
            target_id = existing_id
            action = "matched"
            record = self.speaker_records[target_id]
            record.is_verified = True
        else:
            target_id = self.create_speaker(
                new_display_name, source_type="corrected", is_verified=True
            )
            action = "created"

        added = 0
        if target_id and embeddings:
            added = self.add_embeddings(target_id, embeddings, confidence=confidence, save=False)

        self._save_database()

        result = {
            "action": action,
            "speaker_id": target_id,
            "display_name": self.speaker_records[target_id].display_name if target_id in self.speaker_records else new_display_name,
            "embeddings_added": added,
            "old_speaker_id": old_speaker_id,
        }
        logger.info(f"Name correction {old_speaker_id} -> '{new_display_name}': {result['action']}")
        return result

    def save_transcription_with_corrections(self, transcription_data: Dict,
                                            speaker_corrections: Dict[str, str],
                                            output_file: str = None) -> str:
        """
        Save transcription with corrected speaker names WITHOUT sending feedback.
        This is the "Save Document" operation that just creates the file.
        """
        try:
            corrected_data = dict(transcription_data or {})

            for segment in corrected_data.get('segments', []):
                old_speaker = segment.get('speaker', '')
                if old_speaker in speaker_corrections:
                    segment['speaker'] = speaker_corrections[old_speaker]

            corrected_data['metadata'] = corrected_data.get('metadata', {})
            corrected_data['metadata']['corrections_applied'] = len(speaker_corrections)
            corrected_data['metadata']['correction_timestamp'] = datetime.now().isoformat()

            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"transcription_corrected_{timestamp}.json"

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(corrected_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved corrected transcription to {output_file} with {len(speaker_corrections)} corrections")
            return output_file

        except Exception as e:
            logger.error(f"Error saving corrected transcription: {e}")
            raise

    def send_feedback_for_learning(self, speaker_corrections: Dict[str, str],
                                   audio_segments: List[Dict] = None,
                                   embeddings: Dict[str, List] = None) -> Dict:
        """
        Apply speaker corrections to the database, learning voiceprints when given.

        Args:
            speaker_corrections: {old speaker id or label -> corrected display name}
            audio_segments: optional segment dicts; any entry carrying an
                "embedding" key (and a "speaker" matching a correction key) is
                learned from
            embeddings: optional {old speaker id/label -> [embedding, ...]}

        Returns counts of what was created/merged/renamed plus embeddings_learned.
        """
        results = {
            'processed_corrections': 0,
            'speakers_created': 0,
            'speakers_merged': 0,
            'speakers_renamed': 0,
            'speakers_matched': 0,
            'embeddings_learned': 0,
            'errors': []
        }

        # Fold any segment-carried embeddings into the per-label map.
        vectors_by_label: Dict[str, List] = defaultdict(list)
        for label, vectors in (embeddings or {}).items():
            vectors_by_label[label].extend(vectors or [])
        for segment in audio_segments or []:
            if not isinstance(segment, dict):
                continue
            vector = segment.get('embedding')
            label = segment.get('speaker') or segment.get('speaker_id')
            if vector is not None and label:
                vectors_by_label[label].append(vector)

        for old_speaker_id, new_display_name in (speaker_corrections or {}).items():
            try:
                outcome = self.apply_name_correction(
                    old_speaker_id,
                    new_display_name,
                    embeddings=vectors_by_label.get(old_speaker_id),
                )
                action = outcome['action']
                if action == 'created':
                    results['speakers_created'] += 1
                elif action == 'merged':
                    results['speakers_merged'] += 1
                elif action == 'renamed':
                    results['speakers_renamed'] += 1
                elif action == 'matched':
                    results['speakers_matched'] += 1
                results['embeddings_learned'] += outcome['embeddings_added']
                results['processed_corrections'] += 1

            except Exception as e:
                error_msg = f"Error processing correction {old_speaker_id} -> {new_display_name}: {e}"
                results['errors'].append(error_msg)
                logger.error(error_msg)

        self._save_database()
        logger.info(f"Feedback processing complete: {results}")
        return results

    def trigger_reprocessing(self, audio_file: str = None,
                             segments_to_reprocess: List[Dict] = None) -> Dict:
        """
        Re-run identification over embeddings and report what actually changed.

        Two modes, both returning real numbers (no estimates):
          * ``segments_to_reprocess`` entries carrying an "embedding" are
            re-identified against the current database; a segment counts as an
            improvement when the winning speaker differs from its current label
            or its confidence rises.
          * with no segments, every *stored* embedding is re-scored against the
            database as a self-consistency audit, reporting how many now look
            more like a different speaker than the one holding them.

        Nothing is mutated: this is a report, and acting on it is the caller's
        decision.
        """
        results = {
            'mode': 'segments' if segments_to_reprocess else 'stored_embeddings',
            'audio_file': audio_file,
            'segments_reprocessed': 0,
            'improvements_found': 0,
            'average_confidence_improvement': 0.0,
            'reassignments': []
        }

        improvements: List[float] = []

        if segments_to_reprocess:
            for segment in segments_to_reprocess:
                if not isinstance(segment, dict):
                    continue
                probe = as_vector(segment.get('embedding'))
                if probe is None:
                    continue

                results['segments_reprocessed'] += 1
                speaker_id, display_name, score = self.identify_speaker(
                    probe, auto_create=False, learn=False, save=False
                )
                current_name = segment.get('speaker') or segment.get('current_speaker')
                current_confidence = float(segment.get('confidence', 0.0) or 0.0)

                if speaker_id is None:
                    continue
                if display_name != current_name or score > current_confidence:
                    improvements.append(score - current_confidence)
                    results['improvements_found'] += 1
                    results['reassignments'].append({
                        'segment_start': segment.get('start'),
                        'segment_end': segment.get('end'),
                        'old_speaker': current_name,
                        'new_speaker': display_name,
                        'new_speaker_id': speaker_id,
                        'confidence': round(float(score), 4),
                    })
        else:
            for speaker_id in list(self.speaker_records.keys()):
                for raw in list(self.speaker_embeddings.get(speaker_id, [])):
                    probe = as_vector(raw)
                    if probe is None:
                        continue
                    results['segments_reprocessed'] += 1

                    own_score = self.score_speaker(speaker_id, probe)
                    best_other_id = None
                    best_other_score = 0.0
                    for other_id in self.speaker_records:
                        if other_id == speaker_id:
                            continue
                        score = self.score_speaker(other_id, probe)
                        if score > best_other_score:
                            best_other_score = score
                            best_other_id = other_id

                    if best_other_id is not None and best_other_score > own_score:
                        improvements.append(best_other_score - own_score)
                        results['improvements_found'] += 1
                        results['reassignments'].append({
                            'speaker_id': speaker_id,
                            'current_speaker': self.speaker_records[speaker_id].display_name,
                            'suggested_speaker_id': best_other_id,
                            'suggested_speaker': self.speaker_records[best_other_id].display_name,
                            'current_score': round(float(own_score), 4),
                            'suggested_score': round(float(best_other_score), 4),
                        })

        if improvements:
            results['average_confidence_improvement'] = float(np.mean(improvements))

        # Keep the payload bounded - this can be called on a large database.
        results['reassignments'] = results['reassignments'][:100]

        logger.info(
            f"Reprocessing audit ({results['mode']}): {results['segments_reprocessed']} scored, "
            f"{results['improvements_found']} would change"
        )
        return results

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup(self, min_embeddings: int = 1, merge_duplicate_names: bool = True) -> Dict:
        """
        Housekeeping: fold exact duplicate display names together and drop
        under-sampled auto speakers.

        Enrolled and user-verified speakers are never deleted, however few
        embeddings they hold - a speaker the user named on purpose is data, not
        garbage.
        """
        results = {
            'speakers_before': len(self.speaker_records),
            'duplicates_merged': 0,
            'speakers_removed': 0,
            'removed_speakers': [],
            'speakers_after': 0,
        }

        if merge_duplicate_names:
            by_name: Dict[str, List[str]] = defaultdict(list)
            for speaker_id, record in self.speaker_records.items():
                by_name[record.display_name.strip().lower()].append(speaker_id)

            for name, ids in by_name.items():
                if len(ids) < 2:
                    continue
                # Merge into the best-sampled record; merge_speakers picks the
                # survivor, so re-read the survivor each round.
                ids.sort(key=lambda sid: len(self.speaker_embeddings.get(sid, [])), reverse=True)
                survivor = ids[0]
                for duplicate in ids[1:]:
                    if duplicate not in self.speaker_records or survivor not in self.speaker_records:
                        continue
                    if self.merge_speakers(duplicate, survivor):
                        results['duplicates_merged'] += 1
                        if survivor not in self.speaker_records:
                            survivor = duplicate

        to_remove = []
        for speaker_id, record in self.speaker_records.items():
            if record.is_enrolled or record.is_verified:
                continue
            if record.source_type not in ("auto", "imported"):
                continue
            if len(self.speaker_embeddings.get(speaker_id, [])) < max(int(min_embeddings), 0):
                to_remove.append((speaker_id, record.display_name))

        for speaker_id, display_name in to_remove:
            if self.delete_speaker(speaker_id):
                results['speakers_removed'] += 1
                results['removed_speakers'].append({'speaker_id': speaker_id, 'display_name': display_name})

        self._save_database()
        results['speakers_after'] = len(self.speaker_records)

        logger.info(
            f"Cleanup: merged {results['duplicates_merged']} duplicate-name speakers, "
            f"removed {results['speakers_removed']} under-sampled auto speakers"
        )
        return results

    def reset_auto_speakers(self) -> Dict:
        """
        Destructive reset that keeps identity the user actually established.

        Every auto-created (and never enrolled, never verified) speaker is
        deleted; enrolled/verified/corrected speakers survive. This is what the
        analytics GUI's "reset database" button maps to - a full wipe would
        destroy enrollments that cannot be recovered.
        """
        results = {
            'speakers_before': len(self.speaker_records),
            'speakers_removed': 0,
            'speakers_kept': 0,
            'removed_speakers': [],
        }

        for speaker_id, record in list(self.speaker_records.items()):
            if record.is_enrolled or record.is_verified or record.source_type in ("enrolled", "corrected"):
                results['speakers_kept'] += 1
                continue
            display_name = record.display_name
            if self.delete_speaker(speaker_id):
                results['speakers_removed'] += 1
                results['removed_speakers'].append({'speaker_id': speaker_id, 'display_name': display_name})

        self._save_database()
        results['speakers_after'] = len(self.speaker_records)

        logger.warning(
            f"Reset removed {results['speakers_removed']} auto speakers, "
            f"kept {results['speakers_kept']} enrolled/verified speakers"
        )
        return results
