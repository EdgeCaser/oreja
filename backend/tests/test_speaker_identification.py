"""
Dependency-light unit tests for speaker_database_v2.EnhancedSpeakerDatabase's
identification math: cosine_similarity, score_speaker, identify_speaker
(threshold + auto-create), dimension-safety between mismatched embedding
sizes, and the reinforcement-learning tiering that keeps a long call from
flooding a speaker's bank.

speaker_database_v2.py imports nothing heavier than numpy, so this file needs
no stubbing at all - it runs anywhere pytest + numpy runs.
"""

import tempfile

import numpy as np
import pytest

from speaker_database_v2 import (
    EnhancedSpeakerDatabase,
    cosine_similarity,
    as_vector,
    AUTO_LEARN_CONFIDENCE_CAP,
    REINFORCE_SKIP_ABOVE,
)


def unit_vector(seed: int, dim: int = 32) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def nudge(vector: np.ndarray, amount: float, seed: int = 999) -> np.ndarray:
    """A vector close to `vector` but not identical - simulates the same
    speaker recorded again under slightly different conditions."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(*vector.shape).astype(np.float32)
    noisy = vector + amount * noise
    return (noisy / np.linalg.norm(noisy)).astype(np.float32)


@pytest.fixture
def db(tmp_path):
    return EnhancedSpeakerDatabase(data_dir=str(tmp_path / "speaker_data_v2"))


# ---------------------------------------------------------------------------
# cosine_similarity / as_vector
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_are_one(self):
        v = unit_vector(1)
        assert cosine_similarity(v, v.copy()) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors_are_zero(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_are_negative_one(self):
        v = unit_vector(2)
        assert cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-5)

    def test_zero_vector_is_unusable_not_a_crash(self):
        v = unit_vector(3)
        zero = np.zeros_like(v)
        assert cosine_similarity(v, zero) == 0.0

    def test_mismatched_shapes_are_zero_not_a_crash(self):
        a = unit_vector(1, dim=32)
        b = unit_vector(1, dim=64)
        assert cosine_similarity(a, b) == 0.0

    def test_none_input_is_zero(self):
        assert cosine_similarity(None, unit_vector(1)) == 0.0

    def test_nan_vector_is_unusable(self):
        bad = np.array([np.nan, 1.0, 2.0], dtype=np.float32)
        assert as_vector(bad) is None
        assert cosine_similarity(bad, unit_vector(1, dim=3)) == 0.0


# ---------------------------------------------------------------------------
# identify_speaker: threshold + auto-create
# ---------------------------------------------------------------------------

class TestIdentifySpeaker:
    def test_unusable_embedding_returns_none_unknown(self, db):
        speaker_id, name, confidence = db.identify_speaker(None)
        assert speaker_id is None
        assert name == "Unknown"
        assert confidence == 0.0

    def test_empty_database_auto_creates(self, db):
        probe = unit_vector(1)
        speaker_id, name, confidence = db.identify_speaker(probe)

        assert speaker_id is not None
        assert speaker_id in db.speaker_records
        assert name == "Speaker 1"
        assert confidence == 0.0  # nothing to compare against yet
        assert db.speaker_records[speaker_id].source_type == "auto"
        assert len(db.speaker_embeddings[speaker_id]) == 1

    def test_second_auto_create_gets_a_new_name(self, db):
        db.identify_speaker(unit_vector(1))
        _, name2, _ = db.identify_speaker(unit_vector(2))
        assert name2 == "Speaker 2"

    def test_repeat_probe_matches_existing_speaker(self, db):
        probe = unit_vector(42)
        first_id, _, _ = db.identify_speaker(probe)

        # A near-identical probe (same voice, another chunk) should match, not
        # create a second speaker.
        second_probe = nudge(probe, amount=0.02)
        second_id, name, confidence = db.identify_speaker(second_probe)

        assert second_id == first_id
        assert confidence >= db.similarity_threshold
        assert len(db.speaker_records) == 1

    def test_distinct_voice_creates_a_second_speaker(self, db):
        db.identify_speaker(unit_vector(1))
        second_id, _, _ = db.identify_speaker(unit_vector(2))
        assert len(db.speaker_records) == 2
        assert second_id in db.speaker_records

    def test_auto_create_false_returns_none_without_writing(self, db):
        speaker_id, name, confidence = db.identify_speaker(
            unit_vector(1), auto_create=False
        )
        assert speaker_id is None
        assert name == "Unknown"
        assert len(db.speaker_records) == 0

    def test_custom_threshold_overrides_default(self, db):
        probe = unit_vector(7)
        first_id, _, _ = db.identify_speaker(probe)

        slightly_off = nudge(probe, amount=0.02)
        # An impossible-to-clear threshold forces a "no match" even though the
        # default threshold would have matched.
        speaker_id, name, confidence = db.identify_speaker(
            slightly_off, threshold=1.5, auto_create=False
        )
        assert speaker_id is None
        assert confidence < 1.5

    def test_learn_false_does_not_grow_the_bank(self, db):
        probe = unit_vector(5)
        first_id, _, _ = db.identify_speaker(probe)
        bank_size_before = len(db.speaker_embeddings[first_id])

        db.identify_speaker(nudge(probe, 0.02), learn=False)
        assert len(db.speaker_embeddings[first_id]) == bank_size_before

    def test_save_false_does_not_write_to_disk(self, db):
        db.identify_speaker(unit_vector(1), save=False)
        assert not db.records_file.exists()


# ---------------------------------------------------------------------------
# Dimension safety: legacy 192-d vectors must coexist with 512-d ones
# ---------------------------------------------------------------------------

class TestDimensionSafety:
    def test_mismatched_dimensions_never_match(self, db):
        legacy_192 = unit_vector(1, dim=192)
        speaker_id = db.create_speaker("Legacy Speaker")
        db.add_embedding(speaker_id, legacy_192, confidence=0.9)

        # A 512-d probe (new pyannote embedding space) must not be scored
        # against the 192-d legacy bank at all.
        probe_512 = unit_vector(1, dim=512)
        score = db.score_speaker(speaker_id, probe_512)
        assert score == 0.0

    def test_new_speaker_created_for_incompatible_dimension(self, db):
        legacy_192 = unit_vector(1, dim=192)
        db.identify_speaker(legacy_192)
        assert len(db.speaker_records) == 1

        probe_512 = unit_vector(1, dim=512)
        speaker_id, name, confidence = db.identify_speaker(probe_512)
        # Cannot match the dimension-incompatible legacy speaker - a new one
        # is created instead of a bogus/crashing comparison.
        assert len(db.speaker_records) == 2
        assert confidence == 0.0

    def test_score_speaker_picks_matching_dimension_only(self, db):
        speaker_id = db.create_speaker("Mixed Speaker")
        v192 = unit_vector(3, dim=192)
        v512 = unit_vector(3, dim=512)
        db.add_embedding(speaker_id, v192, confidence=0.9)
        db.add_embedding(speaker_id, v512, confidence=0.9)

        # A 512-d probe should score against the 512-d entry, ignoring the
        # incompatible 192-d one, and match near-perfectly.
        score = db.score_speaker(speaker_id, v512)
        assert score == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Reinforcement tiering (confidence caps + REINFORCE_SKIP_ABOVE)
# ---------------------------------------------------------------------------

class TestReinforcementTiering:
    def test_auto_reinforcement_is_capped_below_enrollment_confidence(self, db):
        probe = unit_vector(9)
        speaker_id, _, _ = db.identify_speaker(probe)
        # Re-observe the same voice - this is a self-confirmed auto match.
        db.identify_speaker(nudge(probe, 0.01))

        stored_confidences = db.confidence_scores[speaker_id]
        assert all(c <= AUTO_LEARN_CONFIDENCE_CAP for c in stored_confidences[1:])

    def test_full_bank_skips_near_perfect_reinforcement(self, db):
        speaker_id = db.create_speaker("Frequent Caller")
        db.max_embeddings_per_speaker = 3
        probe = unit_vector(11)
        for i in range(3):
            db.add_embedding(speaker_id, nudge(probe, 0.01, seed=i), confidence=0.8)

        bank_size = len(db.speaker_embeddings[speaker_id])
        assert bank_size == 3

        # Bank is full; a near-perfect match (same probe, effectively the
        # centroid) teaches nothing new and must not be stored.
        assert db._should_reinforce(speaker_id, REINFORCE_SKIP_ABOVE + 0.01) is False
        # A mid-confidence match (different recording conditions) still earns
        # a slot even when the bank is nominally full... no: once truly full
        # (>= max) only sub-REINFORCE_SKIP_ABOVE scores are worth storing.
        assert db._should_reinforce(speaker_id, REINFORCE_SKIP_ABOVE - 0.1) is True

    def test_bank_with_room_always_reinforces(self, db):
        speaker_id = db.create_speaker("New Speaker")
        db.max_embeddings_per_speaker = 50
        assert db._should_reinforce(speaker_id, 0.99) is True


# ---------------------------------------------------------------------------
# rank_speakers
# ---------------------------------------------------------------------------

class TestRankSpeakers:
    def test_ranks_best_match_first(self, db):
        v1 = unit_vector(1)
        v2 = unit_vector(2)
        id1 = db.create_speaker("Alice")
        id2 = db.create_speaker("Bob")
        db.add_embedding(id1, v1, confidence=0.9)
        db.add_embedding(id2, v2, confidence=0.9)

        ranked = db.rank_speakers(nudge(v1, 0.02), top_k=5)
        assert ranked[0][0] == id1
        assert ranked[0][1] == "Alice"

    def test_top_k_limits_results(self, db):
        for i in range(5):
            speaker_id = db.create_speaker(f"Speaker{i}")
            db.add_embedding(speaker_id, unit_vector(i), confidence=0.9)

        ranked = db.rank_speakers(unit_vector(0), top_k=2)
        assert len(ranked) == 2

    def test_empty_database_returns_empty(self, db):
        assert db.rank_speakers(unit_vector(1)) == []
