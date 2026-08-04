"""
Dependency-light unit tests for word-level speaker attribution in server.py:
extract_diarization_turns, speaker_at_time, _chunk_bounds,
_coalesce_speaker_groups, _smooth_speaker_groups, split_chunk_by_speaker,
merge_transcription_and_diarization, find_speaker_for_segment.

These are pure dict/float/list functions - no waveform math involved - so
they are exercised directly, without a TestClient or real audio. The heavy
ML imports server.py needs at module scope (torch, torchaudio, ...) are
stubbed via _stub_heavy_deps so this file can be collected and run on a
machine without them installed; on a fully provisioned machine the stub is a
no-op and the real packages are used instead.
"""

import _stub_heavy_deps  # noqa: F401 - installs sys.modules stubs before `import server`

import pytest

from server import (
    extract_diarization_turns,
    speaker_at_time,
    _chunk_bounds,
    _coalesce_speaker_groups,
    _smooth_speaker_groups,
    split_chunk_by_speaker,
    merge_transcription_and_diarization,
    find_speaker_for_segment,
)


class FakeTurn:
    """Stand-in for a pyannote.core.Segment: only .start/.end are read."""

    def __init__(self, start, end):
        self.start = start
        self.end = end


class FakeDiarization:
    """Stand-in for a pyannote.core.Annotation: only itertracks() is read."""

    def __init__(self, tracks):
        # tracks: list of (start, end, speaker_label)
        self._tracks = tracks

    def itertracks(self, yield_label=True):
        for start, end, speaker in self._tracks:
            yield FakeTurn(start, end), None, speaker


def word(start, end, text, probability=0.9):
    return {"start": start, "end": end, "word": text, "probability": probability}


# ---------------------------------------------------------------------------
# extract_diarization_turns
# ---------------------------------------------------------------------------

class TestExtractDiarizationTurns:
    def test_none_diarization_returns_empty(self):
        assert extract_diarization_turns(None) == []

    def test_basic_flatten_and_label(self):
        diarization = FakeDiarization([
            (0.0, 2.0, "SPEAKER_00"),
            (2.0, 4.0, "SPEAKER_01"),
        ])
        turns = extract_diarization_turns(diarization)
        assert turns == [
            {"start": 0.0, "end": 2.0, "speaker": "Speaker SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "speaker": "Speaker SPEAKER_01"},
        ]

    def test_sorts_out_of_order_tracks(self):
        diarization = FakeDiarization([
            (5.0, 6.0, "SPEAKER_01"),
            (0.0, 1.0, "SPEAKER_00"),
        ])
        turns = extract_diarization_turns(diarization)
        assert [t["start"] for t in turns] == [0.0, 5.0]

    def test_drops_zero_and_negative_duration_turns(self):
        diarization = FakeDiarization([
            (1.0, 1.0, "SPEAKER_00"),   # zero duration
            (3.0, 2.0, "SPEAKER_01"),   # end before start
            (0.0, 1.0, "SPEAKER_02"),   # valid
        ])
        turns = extract_diarization_turns(diarization)
        assert len(turns) == 1
        assert turns[0]["speaker"] == "Speaker SPEAKER_02"

    def test_broken_itertracks_returns_empty_not_raises(self):
        class Exploding:
            def itertracks(self, yield_label=True):
                raise RuntimeError("boom")

        assert extract_diarization_turns(Exploding()) == []


# ---------------------------------------------------------------------------
# speaker_at_time
# ---------------------------------------------------------------------------

class TestSpeakerAtTime:
    def test_empty_turns_returns_default(self):
        assert speaker_at_time([], 1.0, "DEFAULT") == "DEFAULT"

    def test_moment_inside_a_turn(self):
        turns = [
            {"start": 0.0, "end": 2.0, "speaker": "A"},
            {"start": 2.0, "end": 4.0, "speaker": "B"},
        ]
        assert speaker_at_time(turns, 1.0, "DEFAULT") == "A"
        assert speaker_at_time(turns, 3.0, "DEFAULT") == "B"

    def test_boundary_is_inclusive(self):
        turns = [{"start": 0.0, "end": 2.0, "speaker": "A"}]
        assert speaker_at_time(turns, 2.0, "DEFAULT") == "A"

    def test_moment_outside_uses_nearest_turn(self):
        turns = [
            {"start": 0.0, "end": 1.0, "speaker": "A"},
            {"start": 5.0, "end": 6.0, "speaker": "B"},
        ]
        # 1.4s from A's end, 3.6s from B's start -> nearer to A
        assert speaker_at_time(turns, 2.4, "DEFAULT") == "A"
        # nearer to B
        assert speaker_at_time(turns, 4.4, "DEFAULT") == "B"

    def test_moment_before_all_turns(self):
        turns = [{"start": 10.0, "end": 11.0, "speaker": "A"}]
        assert speaker_at_time(turns, 0.0, "DEFAULT") == "A"


# ---------------------------------------------------------------------------
# _chunk_bounds - the classic falsy-zero trap
# ---------------------------------------------------------------------------

class TestChunkBounds:
    def test_reads_timestamp_list(self):
        assert _chunk_bounds({"timestamp": [1.5, 3.5]}) == (1.5, 3.5)

    def test_zero_start_is_not_treated_as_missing(self):
        # A segment that legitimately starts at t=0.0 must not fall back to
        # "missing" just because 0.0 is falsy.
        assert _chunk_bounds({"timestamp": [0.0, 2.0]}) == (0.0, 2.0)
        assert _chunk_bounds({"start": 0.0, "end": 2.0}) == (0.0, 2.0)

    def test_falls_back_to_start_end_keys(self):
        assert _chunk_bounds({"start": 4.0, "end": 6.0}) == (4.0, 6.0)

    def test_partial_timestamp_falls_back_for_missing_side(self):
        # timestamp end is None -> should be picked up from "end" key instead.
        assert _chunk_bounds({"timestamp": [1.0, None], "end": 9.0}) == (1.0, 9.0)

    def test_nothing_present_returns_none_none(self):
        assert _chunk_bounds({}) == (None, None)

    def test_non_numeric_values_become_none(self):
        assert _chunk_bounds({"start": "not-a-number", "end": 2.0}) == (None, 2.0)


# ---------------------------------------------------------------------------
# _coalesce_speaker_groups / _smooth_speaker_groups
# ---------------------------------------------------------------------------

class TestCoalesceAndSmooth:
    def test_coalesce_merges_adjacent_same_speaker(self):
        groups = [
            {"speaker": "A", "words": [word(0, 1, "hi")]},
            {"speaker": "A", "words": [word(1, 2, "there")]},
            {"speaker": "B", "words": [word(2, 3, "hey")]},
        ]
        coalesced = _coalesce_speaker_groups(groups)
        assert len(coalesced) == 2
        assert [w["word"] for w in coalesced[0]["words"]] == ["hi", "there"]
        assert coalesced[1]["speaker"] == "B"

    def test_coalesce_empty_list(self):
        assert _coalesce_speaker_groups([]) == []

    def test_smooth_absorbs_tiny_single_word_flip(self):
        groups = [
            {"speaker": "A", "words": [word(0.0, 1.0, "one"), word(1.0, 2.0, "two")]},
            {"speaker": "B", "words": [word(2.0, 2.1, "uh")]},  # 100ms blip
            {"speaker": "A", "words": [word(2.1, 3.0, "three")]},
        ]
        smoothed = _smooth_speaker_groups(groups)
        # The B blip should be reassigned to A and coalesced into one run.
        assert len(smoothed) == 1
        assert smoothed[0]["speaker"] == "A"

    def test_smooth_keeps_real_turn_change(self):
        groups = [
            {"speaker": "A", "words": [word(0.0, 1.0, "one")]},
            {"speaker": "B", "words": [word(1.0, 3.0, "a longer utterance here")]},
            {"speaker": "A", "words": [word(3.0, 4.0, "back")]},
        ]
        smoothed = _smooth_speaker_groups(groups)
        # B's turn is 2s long and multi-word - not tiny - must survive.
        assert [g["speaker"] for g in smoothed] == ["A", "B", "A"]

    def test_smooth_short_list_returned_unchanged(self):
        groups = [{"speaker": "A", "words": [word(0, 1, "hi")]}]
        assert _smooth_speaker_groups(groups) == groups


# ---------------------------------------------------------------------------
# split_chunk_by_speaker
# ---------------------------------------------------------------------------

class TestSplitChunkBySpeaker:
    def test_splits_at_word_level_speaker_change(self):
        chunk = {
            "text": "hello there friend",
            "timestamp": [0.0, 3.0],
            "words": [
                word(0.0, 1.0, "hello"),
                word(1.0, 2.0, " there"),
                word(2.0, 3.0, " friend"),
            ],
        }
        turns = [
            # Boundary set so word midpoints (0.5, 1.5, 2.5) are unambiguous:
            # "hello"[0,1]->0.5 in A; "there"[1,2]->1.5 and "friend"[2,3]->2.5
            # both in B.
            {"start": 0.0, "end": 1.2, "speaker": "Speaker A"},
            {"start": 1.2, "end": 3.0, "speaker": "Speaker B"},
        ]
        segments = split_chunk_by_speaker(chunk, turns, diarization=object())

        assert len(segments) == 2
        assert segments[0]["speaker"] == "Speaker A"
        assert segments[0]["text"] == "hello"
        assert segments[1]["speaker"] == "Speaker B"
        assert segments[1]["text"] == "there friend"
        # every segment carries the fields the WPF client / downstream hook expects
        for seg in segments:
            assert set(["start", "end", "text", "speaker"]).issubset(seg.keys())

    def test_single_speaker_chunk_stays_one_segment(self):
        chunk = {
            "text": "all one speaker",
            "timestamp": [0.0, 2.0],
            "words": [word(0.0, 1.0, "all"), word(1.0, 2.0, " one speaker")],
        }
        turns = [{"start": 0.0, "end": 2.0, "speaker": "Speaker A"}]
        segments = split_chunk_by_speaker(chunk, turns)
        assert len(segments) == 1
        assert segments[0]["speaker"] == "Speaker A"

    def test_no_words_no_text_returns_empty(self):
        chunk = {"text": "", "timestamp": [0.0, 1.0], "words": []}
        assert split_chunk_by_speaker(chunk, [], diarization=None) == []

    def test_no_words_falls_back_to_segment_level_midpoint(self):
        chunk = {"text": "legacy shape", "timestamp": [0.0, 3.0], "words": []}
        turns = [
            {"start": 0.0, "end": 1.0, "speaker": "Speaker A"},
            {"start": 1.0, "end": 3.5, "speaker": "Speaker B"},
        ]
        segments = split_chunk_by_speaker(chunk, turns)
        assert len(segments) == 1
        # midpoint of [0, 3] is 1.5, unambiguously inside Speaker B's turn.
        assert segments[0]["speaker"] == "Speaker B"

    def test_no_turns_no_diarization_uses_default_speaker(self):
        chunk = {
            "text": "no diarization available",
            "timestamp": [0.0, 1.0],
            "words": [word(0.0, 1.0, "no diarization available")],
        }
        segments = split_chunk_by_speaker(chunk, [], diarization=None)
        assert len(segments) == 1
        assert segments[0]["speaker"] == "SPEAKER_00"

    def test_zero_start_word_timing_preserved(self):
        # A word starting at exactly t=0.0 must not be dropped/mis-handled by
        # the classic falsy-zero bug this pipeline was rewritten to avoid.
        chunk = {
            "text": "zero start",
            "timestamp": [0.0, 1.0],
            "words": [word(0.0, 1.0, "zero start")],
        }
        turns = [{"start": 0.0, "end": 1.0, "speaker": "Speaker A"}]
        segments = split_chunk_by_speaker(chunk, turns)
        assert segments[0]["start"] == 0.0


# ---------------------------------------------------------------------------
# merge_transcription_and_diarization
# ---------------------------------------------------------------------------

class TestMergeTranscriptionAndDiarization:
    def test_empty_transcription_returns_empty(self):
        assert merge_transcription_and_diarization({}, None) == []
        assert merge_transcription_and_diarization(None, None) == []

    def test_full_pipeline_two_speakers(self):
        transcription = {
            "chunks": [
                {
                    "timestamp": [0.0, 2.0],
                    "text": "hi there",
                    "words": [word(0.0, 1.0, "hi"), word(1.0, 2.0, " there")],
                },
            ],
            "text": "hi there",
        }
        diarization = FakeDiarization([(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")])

        segments = merge_transcription_and_diarization(transcription, diarization)

        assert len(segments) == 2
        assert segments[0]["speaker"] == "Speaker SPEAKER_00"
        assert segments[1]["speaker"] == "Speaker SPEAKER_01"

    def test_malformed_chunk_is_skipped_not_fatal(self):
        transcription = {"chunks": ["not-a-dict", {"timestamp": [0.0, 1.0], "text": "ok", "words": []}]}
        segments = merge_transcription_and_diarization(transcription, None)
        assert len(segments) == 1
        assert segments[0]["text"] == "ok"

    def test_no_chunks_key_falls_back_to_flat_text(self):
        transcription = {"text": "flat text only"}
        segments = merge_transcription_and_diarization(transcription, None, waveform=None, sample_rate=None)
        assert len(segments) == 1
        assert segments[0]["text"] == "flat text only"
        assert segments[0]["speaker"] == "SPEAKER_00"


# ---------------------------------------------------------------------------
# find_speaker_for_segment (legacy no-word-timings fallback)
# ---------------------------------------------------------------------------

class TestFindSpeakerForSegment:
    def test_midpoint_match(self):
        diarization = FakeDiarization([(0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")])
        assert find_speaker_for_segment(diarization, 0.5, 1.5) == "Speaker SPEAKER_00"

    def test_overlap_fallback_when_no_single_turn_covers_midpoint(self):
        # Midpoint of [0.9, 3.1] is 2.0, exactly on the SPEAKER_00/01 boundary,
        # which the first (midpoint) pass in the implementation catches.
        # Use a gap instead so the midpoint truly falls in silence.
        diarization = FakeDiarization([(0.0, 1.0, "SPEAKER_00"), (3.0, 4.0, "SPEAKER_01")])
        # midpoint of [0.5, 3.5] is 2.0 -> no turn covers it -> overlap analysis:
        # overlap with SPEAKER_00 is [0.5,1.0]=0.5s, overlap with SPEAKER_01 is [3.0,3.5]=0.5s (tie -> first wins per max())
        result = find_speaker_for_segment(diarization, 0.5, 3.5)
        assert result in ("Speaker SPEAKER_00", "Speaker SPEAKER_01")

    def test_none_timing_returns_unknown(self):
        diarization = FakeDiarization([(0.0, 1.0, "SPEAKER_00")])
        assert find_speaker_for_segment(diarization, None, 1.0) == "Unknown Speaker"

    def test_no_overlap_returns_unknown(self):
        diarization = FakeDiarization([(10.0, 11.0, "SPEAKER_00")])
        assert find_speaker_for_segment(diarization, 0.0, 1.0) == "Unknown Speaker"
