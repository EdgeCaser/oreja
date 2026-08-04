"""
Dependency-light unit tests for the faster-whisper hallucination/artifact gate
in server.py: _hallucination_reason, _artifact_reason, _float_or_none,
_extract_words.

Pure functions over floats/strings/dicts - no waveform math - so they are
called directly. Heavy ML imports are stubbed via _stub_heavy_deps.
"""

import _stub_heavy_deps  # noqa: F401

from types import SimpleNamespace

from server import (
    _hallucination_reason,
    _artifact_reason,
    _float_or_none,
    _extract_words,
    NO_SPEECH_PROB_THRESHOLD,
    AVG_LOGPROB_THRESHOLD,
    COMPRESSION_RATIO_THRESHOLD,
)


def fw_segment(no_speech_prob=None, avg_logprob=None, compression_ratio=None, words=None, text=""):
    """Stand-in for a faster_whisper.transcribe.Segment - only attribute access is used."""
    return SimpleNamespace(
        no_speech_prob=no_speech_prob,
        avg_logprob=avg_logprob,
        compression_ratio=compression_ratio,
        words=words,
        text=text,
    )


def fw_word(start=None, end=None, text=None, probability=None):
    return SimpleNamespace(start=start, end=end, word=text, probability=probability)


# ---------------------------------------------------------------------------
# _float_or_none
# ---------------------------------------------------------------------------

class TestFloatOrNone:
    def test_none_stays_none(self):
        assert _float_or_none(None) is None

    def test_legit_zero_is_preserved(self):
        # The classic falsy-zero bug: 0.0 must NOT become None.
        assert _float_or_none(0.0) == 0.0

    def test_numeric_string_converts(self):
        assert _float_or_none("1.5") == 1.5

    def test_int_converts(self):
        assert _float_or_none(3) == 3.0

    def test_garbage_becomes_none(self):
        assert _float_or_none("not-a-number") is None
        assert _float_or_none(object()) is None


# ---------------------------------------------------------------------------
# _hallucination_reason
# ---------------------------------------------------------------------------

class TestHallucinationReason:
    def test_confident_no_speech_and_low_logprob_is_dropped(self):
        seg = fw_segment(
            no_speech_prob=NO_SPEECH_PROB_THRESHOLD + 0.1,
            avg_logprob=AVG_LOGPROB_THRESHOLD - 0.1,
        )
        assert _hallucination_reason(seg) == "no_speech"

    def test_high_no_speech_but_confident_logprob_is_kept(self):
        # no_speech_prob alone is not enough - the decode must ALSO be low
        # probability, otherwise a confident short utterance would be dropped.
        seg = fw_segment(
            no_speech_prob=NO_SPEECH_PROB_THRESHOLD + 0.1,
            avg_logprob=AVG_LOGPROB_THRESHOLD + 0.5,
        )
        assert _hallucination_reason(seg) is None

    def test_low_no_speech_prob_is_kept_regardless_of_logprob(self):
        seg = fw_segment(no_speech_prob=0.1, avg_logprob=-5.0)
        assert _hallucination_reason(seg) is None

    def test_repetitive_output_dropped(self):
        seg = fw_segment(compression_ratio=COMPRESSION_RATIO_THRESHOLD + 0.1)
        assert _hallucination_reason(seg) == "repetitive_output"

    def test_compression_ratio_at_threshold_is_kept(self):
        # Strictly-greater-than semantics: exactly at the threshold survives.
        seg = fw_segment(compression_ratio=COMPRESSION_RATIO_THRESHOLD)
        assert _hallucination_reason(seg) is None

    def test_missing_stats_never_raises_and_keeps_segment(self):
        seg = fw_segment()  # everything None
        assert _hallucination_reason(seg) is None

    def test_zero_no_speech_prob_is_not_treated_as_missing(self):
        # no_speech_prob=0.0 is a legitimate "definitely speech" reading, not
        # a missing value - must not accidentally short-circuit the gate.
        seg = fw_segment(no_speech_prob=0.0, avg_logprob=-5.0)
        assert _hallucination_reason(seg) is None


# ---------------------------------------------------------------------------
# _artifact_reason
# ---------------------------------------------------------------------------

class TestArtifactReason:
    def test_empty_text(self):
        assert _artifact_reason("", confident=False) == "empty_text"
        assert _artifact_reason("   ", confident=False) == "empty_text"

    def test_short_unconfident_text_is_dropped(self):
        assert _artifact_reason("Hi", confident=False) == "too_short"

    def test_short_confident_text_is_kept(self):
        # A high-probability decode of a short utterance ("OK.") is real speech.
        assert _artifact_reason("OK.", confident=True) is None

    def test_keyboard_artifacts_detected(self):
        assert _artifact_reason("あいうえおかきくけこ", confident=False) == "keyboard_artifacts"

    def test_character_repetition_detected(self):
        assert _artifact_reason("aaaaaaaaaa", confident=False) == "character_repetition"

    def test_normal_sentence_kept(self):
        assert _artifact_reason("This is a completely normal sentence.", confident=False) is None

    def test_short_confident_still_rejects_pure_artifacts(self):
        # "confident" only rescues short LEGITIMATE text from the length
        # check - it does not bypass the keyboard-artifact check.
        assert _artifact_reason("ああああああああああ", confident=True) == "keyboard_artifacts"


# ---------------------------------------------------------------------------
# _extract_words
# ---------------------------------------------------------------------------

class TestExtractWords:
    def test_no_words_attribute_returns_empty(self):
        seg = fw_segment(words=None)
        assert _extract_words(seg) == []

    def test_valid_words_extracted(self):
        seg = fw_segment(words=[
            fw_word(start=0.0, end=1.0, text="hello", probability=0.9),
            fw_word(start=1.0, end=2.0, text=" world", probability=0.8),
        ])
        words = _extract_words(seg)
        assert len(words) == 2
        assert words[0] == {"start": 0.0, "end": 1.0, "word": "hello", "probability": 0.9}

    def test_word_missing_start_is_skipped(self):
        seg = fw_segment(words=[fw_word(start=None, end=1.0, text="x", probability=0.5)])
        assert _extract_words(seg) == []

    def test_word_missing_text_is_skipped(self):
        seg = fw_segment(words=[fw_word(start=0.0, end=1.0, text=None, probability=0.5)])
        assert _extract_words(seg) == []

    def test_word_missing_text_empty_string_is_skipped(self):
        seg = fw_segment(words=[fw_word(start=0.0, end=1.0, text="", probability=0.5)])
        assert _extract_words(seg) == []

    def test_end_before_start_is_corrected(self):
        seg = fw_segment(words=[fw_word(start=2.0, end=1.0, text="oops", probability=0.5)])
        words = _extract_words(seg)
        assert len(words) == 1
        assert words[0]["end"] == words[0]["start"] == 2.0

    def test_missing_probability_defaults_to_zero(self):
        seg = fw_segment(words=[fw_word(start=0.0, end=1.0, text="x", probability=None)])
        assert _extract_words(seg)[0]["probability"] == 0.0

    def test_zero_start_word_is_preserved(self):
        seg = fw_segment(words=[fw_word(start=0.0, end=0.5, text="zero", probability=0.5)])
        words = _extract_words(seg)
        assert len(words) == 1
        assert words[0]["start"] == 0.0
