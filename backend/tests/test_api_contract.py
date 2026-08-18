"""
Frozen HTTP contract tests for the two endpoints the shipped WPF client
(App.xaml.cs) actually depends on:

  * POST /transcribe  -> {"segments": [{start, end, text, speaker}, ...],
                          "full_text": str, ...}
  * GET  /speakers    -> {"total_speakers": int,
                          "speakers": [{id, name, embedding_count:int}, ...]}

These assertions used to live only in tests/test_api_endpoints.py, which was
built on the removed speaker_embedding_manager architecture and has been
deleted. Nothing else in the suite covers the wire shape, so a rename or a
type change on either endpoint would ship silently and break the client.

Everything heavy is faked:
  * faster-whisper -> FakeWhisperModel yielding FakeSegment/FakeWord objects,
    so the real _transcribe_sync / hallucination gate / word extraction run.
  * pyannote       -> FakeDiarization exposing itertracks(yield_label=True),
    so the real extract_diarization_turns / word attribution run.
  * torchaudio     -> server.load_audio_from_bytes is monkeypatched to return a
    FakeWaveform, since decoding real bytes needs the actual torchaudio wheel.

Speaker identification is left OFF (server.embedding_model is None), which is
exactly the shipped fallback path: segments keep their per-request
"Speaker SPEAKER_XX" diarization labels. That is the contract being frozen -
`speaker` is always a non-empty string.
"""

import _stub_heavy_deps  # noqa: F401  (must precede `import server`)

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

# `import server` below needs the backend/ directory on sys.path. That's
# already true when pytest is invoked from backend/ (cwd is auto-prepended)
# or with PYTHONPATH=.../backend set, but not when running from the repo root
# with neither - pytest's own import-mode insertion only adds tests/ (this
# file's directory has no __init__.py). Insert it explicitly so this module
# collects the same way from the repo root and from backend/, matching its
# sibling test files (e.g. test_enhanced_speaker_database.py).
sys.path.insert(0, str(Path(__file__).parent.parent))

import server
from speaker_database_v2 import EnhancedSpeakerDatabase


SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeWaveform:
    """Minimal stand-in for a (1, N) float32 torch tensor."""

    def __init__(self, samples: np.ndarray, channels: int = 1):
        self._samples = samples.astype(np.float32)
        self.shape = (channels, len(self._samples))

    def dim(self):
        return 2

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def numpy(self):
        return self._samples.reshape(1, -1)

    def numel(self):
        return self._samples.size


class FakeWord:
    def __init__(self, start, end, word, probability=0.9):
        self.start = start
        self.end = end
        self.word = word
        self.probability = probability


class FakeSegment:
    """Shaped like a faster-whisper Segment (only the attributes server.py reads)."""

    def __init__(self, start, end, text, words,
                 avg_logprob=-0.2, no_speech_prob=0.01, compression_ratio=1.4):
        self.start = start
        self.end = end
        self.text = text
        self.words = words
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        self.compression_ratio = compression_ratio


class FakeInfo:
    language = "en"
    language_probability = 0.99


class FakeWhisperModel:
    def __init__(self, segments):
        self._segments = segments

    def transcribe(self, audio_array, **kwargs):
        # faster-whisper returns (generator, info); the generator is what decodes.
        return iter(self._segments), FakeInfo()


class FakeTurn:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class FakeDiarization:
    """Shaped like a pyannote Annotation for extract_diarization_turns()."""

    def __init__(self, turns):
        self._turns = turns  # [(start, end, label), ...]

    def itertracks(self, yield_label=False):
        for start, end, label in self._turns:
            if yield_label:
                yield FakeTurn(start, end), None, label
            else:
                yield FakeTurn(start, end), None


def speechlike_samples(seconds: float = 4.0) -> np.ndarray:
    """Non-silent audio so run_transcription's RMS fast-path does not skip it."""
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(server.app)


@pytest.fixture
def wired_pipeline(monkeypatch):
    """
    Wire server.py's module globals to the fakes above and hand back the
    two-speaker script the assertions expect.
    """
    waveform = FakeWaveform(speechlike_samples(4.0))

    segments = [
        FakeSegment(
            0.0, 1.6, " Hello there, how are you?",
            [
                FakeWord(0.0, 0.4, " Hello"),
                FakeWord(0.4, 0.8, " there,"),
                FakeWord(0.8, 1.1, " how"),
                FakeWord(1.1, 1.35, " are"),
                FakeWord(1.35, 1.6, " you?"),
            ],
        ),
        FakeSegment(
            2.0, 3.6, " I am doing well thanks.",
            [
                FakeWord(2.0, 2.3, " I"),
                FakeWord(2.3, 2.6, " am"),
                FakeWord(2.6, 2.9, " doing"),
                FakeWord(2.9, 3.2, " well"),
                FakeWord(3.2, 3.6, " thanks."),
            ],
        ),
    ]

    diarization = FakeDiarization([
        (0.0, 1.8, "SPEAKER_00"),
        (1.9, 3.8, "SPEAKER_01"),
    ])

    monkeypatch.setattr(
        server, "load_audio_from_bytes",
        lambda data: (waveform, SAMPLE_RATE), raising=False
    )
    monkeypatch.setattr(server, "whisper_model", FakeWhisperModel(segments), raising=False)
    monkeypatch.setattr(server, "diarization_pipeline", object(), raising=False)
    monkeypatch.setattr(
        server, "_diarize_sync",
        lambda wf, sr, max_speakers=None: diarization, raising=False
    )
    # No embedding model / database -> identify_speakers_hook is a pass-through
    # and the diarization labels survive. This is the shipped fallback path.
    monkeypatch.setattr(server, "embedding_model", None, raising=False)
    monkeypatch.setattr(server, "enhanced_speaker_database", None, raising=False)
    monkeypatch.setattr(server, "enhanced_service", None, raising=False)

    return {"waveform": waveform, "segments": segments, "diarization": diarization}


def post_transcribe(client, **params):
    files = {"audio": ("chunk.wav", io.BytesIO(b"RIFFfake-wav-bytes"), "audio/wav")}
    return client.post("/transcribe", files=files, params=params or None)


# ---------------------------------------------------------------------------
# POST /transcribe
# ---------------------------------------------------------------------------

class TestTranscribeContract:
    def test_transcribe_returns_wpf_client_shape(self, client, wired_pipeline):
        response = post_transcribe(client)

        assert response.status_code == 200
        body = response.json()

        # Top level: the client reads segments + full_text.
        assert "segments" in body and isinstance(body["segments"], list)
        assert "full_text" in body and isinstance(body["full_text"], str)
        assert body["segments"], "expected at least one segment"

        for segment in body["segments"]:
            for field in ("start", "end", "text", "speaker"):
                assert field in segment, f"missing segment field: {field}"
            assert isinstance(segment["start"], (int, float))
            assert isinstance(segment["end"], (int, float))
            assert isinstance(segment["text"], str)
            # `speaker` must always be a usable non-empty string - the client
            # puts it straight into a ComboBox.
            assert isinstance(segment["speaker"], str)
            assert segment["speaker"].strip()
            assert segment["end"] >= segment["start"]

    def test_full_text_is_the_concatenated_segment_text(self, client, wired_pipeline):
        body = post_transcribe(client).json()
        expected = " ".join(s["text"] for s in body["segments"])
        assert body["full_text"] == expected

    def test_response_is_json_serializable_scalars_not_numpy(self, client, wired_pipeline):
        # A numpy scalar anywhere in the payload 500s the endpoint; getting a
        # 200 with a parsed body already proves JSON encoding succeeded, but
        # assert the types the client's deserializer is strict about.
        body = post_transcribe(client).json()
        assert isinstance(body["processing_time"], (int, float))
        assert isinstance(body["timestamp"], (int, float))  # time.time(), not a string
        assert isinstance(body["sample_rate"], int)

    def test_diarization_speakers_are_carried_through(self, client, wired_pipeline):
        body = post_transcribe(client).json()
        speakers = {s["speaker"] for s in body["segments"]}
        # Two diarization turns with different labels -> two distinct speakers.
        assert speakers == {"Speaker SPEAKER_00", "Speaker SPEAKER_01"}

    def test_plain_multipart_post_defaults_to_the_fast_path(self, client, wired_pipeline):
        # The shipped client sends nothing but the file. include_analysis must
        # default to False, so no analysis keys appear.
        body = post_transcribe(client).json()
        assert "conversation_analysis" not in body
        assert "enhancement_info" not in body

    def test_empty_upload_is_rejected(self, client, wired_pipeline):
        files = {"audio": ("chunk.wav", io.BytesIO(b""), "audio/wav")}
        response = client.post("/transcribe", files=files)
        assert response.status_code == 400

    def test_silence_returns_empty_segments_not_an_error(self, client, wired_pipeline, monkeypatch):
        # Digital silence trips run_transcription's RMS fast-path. The client
        # polls every 5 seconds and must get a well-formed empty result.
        silent = FakeWaveform(np.zeros(SAMPLE_RATE * 4, dtype=np.float32))
        monkeypatch.setattr(
            server, "load_audio_from_bytes",
            lambda data: (silent, SAMPLE_RATE), raising=False
        )

        body = post_transcribe(client).json()
        assert body["segments"] == []
        assert body["full_text"] == ""
        assert body["skipped_reason"] == "silence"

    def test_diarization_failure_still_returns_a_transcript(self, client, wired_pipeline, monkeypatch):
        # asyncio.gather(..., return_exceptions=True): a diarization blow-up must
        # not fail the request or orphan the transcription task.
        def boom(waveform, sample_rate):
            raise RuntimeError("pyannote exploded")

        monkeypatch.setattr(server, "_diarize_sync", boom, raising=False)

        response = post_transcribe(client)
        assert response.status_code == 200
        body = response.json()
        assert body["segments"]
        assert body["full_text"].strip()


# ---------------------------------------------------------------------------
# GET /speakers
# ---------------------------------------------------------------------------

class TestSpeakersContract:
    @pytest.fixture
    def populated_db(self, tmp_path, monkeypatch):
        db = EnhancedSpeakerDatabase(data_dir=str(tmp_path / "speaker_data_v2"))
        rng = np.random.RandomState(7)
        for name in ("Alice", "Bob"):
            vector = rng.randn(32).astype(np.float32)
            db.enroll_speaker(name, embeddings=[vector / np.linalg.norm(vector)])
        monkeypatch.setattr(server, "enhanced_speaker_database", db, raising=False)
        return db

    def test_speakers_returns_wpf_client_shape(self, client, populated_db):
        response = client.get("/speakers")

        assert response.status_code == 200
        body = response.json()

        assert isinstance(body["total_speakers"], int)
        assert isinstance(body["speakers"], list)
        assert body["total_speakers"] == len(body["speakers"]) == 2

        for speaker in body["speakers"]:
            for field in ("id", "name", "embedding_count"):
                assert field in speaker, f"missing speaker field: {field}"
            assert isinstance(speaker["id"], str) and speaker["id"]
            assert isinstance(speaker["name"], str) and speaker["name"]
            # int, never a numpy integer - the client deserializes this as int.
            assert isinstance(speaker["embedding_count"], int)
            assert not isinstance(speaker["embedding_count"], bool)

        assert {s["name"] for s in body["speakers"]} == {"Alice", "Bob"}

    def test_empty_database_returns_an_empty_list_not_an_error(self, client, tmp_path, monkeypatch):
        db = EnhancedSpeakerDatabase(data_dir=str(tmp_path / "empty_v2"))
        monkeypatch.setattr(server, "enhanced_speaker_database", db, raising=False)

        body = client.get("/speakers").json()
        assert body["total_speakers"] == 0
        assert body["speakers"] == []

    def test_missing_database_returns_503(self, client, monkeypatch):
        monkeypatch.setattr(server, "enhanced_speaker_database", None, raising=False)
        assert client.get("/speakers").status_code == 503
