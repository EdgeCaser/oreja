"""
Dependency-light unit tests for POST /speakers/name_mapping - the endpoint the
shipped WPF client calls (see App.xaml.cs) with query params old_speaker_id
and new_speaker_name. server.enhanced_speaker_database is monkeypatched to a
lightweight fake so no real model or on-disk database is needed.

Heavy ML imports are stubbed via _stub_heavy_deps so `server.app` can be
built and driven with FastAPI's TestClient without torch/pyannote installed.
"""

import _stub_heavy_deps  # noqa: F401

import pytest
from fastapi.testclient import TestClient

import server


class FakeSpeakerDB:
    """Records apply_name_correction() calls and returns a scripted outcome."""

    def __init__(self, outcome=None, raise_error=None):
        self.calls = []
        self._outcome = outcome
        self._raise_error = raise_error
        self.speaker_records = {}

    def apply_name_correction(self, old_speaker_id, new_display_name, embeddings=None):
        self.calls.append((old_speaker_id, new_display_name))
        if self._raise_error is not None:
            raise self._raise_error
        return self._outcome


@pytest.fixture
def client():
    return TestClient(server.app)


@pytest.fixture(autouse=True)
def restore_database(monkeypatch):
    # Every test sets server.enhanced_speaker_database itself; make sure a
    # stray None from a previous test module doesn't leak across tests.
    yield
    monkeypatch.setattr(server, "enhanced_speaker_database", None, raising=False)


def install_fake_db(monkeypatch, **kwargs):
    fake = FakeSpeakerDB(**kwargs)
    monkeypatch.setattr(server, "enhanced_speaker_database", fake, raising=False)
    return fake


class TestNameMappingEndpoint:
    def test_rename_maps_to_name_updated(self, client, monkeypatch):
        fake = install_fake_db(monkeypatch, outcome={
            "action": "renamed",
            "speaker_id": "spk_abc123",
            "display_name": "Alice",
            "embeddings_added": 0,
        })

        response = client.post(
            "/speakers/name_mapping",
            params={"old_speaker_id": "spk_abc123", "new_speaker_name": "Alice"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "name_updated"
        assert body["action"] == "renamed"
        assert body["speaker_id"] == "spk_abc123"
        assert body["target_speaker_id"] == "spk_abc123"
        assert body["speaker_name"] == "Alice"
        assert body["new_name"] == "Alice"
        assert body["embeddings_learned"] == 0
        assert fake.calls == [("spk_abc123", "Alice")]

    def test_anonymous_session_label_matched(self, client, monkeypatch):
        # old_speaker_id can be a per-session label like "Speaker SPEAKER_00",
        # not a real v2 ID - the endpoint passes it straight through.
        fake = install_fake_db(monkeypatch, outcome={
            "action": "matched",
            "speaker_id": "spk_existing",
            "display_name": "Bob",
            "embeddings_added": 0,
        })

        response = client.post(
            "/speakers/name_mapping",
            params={"old_speaker_id": "Speaker SPEAKER_00", "new_speaker_name": "Bob"},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "name_updated"
        assert fake.calls == [("Speaker SPEAKER_00", "Bob")]

    def test_merge_maps_to_speakers_merged(self, client, monkeypatch):
        install_fake_db(monkeypatch, outcome={
            "action": "merged",
            "speaker_id": "spk_target",
            "display_name": "Carol",
            "embeddings_added": 2,
        })

        response = client.post(
            "/speakers/name_mapping",
            params={"old_speaker_id": "spk_source", "new_speaker_name": "Carol"},
        )

        body = response.json()
        assert response.status_code == 200
        assert body["status"] == "speakers_merged"
        assert body["embeddings_learned"] == 2

    def test_created_maps_to_speaker_created(self, client, monkeypatch):
        install_fake_db(monkeypatch, outcome={
            "action": "created",
            "speaker_id": "spk_new",
            "display_name": "Dave",
            "embeddings_added": 0,
        })

        response = client.post(
            "/speakers/name_mapping",
            params={"old_speaker_id": "AUTO_SPEAKER_003", "new_speaker_name": "Dave"},
        )

        assert response.json()["status"] == "speaker_created"

    def test_merge_failed_maps_through(self, client, monkeypatch):
        install_fake_db(monkeypatch, outcome={
            "action": "merge_failed",
            "speaker_id": "spk_target",
            "display_name": "Eve",
            "embeddings_added": 0,
        })

        response = client.post(
            "/speakers/name_mapping",
            params={"old_speaker_id": "spk_source", "new_speaker_name": "Eve"},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "merge_failed"

    def test_empty_new_name_is_rejected_before_touching_database(self, client, monkeypatch):
        fake = install_fake_db(monkeypatch, outcome={
            "action": "noop", "speaker_id": None, "display_name": "", "embeddings_added": 0
        })

        response = client.post(
            "/speakers/name_mapping",
            params={"old_speaker_id": "spk_abc123", "new_speaker_name": "   "},
        )

        assert response.status_code == 400
        assert fake.calls == []  # never reached the database

    def test_missing_database_returns_503(self, client, monkeypatch):
        monkeypatch.setattr(server, "enhanced_speaker_database", None, raising=False)

        response = client.post(
            "/speakers/name_mapping",
            params={"old_speaker_id": "spk_abc123", "new_speaker_name": "Alice"},
        )

        assert response.status_code == 503

    def test_value_error_from_database_becomes_400(self, client, monkeypatch):
        install_fake_db(monkeypatch, raise_error=ValueError("Speaker name cannot be empty"))

        response = client.post(
            "/speakers/name_mapping",
            params={"old_speaker_id": "spk_abc123", "new_speaker_name": "Alice"},
        )

        assert response.status_code == 400

    def test_unexpected_exception_becomes_500(self, client, monkeypatch):
        install_fake_db(monkeypatch, raise_error=RuntimeError("disk on fire"))

        response = client.post(
            "/speakers/name_mapping",
            params={"old_speaker_id": "spk_abc123", "new_speaker_name": "Alice"},
        )

        assert response.status_code == 500

    def test_params_are_query_params_not_body(self, client, monkeypatch):
        # HARD contract: the shipped WPF client sends these as query params on
        # a POST with no body. Sending them as a JSON body instead must NOT work.
        install_fake_db(monkeypatch, outcome={
            "action": "renamed", "speaker_id": "spk_1", "display_name": "Alice", "embeddings_added": 0
        })

        response = client.post(
            "/speakers/name_mapping",
            json={"old_speaker_id": "spk_1", "new_speaker_name": "Alice"},
        )

        assert response.status_code == 422  # FastAPI: missing required query params
