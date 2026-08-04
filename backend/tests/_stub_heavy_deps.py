"""
Install lightweight stand-ins for heavy ML packages into sys.modules, ONLY
when the real package is not already installed.

server.py (and its siblings) import torch / torchaudio / faster_whisper /
pyannote.audio / scipy at module scope - real, unavoidable dependencies on a
fully provisioned dev/CI machine (see backend/requirements.txt). This module
lets the *pure logic* in those files (dict/float bookkeeping: word-to-speaker
attribution, hallucination gating thresholds, ...) be imported and unit
tested on a machine that only has numpy + fastapi, by handing back
unittest.mock.MagicMock objects for names those modules merely need to exist
(e.g. `torch.Tensor` in a type hint, `import torchaudio` succeeding) without
ever doing real tensor math.

Tests built on this module MUST stick to pure control-flow/dict/float logic;
they must never assert on real numeric output from a stubbed call, since a
MagicMock happily returns another MagicMock for anything you ask of it.

When the real package IS installed (a real dev/CI box), this is a no-op:
every stub is installed with sys.modules.setdefault() after checking
importlib.util.find_spec(), so the genuine package always wins.

Usage: `import _stub_heavy_deps` as the FIRST import in a test module, before
`import server` / `import batch_transcription` / etc.
"""

import importlib.util
import sys
import types
from unittest.mock import MagicMock


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        # A previously-installed stub sub-module (e.g. "torchaudio.transforms"
        # without a real "torchaudio" package on disk) makes find_spec raise
        # instead of returning None - treat that as "not really installed".
        return False


# Names of heavy packages this module actually replaced with a MagicMock, filled
# in as install() runs. Reliable, order-independent way for tests/conftest to
# tell "torch is importable" (always true after install() - real or fake) apart
# from "torch is the REAL package" (false on a machine with no ML wheels).
# `TORCH_STUBBED` / `HEAVY_DEPS_STUBBED` below are the flags tests should key
# skips off; do not re-derive this by probing sys.modules, since a MagicMock is
# indistinguishable from a real module by simple presence checks.
STUBBED_MODULES: set = set()


class FakeTensor(list):
    """Real (non-Mock) placeholder type so isinstance(x, torch.Tensor) works."""


def _install_torch():
    if _installed("torch"):
        return
    STUBBED_MODULES.add("torch")
    torch_mod = MagicMock(name="torch")
    torch_mod.Tensor = FakeTensor
    torch_mod.cuda.is_available.return_value = False
    sys.modules.setdefault("torch", torch_mod)

    torchaudio_mod = MagicMock(name="torchaudio")
    transforms_mod = MagicMock(name="torchaudio.transforms")
    transforms_mod.Resample = MagicMock(name="Resample")
    torchaudio_mod.transforms = transforms_mod
    sys.modules.setdefault("torchaudio", torchaudio_mod)
    sys.modules.setdefault("torchaudio.transforms", transforms_mod)


def _install_faster_whisper():
    if _installed("faster_whisper"):
        return
    STUBBED_MODULES.add("faster_whisper")
    mod = MagicMock(name="faster_whisper")
    mod.WhisperModel = MagicMock(name="WhisperModel")
    sys.modules.setdefault("faster_whisper", mod)


def _install_pyannote():
    if _installed("pyannote"):
        return
    STUBBED_MODULES.add("pyannote")
    pkg = types.ModuleType("pyannote")
    pkg.__path__ = []  # mark as a package so "pyannote.audio" submodule imports resolve
    audio_mod = MagicMock(name="pyannote.audio")
    sys.modules.setdefault("pyannote", pkg)
    sys.modules.setdefault("pyannote.audio", audio_mod)
    pkg.audio = audio_mod

    sv_mod = MagicMock(name="pyannote.audio.pipelines.speaker_verification")
    sv_mod.PretrainedSpeakerEmbedding = MagicMock(name="PretrainedSpeakerEmbedding")
    sys.modules.setdefault("pyannote.audio.pipelines.speaker_verification", sv_mod)
    pipelines_mod = MagicMock(name="pyannote.audio.pipelines")
    pipelines_mod.speaker_verification = sv_mod
    sys.modules.setdefault("pyannote.audio.pipelines", pipelines_mod)
    audio_mod.pipelines = pipelines_mod


def _install_scipy():
    if _installed("scipy"):
        return
    STUBBED_MODULES.add("scipy")
    scipy_mod = MagicMock(name="scipy")
    spatial_mod = MagicMock(name="scipy.spatial")
    distance_mod = MagicMock(name="scipy.spatial.distance")
    distance_mod.cosine = MagicMock(name="cosine", return_value=0.5)
    spatial_mod.distance = distance_mod
    scipy_mod.spatial = spatial_mod
    signal_mod = MagicMock(name="scipy.signal")
    scipy_mod.signal = signal_mod
    sys.modules.setdefault("scipy", scipy_mod)
    sys.modules.setdefault("scipy.spatial", spatial_mod)
    sys.modules.setdefault("scipy.spatial.distance", distance_mod)
    sys.modules.setdefault("scipy.signal", signal_mod)


def _install_simple(name: str):
    if _installed(name):
        return
    STUBBED_MODULES.add(name)
    sys.modules.setdefault(name, MagicMock(name=name))


def install():
    _install_torch()
    _install_faster_whisper()
    _install_pyannote()
    _install_scipy()
    _install_simple("sklearn")
    _install_simple("librosa")
    _install_simple("nltk")
    _install_simple("textblob")
    _install_simple("speechbrain")


install()

# Reliable, order-independent detection flags for tests/conftest.py. Computed
# once, right here, from what install() actually did (backed by
# importlib.util.find_spec) - NOT by probing sys.modules later, since by then
# every heavy package name resolves to *something* (real or MagicMock) and the
# two are indistinguishable by presence alone.
TORCH_STUBBED = "torch" in STUBBED_MODULES
HEAVY_DEPS_STUBBED = bool(STUBBED_MODULES)
