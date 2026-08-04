"""
FFmpeg-free audio I/O helpers.

torchaudio 2.9+ delegates load()/save() to torchcodec, which needs FFmpeg
shared libraries installed on the machine and raises at call time when they
are absent. Every audio payload Oreja moves internally is WAV (the WPF client
uploads 16-bit PCM WAV chunks), which soundfile decodes natively through its
bundled libsndfile - no system dependency. These helpers therefore try
soundfile first and only fall back to torchaudio for formats soundfile cannot
read (exotic containers, on machines that do have FFmpeg installed).
"""

import io
import logging

import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)


def load_audio(source):
    """
    Drop-in replacement for ``torchaudio.load()``.

    Accepts a file path or a binary stream and returns
    ``(waveform, sample_rate)`` where waveform is a float32 tensor shaped
    (channels, frames) in [-1, 1] - the same contract torchaudio.load has.
    """
    try:
        data, sample_rate = sf.read(source, dtype="float32", always_2d=True)
        return torch.from_numpy(np.ascontiguousarray(data.T)), int(sample_rate)
    except Exception as sf_error:
        if hasattr(source, "seek"):
            source.seek(0)
        try:
            return torchaudio.load(source)
        except Exception:
            # torchaudio's own error here is usually "install FFmpeg", which
            # buries the more relevant soundfile failure - surface both.
            logger.error(f"soundfile could not decode audio: {sf_error}")
            raise


def save_wav(target, waveform: torch.Tensor, sample_rate: int) -> None:
    """
    Drop-in replacement for ``torchaudio.save(..., format="wav")``.

    Writes 16-bit PCM WAV to a file path or writable binary stream.
    """
    array = waveform.detach().cpu().numpy()
    if array.ndim == 1:
        array = array[np.newaxis, :]
    sf.write(target, array.T, int(sample_rate), format="WAV", subtype="PCM_16")


def wav_bytes(waveform: torch.Tensor, sample_rate: int) -> bytes:
    """Encode a waveform tensor as 16-bit PCM WAV bytes."""
    buffer = io.BytesIO()
    save_wav(buffer, waveform, sample_rate)
    return buffer.getvalue()
