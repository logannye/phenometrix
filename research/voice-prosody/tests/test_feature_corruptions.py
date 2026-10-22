import numpy as np
import pytest
from phenometrix_prosody.features import transform


def test_declared_corruptions_preserve_length_and_fixed_construction():
    pcm = (.3 * np.sin(np.arange(48000) * 2 * np.pi / 240)).astype(np.float32)
    clean, _ = transform(pcm, "clean", "F04_si1146", "train")
    assert np.array_equal(clean, pcm)
    noisy, meta = transform(pcm, "noise10db", "F04_si1146", "train")
    assert meta["achieved_snr_db"] == pytest.approx(10, abs=1e-5)
    assert np.array_equal(noisy, transform(pcm, "noise10db", "F04_si1146", "train")[0])
    assert not np.array_equal(noisy, transform(pcm, "noise10db", "F04_si1146", "development")[0])
    filtered, _ = transform(pcm, "first_difference", "x", "train")
    assert filtered[0] == pcm[0]
    assert np.array_equal(filtered[1:], np.diff(pcm))
    clipped, _ = transform(pcm, "clip01", "x", "train")
    assert np.max(np.abs(clipped)) <= np.float32(.1)
    assert all(len(x) == len(pcm) and np.isfinite(x).all() for x in (noisy, filtered, clipped))


def test_silence_and_invalid_configuration():
    result, meta = transform(np.zeros(10, np.float32), "noise10db", "x", "train")
    assert not result.any() and meta["achieved_snr_db"] is None
    with pytest.raises(ValueError):
        transform(np.ones(10), "unknown", "x", "train")
