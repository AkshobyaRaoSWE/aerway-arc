"""Deterministic window features, shared by training and replay."""

import numpy as np
from scipy.fft import dct


def _samples(samples, sample_rate_hz):
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("samples must have shape (at least 2 samples, channels)")
    if not np.isfinite(x).all() or np.max(np.abs(x)) > 1e100:
        raise ValueError("samples must contain finite, representable physical values")
    if isinstance(sample_rate_hz, bool) or not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive and finite")
    return x


def awake_features(samples, sample_rate_hz, calibration, motion=None):
    """Use only the supplied trailing window and pre-recorded calibration."""
    x = _samples(samples, sample_rate_hz)
    if not isinstance(calibration, dict) or any(
        not isinstance(calibration.get(key), str) or not calibration[key].strip()
        for key in ("id", "source")
    ):
        raise ValueError("calibration requires id and source")
    reference = np.asarray(calibration.get("rms_reference", []), dtype=float)
    if reference.shape != (x.shape[1],) or not np.isfinite(reference).all() or np.any(reference <= 0):
        raise ValueError("one finite, positive calibration RMS required per channel")
    x = x - x.mean(axis=0)
    rms = np.sqrt(np.mean(x * x, axis=0))
    midpoint = len(x) // 2
    slope = (np.sqrt(np.mean(x[midpoint:] ** 2, axis=0))
             - np.sqrt(np.mean(x[:midpoint] ** 2, axis=0))) / (len(x) / (2 * sample_rate_hz))
    metrics = (
        ("rms", rms),
        ("mav", np.mean(np.abs(x), axis=0)),
        ("waveform_length_per_s", np.sum(np.abs(np.diff(x, axis=0)), axis=0) * sample_rate_hz / (len(x) - 1)),
        ("relative_rms", rms / reference),
        ("rms_slope_per_s", slope),
    )
    names = [f"ch{channel}.{name}" for channel in range(x.shape[1]) for name, _ in metrics]
    values = [metric[channel] for channel in range(x.shape[1]) for _, metric in metrics]
    if motion is not None:
        m = _samples(motion, sample_rate_hz)
        if m.shape[1] != 6:
            raise ValueError("motion requires ax, ay, az, gx, gy, gz in that order")
        # Acceleration includes gravity; deployment units/order belong in the bundle contract.
        values.extend((np.sqrt(np.mean(np.sum(m[:, :3] ** 2, axis=1))),
                       np.sqrt(np.mean(np.sum(m[:, 3:] ** 2, axis=1)))))
        names.extend(("motion.accel_rms", "motion.gyro_rms"))
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("feature overflow; check units and calibration")
    return values, names


def log_mel(samples, sample_rate_hz=16000):
    """Return complete, unpadded 32 ms frames at 10 ms hops, 64 log-mel bins."""
    x = _samples(samples, sample_rate_hz)
    if x.shape[1] != 1:
        raise ValueError("night audio must be mono; mixdown must be explicit upstream")
    frame = round(0.032 * sample_rate_hz)
    hop = round(0.010 * sample_rate_hz)
    if frame < 128 or hop < 1 or len(x) < frame:
        raise ValueError("audio sample rate or window too short for the 64-bin frontend")
    frames = np.lib.stride_tricks.sliding_window_view(x[:, 0], frame)[::hop]
    window = np.hanning(frame)
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2 / np.sum(window ** 2)
    frequencies = np.fft.rfftfreq(frame, 1 / sample_rate_hz)
    high_hz = min(7600, sample_rate_hz / 2)
    mel_bounds = np.linspace(2595 * np.log10(1 + 20 / 700),
                             2595 * np.log10(1 + high_hz / 700), 66)
    bounds = 700 * (10 ** (mel_bounds / 2595) - 1)
    rising = (frequencies[None, :] - bounds[:-2, None]) / (bounds[1:-1] - bounds[:-2])[:, None]
    falling = (bounds[2:, None] - frequencies[None, :]) / (bounds[2:] - bounds[1:-1])[:, None]
    filters = np.maximum(0, np.minimum(rising, falling))
    if np.any(filters.sum(axis=1) == 0):
        raise ValueError("sample rate cannot resolve the configured mel bands")
    result = np.log(np.maximum(power @ filters.T, 1e-12))
    if not np.isfinite(result).all():
        raise ValueError("audio feature overflow")
    return result


def night_features(samples, sample_rate_hz):
    """Energy plus MFCC statistics. Quiet audio alone is not an apnea label."""
    x = _samples(samples, sample_rate_hz)
    mel = log_mel(x, sample_rate_hz)
    mfcc = dct(mel, type=2, axis=1, norm="ortho")[:, :13]
    values = np.r_[np.log(max(float(np.sqrt(np.mean(x * x))), 1e-12)),
                   np.mean(np.signbit(x[1:, 0]) != np.signbit(x[:-1, 0])),
                   mfcc.mean(axis=0), mfcc.std(axis=0)]
    names = ["audio.log_rms", "audio.zero_crossing_rate"]
    names += [f"audio.mfcc{i}.mean" for i in range(13)]
    names += [f"audio.mfcc{i}.std" for i in range(13)]
    return values, names
