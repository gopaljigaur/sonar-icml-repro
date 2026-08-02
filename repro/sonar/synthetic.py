"""Offline synthetic proxy task (used because both the real ASVspoof2021/ITW
downloads and HF Jobs GPU compute were unavailable in this environment --
see logbook Claim 1/2/3 pages for the stated blockers).

Encodes the paper's core hypothesis directly: genuine audio has phase-coherent
harmonic structure shared across low- and high-frequency bands (same f0
carried through); synthesized/fake audio decouples the two bands (the
high-frequency content comes from a mismatched source), which is exactly the
"spectral bias" artifact SONAR is designed to detect via cross-band
alignment/separation (Eq. 8).

This is a toy/proxy task, NOT a substitute for ASVspoof -- absolute EER
numbers here are not comparable to Table 1. It exists to test whether the
dual-path architecture + Eq. 8 loss can learn the intended cross-band
consistency signal at all, and how fast, on real gradient-based training.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

SR = 16000
CLIP_LEN = 16000  # 1s toy clips (paper uses ~4s; shortened for CPU-only toy run)


def _harmonic_signal(f0: float, n_harmonics: int, t: np.ndarray, decay: float = 0.85) -> np.ndarray:
    sig = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        amp = decay ** (k - 1)
        sig += amp * np.sin(2 * np.pi * f0 * k * t + np.random.uniform(0, 2 * np.pi))
    return sig


def make_clip(genuine: bool, rng: np.random.RandomState) -> np.ndarray:
    t = np.arange(CLIP_LEN) / SR
    f0 = rng.uniform(100, 250)  # natural voice-like fundamental
    n_harm = int(SR / 2 / f0)
    full = _harmonic_signal(f0, n_harm, t)
    if not genuine:
        # decouple: replace high-frequency content with harmonics of a
        # mismatched, unrelated fundamental (simulates vocoder HF artifacts)
        f0_mismatch = rng.uniform(100, 250)
        while abs(f0_mismatch - f0) < 40:
            f0_mismatch = rng.uniform(100, 250)
        full_mismatch = _harmonic_signal(f0_mismatch, n_harm, t)
        spec = np.fft.rfft(full)
        spec_mismatch = np.fft.rfft(full_mismatch)
        freqs = np.fft.rfftfreq(CLIP_LEN, d=1.0 / SR)
        high = freqs > 4000
        spec[high] = spec_mismatch[high]
        full = np.fft.irfft(spec, n=CLIP_LEN)
    full = full / (np.abs(full).max() + 1e-6)
    full = full + rng.normal(0, 0.01, size=full.shape)  # light sensor noise
    return full.astype(np.float32)


class SyntheticCoherenceDataset(Dataset):
    def __init__(self, n: int, seed: int = 0):
        self.n = n
        self.rng = np.random.RandomState(seed)
        self.labels = (self.rng.rand(n) < 0.5).astype(np.int64)  # 1=genuine, 0=fake

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.RandomState(self.rng.randint(0, 2**31 - 1) + i)
        genuine = bool(self.labels[i])
        wav = make_clip(genuine, rng)
        return torch.from_numpy(wav), torch.tensor(int(genuine), dtype=torch.long)
