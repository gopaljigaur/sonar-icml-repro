"""Loaders for the public HF-hub mirrors used in place of the gated
datashare.ed.ac.uk ASVspoof downloads (documented substitution, same
underlying corpora):

  train (ASVspoof2019 LA train)     -> LanceaKing/asvspoof2019
  eval  (ASVspoof2021 DF)           -> MoaazTalab/ASVspoof_2021_DF_Balanced_Normalized
  eval  (ASVspoof2021 LA)           -> MoaazTalab/ASVspoof_2021_LA_Balanced_Normalized
  eval  (In-The-Wild)               -> sarkarbkl/In_the_wild_audio_deepfake
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

CLIP_LEN = 64600  # ~4s @ 16kHz, per paper Sec 4


def _find_audio_col(example: dict) -> str:
    for k in ("audio", "speech", "file", "wav", "input"):
        if k in example:
            return k
    raise KeyError(f"no audio column found among {list(example.keys())}")


def _find_label_col(example: dict) -> str:
    for k in ("label", "labels", "target", "class", "bonafide"):
        if k in example:
            return k
    raise KeyError(f"no label column found among {list(example.keys())}")


def _to_binary_label(value, label_col: str) -> int:
    """Normalize arbitrary label encodings to 1=genuine/bonafide, 0=spoof."""
    if isinstance(value, str):
        v = value.lower()
        return 1 if v in ("bonafide", "genuine", "real", "1") else 0
    return int(value)


class HFAudioDataset(Dataset):
    def __init__(self, hf_dataset, n_samples: int | None = None, clip_len: int = CLIP_LEN, seed: int = 0):
        self.ds = hf_dataset
        if n_samples is not None and n_samples < len(self.ds):
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(self.ds), size=n_samples, replace=False)
            self.ds = self.ds.select(idx.tolist())
        self.clip_len = clip_len
        self._audio_col = None
        self._label_col = None

    def __len__(self):
        return len(self.ds)

    def _pad_or_trim(self, wav: np.ndarray) -> np.ndarray:
        if len(wav) >= self.clip_len:
            start = (len(wav) - self.clip_len) // 2
            return wav[start : start + self.clip_len]
        reps = int(np.ceil(self.clip_len / len(wav)))
        return np.tile(wav, reps)[: self.clip_len]

    def __getitem__(self, i):
        ex = self.ds[i]
        if self._audio_col is None:
            self._audio_col = _find_audio_col(ex)
            self._label_col = _find_label_col(ex)
        audio_field = ex[self._audio_col]
        wav = np.asarray(audio_field["array"] if isinstance(audio_field, dict) else audio_field, dtype=np.float32)
        wav = self._pad_or_trim(wav)
        label = _to_binary_label(ex[self._label_col], self._label_col)
        return torch.from_numpy(wav), torch.tensor(label, dtype=torch.long)


def load_split(dataset_id: str, split: str = "train", n_samples: int | None = None):
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split)
    return HFAudioDataset(ds, n_samples=n_samples)
