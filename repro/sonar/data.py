"""Loaders for the public HF-hub mirrors used in place of the gated
datashare.ed.ac.uk ASVspoof downloads (documented substitution, same
underlying corpora):

  train (ASVspoof2019 LA train)     -> Bisher/ASVspoof_2019_LA           (key: 0=bonafide, 1=spoof)
  eval  (ASVspoof2021 DF)           -> MoaazTalab/ASVspoof_2021_DF_Balanced_Normalized  (label: 0=fake, 1=real)
  eval  (ASVspoof2021 LA)           -> MoaazTalab/ASVspoof_2021_LA_Balanced_Normalized  (label: 0=fake, 1=real)

In-The-Wild (Muller et al.) has no usable public HF mirror at time of writing
(the one candidate, sarkarbkl/In_the_wild_audio_deepfake, contains no data
files) and the official source (deepfake-total.com) requires a manual
download step outside this environment. We substitute the ASVspoof2021 DF
eval set (unseen-condition, cross-corpus-flavored) as an ITW proxy for the
toy run and label results accordingly -- this is NOT the real ITW corpus.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

CLIP_LEN = 64600  # ~4s @ 16kHz, per paper Sec 4

# dataset_id -> (label_column, value_meaning_genuine)
LABEL_CONFIG = {
    "Bisher/ASVspoof_2019_LA": ("key", 0),  # 0=bonafide
    "MoaazTalab/ASVspoof_2021_DF_Balanced_Normalized": ("label", 1),  # 1=real
    "MoaazTalab/ASVspoof_2021_LA_Balanced_Normalized": ("label", 1),  # 1=real
}


def _decode_audio(audio_field) -> np.ndarray:
    if hasattr(audio_field, "get_all_samples"):  # torchcodec AudioDecoder
        samples = audio_field.get_all_samples()
        wav = samples.data.mean(dim=0).numpy()  # collapse to mono if needed
        sr = samples.sample_rate
    elif isinstance(audio_field, dict):
        wav = np.asarray(audio_field["array"], dtype=np.float32)
        sr = audio_field.get("sampling_rate", 16000)
    else:
        raise TypeError(f"unrecognized audio field type: {type(audio_field)}")
    if sr != 16000:
        import torchaudio

        wav_t = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
        wav = torchaudio.functional.resample(wav_t, sr, 16000).squeeze(0).numpy()
    return np.asarray(wav, dtype=np.float32)


class HFAudioDataset(Dataset):
    def __init__(self, dataset_id: str, hf_dataset, n_samples: int | None = None,
                 clip_len: int = CLIP_LEN, seed: int = 0):
        self.dataset_id = dataset_id
        self.ds = hf_dataset
        if n_samples is not None and n_samples < len(self.ds):
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(self.ds), size=n_samples, replace=False)
            self.ds = self.ds.select(idx.tolist())
        self.clip_len = clip_len
        if dataset_id not in LABEL_CONFIG:
            raise KeyError(f"no label config for {dataset_id}; add it to LABEL_CONFIG")
        self.label_col, self.genuine_value = LABEL_CONFIG[dataset_id]

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
        wav = self._pad_or_trim(_decode_audio(ex["audio"]))
        label = 1 if ex[self.label_col] == self.genuine_value else 0
        return torch.from_numpy(wav), torch.tensor(label, dtype=torch.long)


def load_split(dataset_id: str, split: str = "train", n_samples: int | None = None):
    from datasets import load_dataset

    # slice server-side so we don't pull entire multi-GB shards for a toy run
    split_expr = f"{split}[:{n_samples}]" if n_samples else split
    ds = load_dataset(dataset_id, split=split_expr)
    return HFAudioDataset(dataset_id, ds, n_samples=None)
