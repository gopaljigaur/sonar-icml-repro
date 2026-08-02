"""SONAR architecture reimplementation (independent reproduction, no official code released).

Follows arXiv:2511.21325 Sec 4.1 / Fig 1:
  - ideal band-pass split of raw waveform into low-freq (L) and high-freq (H) bands
  - Content Feature Extractor (CFE): XLSR encoder on L        -> z_content
  - Noise Feature Extractor (NFE): constrained SRM filters on H, then a second
    (unshared) XLSR encoder                                   -> z_noise
  - frequency cross-attention fusion of z_content / z_noise    -> e_out
  - AASIST-inspired graph-attention classifier head on e_out   -> logits

Deviations from the paper (documented, not hidden):
  - AASIST head is a simplified single-layer graph-attention pooling block
    (spectral + temporal attention pooling, a la Jung et al. 2022), not the
    verbatim clovaai/aasist implementation (which expects raw-waveform sinc
    front-end input, not a pre-fused embedding sequence).
  - Ideal band-pass filter implemented via FFT bin masking (rectangular /
    "ideal" filter as literally named in the paper) rather than an analog
    filter design.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def ideal_bandpass_split(x: torch.Tensor, sr: int = 16000, cutoff_hz: float = 4000.0):
    """Split waveform x (B, T) into low-freq and high-freq bands via ideal FFT masking."""
    n = x.shape[-1]
    spec = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(n, d=1.0 / sr).to(x.device)
    low_mask = (freqs <= cutoff_hz).float()
    high_mask = 1.0 - low_mask
    low = torch.fft.irfft(spec * low_mask, n=n, dim=-1)
    high = torch.fft.irfft(spec * high_mask, n=n, dim=-1)
    return low, high


class ConstrainedSRM(nn.Module):
    """Learnable constrained high-pass SRM filters (Sec 4.1).

    M filters, kernel length 5. Constraints, enforced by projection after every
    optimizer step (paper: "after every optimizer step we project the filters
    back to the constraint set"):
      - center tap fixed at -1
      - remaining taps sum to +1 (equivalently, all 5 taps sum to 0)
    """

    def __init__(self, num_filters: int = 30, kernel_size: int = 5):
        super().__init__()
        assert kernel_size % 2 == 1
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.center_idx = kernel_size // 2
        free = torch.randn(num_filters, kernel_size - 1) * 0.1
        self.free_taps = nn.Parameter(free)
        self.register_buffer("center_val", torch.tensor(-1.0))
        with torch.no_grad():
            self.project()

    def _assemble(self) -> torch.Tensor:
        free_ptr = 0
        cols = []
        for k in range(self.kernel_size):
            if k == self.center_idx:
                cols.append(self.center_val.expand(self.num_filters, 1))
            else:
                cols.append(self.free_taps[:, free_ptr : free_ptr + 1])
                free_ptr += 1
        return torch.cat(cols, dim=1)  # (M, kernel_size)

    @torch.no_grad()
    def project(self):
        """Project free taps so each filter's taps sum to 0 (center fixed at -1)."""
        target_sum = 1.0  # sum of free taps must equal +1 so total (with -1) is 0
        residual = target_sum - self.free_taps.sum(dim=1, keepdim=True)
        n_free = self.free_taps.shape[1]
        self.free_taps.add_(residual / n_free)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) -> (B, M, T)
        weight = self._assemble().unsqueeze(1)  # (M, 1, kernel_size)
        x = x.unsqueeze(1)  # (B, 1, T)
        pad = self.kernel_size // 2
        return F.conv1d(x, weight, padding=pad)


class TinyEncoder(nn.Module):
    """Lightweight stand-in encoder for structural smoke tests (no network access,
    no GPU needed). Real reproduction runs swap this for the actual XLSR encoder
    via `build_encoder(use_real_xlsr=True)`.
    """

    def __init__(self, out_dim: int = 1024, hop: int = 320):
        super().__init__()
        self.hop = hop
        self.proj = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=hop, stride=hop),
            nn.GELU(),
            nn.Conv1d(64, out_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) or (B, M, T) -> flatten channel dim into batch, then restore
        if x.dim() == 3:
            b, m, t = x.shape
            x = x.reshape(b * m, 1, t)
            out = self.proj(x)  # (B*M, D, F)
            out = out.transpose(1, 2)  # (B*M, F, D)
            f, d = out.shape[1], out.shape[2]
            return out.reshape(b, m, f, d).mean(dim=1)  # collapse M filters -> (B, F, D)
        x = x.unsqueeze(1)
        out = self.proj(x).transpose(1, 2)  # (B, F, D)
        return out


def build_encoder(use_real_xlsr: bool, dim: int = 1024, model_name: str = "facebook/wav2vec2-large-xlsr-53"):
    if not use_real_xlsr:
        return TinyEncoder(out_dim=dim)
    from transformers import Wav2Vec2Model

    class XLSREncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Wav2Vec2Model.from_pretrained(model_name)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 3:  # (B, M, T) from SRM output -> average filters first
                x = x.mean(dim=1)
            out = self.backbone(x).last_hidden_state  # (B, F, 1024)
            return out

    return XLSREncoder()


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int = 1024, heads: int = 8):
        super().__init__()
        self.c2n = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2c = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, z_content: torch.Tensor, z_noise: torch.Tensor) -> torch.Tensor:
        a, _ = self.c2n(z_content, z_noise, z_noise)
        b, _ = self.n2c(z_noise, z_content, z_content)
        return self.norm(a + b)


class AASISTInspiredHead(nn.Module):
    """Simplified graph-attention pooling classifier head (see module docstring
    for deviations from clovaai/aasist).
    """

    def __init__(self, dim: int = 1024, hidden: int = 128):
        super().__init__()
        self.spectral_attn = nn.Linear(dim, 1)
        self.temporal_attn = nn.Linear(dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, e_out: torch.Tensor) -> torch.Tensor:
        # e_out: (B, F, D)
        spec_w = torch.softmax(self.spectral_attn(e_out), dim=1)
        spec_pooled = (spec_w * e_out).sum(dim=1)
        temp_w = torch.softmax(self.temporal_attn(e_out), dim=1)
        temp_pooled = (temp_w * e_out).sum(dim=1)
        pooled = torch.cat([spec_pooled, temp_pooled], dim=-1)
        return self.mlp(pooled)  # (B, 2) logits


class SONAR(nn.Module):
    def __init__(
        self,
        use_real_xlsr: bool = False,
        num_srm_filters: int = 30,
        srm_kernel_size: int = 5,
        dim: int = 1024,
        heads: int = 8,
        cutoff_hz: float = 4000.0,
        sr: int = 16000,
    ):
        super().__init__()
        self.cutoff_hz = cutoff_hz
        self.sr = sr
        self.content_encoder = build_encoder(use_real_xlsr, dim=dim)
        self.srm = ConstrainedSRM(num_srm_filters, srm_kernel_size)
        self.noise_encoder = build_encoder(use_real_xlsr, dim=dim)
        self.fusion = CrossAttentionFusion(dim, heads)
        self.head = AASISTInspiredHead(dim)

    @torch.no_grad()
    def project_srm(self):
        self.srm.project()

    def forward(self, x: torch.Tensor):
        low, high = ideal_bandpass_split(x, sr=self.sr, cutoff_hz=self.cutoff_hz)
        z_content = self.content_encoder(low)  # (B, F, D)
        h_srm = self.srm(high)  # (B, M, T)
        z_noise = self.noise_encoder(h_srm)  # (B, F, D)
        f = min(z_content.shape[1], z_noise.shape[1])
        z_content, z_noise = z_content[:, :f], z_noise[:, :f]
        e_out = self.fusion(z_content, z_noise)
        logits = self.head(e_out)
        return logits, z_content, z_noise


class BaselineSinglePath(nn.Module):
    """Single-path XLSR + AASIST-inspired head, no NFE/SRM, no cross-attention,
    no JS loss -- stands in for the "XLSR+AASIST" prior-work baseline of
    Table 1 as a convergence-speed counterfactual for Claim 3. Not tuned to
    match the original XLSR+AASIST paper's exact architecture; only the
    single-path-vs-dual-path convergence comparison is being tested here.
    """

    def __init__(self, use_real_xlsr: bool = False, dim: int = 1024):
        super().__init__()
        self.encoder = build_encoder(use_real_xlsr, dim=dim)
        self.head = AASISTInspiredHead(dim)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        logits = self.head(z)
        return logits, z, z  # z_content==z_noise placeholder; L_JS unused for this variant

    def project_srm(self):
        pass
