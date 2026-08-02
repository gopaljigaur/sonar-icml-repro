"""Eq. 8 loss: Jensen-Shannon distributional alignment/separation loss + WCE.

L(x,y) = WCE(y_hat, y) + lambda_JS * L_JS(z_content, z_noise)
L_JS(x,y) = y * JS(z_c, z_n) + (1-y) * (1 - JS(z_c, z_n))
  y = 1 -> genuine  : minimizing loss pulls JS -> 0 (content/noise embeddings aligned)
  y = 0 -> spoofed   : minimizing loss pushes JS -> 1 (content/noise embeddings separated)

JS(p,q) here uses base-2 log so the divergence is bounded in [0, 1], which is
required for "(1 - JS)" to behave as a complementary probability-like term as
Eq. 8 implies (the paper does not spell out the log base; base-2 is the
standard convention for a [0,1]-normalized JS divergence).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _kl_div_base2(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (torch.log2(p) - torch.log2(q))).sum(dim=-1)


def frame_js_divergence(z_content: torch.Tensor, z_noise: torch.Tensor) -> torch.Tensor:
    """Frame-wise JS divergence averaged over frames F.

    z_content, z_noise: (B, F, D) raw embeddings -> softmax over D per frame to
    get discrete per-frame distributions p_content[i], p_noise[i].
    Returns: (B,) scalar JS divergence per sample.
    """
    p = torch.softmax(z_content, dim=-1)
    q = torch.softmax(z_noise, dim=-1)
    m = 0.5 * (p + q)
    js_per_frame = 0.5 * _kl_div_base2(p, m) + 0.5 * _kl_div_base2(q, m)  # (B, F)
    return js_per_frame.mean(dim=1)  # (B,)


def js_alignment_loss(z_content: torch.Tensor, z_noise: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Eq. 8's L_JS term. y: (B,) with 1=genuine, 0=spoofed."""
    js = frame_js_divergence(z_content, z_noise)
    y = y.float()
    return (y * js + (1 - y) * (1 - js)).mean()


def sonar_loss(
    logits: torch.Tensor,
    z_content: torch.Tensor,
    z_noise: torch.Tensor,
    y: torch.Tensor,
    class_weights: torch.Tensor,
    lambda_js: float = 1.0,
):
    """Full Eq. 8 objective: WCE + lambda_JS * L_JS."""
    wce = F.cross_entropy(logits, y.long(), weight=class_weights)
    l_js = js_alignment_loss(z_content, z_noise, y)
    total = wce + lambda_js * l_js
    return total, {"wce": wce.item(), "l_js": l_js.item(), "total": total.item()}
