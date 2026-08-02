"""Unit tests verifying Eq. 8's intended behavior (Claim 5) without any dataset
or GPU: genuine pairs should be rewarded for aligning content/noise embeddings,
fake pairs for separating them.
"""
import torch

from sonar.losses import frame_js_divergence, js_alignment_loss, sonar_loss


def test_js_divergence_bounds():
    torch.manual_seed(0)
    z1 = torch.randn(4, 10, 16)
    z2 = torch.randn(4, 10, 16)
    js = frame_js_divergence(z1, z2)
    assert (js >= -1e-6).all() and (js <= 1 + 1e-6).all(), js

    js_same = frame_js_divergence(z1, z1)
    assert torch.allclose(js_same, torch.zeros_like(js_same), atol=1e-5), js_same


def test_genuine_loss_prefers_aligned_embeddings():
    torch.manual_seed(0)
    z_content = torch.randn(1, 8, 16)
    aligned = z_content.clone()
    separated = torch.randn(1, 8, 16) * 5 + 10  # far from z_content in softmax space
    y_genuine = torch.tensor([1.0])

    loss_aligned = js_alignment_loss(z_content, aligned, y_genuine)
    loss_separated = js_alignment_loss(z_content, separated, y_genuine)
    assert loss_aligned < loss_separated, (loss_aligned.item(), loss_separated.item())


def test_fake_loss_prefers_separated_embeddings():
    torch.manual_seed(0)
    z_content = torch.randn(1, 8, 16)
    aligned = z_content.clone()
    separated = torch.randn(1, 8, 16) * 5 + 10
    y_fake = torch.tensor([0.0])

    loss_aligned = js_alignment_loss(z_content, aligned, y_fake)
    loss_separated = js_alignment_loss(z_content, separated, y_fake)
    assert loss_separated < loss_aligned, (loss_aligned.item(), loss_separated.item())


def test_sonar_loss_combines_wce_and_js():
    torch.manual_seed(0)
    logits = torch.randn(3, 2)
    z_content = torch.randn(3, 5, 8)
    z_noise = torch.randn(3, 5, 8)
    y = torch.tensor([1, 0, 1])
    weights = torch.tensor([9.0, 1.0])  # upweight minority genuine class (1:9 ratio)
    total, parts = sonar_loss(logits, z_content, z_noise, y, weights, lambda_js=1.0)
    assert torch.isfinite(total)
    assert set(parts.keys()) == {"wce", "l_js", "total"}


if __name__ == "__main__":
    test_js_divergence_bounds()
    test_genuine_loss_prefers_aligned_embeddings()
    test_fake_loss_prefers_separated_embeddings()
    test_sonar_loss_combines_wce_and_js()
    print("ALL EQ8 TESTS PASSED")
