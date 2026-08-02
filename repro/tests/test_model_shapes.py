"""Structural smoke test for Claim 4 (dual-path architecture): confirms the
CFE/NFE/cross-attention/head data flow runs end-to-end and the SRM projection
constraint holds. Uses TinyEncoder (no network/GPU) -- real XLSR swapped in for
the HF Jobs run via SONAR(use_real_xlsr=True).
"""
import torch

from sonar.model import SONAR, ConstrainedSRM


def test_srm_constraint_holds_after_init_and_after_step():
    srm = ConstrainedSRM(num_filters=30, kernel_size=5)
    w = srm._assemble()
    assert w.shape == (30, 5)
    assert torch.allclose(w[:, srm.center_idx], torch.full((30,), -1.0))
    assert torch.allclose(w.sum(dim=1), torch.zeros(30), atol=1e-5)

    # simulate an optimizer step perturbing free_taps, then re-project
    with torch.no_grad():
        srm.free_taps.add_(torch.randn_like(srm.free_taps) * 0.5)
    srm.project()
    w2 = srm._assemble()
    assert torch.allclose(w2[:, srm.center_idx], torch.full((30,), -1.0))
    assert torch.allclose(w2.sum(dim=1), torch.zeros(30), atol=1e-5)


def test_forward_shapes_dim_128():
    model = SONAR(use_real_xlsr=False, dim=128, num_srm_filters=8, heads=4)
    x = torch.randn(2, 16000)  # 1s @ 16kHz toy clip
    logits, z_content, z_noise = model(x)
    assert logits.shape == (2, 2)
    assert z_content.shape[0] == 2 and z_content.shape[-1] == 128
    assert z_noise.shape[0] == 2 and z_noise.shape[-1] == 128


if __name__ == "__main__":
    test_srm_constraint_holds_after_init_and_after_step()
    test_forward_shapes_dim_128()
    print("ALL MODEL SHAPE TESTS PASSED")
