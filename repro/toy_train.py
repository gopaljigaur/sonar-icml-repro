"""CPU-only toy run on the synthetic LF/HF-coherence proxy task (see
sonar/synthetic.py docstring for why: real ASVspoof downloads and HF Jobs
GPU access were both unavailable in this environment).

Compares SONAR-Full (dual-path + Eq.8 loss) against a single-path baseline
(no NFE/SRM, no cross-attention, no JS loss) on identical data/epochs/optimizer
settings, to test the convergence-speed claim (Claim 3) directionally.
Uses TinyEncoder (not real XLSR -- also a network/compute-constrained
substitution, documented) so the whole run finishes in CPU seconds.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from sonar.eval_eer import compute_eer
from sonar.losses import sonar_loss
from sonar.model import SONAR, BaselineSinglePath
from sonar.synthetic import SyntheticCoherenceDataset


def evaluate(model, loader):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits, _, _ = model(x)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            scores.append(probs.numpy())
            labels.append(y.numpy())
    return compute_eer(np.concatenate(scores), np.concatenate(labels))


def run(variant: str, epochs: int, seed: int = 0):
    torch.manual_seed(seed)
    is_baseline = variant == "baseline-nofusion"
    model = (BaselineSinglePath(use_real_xlsr=False, dim=128)
             if is_baseline else SONAR(use_real_xlsr=False, dim=128, num_srm_filters=8, heads=4))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)  # higher LR than paper's 1e-5: toy task, tiny model, few steps
    lambda_js = 0.0 if is_baseline else 1.0
    class_weights = torch.tensor([1.0, 1.0])  # synthetic set is class-balanced (paper's real data is 1:9, see docstring)

    train_ds = SyntheticCoherenceDataset(n=256, seed=seed)
    eval_ds = SyntheticCoherenceDataset(n=128, seed=seed + 1000)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=16)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for x, y in train_loader:
            logits, z_c, z_n = model(x)
            loss, parts = sonar_loss(logits, z_c, z_n, y, class_weights, lambda_js=lambda_js)
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.project_srm()
            losses.append(parts["total"])
        eer = evaluate(model, eval_loader)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "eer": eer, "time_s": time.time() - t0}
        history.append(row)
        print(f"[{variant}] epoch={epoch} loss={row['train_loss']:.4f} eer={eer:.2f}% time={row['time_s']:.2f}s")
    return history


def main():
    epochs = 15
    results = {}
    for variant in ["SONAR-Full", "baseline-nofusion"]:
        print(f"=== {variant} ===")
        results[variant] = run(variant, epochs)

    import os
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/toy_synthetic_run.json", "w") as f:
        json.dump(results, f, indent=2)

    for variant, hist in results.items():
        final_eer = hist[-1]["eer"]
        # epoch where EER first comes within 1 point of its own final value (plateau proxy)
        plateau_epoch = next((r["epoch"] for r in hist if abs(r["eer"] - final_eer) <= 1.0), hist[-1]["epoch"])
        print(f"RESULT variant={variant} final_eer={final_eer:.2f}% plateau_epoch={plateau_epoch}")


if __name__ == "__main__":
    main()
