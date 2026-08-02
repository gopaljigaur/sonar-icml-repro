"""Toy-scale SONAR training + eval, run on an HF GPU Job (see jobs/run_toy_job.sh).

Prints per-epoch CSV lines to stdout (captured verbatim by `trackio logbook
run`) so the logbook has a hard record of the convergence trend (Claim 3) and
final EER (Claim 1/2) independent of any file artifact.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from sonar.data import load_split
from sonar.eval_eer import compute_eer
from sonar.losses import sonar_loss
from sonar.model import SONAR, BaselineSinglePath


def run_epoch_eval(model, loader, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, _, _ = model(x)
            probs = torch.softmax(logits, dim=-1)[:, 1]  # P(genuine)
            scores.append(probs.cpu().numpy())
            labels.append(y.numpy())
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    return compute_eer(scores, labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dataset", default="LanceaKing/asvspoof2019")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-dataset", action="append", required=True,
                     help="name=hf_dataset_id[:split], repeatable")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--use-real-xlsr", action="store_true")
    ap.add_argument("--variant", default="SONAR-Full", choices=["SONAR-Full", "SONAR-Lite", "baseline-nofusion"])
    ap.add_argument("--out", default="outputs/toy_run.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"# device={device} variant={args.variant} use_real_xlsr={args.use_real_xlsr}")

    train_ds = load_split(args.train_dataset, args.train_split, n_samples=args.n_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    eval_loaders = {}
    for spec in args.eval_dataset:
        name, rest = spec.split("=", 1)
        if ":" in rest:
            ds_id, split = rest.split(":", 1)
        else:
            ds_id, split = rest, "train"
        eval_ds = load_split(ds_id, split, n_samples=args.n_eval)
        eval_loaders[name] = DataLoader(eval_ds, batch_size=args.batch_size)

    is_baseline = args.variant == "baseline-nofusion"
    if is_baseline:
        model = BaselineSinglePath(use_real_xlsr=args.use_real_xlsr).to(device)
    else:
        model = SONAR(use_real_xlsr=args.use_real_xlsr).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-8)
    class_weights = torch.tensor([9.0, 1.0], device=device)  # [spoof, genuine], 1:9 real:fake
    lambda_js = 0.0 if is_baseline else 1.0

    history = []
    t0 = time.time()
    print("epoch,train_loss,wce,l_js," + ",".join(f"eer_{k}" for k in eval_loaders) + ",epoch_time_s")
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_t0 = time.time()
        losses, wces, ljss = [], [], []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, z_c, z_n = model(x)
            loss, parts = sonar_loss(logits, z_c, z_n, y, class_weights, lambda_js=lambda_js)
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.project_srm()
            losses.append(parts["total"])
            wces.append(parts["wce"])
            ljss.append(parts["l_js"])
        sched.step()

        eers = {name: run_epoch_eval(model, loader, device) for name, loader in eval_loaders.items()}
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "wce": float(np.mean(wces)),
            "l_js": float(np.mean(ljss)),
            "epoch_time_s": time.time() - ep_t0,
            **{f"eer_{k}": v for k, v in eers.items()},
        }
        history.append(row)
        print(",".join(str(row[k]) for k in ["epoch", "train_loss", "wce", "l_js"]) +
              "," + ",".join(str(eers[k]) for k in eval_loaders) + f",{row['epoch_time_s']:.1f}")

    total_time = time.time() - t0
    result = {
        "variant": args.variant,
        "use_real_xlsr": args.use_real_xlsr,
        "n_train": args.n_train,
        "n_eval": args.n_eval,
        "epochs": args.epochs,
        "total_time_s": total_time,
        "history": history,
        "final_eer": {k: history[-1][f"eer_{k}"] for k in eval_loaders},
    }
    import os
    os.makedirs("outputs", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("FINAL_RESULT_JSON=" + json.dumps(result["final_eer"]))
    print(f"TOTAL_TIME_S={total_time:.1f}")


if __name__ == "__main__":
    main()
