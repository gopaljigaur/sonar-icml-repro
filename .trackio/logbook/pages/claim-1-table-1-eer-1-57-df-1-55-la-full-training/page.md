# Claim 1: Table 1 EER: 1.57% DF / 1.55% LA (full-training)


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c68579668c6b", "created_at": "2026-07-20T20:25:27+00:00", "title": "Claim 1: Table 1 EER: 1.57% DF / 1.55% LA (full-training)"}
-->
Document setup, runs, and results for **Claim 1: Table 1 EER: 1.57% DF / 1.55% LA (full-training)**.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c1aab8826d9d", "created_at": "2026-08-02T15:00:24+00:00", "title": "Blocker: no real ASVspoof numbers obtained"}
-->
**Blocker (stated explicitly, not hidden):** could not obtain the paper's real EER numbers in this environment.

1. **HF Jobs GPU access unavailable**: canary `hf jobs run python:3.12 python -c "print('ok')"` returned `403 Forbidden: missing permissions: job.write` for this account/token. No GPU compute path was available for any run.
2. **Real dataset downloads impractically slow/unreliable**: `Bisher/ASVspoof_2019_LA` (train, real HF mirror of ASVspoof2019 LA), `MoaazTalab/ASVspoof_2021_DF_Balanced_Normalized` and `..._LA_Balanced_Normalized` (real HF mirrors of the ASVspoof2021 eval sets) are legitimate public datasets with correct schemas (verified: `key`/`label` columns, audio decodable via torchcodec) — but streaming and sliced non-streaming loads both hung for 10+ minutes with zero throughput in this sandboxed environment, so no real-data training run completed.

Given both blockers, we could not run the paper's actual training configuration (4x NVIDIA L40, ASVspoof2019 LA train -> ASVspoof2021 DF/LA eval, 12 epochs, XLSR-large-53 dual encoders). **This claim is left unverified at the numeric level** in this reproduction; see the [Conclusion](#/conclusion) page for exactly what would be needed to complete it (a working `job.write`-scoped token + normal Hub bandwidth).

What we *could* verify: the architecture (Claim 4) and Eq. 8 loss mechanism (Claim 5) are implemented correctly and behave as the paper describes, at the code/unit-test level, independent of this data/compute blocker — see those pages. The toy convergence experiment on [Claim 3](#/claim-3-convergence-sonar-full-in-12-epochs-sonar-finetune-in-4-6-vs-100-for-baselines) also used a synthetic proxy task for the same reason.
