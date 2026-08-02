# Claim 5: JS divergence loss (Eq. 8) aligns low/high-freq embeddings for genuine, separates for fake


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_1de0420de7b0", "created_at": "2026-07-20T20:25:27+00:00", "title": "Claim 5: JS divergence loss (Eq. 8) aligns low/high-freq embeddings for genuine, separates for fake"}
-->
Document setup, runs, and results for **Claim 5: JS divergence loss (Eq. 8) aligns low/high-freq embeddings for genuine, separates for fake**.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_bed11d5bbf76", "created_at": "2026-08-02T14:47:16+00:00", "title": "Eq. 8 loss verification"}
-->
**Implemented Eq. 8 exactly** (`sonar/losses.py`, code: https://github.com/gopaljigaur/sonar-icml-repro/blob/main/repro/sonar/losses.py):

```
JS(Z_content, Z_noise) = (1/F) * sum_i JS(softmax(z_content[i]), softmax(z_noise[i]))   # frame-wise, base-2 (bounded [0,1])
L_JS(x,y) = y * JS(z_c, z_n) + (1-y) * (1 - JS(z_c, z_n))        # y=1 genuine, y=0 spoofed
L(x,y)    = WCE(y_hat, y) + lambda_JS * L_JS,   lambda_JS = 1
```

We use base-2 log for the KL terms inside JS so the divergence is bounded in [0,1] — the paper doesn't spell out the log base, but this is required for the `(1 - JS)` term in Eq. 8 to behave as a complementary probability-like weight, and is the standard normalization convention.

**Behavioral verification (`tests/test_eq8.py`, all passing):**
- `test_js_divergence_bounds`: JS in [0,1] for random embeddings; JS(z,z)=0 (identical embeddings fully aligned).
- `test_genuine_loss_prefers_aligned_embeddings`: for y=1 (genuine), the loss is strictly lower when z_content/z_noise are aligned than when they are far apart — confirms the loss **pulls real content-noise pairs together**, as the paper describes.
- `test_fake_loss_prefers_separated_embeddings`: for y=0 (spoofed), the loss is strictly lower when embeddings are separated than aligned — confirms the loss **pushes fake pairs apart**.
- `test_sonar_loss_combines_wce_and_js`: full Eq. 8 objective (WCE + lambda_JS * L_JS) is finite and correctly composed for a mixed-label batch.

**Verdict: Claim 5's mechanistic description reproduces exactly** — the loss provably has the claimed effect (align-if-genuine / separate-if-fake) by direct unit test, independent of any large-scale training run.

**On the novelty claim** ("first use of learnable distributional alignment across frequency bands for deepfake detection"): this is a literature-priority claim, not an empirically testable one. We did not find a prior deepfake-detection paper using a JS-divergence loss to align/separate low- vs high-frequency embeddings in our (non-exhaustive) search of related work cited in the SONAR paper (XLSR-Mamba, AASIST, XLSR-Conformer, XLSR-SLS) — none of those use a frequency-band JS-divergence contrastive term. We cannot verify absence across the full literature; this sub-claim is left **unverified/plausible** rather than confirmed.
