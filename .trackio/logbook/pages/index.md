# Reproduction: SONAR: Spectral-Contrastive Audio Residuals for Generalizable Deepfake Detection

[HF paper page](https://huggingface.co/papers/2511.21325)

## Pages

| Page |
| --- |
| [Executive summary](#/executive-summary) |
| [Claim 1: Table 1 EER: 1.57% DF / 1.55% LA (full-training)](#/claim-1-table-1-eer-1-57-df-1-55-la-full-training) |
| [Claim 2: ITW EER: SONAR-Full 6.00% vs XLSR+AASIST 10.46%](#/claim-2-itw-eer-sonar-full-6-00-vs-xlsr-aasist-10-46) |
| [Claim 3: Convergence: SONAR-Full in 12 epochs, SONAR-Finetune in 4-6, vs ~100 for baselines](#/claim-3-convergence-sonar-full-in-12-epochs-sonar-finetune-in-4-6-vs-100-for-baselines) |
| [Claim 4: Dual-path architecture: CFE (XLSR) + NFE (SRM filters + XLSR), cross-attention fusion, AASIST head](#/claim-4-dual-path-architecture-cfe-xlsr-nfe-srm-filters-xlsr-cross-attention-fusion-aasist-head) |
| [Claim 5: JS divergence loss (Eq. 8) aligns low/high-freq embeddings for genuine, separates for fake](#/claim-5-js-divergence-loss-eq-8-aligns-low-high-freq-embeddings-for-genuine-separates-for-fake) |
| [Conclusion](#/conclusion) |
