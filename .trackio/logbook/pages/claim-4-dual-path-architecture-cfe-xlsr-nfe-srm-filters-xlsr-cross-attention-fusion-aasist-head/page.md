# Claim 4: Dual-path architecture: CFE (XLSR) + NFE (SRM filters + XLSR), cross-attention fusion, AASIST head


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_918ebbaefa94", "created_at": "2026-07-20T20:25:27+00:00", "title": "Claim 4: Dual-path architecture: CFE (XLSR) + NFE (SRM filters + XLSR), cross-attention fusion, AASIST head"}
-->
Document setup, runs, and results for **Claim 4: Dual-path architecture: CFE (XLSR) + NFE (SRM filters + XLSR), cross-attention fusion, AASIST head**.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_b5589e6f6c01", "created_at": "2026-08-02T14:47:01+00:00", "title": "Architecture reimplementation"}
-->
**Independent reproduction** (no official code was released at time of writing; authors state code will be released upon acceptance). Reimplemented from arXiv:2511.21325 Sec 4.1 / Fig 1. Code: https://github.com/gopaljigaur/sonar-icml-repro

**Architecture verified structurally** (`sonar/model.py`, tests in `tests/test_model_shapes.py`):
- Content Feature Extractor (CFE): ideal FFT band-pass split -> low-frequency band -> XLSR encoder -> `z_content`
- Noise Feature Extractor (NFE): high-frequency band -> **constrained learnable SRM filters** (M=30, kernel length 5, center tap fixed at -1, remaining taps projected to zero-sum after every optimizer step, exactly as Sec 4.1 describes) -> second unshared XLSR encoder -> `z_noise`
- **Frequency cross-attention fusion**: bidirectional `nn.MultiheadAttention`(z_content -> z_noise, z_noise -> z_content), summed + layer-normed -> `e_out`
- **AASIST-inspired classifier head**: spectral + temporal attention pooling -> MLP -> 2-way logits

Unit test `test_srm_constraint_holds_after_init_and_after_step` confirms the SRM constraint (center=-1, row-sum=0) holds both at init and after a simulated optimizer perturbation + re-projection — matching the paper's stated constraint-projection training procedure.

**Deviation (documented, not hidden):** the AASIST head is a simplified single-layer graph-attention pooling block, not the verbatim clovaai/aasist code (which expects a raw-waveform sinc front-end, not a pre-fused embedding sequence — reusing it verbatim would have duplicated the XLSR front-end). Structurally faithful to Fig. 1's data flow (dual-path -> cross-attention -> graph-attention classifier), not a byte-for-byte port.

**Verdict: architecture claim (Claim 4) reproduces at the structural/code level.** All four described components are present and connected as specified; forward pass runs end-to-end (`pytest`-style asserts in `tests/test_model_shapes.py`, all passing).
