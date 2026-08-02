# Claim 2: ITW EER: SONAR-Full 6.00% vs XLSR+AASIST 10.46%


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c63c22388c9d", "created_at": "2026-07-20T20:25:27+00:00", "title": "Claim 2: ITW EER: SONAR-Full 6.00% vs XLSR+AASIST 10.46%"}
-->
Document setup, runs, and results for **Claim 2: ITW EER: SONAR-Full 6.00% vs XLSR+AASIST 10.46%**.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_7f969b2fd6dc", "created_at": "2026-08-02T15:00:34+00:00", "title": "Blocker: no ITW data source available"}
-->
**Same blocker as Claim 1** (no working HF Jobs GPU token, real dataset downloads impractically slow in this environment) — see [Claim 1](#/claim-1-table-1-eer-1-57-df-1-55-la-full-training) for the full detail.

Additionally: the real **In-The-Wild** corpus (Muller et al.) has no usable public HF Hub mirror at time of writing. We checked every HF dataset matching `in-the-wild`/`wild audio`/`deepfake` search terms; the one plausible candidate, `sarkarbkl/In_the_wild_audio_deepfake`, contains **no data files** (only a `.gitattributes` stub — confirmed via the Hub tree API). The paper's official ITW source (deepfake-total.com/in_the_wild) requires a manual out-of-band download not available in this sandboxed environment.

**Left unverified at the numeric level.** No EER numbers for SONAR-Full vs XLSR+AASIST on real or proxy ITW data were produced in this reproduction.
