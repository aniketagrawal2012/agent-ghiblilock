# Agent-GhibliLock (`IDF-GGL-2026-AGENT-GHIBLILOCK-C2PA`)

**Deterministic Wavelet-Locked Cross-Modal Style Transfer, Iterative Sliced-Wasserstein Chromaticity Transport, and ISO/IEC 19566-5 C2PA JUMBF + Mid-Band 2D-DCT Cryptographic Provenance Compiler**

* **Author:** Aniket Agrawal (AI Cloud Engineer, Google Cloud, Bangalore)
* **Paper Manuscript:** [`Agent_GhibliLock_Paper.pdf`](./Agent_GhibliLock_Paper.pdf) | [`agent_ghiblilock_paper.tex`](./agent_ghiblilock_paper.tex)
* **Google IDF Dossier:** [`GOOGLE_IDF_PORTAL_SUBMISSION.md`](./GOOGLE_IDF_PORTAL_SUBMISSION.md)

## Overview
`Agent-GhibliLock` resolves **The Captioning Conundrum** (high-frequency spatial/geometric drift across `Pixels -> Prose -> Pixels/Video`) and **C2PA Content Credential Severance** when chaining Vision-Language Models (`gemini-3.8-flash`, `gemini-4-argon`) with image/video diffusion synthesizers (`gemini-nano-banana-2.1`, `imagen-3.0`, `veo-3.1-generate-001`).

### Key Components
1. **Orthonormal Multi-Level 2D Haar Wavelet Transform (`DWT2` / `IDWT2`)**: Exact Parseval energy-conserving multiresolution decomposition in perceptual CIELAB ($L^*a^*b^*$) space.
2. **Iterative 3D Sliced-Wasserstein Optimal Transport (`SWOT`)**: Aligns candidate chromaticity distributions with target artistic gouache palettes across orthonormal $SO(3)$ projections while locking high-frequency structural subbands ($LH_k, HL_k, HH_k$).
3. **Mid-Band $8\times 8$ 2D-DCT QIM Cryptographic Watermark**: Embeds a 64-bit structural-mask digest at mid-frequency coordinates $\{(2,3),(3,2),(1,3),(3,1)\}$ with $0.00\%$ Bit Error Rate ($\text{BER} = 0.0$) under $\pm 2\text{ LSB}$ quantization noise.
4. **ISO/IEC 19566-5 JUMBF (`caBX`) C2PA v2.1 Binary Compiler**: Injects a signed `jumb` superbox (`jumd` + `json` + `c2sh` HMAC-SHA256) into ISO/IEC 15948 PNG streams with IEEE 802.3 CRC-32 validation.

## Reproducing Benchmarks & Tests
```bash
python3 test_agent_ghiblilock.py
python3 run_live_benchmark.py
python3 run_live_gemini_multimodel_eval.py
```
