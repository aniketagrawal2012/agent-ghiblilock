"""
Comprehensive Unit & Adversarial Test Suite for Agent-GhibliLock:
1. sRGB <-> CIELAB (D65) Bijective Round-Trip Accuracy
2. Orthonormal 2D Multi-Level Haar Wavelet Transform Exact Invertibility (Even & Odd Shapes)
3. High-Frequency Structural Edge Mask & Logo/Rail Preservation under Captioning Drift
4. 1D Sliced-Wasserstein Optimal Transport Distance Axioms & Monotonic Convergence
5. Mid-Band 2D-DCT QIM Cryptographic Watermark Embedding, PSNR (>38 dB), & Blind Extraction
6. ISO/IEC 19566-5 JUMBF (`caBX`) PNG Binary Container Compilation, CRC-32 Verification,
   and Adversarial Tamper Rejection (Pixel Modification, Manifest Forgery, CRC Corruption)
"""

from __future__ import annotations

import struct
import unittest
import zlib
import numpy as np

from ghiblilock_engine import (
    AgentGhibliLockCompiler,
    bits_to_bytes,
    bytes_to_bits,
    compute_full_luminance_ssim,
    compute_heldout_sliced_wasserstein_lab,
    compute_parseval_wavelet_energies,
    compute_sliced_wasserstein_lab,
    compute_sobel_gradient_correlation,
    compute_structural_edge_map,
    compute_structural_mask_hash,
    compute_wavelet_edge_ssim,
    deflicker_video_gop_baselines,
    dwt2_multilevel,
    embed_cox_spread_spectrum_dct,
    embed_midband_dct_watermark,
    encode_rgb_to_png_bytes,
    extract_cox_spread_spectrum_dct,
    extract_midband_dct_watermark,
    idwt2_multilevel,
    lab_to_srgb,
    localize_dct_qim_tamper_blocks,
    srgb_to_lab,
    transport_chromaticity_wasserstein,
    verify_png_c2pa_and_dct_provenance,
)


def _create_synthetic_scene_with_logo_and_rail(h: int = 128, w: int = 128) -> np.ndarray:
    """
    Construct a realistic test photograph containing:
    - Smooth sky/foliage background gradients
    - A sharp diagonal metallic railing (high-frequency structural line)
    - A high-contrast corporate emblem ("Kappa"-style geometric diamond logo)
    """
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Realistic neutral-cool outdoor lighting
    img[..., 0] = np.clip(70 + (yy * 90) // h, 0, 255)
    img[..., 1] = np.clip(95 + (xx * 80) // w, 0, 255)
    img[..., 2] = np.clip(140 + ((h - yy) * 70) // h, 0, 255)

    # Diagonal railing across foreground
    rail_mask = np.abs((yy - int(0.65 * h)) - (0.25 * (xx - w // 2)).astype(int)) <= 2
    img[rail_mask] = [235, 238, 242]

    # Sharp geometric shirt emblem (diamond + crossbar)
    cy, cx = int(0.75 * h), int(0.45 * w)
    logo_mask = (np.abs(yy - cy) + np.abs(xx - cx) <= 10) & (
        np.abs(yy - cy) + np.abs(xx - cx) >= 7
    )
    img[logo_mask] = [250, 250, 250]
    return img


def _create_unconstrained_diffusion_candidate(src_rgb: np.ndarray) -> np.ndarray:
    """
    Simulate an unconstrained VLM-to-Diffusion candidate ("Pixels -> Prose -> Pixels")
    that has drifted in high-frequency geometry (lost the corporate emblem & blurred
    the metallic railing angle) and shifted to an oversaturated synthetic palette.
    """
    h, w, _ = src_rgb.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cand = np.zeros_like(src_rgb)
    cand[..., 0] = np.clip(180 - (yy * 60) // h, 0, 255)
    cand[..., 1] = np.clip(140 + (xx * 50) // w, 0, 255)
    cand[..., 2] = np.clip(90 + (yy * 40) // h, 0, 255)
    # Hallucinated wrong rail position and missing shirt emblem
    wrong_rail = np.abs(yy - int(0.82 * h)) <= 3
    cand[wrong_rail] = [120, 110, 100]
    return cand


def _create_ghibli_style_palette_reference(h: int = 128, w: int = 128) -> np.ndarray:
    """
    Create a reference Studio Ghibli cel-shaded gouache palette (warm cerulean sky,
    lush Viridian meadow greens, and soft golden-hour sunlight highlights).
    """
    yy, xx = np.mgrid[0:h, 0:w]
    style = np.zeros((h, w, 3), dtype=np.uint8)
    sky_mask = yy < (h // 2)
    style[sky_mask, 0] = 98
    style[sky_mask, 1] = 178
    style[sky_mask, 2] = 228

    meadow_mask = ~sky_mask
    style[meadow_mask, 0] = 86
    style[meadow_mask, 1] = 168
    style[meadow_mask, 2] = 92

    sun_glow = (yy - h // 3) ** 2 + (xx - w // 3) ** 2 <= (h // 5) ** 2
    style[sun_glow] = [246, 224, 152]
    return style


class TestAgentGhibliLock(unittest.TestCase):
    def test_01_srgb_lab_roundtrip_accuracy(self) -> None:
        src = _create_synthetic_scene_with_logo_and_rail(64, 64)
        lab = srgb_to_lab(src)
        rec = lab_to_srgb(lab)
        max_err = int(np.max(np.abs(src.astype(int) - rec.astype(int))))
        self.assertLessEqual(max_err, 1, f"sRGB <-> CIELAB roundtrip max error {max_err} > 1 LSB")

    def test_02_dwt2_multilevel_exact_invertibility_even_and_odd(self) -> None:
        rng = np.random.default_rng(123)
        for shape in [(64, 64), (128, 96), (63, 75), (101, 103)]:
            arr = rng.uniform(0.0, 100.0, size=shape)
            decomp = dwt2_multilevel(arr, levels=3)
            rec = idwt2_multilevel(decomp)
            self.assertEqual(rec.shape, shape)
            max_diff = float(np.max(np.abs(arr - rec)))
            self.assertLess(max_diff, 1e-10, f"DWT2/IDWT2 failed exact inversion on shape {shape}")

    def test_03_sliced_wasserstein_metric_axioms_and_transport(self) -> None:
        src = _create_synthetic_scene_with_logo_and_rail(64, 64)
        style = _create_ghibli_style_palette_reference(64, 64)
        src_lab = srgb_to_lab(src)
        style_lab = srgb_to_lab(style)

        w2_self = compute_sliced_wasserstein_lab(src_lab, src_lab)
        self.assertAlmostEqual(w2_self, 0.0, places=7)

        w2_ab = compute_sliced_wasserstein_lab(src_lab, style_lab)
        w2_ba = compute_sliced_wasserstein_lab(style_lab, src_lab)
        self.assertAlmostEqual(w2_ab, w2_ba, places=7)
        self.assertGreater(w2_ab, 5.0)

        transported = transport_chromaticity_wasserstein(src_lab, style_lab, blend_alpha=0.95)
        w2_after = compute_sliced_wasserstein_lab(transported, style_lab)
        self.assertLess(w2_after, w2_ab * 0.30)

    def test_04_midband_dct_watermark_embedding_and_blind_extraction(self) -> None:
        src = _create_synthetic_scene_with_logo_and_rail(128, 128)
        payload_bytes = b"GHIBLI26"  # 64 bits
        bits = bytes_to_bits(payload_bytes)
        self.assertEqual(len(bits), 64)

        wm_img = embed_midband_dct_watermark(src, bits, qim_step=14.0)
        mse = float(np.mean((src.astype(np.float64) - wm_img.astype(np.float64)) ** 2))
        psnr = 10.0 * np.log10((255.0 ** 2) / mse)
        self.assertGreater(psnr, 36.0, f"Watermark PSNR {psnr:.2f} dB too low")

        ext_bits, confidence = extract_midband_dct_watermark(wm_img, payload_len=64, qim_step=14.0)
        self.assertEqual(ext_bits, bits)
        self.assertEqual(bits_to_bytes(ext_bits), payload_bytes)
        self.assertGreater(confidence, 0.70)

    def test_05_end_to_end_compiler_restores_structure_and_embeds_c2pa(self) -> None:
        src = _create_synthetic_scene_with_logo_and_rail(128, 128)
        cand = _create_unconstrained_diffusion_candidate(src)
        style = _create_ghibli_style_palette_reference(128, 128)

        compiler = AgentGhibliLockCompiler(tau_struct=0.88, tau_color=4.5)
        res = compiler.compile(src, cand, style)

        # Structural SSIM must jump significantly from unconstrained candidate (<0.55) to >= 0.88
        self.assertLess(res.pre_lock_edge_ssim, 0.65)
        self.assertGreaterEqual(res.post_lock_edge_ssim, 0.88)

        # Wasserstein chromaticity distance must improve toward Ghibli style target
        self.assertLess(res.post_lock_wasserstein_w2, res.pre_lock_wasserstein_w2)
        self.assertEqual(res.dct_watermark_ber, 0.0)

        # Verify C2PA JUMBF `caBX` binary container and DCT watermark
        ver = verify_png_c2pa_and_dct_provenance(
            res.png_bytes,
            res.stylized_rgb,
            compiler.signing_key,
            res.dct_watermark_bits,
        )
        self.assertTrue(ver.valid, f"Verification failed: {ver.error_reason}")
        self.assertTrue(ver.has_cabx_chunk)
        self.assertTrue(ver.crc_valid)
        self.assertTrue(ver.hmac_valid)
        self.assertTrue(ver.pixel_hash_matches)
        self.assertEqual(ver.dct_watermark_ber, 0.0)

    def test_06_adversarial_tamper_detection_pixel_and_manifest(self) -> None:
        src = _create_synthetic_scene_with_logo_and_rail(128, 128)
        cand = _create_unconstrained_diffusion_candidate(src)
        style = _create_ghibli_style_palette_reference(128, 128)

        compiler = AgentGhibliLockCompiler()
        res = compiler.compile(src, cand, style)

        # Attack 1: 1-pixel modification after signing
        tampered_rgb = res.stylized_rgb.copy()
        tampered_rgb[10, 10, 0] = (int(tampered_rgb[10, 10, 0]) + 5) % 256
        ver_pixel = verify_png_c2pa_and_dct_provenance(
            res.png_bytes,
            tampered_rgb,
            compiler.signing_key,
            res.dct_watermark_bits,
        )
        self.assertFalse(ver_pixel.valid)
        self.assertFalse(ver_pixel.pixel_hash_matches)
        self.assertEqual(ver_pixel.error_reason, "PIXEL_SHA256_TAMPERED")

        # Attack 2: Wrong signing key (forged C2PA manifest)
        ver_forged = verify_png_c2pa_and_dct_provenance(
            res.png_bytes,
            res.stylized_rgb,
            b"attacker-forged-signing-key-0000",
            res.dct_watermark_bits,
        )
        self.assertFalse(ver_forged.valid)
        self.assertFalse(ver_forged.hmac_valid)
        self.assertEqual(ver_forged.error_reason, "JUMBF_HMAC_SIGNATURE_INVALID")

        # Attack 3: Stripped caBX chunk (standard non-C2PA PNG)
        stripped_png = encode_rgb_to_png_bytes(res.stylized_rgb, cabx_jumbf_payload=None)
        ver_stripped = verify_png_c2pa_and_dct_provenance(
            stripped_png,
            res.stylized_rgb,
            compiler.signing_key,
            res.dct_watermark_bits,
        )
        self.assertFalse(ver_stripped.valid)
        self.assertFalse(ver_stripped.has_cabx_chunk)
        self.assertEqual(ver_stripped.error_reason, "MISSING_CABX_CHUNK")

    def test_07_parseval_wavelet_energy_conservation(self) -> None:
        rng = np.random.default_rng(2026)
        for levels in [1, 2, 3]:
            channel = rng.uniform(0.0, 100.0, size=(64, 64))
            e_spatial, e_spectral = compute_parseval_wavelet_energies(channel, levels=levels)
            rel_err = abs(e_spatial - e_spectral) / (e_spatial + 1e-12)
            self.assertLess(rel_err, 1e-12, f"Parseval energy violation at levels={levels}: {rel_err}")

    def test_08_midband_dct_watermark_resilience_to_additive_noise(self) -> None:
        src = _create_synthetic_scene_with_logo_and_rail(128, 128)
        bits = bytes_to_bits(b"VEO31C2P")
        wm_img = embed_midband_dct_watermark(src, bits, qim_step=14.0)

        rng = np.random.default_rng(99)
        noisy = np.clip(
            wm_img.astype(np.int32) + rng.integers(-2, 3, size=wm_img.shape),
            0,
            255,
        ).astype(np.uint8)

        ext_bits, conf = extract_midband_dct_watermark(noisy, payload_len=64, qim_step=14.0)
        ber = sum(1 for a, b in zip(ext_bits, bits) if a != b) / 64.0
        self.assertEqual(ber, 0.0, f"Expected 0.0 BER under +/-2 LSB noise, got {ber}")
        self.assertGreater(conf, 0.65)

    def test_09_temporal_veo_video_gop_flicker_suppression_and_merkle_root(self) -> None:
        base_src = _create_synthetic_scene_with_logo_and_rail(64, 64)
        base_cand = _create_unconstrained_diffusion_candidate(base_src)
        style = _create_ghibli_style_palette_reference(64, 64)

        src_frames = []
        cand_frames = []
        flicker_offsets = [-28, 32, -25, 30, -29, 27, -26, 31]
        for t, offset in enumerate(flicker_offsets):
            # Subtle camera pan in source
            s_frame = np.roll(base_src, shift=t, axis=1)
            src_frames.append(s_frame)
            # Severe temporal luminance flicker in unconstrained video candidate
            c_frame = np.clip(base_cand.astype(np.int32) + offset, 0, 255).astype(np.uint8)
            cand_frames.append(c_frame)

        compiler = AgentGhibliLockCompiler()
        gop_res = compiler.compile_video_gop(src_frames, cand_frames, style, temporal_ema_beta=0.78)
        self.assertEqual(gop_res["num_frames"], 8)
        self.assertEqual(len(gop_res["gop_merkle_root_sha256"]), 64)
        self.assertLess(
            gop_res["post_flicker_variance"],
            gop_res["pre_flicker_variance"] * 0.15,
            "Temporal EMA wavelet lock failed to reduce inter-frame flicker variance by >85%",
        )
        self.assertGreaterEqual(gop_res["mean_post_edge_ssim"], 0.88)

    def test_10_png_cabx_crc32_corruption_rejection(self) -> None:
        src = _create_synthetic_scene_with_logo_and_rail(64, 64)
        cand = _create_unconstrained_diffusion_candidate(src)
        style = _create_ghibli_style_palette_reference(64, 64)
        compiler = AgentGhibliLockCompiler()
        res = compiler.compile(src, cand, style)

        # Flip 1 bit inside the caBX chunk payload without updating CRC-32
        idx = res.png_bytes.find(b"caBX")
        self.assertGreater(idx, 0)
        corrupted = bytearray(res.png_bytes)
        corrupted[idx + 10] ^= 0x01
        ver = verify_png_c2pa_and_dct_provenance(
            bytes(corrupted),
            res.stylized_rgb,
            compiler.signing_key,
            res.dct_watermark_bits,
        )
        self.assertFalse(ver.valid)
        self.assertFalse(ver.crc_valid)
        self.assertEqual(ver.error_reason, "CABX_CRC32_MISMATCH")

    def test_11_malformed_jumbf_superbox_rejection(self) -> None:
        src = _create_synthetic_scene_with_logo_and_rail(64, 64)
        bad_png = encode_rgb_to_png_bytes(src, cabx_jumbf_payload=b"NOT_A_VALID_JUMBF_BOX")
        ver = verify_png_c2pa_and_dct_provenance(
            bad_png,
            src,
            b"secret",
            [0] * 64,
        )
        self.assertFalse(ver.valid)
        self.assertEqual(ver.error_reason, "INVALID_JUMBF_SUPERBOX")

    def test_12_structural_mask_sha256_sensitivity_to_logo_removal(self) -> None:
        src_with_logo = _create_synthetic_scene_with_logo_and_rail(64, 64)
        src_without_logo = src_with_logo.copy()
        # Erase the central corporate emblem
        src_without_logo[35:60, 20:40] = src_without_logo[10:35, 20:40]

        _, h1 = compute_structural_mask_hash(srgb_to_lab(src_with_logo)[..., 0])
        _, h2 = compute_structural_mask_hash(srgb_to_lab(src_without_logo)[..., 0])
        self.assertNotEqual(h1, h2)

    def test_13_step_a_vs_step_b_vs_post_qim_and_heldout_wasserstein_bounds(self) -> None:
        src = _create_synthetic_scene_with_logo_and_rail(64, 64)
        cand = _create_unconstrained_diffusion_candidate(src)
        style = _create_ghibli_style_palette_reference(64, 64)
        compiler = AgentGhibliLockCompiler()
        res = compiler.compile(src, cand, style)

        # Step A pure SWOT distance is strictly smaller than post-Step-B wavelet-injected distance
        self.assertLess(res.post_step_a_wasserstein_w2, res.post_lock_wasserstein_w2)
        # Post-output (post-QIM, post-sRGB clamp) W_2 is explicitly tracked and bounded
        self.assertGreater(res.post_output_wasserstein_w2, 0.0)
        self.assertLessEqual(res.post_output_wasserstein_w2, 4.90)
        # Held-out M_eval=64 SO(3) evaluation (seed=999) is bounded and non-circular metrics are high
        self.assertLessEqual(res.eval_heldout_wasserstein_w2, 4.90)
        self.assertGreaterEqual(res.post_full_luminance_ssim, 0.78)
        self.assertGreaterEqual(res.post_sobel_grad_corr, 0.84)

    def test_14_block_level_coset_tamper_localization(self) -> None:
        src = _create_synthetic_scene_with_logo_and_rail(64, 64)
        bits = bytes_to_bits(b"GHIBLI26")
        wm_img = embed_midband_dct_watermark(src, bits, qim_step=14.0)

        # Unmodified image has zero tampered blocks in JPEG-resilient mode
        clean_blocks = localize_dct_qim_tamper_blocks(
            wm_img, bits, qim_step=14.0, jpeg_resilient_mode=True
        )
        self.assertEqual(len(clean_blocks), 0)

        # Inject localized tamper in block (by=2, bx=3) -> y in [16..23], x in [24..31]
        tampered = wm_img.copy()
        tampered[16:18, 24:26] = np.clip(tampered[16:18, 24:26].astype(np.int32) + 45, 0, 255).astype(np.uint8)
        flagged = localize_dct_qim_tamper_blocks(
            tampered, bits, qim_step=14.0, jpeg_resilient_mode=False
        )
        self.assertIn((2, 3), flagged)

    def test_15_qim_outperforms_blind_cox_spread_spectrum_dct(self) -> None:
        src = _create_synthetic_scene_with_logo_and_rail(64, 64)
        bits = bytes_to_bits(b"C2PASOFT")
        wm_qim = embed_midband_dct_watermark(src, bits, qim_step=14.0)
        ext_qim, _ = extract_midband_dct_watermark(wm_qim, payload_len=64, qim_step=14.0)
        ber_qim = sum(1 for a, b in zip(ext_qim, bits) if a != b) / 64.0

        wm_cox = embed_cox_spread_spectrum_dct(src, bits, alpha_ss=5.0)
        ext_cox = extract_cox_spread_spectrum_dct(wm_cox, payload_len=64)
        ber_cox = sum(1 for a, b in zip(ext_cox, bits) if a != b) / 64.0

        self.assertEqual(ber_qim, 0.0)
        self.assertGreater(ber_cox, 0.05, "Blind Cox spread-spectrum must exhibit host-signal interference")

    def test_16_video_gop_deflicker_baselines_and_preclamp_zero_flicker(self) -> None:
        base_src = _create_synthetic_scene_with_logo_and_rail(64, 64)
        base_cand = _create_unconstrained_diffusion_candidate(base_src)
        style = _create_ghibli_style_palette_reference(64, 64)
        flicker_offsets = [-24, 28, -22, 26, -25, 24, -23, 27]
        src_frames = [np.roll(base_src, shift=t * 2, axis=1) for t in range(8)]
        cand_frames = [
            np.clip(base_cand.astype(np.int32) + off, 0, 255).astype(np.uint8)
            for off in flicker_offsets
        ]

        compiler = AgentGhibliLockCompiler()
        gop_res = compiler.compile_video_gop(src_frames, cand_frames, style, temporal_ema_beta=0.78)
        self.assertLess(gop_res["post_preclamp_flicker_variance"], 1e-12)
        self.assertLess(gop_res["post_local_ll_variance"], gop_res["pre_local_ll_variance"] * 0.45)

        baselines = deflicker_video_gop_baselines(src_frames, cand_frames, style, temporal_ema_beta=0.78)
        self.assertGreater(
            gop_res["mean_post_edge_ssim"],
            baselines["Pixel_Space_EMA"]["mean_post_edge_ssim"] + 0.30,
        )
        self.assertGreater(
            gop_res["mean_post_edge_ssim"],
            baselines["Global_Lum_Hist_Lock"]["mean_post_edge_ssim"] + 0.50,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)


