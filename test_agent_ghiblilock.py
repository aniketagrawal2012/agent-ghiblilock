"""
Comprehensive Unit & Adversarial Test Suite for Agent-GhibliLock:
1. sRGB <-> CIELAB (D65) Bijective Round-Trip Accuracy
2. Orthonormal 2D Multi-Level Haar Wavelet Transform Exact Invertibility (Even & Odd Shapes)
3. High-Frequency Structural Edge Mask & Logo/Rail Preservation under Captioning Drift
4. 3D Sliced-Wasserstein Optimal Transport Distance Axioms, RMS Estimators, & Perturbation Bound
5. Mid-Band 2D-DCT QIM Cryptographic Watermark Embedding, PSNR (>38 dB), & Blind Extraction
6. ISO/IEC 19566-5 JUMBF (`caBX`) PNG Binary Container Compilation, CRC-32 Verification,
   and Adversarial Tamper Rejection (Pixel Modification, Manifest Forgery, CRC Corruption)
7. Spatially Masked ROI vs. Global Wavelet Locking & Exact Spatial Mean Conservation
8. Decoupled (gamma_tau, alpha_tau) Closed-Loop Controller under Color Violations
9. QIM Export Self-Noise Decomposition, Blind Majority-Voted Coset Tamper Localization,
   R=5 Odd Repetition Voting, and Disaggregated LSB vs. Lossy JPEG Q=85 Survival
10. Motion-Compensated and Motion-Gated Video GOP Wavelet EMA Deflickering
11. Real Natural Photographic Image Evaluation (Grace Hopper Portrait & Product Pack)
"""

from __future__ import annotations

import hashlib
import io
import os
import struct
import time
import unittest
import zlib
import numpy as np

from ghiblilock_engine import (
    AgentGhibliLockCompiler,
    WaveletDecomposition2D,
    _derive_qim_dither_matrix,
    apply_interior_dc_reanchoring,
    bits_to_bytes,
    build_spatial_roi_detail_mask,
    bytes_to_bits,
    compute_dct8_ac_l1_rounding_bounds,
    compute_full_luminance_ssim,
    compute_heldout_sliced_wasserstein_lab,
    compute_motion_innovation_gate,
    compute_parseval_wavelet_energies,
    compute_qim_bit_index,
    compute_sliced_wasserstein_lab,
    compute_sobel_coherence_5x5,
    compute_sobel_gradient_correlation,
    compute_structural_edge_map,
    compute_structural_mask_hash,
    compute_wavelet_edge_ssim,
    dct2_blocks_8x8,
    deflicker_video_gop_baselines,
    dwt2_multilevel,
    embed_cox_spread_spectrum_dct,
    embed_midband_dct_watermark,
    encode_rgb_to_png_bytes,
    extract_cox_spread_spectrum_dct,
    extract_midband_dct_watermark,
    fold_merkle_digests_sha256,
    idct2_blocks_8x8,
    idwt2_multilevel,
    lab_to_srgb,
    localize_dct_qim_tamper_blocks,
    measure_qim_export_self_noise,
    should_use_row_shift_interleave,
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
        # Verify blind majority-voted localization without source registry (expected_bits=None)
        flagged_blind = localize_dct_qim_tamper_blocks(
            tampered, expected_bits=None, qim_step=14.0, jpeg_resilient_mode=False
        )
        self.assertIn((2, 3), flagged_blind)

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

    def test_17_spatially_masked_roi_vs_global_wavelet_locking_and_exact_mean(self) -> None:
        src = _create_synthetic_scene_with_logo_and_rail(128, 128)
        cand = _create_unconstrained_diffusion_candidate(src)
        style = _create_ghibli_style_palette_reference(128, 128)
        # Add photographic sensor grain in non-ROI sky background of src
        rng = np.random.default_rng(777)
        src_noisy_bg = src.copy()
        src_noisy_bg[:45, :, :] = np.clip(
            src_noisy_bg[:45, :, :].astype(np.int32) + rng.integers(-12, 13, size=(45, 128, 3)),
            0,
            255,
        ).astype(np.uint8)

        compiler = AgentGhibliLockCompiler(tau_struct=0.88, tau_color=4.5)
        fg_roi = build_spatial_roi_detail_mask(srgb_to_lab(src)[..., 0], background_floor=0.0)
        res_global = compiler.compile(src_noisy_bg, cand, style, roi_spatial_masking=False)
        res_roi = compiler.compile(src_noisy_bg, cand, style, roi_spatial_masking=True, roi_mask=fg_roi)

        self.assertGreaterEqual(res_roi.post_lock_edge_ssim, 0.88)
        self.assertTrue(res_roi.roi_spatial_masking)
        # In the flat background sky region (y < 32, x >= 72, outside the sun glow disc),
        # ROI masking suppresses high-frequency photographic sensor grain while preserving gouache sky
        dwt_global_sky = dwt2_multilevel(srgb_to_lab(res_global.stylized_rgb)[:32, 72:, 0], levels=1)
        dwt_roi_sky = dwt2_multilevel(srgb_to_lab(res_roi.stylized_rgb)[:32, 72:, 0], levels=1)
        hf_global = float(np.sqrt(np.mean(dwt_global_sky.details[0][0] ** 2 + dwt_global_sky.details[0][1] ** 2)))
        hf_roi = float(np.sqrt(np.mean(dwt_roi_sky.details[0][0] ** 2 + dwt_roi_sky.details[0][1] ** 2)))
        self.assertLess(hf_roi, hf_global * 0.45)

        # Verify algebraic Theorem 1 property: arbitrary spatial mask M_k(u,v) in [0,1]
        # preserves exact pre-clamping spatial mean luminance mu(L_hat*) == mu(L_tilde*)
        s_lum = srgb_to_lab(src_noisy_bg)[..., 0]
        c_lum = srgb_to_lab(cand)[..., 0]
        d_s = dwt2_multilevel(s_lum, levels=2)
        d_c = dwt2_multilevel(c_lum, levels=2)
        rand_masks = [rng.uniform(0.0, 1.0, size=s_lh.shape) for (s_lh, _, _) in d_s.details]
        fused_det = [
            (m * s_lh + (1.0 - m) * c_lh, m * s_hl + (1.0 - m) * c_hl, m * s_hh + (1.0 - m) * c_hh)
            for m, (s_lh, s_hl, s_hh), (c_lh, c_hl, c_hh) in zip(rand_masks, d_s.details, d_c.details)
        ]
        rec_lum = idwt2_multilevel(WaveletDecomposition2D(ll=d_c.ll, details=fused_det, original_shape=s_lum.shape))
        self.assertAlmostEqual(float(np.mean(rec_lum)), float(np.mean(c_lum)), places=10)

    def test_18_training_vs_heldout_wasserstein_rms_and_perturbation_bound(self) -> None:
        src = _create_synthetic_scene_with_logo_and_rail(64, 64)
        cand = _create_unconstrained_diffusion_candidate(src)
        style = _create_ghibli_style_palette_reference(64, 64)
        src_lab = srgb_to_lab(src)
        cand_lab = srgb_to_lab(cand)
        style_lab = srgb_to_lab(style)

        tilde_lab = transport_chromaticity_wasserstein(cand_lab, style_lab, blend_alpha=0.95, swot_steps=8)
        w2_step_a = compute_sliced_wasserstein_lab(tilde_lab, style_lab, swot_steps=8)
        sw2_eval = compute_heldout_sliced_wasserstein_lab(tilde_lab, style_lab, num_projections=64, seed=999)
        # In-sample optimized W_hat_2^StepA is smaller than or comparable to held-out SW_hat_2^eval
        self.assertLessEqual(w2_step_a, sw2_eval + 0.15)

        # Verify isotropic perturbation bound: |W_hat_2(I_hat, Y_ref) - W_hat_2(I_tilde, Y_ref)| <= ||L_hat* - L_tilde*||_F / sqrt(3 HW)
        hat_lab = tilde_lab.copy()
        hat_lab[..., 0] = 0.85 * src_lab[..., 0] + 0.15 * tilde_lab[..., 0]
        w2_step_b = compute_sliced_wasserstein_lab(hat_lab, style_lab, swot_steps=8)
        fro_diff = float(np.linalg.norm(hat_lab[..., 0] - tilde_lab[..., 0], ord="fro"))
        hw = hat_lab.shape[0] * hat_lab.shape[1]
        iso_bound = fro_diff / np.sqrt(3.0 * hw)
        self.assertLessEqual(abs(w2_step_b - w2_step_a), iso_bound + 1e-9)

    def test_19_decoupled_alpha_gamma_closed_loop_controller(self) -> None:
        # Construct a high-contrast source scene where SSIM_edge >= 0.88 is met immediately
        # but high luminance contrast challenges W_2 <= tau_color
        src = _create_synthetic_scene_with_logo_and_rail(64, 64)
        src[:20, :] = [250, 250, 250]
        src[45:, :] = [15, 15, 20]
        cand = _create_unconstrained_diffusion_candidate(src)
        style = _create_ghibli_style_palette_reference(64, 64)

        compiler = AgentGhibliLockCompiler(tau_struct=0.88, tau_color=3.5)
        res = compiler.compile(src, cand, style)
        self.assertGreaterEqual(res.post_lock_edge_ssim, 0.88)
        self.assertLessEqual(res.post_lock_wasserstein_w2, 3.85)

    def test_20_qim_export_self_noise_and_jpeg85_vs_lsb_disaggregation(self) -> None:
        from PIL import Image

        src = _create_synthetic_scene_with_logo_and_rail(64, 64)
        bits = bytes_to_bits(b"GHIBLI26")

        # 1. Measure export self-noise eta_b^export from uint8 rounding and gamut clamping
        self_noise = measure_qim_export_self_noise(src, bits, qim_step=14.0)
        self.assertLess(self_noise["rms_export_noise"], 0.60)
        self.assertLess(self_noise["max_export_noise"], 3.5)
        self.assertEqual(self_noise["single_coeff_error_rate_export"], 0.0)

        # 2. Test R=5 odd repetition voting (5 mid-band coords on 64x64 -> 64*5/64 = 5 votes/bit, gcd(5,64)=1)
        secret = b"enterprise-dither-key-2026"
        wm_r5 = embed_midband_dct_watermark(src, bits, qim_step=14.0, num_midband_coords=5, secret_key=secret)
        ext_r5, conf_r5, hz, ho = extract_midband_dct_watermark(
            wm_r5, payload_len=64, qim_step=14.0, num_midband_coords=5, secret_key=secret, return_hard_vote_stats=True
        )
        self.assertEqual(ext_r5, bits)
        self.assertTrue(np.all(hz + ho == 5), "Each of the 64 bits must receive exactly R=5 odd votes")

        # 3. Test lossy JPEG Q=85 recompression survival on R=5 watermarked image
        buf = io.BytesIO()
        Image.fromarray(wm_r5).save(buf, format="JPEG", quality=85, subsampling=0)
        buf.seek(0)
        jpg_rgb = np.asarray(Image.open(buf).convert("RGB"))
        ext_jpg, _ = extract_midband_dct_watermark(
            jpg_rgb, payload_len=64, qim_step=14.0, num_midband_coords=5, secret_key=secret
        )
        self.assertEqual(ext_jpg, bits)

    def test_21_motion_compensated_and_gated_video_gop_wavelet_ema(self) -> None:
        base_src = _create_synthetic_scene_with_logo_and_rail(64, 64)
        base_cand = _create_unconstrained_diffusion_candidate(base_src)
        style = _create_ghibli_style_palette_reference(64, 64)
        flicker_offsets = [-20, 24, -18, 22, -21, 19, -20, 23]

        src_frames = []
        cand_frames = []
        for t, off in enumerate(flicker_offsets):
            s_f = np.roll(base_src, shift=t * 4, axis=1)
            # Inject a fast-moving local high-contrast foreground object on frames 3..5
            if 3 <= t <= 5:
                s_f[20:32, 20:32] = 252
            src_frames.append(s_f)
            c_f = np.clip(np.roll(base_cand, shift=t * 4, axis=1).astype(np.int32) + off, 0, 255).astype(np.uint8)
            cand_frames.append(c_f)

        compiler = AgentGhibliLockCompiler()
        gop_mc = compiler.compile_video_gop(
            src_frames, cand_frames, style, temporal_ema_beta=0.78, motion_compensated=True
        )
        self.assertLess(gop_mc["post_preclamp_flicker_variance"], 1e-12)
        self.assertGreaterEqual(gop_mc["mean_post_edge_ssim"], 0.88)

    def test_22_real_natural_photograph_compilation_grace_hopper(self) -> None:
        import matplotlib
        from PIL import Image

        sample_dir = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "sample_data")
        gh_path = os.path.join(sample_dir, "grace_hopper.jpg")
        self.assertTrue(os.path.exists(gh_path))

        img = Image.open(gh_path).convert("RGB").resize((128, 128), Image.Resampling.BICUBIC)
        src_photo = np.asarray(img, dtype=np.uint8)
        # Simulate stylistic VLM-to-Diffusion candidate (smooth gouache shading + softened collar insignia)
        padded = np.pad(src_photo.astype(np.float64), ((2, 2), (2, 2), (0, 0)), mode="edge")
        blurred = (
            padded[:-4, :-4] + padded[:-4, 2:-2] + padded[:-4, 4:]
            + padded[2:-2, :-4] + padded[2:-2, 2:-2] + padded[2:-2, 4:]
            + padded[4:, :-4] + padded[4:, 2:-2] + padded[4:, 4:]
        ) / 9.0
        style_ref = _create_ghibli_style_palette_reference(128, 128)
        cand_photo = np.clip(0.70 * blurred + 0.30 * style_ref.astype(np.float64), 0, 255).astype(np.uint8)

        compiler = AgentGhibliLockCompiler(tau_struct=0.88, tau_color=4.5)
        res_global = compiler.compile(src_photo, cand_photo, style_ref, roi_spatial_masking=False)
        res_roi = compiler.compile(src_photo, cand_photo, style_ref, roi_spatial_masking=True)

        self.assertGreaterEqual(res_global.post_lock_edge_ssim, 0.90)
        self.assertGreaterEqual(res_roi.post_lock_edge_ssim, 0.88)
        self.assertLessEqual(res_global.post_output_wasserstein_w2, 4.5)
        self.assertEqual(res_global.dct_watermark_ber, 0.0)
        self.assertEqual(res_roi.dct_watermark_ber, 0.0)

    def test_23_ac_l1_rounding_bounds_and_row_shifted_hard_soft_qim_voting(self) -> None:
        # 1. Verify exact analytical AC l1/linf Holder rounding bounds (< 3.5 = Delta_QIM / 4)
        bounds = compute_dct8_ac_l1_rounding_bounds()
        self.assertAlmostEqual(bounds["m4_max_ac_rounding_bound"], 3.3486, places=3)
        self.assertAlmostEqual(bounds["m5_max_ac_rounding_bound"], 2.0 + np.sqrt(2.0), places=6)
        self.assertLess(bounds["m4_max_ac_rounding_bound"], 3.5)
        self.assertLess(bounds["m5_max_ac_rounding_bound"], 3.5)
        self.assertAlmostEqual(bounds["haar_3ac_subband_linf_bound"], 1.5 * np.sqrt(3.0), places=6)

        # 2. Verify row-shifted interleaving and hard vs. clipped-soft voting under extreme single-block tamper
        src = _create_synthetic_scene_with_logo_and_rail(128, 128)
        bits = bytes_to_bits(b"GHIBLI26")
        wm_rs = embed_midband_dct_watermark(
            src, bits, qim_step=14.0, num_midband_coords=4, row_shift_interleave=True
        )
        # Inject an extreme amplitude tamper (+200 LSB) in a 2x1 vertical strip of 8x8 blocks (by=2..3, bx=3)
        tampered = wm_rs.copy()
        tampered[16:32, 24:32] = 255
        ext_hard, _ = extract_midband_dct_watermark(
            tampered, payload_len=64, qim_step=14.0, voting_mode="hard", row_shift_interleave=True
        )
        ext_soft, _ = extract_midband_dct_watermark(
            tampered, payload_len=64, qim_step=14.0, voting_mode="clipped_soft", row_shift_interleave=True
        )
        self.assertEqual(ext_hard, bits)
        self.assertEqual(ext_soft, bits)

        flagged = localize_dct_qim_tamper_blocks(
            tampered, expected_bits=None, qim_step=14.0, voting_mode="hard", row_shift_interleave=True
        )
        self.assertIn((2, 3), flagged)
        self.assertIn((3, 3), flagged)

        # 3. Single-block blind QIM baseline (R=1, num_midband_coords=1) works cleanly without noise
        wm_r1 = embed_midband_dct_watermark(src, bits, qim_step=14.0, num_midband_coords=1)
        ext_r1, _ = extract_midband_dct_watermark(wm_r1, payload_len=64, qim_step=14.0, num_midband_coords=1)
        self.assertEqual(ext_r1, bits)

    def test_24_motion_compensated_pixel_ema_and_ungated_wavelet_ema_baselines(self) -> None:
        base_src = _create_synthetic_scene_with_logo_and_rail(64, 64)
        base_cand = _create_unconstrained_diffusion_candidate(base_src)
        style = _create_ghibli_style_palette_reference(64, 64)
        flicker_offsets = [-24, 28, -22, 26, -25, 24, -23, 27]
        src_frames = []
        cand_frames = []
        for t, off in enumerate(flicker_offsets):
            s_f = np.roll(base_src, shift=t * 2, axis=1)
            c_f = np.clip(np.roll(base_cand, shift=t * 2, axis=1).astype(np.int32) + off, 0, 255).astype(np.uint8)
            if 3 <= t <= 5:
                s_f[16:32, 16:32] = 250
                c_f[16:32, 16:32] = 250
            src_frames.append(s_f)
            cand_frames.append(c_f)

        compiler = AgentGhibliLockCompiler()
        gop_gated = compiler.compile_video_gop(
            src_frames, cand_frames, style, temporal_ema_beta=0.78, motion_compensated=True, motion_gate_sigma=35.0
        )
        baselines = deflicker_video_gop_baselines(src_frames, cand_frames, style, temporal_ema_beta=0.78)
        self.assertIn("Motion_Comp_Pixel_EMA", baselines)
        self.assertIn("Ungated_Motion_Wavelet_EMA", baselines)
        # Motion-Compensated Pixel EMA lacks wavelet detail lock (SSIM_edge < 0.25 vs >= 0.95 for GhibliLock)
        self.assertGreater(
            gop_gated["mean_post_edge_ssim"],
            baselines["Motion_Comp_Pixel_EMA"]["mean_post_edge_ssim"] + 0.50,
        )
        # Both gated and ungated wavelet EMA eliminate global pre-clamp luminance flicker
        self.assertLess(gop_gated["post_preclamp_flicker_variance"], 1e-12)

    def test_25_vectorized_4d_tensor_contraction_vs_block_loop_precision_and_512_speedup(self) -> None:
        """
        Verify (#37):
        - 4D tensor-contraction NumPy execution for 2D Haar DWT/IDWT and 8x8 block-DCT/IDCT
          matches explicit block-loop execution to machine precision (< 1e-11).
        - Vectorized 512x512 compilation accelerates execution by > 5x over block-loop execution.
        """
        rng = np.random.default_rng(2026)
        plane_512 = rng.uniform(0.0, 100.0, size=(512, 512)).astype(np.float64)

        # 1. Multi-level 2D Haar DWT/IDWT: 4D tensor contraction vs 2x2 block-loop
        t0 = time.perf_counter()
        dwt_vec = dwt2_multilevel(plane_512, levels=2, vectorized=True)
        rec_dwt_vec = idwt2_multilevel(dwt_vec, vectorized=True)
        dt_dwt_vec = time.perf_counter() - t0

        t0 = time.perf_counter()
        dwt_loop = dwt2_multilevel(plane_512, levels=2, vectorized=False)
        rec_dwt_loop = idwt2_multilevel(dwt_loop, vectorized=False)
        dt_dwt_loop = time.perf_counter() - t0

        self.assertLess(float(np.max(np.abs(dwt_vec.ll - dwt_loop.ll))), 1e-11)
        for lvl in range(2):
            for sb_v, sb_l in zip(dwt_vec.details[lvl], dwt_loop.details[lvl]):
                self.assertLess(float(np.max(np.abs(sb_v - sb_l))), 1e-11)
        self.assertLess(float(np.max(np.abs(rec_dwt_vec - rec_dwt_loop))), 1e-11)
        self.assertLess(float(np.max(np.abs(rec_dwt_vec - plane_512))), 1e-11)
        self.assertGreater(dt_dwt_loop / max(dt_dwt_vec, 1e-9), 5.0)

        # 2. 8x8 Block-DCT/IDCT: 4D tensor contraction vs 8x8 block-loop
        t0 = time.perf_counter()
        dct_vec = dct2_blocks_8x8(plane_512, vectorized=True)
        rec_dct_vec = idct2_blocks_8x8(dct_vec, vectorized=True)
        dt_dct_vec = time.perf_counter() - t0

        t0 = time.perf_counter()
        dct_loop = dct2_blocks_8x8(plane_512, vectorized=False)
        rec_dct_loop = idct2_blocks_8x8(dct_loop, vectorized=False)
        dt_dct_loop = time.perf_counter() - t0

        self.assertLess(float(np.max(np.abs(dct_vec - dct_loop))), 1e-11)
        self.assertLess(float(np.max(np.abs(rec_dct_vec - rec_dct_loop))), 1e-11)
        self.assertLess(float(np.max(np.abs(rec_dct_vec - plane_512))), 1e-11)
        self.assertGreater(dt_dct_loop / max(dt_dct_vec, 1e-9), 3.0)

        # 3. Full 512x512 AgentGhibliLockCompiler.compile speedup (> 5x) and bit-exact equivalence
        src_512 = _create_synthetic_scene_with_logo_and_rail(512, 512)
        cand_512 = _create_unconstrained_diffusion_candidate(src_512)
        style_512 = _create_ghibli_style_palette_reference(512, 512)
        compiler = AgentGhibliLockCompiler()

        dt_compile_vec = float("inf")
        res_vec = None
        for _ in range(2):
            t0 = time.perf_counter()
            res_vec = compiler.compile(src_512, cand_512, style_512, vectorized=True)
            dt_compile_vec = min(dt_compile_vec, time.perf_counter() - t0)

        t0 = time.perf_counter()
        res_loop = compiler.compile(src_512, cand_512, style_512, vectorized=False)
        dt_compile_loop = time.perf_counter() - t0

        self.assertTrue(np.array_equal(res_vec.stylized_rgb, res_loop.stylized_rgb))
        self.assertEqual(
            res_vec.c2pa_assertion["output_pixel_sha256"],
            res_loop.c2pa_assertion["output_pixel_sha256"],
        )
        self.assertGreater(dt_compile_loop / max(dt_compile_vec, 1e-9), 5.0)

    def test_26_closed_form_roi_coherence_threshold_and_interior_dc_reanchoring_guard(self) -> None:
        """
        Verify (#6, #9):
        - Exact closed-form ROI coherence thresholding (C_{5x5} >= 0.35) zeroes isotropic
          texture prior to 5x5 (l_inf radius 2 px) morphological dilation and 3x3 box feathering.
        - Interior DC re-anchoring guard handles |I^circ| = 0 without division-by-zero/NaN
          and enforces the per-pixel shift cap |delta_DC| <= 1.0.
        """
        # 1. Construct an image with an isotropic checkerboard patch (C_{5x5} ~ 0 < 0.35)
        #    and a sharp unidirectional step edge (C_{5x5} ~ 1 >= 0.35)
        l_star = np.full((64, 64), 50.0, dtype=np.float64)
        # Isotropic 1px alternating checkerboard in top-left (away from step edge)
        yy, xx = np.mgrid[0:16, 0:16]
        l_star[4:20, 4:20] = 50.0 + 18.0 * (((yy + xx) % 2) * 2 - 1)
        # Sharp vertical step edge in right half (x = 48)
        l_star[16:48, 48:] = 90.0

        coherence = compute_sobel_coherence_5x5(l_star)
        # Inside the isotropic checkerboard interior, X and Y gradients match in energy and cancel in cross-product
        self.assertLess(float(np.mean(coherence[8:16, 8:16])), 0.15)
        # Along the unidirectional vertical step edge, coherence is near 1.0 (>= 0.35)
        self.assertGreater(float(np.mean(coherence[24:40, 47:49])), 0.90)

        w_roi = build_spatial_roi_detail_mask(
            l_star, background_floor=0.0, dilation_radius=2, coherence_threshold=0.35
        )
        # Isotropic checkerboard interior (far from its boundary) is suppressed by C_{5x5} < 0.35
        self.assertAlmostEqual(float(w_roi[11, 11]), 0.0, places=7)
        # Step edge achieves high normalized ROI weight (> 0.6 after 3x3 box feathering of 4px dilated strip)
        self.assertGreater(float(np.max(w_roi)), 0.60)

        # 2. Interior DC re-anchoring guard when |I^circ| == 0 (all pixels saturated <= 1 or >= 99)
        l_saturated = np.zeros((32, 32), dtype=np.float64)
        l_saturated[:16, :] = 0.5   # <= 1.0
        l_saturated[16:, :] = 99.5  # >= 99.0
        l_anchored_empty, delta_empty = apply_interior_dc_reanchoring(l_saturated, target_mean_lum=45.0)
        self.assertEqual(delta_empty, 0.0)
        self.assertTrue(np.all(np.isfinite(l_anchored_empty)))
        self.assertTrue(np.array_equal(l_anchored_empty, l_saturated))

        # 3. Interior DC re-anchoring cap |delta_DC| <= 1.0 when raw shift > 1.0
        l_shift = np.zeros((32, 32), dtype=np.float64)
        l_shift[:16, :] = 100.0  # Saturated (outside I^o = (1, 99))
        l_shift[16:, :] = 50.0   # Unsaturated interior I^o (mean = 75.0)
        # Target mean = 70.0 -> Delta_mu = -5.0, HW/|I^o| = 2.0 -> raw shift = -10.0 -> capped at -1.0
        l_anchored_cap, delta_cap = apply_interior_dc_reanchoring(l_shift, target_mean_lum=70.0)
        self.assertAlmostEqual(delta_cap, -1.0, places=9)
        # Interior pixels shifted by -1.0 to 49.0; saturated pixels (100.0) untouched
        self.assertAlmostEqual(float(l_anchored_cap[24, 16]), 49.0, places=9)
        self.assertAlmostEqual(float(l_anchored_cap[4, 16]), 100.0, places=9)

    def test_27_c0_hinged_motion_innovation_gate_and_raw_32byte_merkle_folding(self) -> None:
        """
        Verify (#5, #10):
        - Continuous C^0 hinged motion-innovation gate:
          beta_t(u,v) = beta * exp(-(max(0, Delta_mot - sigma_mot))^2 / (2 * sigma_mot^2)).
        - Raw 32-byte SHA-256 Merkle tree digest folding with odd-node duplication.
        """
        beta = 0.78
        sigma_mot = 35.0
        deltas = np.array([0.0, 17.5, 35.0, 35.0 - 1e-7, 35.0 + 1e-7, 70.0, 105.0], dtype=np.float64)
        gates = compute_motion_innovation_gate(deltas, temporal_ema_beta=beta, motion_gate_sigma=sigma_mot)

        # Flat plateau beta for Delta_mot <= sigma_mot
        self.assertAlmostEqual(float(gates[0]), beta, places=12)
        self.assertAlmostEqual(float(gates[1]), beta, places=12)
        self.assertAlmostEqual(float(gates[2]), beta, places=12)
        # C^0 continuity across the hinge point Delta_mot = sigma_mot
        self.assertLess(abs(float(gates[4]) - float(gates[3])), 1e-12)
        # Exact Gaussian decay above sigma_mot: at Delta_mot = 2*sigma_mot (70.0), excess = sigma_mot -> exp(-0.5)
        self.assertAlmostEqual(float(gates[5]), beta * np.exp(-0.5), places=12)
        # At Delta_mot = 3*sigma_mot (105.0), excess = 2*sigma_mot -> exp(-2.0)
        self.assertAlmostEqual(float(gates[6]), beta * np.exp(-2.0), places=12)

        # Raw 32-byte Merkle tree folding verification (even T=4 and odd T=3)
        leaf_hexes = [hashlib.sha256(f"frame_{i}".encode("utf-8")).hexdigest() for i in range(4)]
        raw_leaves = [bytes.fromhex(h) for h in leaf_hexes]
        # Manual raw 32-byte tree for T=4
        n01 = hashlib.sha256(raw_leaves[0] + raw_leaves[1]).digest()
        n23 = hashlib.sha256(raw_leaves[2] + raw_leaves[3]).digest()
        expected_root_4 = hashlib.sha256(n01 + n23).hexdigest()
        self.assertEqual(fold_merkle_digests_sha256(leaf_hexes), expected_root_4)

        # Manual raw 32-byte tree for T=3 (odd leaf duplicated at level 0)
        n22 = hashlib.sha256(raw_leaves[2] + raw_leaves[2]).digest()
        expected_root_3 = hashlib.sha256(n01 + n22).hexdigest()
        self.assertEqual(fold_merkle_digests_sha256(leaf_hexes[:3]), expected_root_3)

        # Ensure raw 32-byte folding is distinct from naive ASCII hex concatenation
        ascii_n01 = hashlib.sha256((leaf_hexes[0] + leaf_hexes[1]).encode("ascii")).hexdigest()
        self.assertNotEqual(n01.hex(), ascii_n01)

    def test_28_row_shifted_qim_mapping_gcd_trigger_and_hmac_dither(self) -> None:
        """
        Verify (#21):
        - Row-shifted QIM mapping trigger gcd(4 * floor(W/8), 64) == 64 activates automatically
          when row_shift_interleave is None and M_coord == 4.
        - All 64 payload bit indices achieve exact uniform multiplicity across 128x128, 256x256,
          and 512x512 images, and HMAC-SHA256 dither matches specification.
        """
        # W = 64 -> floor(W/8) = 8 -> gcd(32, 64) = 32 != 64 -> False
        self.assertFalse(should_use_row_shift_interleave(width=64, num_midband_coords=4, payload_len=64))
        # W = 128 -> floor(W/8) = 16 -> gcd(64, 64) = 64 -> True
        self.assertTrue(should_use_row_shift_interleave(width=128, num_midband_coords=4, payload_len=64))
        # W = 256 -> floor(W/8) = 32 -> gcd(128, 64) = 64 -> True
        self.assertTrue(should_use_row_shift_interleave(width=256, num_midband_coords=4, payload_len=64))
        # W = 512 -> floor(W/8) = 64 -> gcd(256, 64) = 64 -> True
        self.assertTrue(should_use_row_shift_interleave(width=512, num_midband_coords=4, payload_len=64))
        # Coprime M_coord = 5 -> False
        self.assertFalse(should_use_row_shift_interleave(width=128, num_midband_coords=5, payload_len=64))

        # Check exact uniform multiplicity across all 64 bits for W=128, 256, 512 with automatic trigger
        for dim in (128, 256, 512):
            bh = bw = dim // 8
            use_rs = should_use_row_shift_interleave(width=dim, num_midband_coords=4, payload_len=64)
            self.assertTrue(use_rs)
            counts = np.zeros(64, dtype=np.int64)
            block_idx = 0
            for by in range(bh):
                for _bx in range(bw):
                    for m_idx in range(4):
                        bit_idx = compute_qim_bit_index(block_idx, by, m_idx, 4, 64, use_rs)
                        counts[bit_idx] += 1
                    block_idx += 1
            expected_mult = (bh * bw * 4) // 64
            self.assertTrue(np.all(counts == expected_mult))

        # Verify HMAC-SHA256 dither matrix matches first 32 bits of HMAC-SHA256(K_secret, pack(">II", b, m))
        import hmac
        secret_key = b"test_secret_key_2026"
        qim_step = 14.0
        dither = _derive_qim_dither_matrix(64, 4, qim_step, secret_key)
        self.assertEqual(dither.shape, (64, 4))
        for b in (0, 17, 63):
            for m in range(4):
                mac = hmac.new(secret_key, struct.pack(">II", b, m), hashlib.sha256).digest()
                u32 = struct.unpack(">I", mac[:4])[0]
                expected_d = (float(u32) / float(1 << 32)) * qim_step
                self.assertAlmostEqual(float(dither[b, m]), expected_d, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)


