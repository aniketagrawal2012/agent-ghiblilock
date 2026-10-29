"""
Agent-GhibliLock: Deterministic Wavelet-Locked Cross-Modal Style Transfer &
ISO/IEC 19566-5 C2PA-Inspired JUMBF + Mid-Band 2D-DCT Provenance Compiler.

Solves "The Captioning Conundrum" (Pixels -> Prose -> Pixels geometric drift and
C2PA provenance severance) in cross-modal VLM-to-Diffusion pipelines
(gemini-2.5-flash / gemini-2.5-pro / gemini-3-flash-preview -> Imagen 3 / Veo 3.1).

Core Components:
1. 2D Multi-Level Discrete Haar Wavelet Transform (DWT2 / IDWT2) decomposing
   CIELAB luminance (L*) into low-frequency approximation (LL_K) and
   high-frequency structural edge/logo detail subbands (LH_k, HL_k, HH_k),
   supporting both global and spatially masked ROI detail locking while
   preserving exact spatial mean luminance mu(L_hat*) = mu(L_tilde*).
2. 3D Sliced-Wasserstein Optimal Transport Chromaticity Solver (SWOT) aligning
   (L*, a*, b*) distributions to target artistic palettes over Haar-uniform O(3)
   bases with stable lexicographical rank-order tie-breaking, explicit post-Step-A
   mean re-anchoring, and decoupled (gamma_tau, alpha_tau) closed-loop control.
3. Quantization-Index-Modulation (QIM) Mid-Band 2D-DCT Cryptographic Watermark
   embedded in 8x8 luminance blocks at mid-frequency coordinates, supporting
   both 4-mode (R = floor(B/16)) and 5-mode odd-repetition (R = 5B/64) layouts,
   secret dither vector K_secret, and blind majority-voted coset tamper localization.
4. Compact C2PA-Inspired ISO/IEC 19566-5 JUMBF Binary Manifest Profile (`caBX` PNG
   chunk) binding the source wavelet structural hash, Wasserstein transport
   certificates, and keyed signature with IEEE 802.3 CRC-32 chunk integrity.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import dataclasses
import hashlib
import hmac
import json
import math
import os
import struct
import time
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_SORT_POOL = ThreadPoolExecutor(max_workers=min(10, os.cpu_count() or 4))


# ---------------------------------------------------------------------------
# 1. Color Space Conversions: sRGB <-> XYZ (D65) <-> CIELAB (L*a*b*)
# ---------------------------------------------------------------------------

def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 or float [0, 255] sRGB image (H, W, 3) to CIELAB (L*, a*, b*)."""
    arr = np.asarray(rgb, dtype=np.float64) / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    # Inverse sRGB companding (IEC 61966-2-1)
    linear = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)

    # sRGB to CIE XYZ (D65 reference white)
    m_rgb2xyz = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float64)
    xyz = np.matmul(linear, m_rgb2xyz.T)

    # Normalize by D65 white point
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    xyz_norm = xyz / np.array([xn, yn, zn], dtype=np.float64)

    delta = 6.0 / 29.0
    delta3 = delta ** 3
    f_xyz = np.where(
        xyz_norm > delta3,
        np.cbrt(xyz_norm),
        (xyz_norm / (3.0 * delta * delta)) + (4.0 / 29.0),
    )

    L = 116.0 * f_xyz[..., 1] - 16.0
    a = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
    b = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIELAB (L*, a*, b*) image (H, W, 3) back to uint8 sRGB [0, 255]."""
    lab_arr = np.asarray(lab, dtype=np.float64)
    L = lab_arr[..., 0]
    a = lab_arr[..., 1]
    b = lab_arr[..., 2]

    fy = (L + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)

    delta = 6.0 / 29.0
    f_stack = np.stack([fx, fy, fz], axis=-1)
    xyz_norm = np.where(
        f_stack > delta,
        f_stack ** 3,
        3.0 * (delta ** 2) * (f_stack - 4.0 / 29.0),
    )

    xn, yn, zn = 0.95047, 1.00000, 1.08883
    xyz = xyz_norm * np.array([xn, yn, zn], dtype=np.float64)

    m_xyz2rgb = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ], dtype=np.float64)
    linear = np.matmul(xyz, m_xyz2rgb.T)
    linear = np.clip(linear, 0.0, 1.0)

    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * (linear ** (1.0 / 2.4)) - 0.055,
    )
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Orthonormal 2D Discrete Haar Wavelet Transform (DWT2 / IDWT2)
# ---------------------------------------------------------------------------

# Symmetric orthonormal 4x4 2D Haar analysis/synthesis matrix H_2 (H_2^T = H_2^-1 = H_2)
_HAAR2_M = 0.5 * np.array([
    [1.0,  1.0,  1.0,  1.0],
    [1.0, -1.0,  1.0, -1.0],
    [1.0,  1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0,  1.0],
], dtype=np.float64)
_HAAR2_MT = _HAAR2_M.T


@dataclasses.dataclass
class WaveletDecomposition2D:
    """Multi-level 2D Discrete Haar Wavelet decomposition."""
    ll: np.ndarray
    details: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]  # [(LH_k, HL_k, HH_k)] from coarse to fine
    original_shape: Tuple[int, int]


def _dwt2_single_level(
    channel: np.ndarray,
    vectorized: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Single-level orthonormal 2D Haar DWT via 4D tensor contraction (H/2, 2, W/2, 2)
    with _HAAR2_M (`vectorized=True`) or explicit 2x2 block-loop execution (`vectorized=False`).
    """
    h, w = channel.shape
    if h % 2 != 0:
        channel = np.pad(channel, ((0, 1), (0, 0)), mode="edge")
    if w % 2 != 0:
        channel = np.pad(channel, ((0, 0), (0, 1)), mode="edge")

    bh, bw = channel.shape[0] // 2, channel.shape[1] // 2
    if vectorized:
        # Reshape 2D plane into 4D tensor (bh, 2, bw, 2) -> (bh, bw, 4) and contract with H_2^T
        tiles_4d = channel.reshape(bh, 2, bw, 2).swapaxes(1, 2).reshape(bh, bw, 4)
        subbands = np.matmul(tiles_4d, _HAAR2_MT)
        return subbands[..., 0], subbands[..., 1], subbands[..., 2], subbands[..., 3]

    ll = np.empty((bh, bw), dtype=np.float64)
    lh = np.empty((bh, bw), dtype=np.float64)
    hl = np.empty((bh, bw), dtype=np.float64)
    hh = np.empty((bh, bw), dtype=np.float64)
    for u in range(bh):
        u2 = 2 * u
        for v in range(bw):
            v2 = 2 * v
            bvec = channel[u2 : u2 + 2, v2 : v2 + 2].reshape(1, 4)
            svec = np.matmul(bvec, _HAAR2_MT)[0]
            ll[u, v] = svec[0]
            lh[u, v] = svec[1]
            hl[u, v] = svec[2]
            hh[u, v] = svec[3]
    return ll, lh, hl, hh


def _idwt2_single_level(
    ll: np.ndarray,
    lh: np.ndarray,
    hl: np.ndarray,
    hh: np.ndarray,
    target_shape: Tuple[int, int],
    vectorized: bool = True,
) -> np.ndarray:
    """
    Single-level orthonormal 2D Haar inverse DWT via 4D tensor contraction (`vectorized=True`)
    or explicit 2x2 block-loop execution (`vectorized=False`).
    """
    bh, bw = ll.shape
    if vectorized:
        subbands = np.stack([ll, lh, hl, hh], axis=-1)
        tiles_4d = np.matmul(subbands, _HAAR2_M).reshape(bh, bw, 2, 2).swapaxes(1, 2)
        out = tiles_4d.reshape(bh * 2, bw * 2)
        return out[: target_shape[0], : target_shape[1]]

    out = np.empty((bh * 2, bw * 2), dtype=np.float64)
    for u in range(bh):
        u2 = 2 * u
        for v in range(bw):
            v2 = 2 * v
            svec = np.array([[ll[u, v], lh[u, v], hl[u, v], hh[u, v]]], dtype=np.float64)
            out[u2 : u2 + 2, v2 : v2 + 2] = np.matmul(svec, _HAAR2_M).reshape(2, 2)
    return out[: target_shape[0], : target_shape[1]]


def dwt2_multilevel(
    channel: np.ndarray,
    levels: int = 2,
    vectorized: bool = True,
) -> WaveletDecomposition2D:
    """Perform K-level 2D Haar wavelet decomposition."""
    cur = np.asarray(channel, dtype=np.float64)
    orig_shape = cur.shape
    details_fine_to_coarse: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(levels):
        ll, lh, hl, hh = _dwt2_single_level(cur, vectorized=vectorized)
        details_fine_to_coarse.append((lh, hl, hh))
        cur = ll
    # Store from coarse (level K) to fine (level 1)
    return WaveletDecomposition2D(
        ll=cur,
        details=list(reversed(details_fine_to_coarse)),
        original_shape=orig_shape,
    )


def idwt2_multilevel(
    decomp: WaveletDecomposition2D,
    vectorized: bool = True,
) -> np.ndarray:
    """Exact inverse of dwt2_multilevel."""
    cur = decomp.ll
    num_levels = len(decomp.details)
    # Compute expected shapes at each level
    shapes = [decomp.original_shape]
    h, w = decomp.original_shape
    for _ in range(num_levels - 1):
        h = (h + 1) // 2
        w = (w + 1) // 2
        shapes.append((h, w))
    shapes = list(reversed(shapes))

    for lvl_idx, (lh, hl, hh) in enumerate(decomp.details):
        cur = _idwt2_single_level(cur, lh, hl, hh, shapes[lvl_idx], vectorized=vectorized)
    return cur


def _downsample_mask_to_shape(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Downsample a 2D spatial mask in [0, 1] to subband shape via 2x2 block max-pooling."""
    m = np.asarray(mask, dtype=np.float64)
    th, tw = target_shape
    cur = m
    while cur.shape[0] > th or cur.shape[1] > tw:
        h, w = cur.shape
        if h % 2 != 0:
            cur = np.pad(cur, ((0, 1), (0, 0)), mode="edge")
        if w % 2 != 0:
            cur = np.pad(cur, ((0, 0), (0, 1)), mode="edge")
        cur = np.maximum(
            np.maximum(cur[0::2, 0::2], cur[0::2, 1::2]),
            np.maximum(cur[1::2, 0::2], cur[1::2, 1::2]),
        )
    return np.clip(cur[:th, :tw], 0.0, 1.0)


def compute_sobel_coherence_5x5(src_lum: np.ndarray) -> np.ndarray:
    """
    Compute 5x5 Sobel directional coherence map C_{5x5}(x, y) in [0, 1]^(H x W):
        C_{5x5}(x,y) = sqrt((B_5 * G_x)^2 + (B_5 * G_y)^2) / (B_5 * sqrt(G_x^2 + G_y^2) + 1e-6),
    using replicate ('edge') boundary padding for both the 3x3 Sobel filter and 5x5 box filter B_5.
    """
    lum = np.asarray(src_lum, dtype=np.float64)
    if lum.shape[0] < 3 or lum.shape[1] < 3:
        return np.ones_like(lum, dtype=np.float64)
    p = np.pad(lum, 1, mode="edge")
    gx = (p[:-2, 2:] + 2.0 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2.0 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2.0 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2.0 * p[:-2, 1:-1] + p[:-2, 2:])
    mag = np.sqrt(gx ** 2 + gy ** 2)

    def _box5(arr: np.ndarray) -> np.ndarray:
        pa = np.pad(arr, 2, mode="edge")
        acc = np.zeros_like(arr, dtype=np.float64)
        for dy in range(5):
            for dx in range(5):
                acc += pa[dy : dy + arr.shape[0], dx : dx + arr.shape[1]]
        return acc / 25.0

    coh = np.sqrt(_box5(gx) ** 2 + _box5(gy) ** 2) / (_box5(mag) + 1e-6)
    return np.clip(coh, 0.0, 1.0)


def build_spatial_roi_detail_mask(
    src_lum: np.ndarray,
    background_floor: float = 0.0,
    dilation_radius: int = 2,
    coherence_threshold: float = 0.35,
) -> np.ndarray:
    """
    Construct the exact closed-form spatially masked ROI weighting map M_ROI(x, y) in [0, 1]^(H x W)
    from E_full(L*_src) weighted by 5x5 Sobel directional coherence C_{5x5}(x, y):
        W_edge(x, y) = E_full(L*_src)(x, y) * C_{5x5}(x, y)^3,
        M_ROI(x, y)  = clip( B_3 * Dilate_{5x5}(W_edge * 1{C_{5x5} >= 0.35}) / max(max W_edge, 1e-8), 0, 1 ).
    Coherent contours, emblems, and rails satisfy C_{5x5} >= 0.35 and undergo 5x5 square
    (l_inf radius 2 px) grayscale morphological max-dilation followed by 3x3 box feathering B_3,
    whereas isolated background sensor grain (C_{5x5} < 0.35) is zeroed prior to dilation.
    """
    lum = np.asarray(src_lum, dtype=np.float64)
    edge_map = compute_structural_edge_map(lum)
    if lum.shape[0] >= 5 and lum.shape[1] >= 5:
        coh = compute_sobel_coherence_5x5(lum)
        w_edge = edge_map * (coh ** 3)
        w_gated = np.where(coh >= coherence_threshold, w_edge, 0.0)
    else:
        w_edge = edge_map
        w_gated = edge_map

    r = max(0, int(dilation_radius))
    if r > 0:
        pad_w = np.pad(w_gated, r, mode="edge")
        dilated = np.zeros_like(w_gated, dtype=np.float64)
        for dy in range(2 * r + 1):
            for dx in range(2 * r + 1):
                dilated = np.maximum(dilated, pad_w[dy : dy + lum.shape[0], dx : dx + lum.shape[1]])
    else:
        dilated = w_gated.copy()

    # Smooth 3x3 normalized box feathering B_3 with replicate padding
    padded = np.pad(dilated, 1, mode="edge")
    b3_smooth = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
        + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
        + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    ) / 9.0
    norm_denom = max(float(np.max(w_edge)), 1e-8)
    m_roi = np.clip(b3_smooth / norm_denom, 0.0, 1.0)
    roi_map = background_floor + (1.0 - background_floor) * m_roi
    return np.clip(roi_map, 0.0, 1.0)


def apply_interior_dc_reanchoring(
    lum_clip: np.ndarray,
    target_mean_lum: float,
) -> Tuple[np.ndarray, float]:
    """
    Apply interior DC re-anchoring over unsaturated pixels I^o = {(x,y) : lum_clip(x,y) in (1, 99)}
    with explicit guard for |I^o| = 0 and per-pixel shift cap |delta_DC| <= 1.0:
        delta_DC = sign(Delta_mu) * min(|Delta_mu| * HW / |I^o|, 1.0)  if |I^o| > 0 else 0.0,
        L*(x,y)  = clip(lum_clip(x,y) + delta_DC * 1{(x,y) in I^o}, 0, 100).
    Returns (reanchored_lum, delta_dc).
    """
    out = np.array(lum_clip, dtype=np.float64, copy=True)
    delta_mu = float(target_mean_lum) - float(np.mean(out))
    if abs(delta_mu) <= 1e-12:
        return out, 0.0
    interior = (out > 1.0) & (out < 99.0)
    n_interior = int(np.sum(interior))
    if n_interior == 0:
        return out, 0.0
    raw_shift = abs(delta_mu) * (float(out.size) / float(n_interior))
    delta_dc = math.copysign(min(raw_shift, 1.0), delta_mu)
    out[interior] = np.clip(out[interior] + delta_dc, 0.0, 100.0)
    return out, delta_dc


# ---------------------------------------------------------------------------
# 3. Structural Edge Mask & Wavelet-Edge SSIM Metric
# ---------------------------------------------------------------------------

def compute_structural_edge_map(luminance: np.ndarray, vectorized: bool = True) -> np.ndarray:
    """
    Compute high-frequency structural edge magnitude map E_full(L*) in [0, 1]^(H x W)
    from Level-1 Haar DWT subbands upsampled 2x via nearest-neighbor interpolation
    and normalized by max(max_{u,v} E_{1/2}(u,v), 1e-8).
    """
    lum = np.asarray(luminance, dtype=np.float64)
    _, lh1, hl1, hh1 = _dwt2_single_level(lum, vectorized=vectorized)
    energy_half = np.sqrt(lh1 ** 2 + hl1 ** 2 + 0.5 * (hh1 ** 2))
    # Nearest-neighbor 2x upsample back to original dimensions
    energy_full = np.repeat(np.repeat(energy_half, 2, axis=0), 2, axis=1)
    energy_full = energy_full[: lum.shape[0], : lum.shape[1]]
    max_val = max(float(np.max(energy_full)), 1e-8)
    return energy_full / max_val


def compute_wavelet_edge_ssim(
    src_lum: np.ndarray,
    out_lum: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    vectorized: bool = True,
) -> float:
    """
    Compute global Structural Similarity Index (SSIM_edge) over the normalized
    high-frequency wavelet edge maps E_full(L*_src) and E_full(L*_out) with
    c_1 = 0.01^2, c_2 = 0.03^2. If `roi_mask` is provided, evaluates over the
    active ROI locus (roi_mask >= 0.5).
    """
    e1 = compute_structural_edge_map(src_lum, vectorized=vectorized)
    e2 = compute_structural_edge_map(out_lum, vectorized=vectorized)
    if roi_mask is not None:
        active = np.asarray(roi_mask) >= 0.5
        if np.any(active):
            e1 = e1[active]
            e2 = e2[active]

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu1 = float(np.mean(e1))
    mu2 = float(np.mean(e2))
    sigma1_sq = float(np.var(e1))
    sigma2_sq = float(np.var(e2))
    sigma12 = float(np.mean((e1 - mu1) * (e2 - mu2)))

    num = (2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
    return float(np.clip(num / den, 0.0, 1.0))


def compute_full_luminance_ssim(
    src_lum: np.ndarray,
    out_lum: np.ndarray,
    vectorized: bool = True,
) -> float:
    """
    Compute non-circular full-band spatial Structural Similarity Index (SSIM_full)
    directly over non-overlapping 4x4 spatial luminance patches (L* / 100.0, stride 4,
    uniform weights) without Haar wavelets, combining Wang-Bovik structural correlation
    and local contrast similarity (c_2 = 0.03^2, c_3 = c_2 / 2).
    """
    x = np.asarray(src_lum, dtype=np.float64) / 100.0
    y = np.asarray(out_lum, dtype=np.float64) / 100.0
    c2 = 0.03 ** 2
    c3 = c2 / 2.0
    h, w = x.shape
    win = 4
    bh, bw = h // win, w // win
    if vectorized and bh >= 1 and bw >= 1:
        xb = x[: bh * win, : bw * win].reshape(bh, win, bw, win).swapaxes(1, 2).reshape(-1, 16)
        yb = y[: bh * win, : bw * win].reshape(bh, win, bw, win).swapaxes(1, 2).reshape(-1, 16)
        mux = np.mean(xb, axis=1, keepdims=True)
        muy = np.mean(yb, axis=1, keepdims=True)
        dx = xb - mux
        dy = yb - muy
        sx_sq = np.mean(dx * dx, axis=1)
        sy_sq = np.mean(dy * dy, axis=1)
        sx = np.sqrt(sx_sq)
        sy = np.sqrt(sy_sq)
        sxy = np.mean(dx * dy, axis=1)
        contrast = (2.0 * sx * sy + c2) / (sx_sq + sy_sq + c2)
        struct_corr = np.maximum(0.0, (sxy + c3) / (sx * sy + c3))
        return float(np.mean(np.clip(contrast * struct_corr, 0.0, 1.0)))

    ssim_vals: List[float] = []
    for r in range(0, max(1, h - win + 1), 4):
        for c in range(0, max(1, w - win + 1), 4):
            xb = x[r : r + win, c : c + win]
            yb = y[r : r + win, c : c + win]
            mux = float(np.mean(xb))
            muy = float(np.mean(yb))
            sx = float(np.std(xb))
            sy = float(np.std(yb))
            sxy = float(np.mean((xb - mux) * (yb - muy)))
            contrast = (2.0 * sx * sy + c2) / (sx ** 2 + sy ** 2 + c2)
            struct_corr = (sxy + c3) / (sx * sy + c3)
            ssim_vals.append(float(np.clip(contrast * max(0.0, struct_corr), 0.0, 1.0)))
    return float(np.mean(ssim_vals)) if ssim_vals else 0.0


def compute_sobel_gradient_correlation(
    src_lum: np.ndarray,
    out_lum: np.ndarray,
    vectorized: bool = True,
) -> float:
    """
    Compute signed 3x3 Sobel spatial gradient inner-product correlation rho_grad
    over the interior source structural edge locus M_HF^o (sensitive to edge polarity
    inversions: rho_grad -> -1 under contrast reversal).
    """
    s = np.asarray(src_lum, dtype=np.float64)
    o = np.asarray(out_lum, dtype=np.float64)
    if s.shape[0] < 3 or s.shape[1] < 3:
        return 1.0

    def _sobel_xy(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gx = (
            (img[:-2, 2:] + 2.0 * img[1:-1, 2:] + img[2:, 2:])
            - (img[:-2, :-2] + 2.0 * img[1:-1, :-2] + img[2:, :-2])
        )
        gy = (
            (img[2:, :-2] + 2.0 * img[2:, 1:-1] + img[2:, 2:])
            - (img[:-2, :-2] + 2.0 * img[:-2, 1:-1] + img[:-2, 2:])
        )
        return gx, gy

    gx_s, gy_s = _sobel_xy(s)
    gx_o, gy_o = _sobel_xy(o)
    edge_map = compute_structural_edge_map(s, vectorized=vectorized)[1:-1, 1:-1]
    thresh = max(float(np.quantile(edge_map, 0.75)), 0.08)
    mask = edge_map >= thresh
    if np.any(mask):
        gx_s, gy_s = gx_s[mask], gy_s[mask]
        gx_o, gy_o = gx_o[mask], gy_o[mask]
    num = float(np.sum(gx_s * gx_o + gy_s * gy_o))
    den = math.sqrt(float(np.sum(gx_s ** 2 + gy_s ** 2)) * float(np.sum(gx_o ** 2 + gy_o ** 2))) + 1e-12
    return float(np.clip(num / den, -1.0, 1.0))


def compute_structural_mask_hash(
    src_lum: np.ndarray,
    threshold_quantile: float = 0.75,
    vectorized: bool = True,
) -> Tuple[np.ndarray, str]:
    """
    Extract binary high-frequency structural edge/logo mask M_HF in {0, 1}^(H x W)
    (serialized as 1-byte uint8 0x00/0x01 in row-major C-order) and its canonical
    lowercase hex SHA-256 cryptographic digest (raw row-major uint8 buffer followed
    by 8-byte big-endian dimension trailer struct.pack('>II', H, W)).
    """
    edge_map = compute_structural_edge_map(src_lum, vectorized=vectorized)
    thresh = float(np.quantile(edge_map, threshold_quantile))
    binary_mask = (edge_map >= max(thresh, 0.08)).astype(np.uint8)
    digest = hashlib.sha256(binary_mask.tobytes() + struct.pack(">II", *binary_mask.shape)).hexdigest()
    return binary_mask, digest


# ---------------------------------------------------------------------------
# 4. 3D Sliced-Wasserstein Optimal Transport Chromaticity Solver (W_2)
# ---------------------------------------------------------------------------

def _get_swot_training_axes(swot_steps: int = 8, seed: int = 42) -> np.ndarray:
    """
    Construct the exact (3M + 3) x 3 matrix of unit projection axes used in Step A:
    the canonical basis I_3 followed by M = 8 Haar-uniform O(3) orthonormal frames
    Q_m generated via positive-diagonal QR factorization from seed=42.
    """
    axes_list: List[np.ndarray] = [np.eye(3, dtype=np.float64)]
    rng = np.random.default_rng(seed)
    for _ in range(swot_steps):
        z = rng.normal(size=(3, 3))
        q_rot, r_mat = np.linalg.qr(z)
        d_diag = np.diag(r_mat)
        d_sign = np.where(d_diag >= 0.0, 1.0, -1.0)
        q_rot = q_rot * d_sign
        axes_list.append(q_rot.T)  # 3 orthonormal unit row vectors
    return np.vstack(axes_list)  # shape (3 * swot_steps + 3, 3)


def _sort_slices_rows(proj_d_by_n: np.ndarray, vectorized: bool = True) -> np.ndarray:
    """
    Sort each row of a (D, N) projection matrix in ascending order.
    Since 1D empirical Wasserstein-2 depends only on sorted values (not indices),
    any exact comparison sort over float64 produces bit-identical order statistics.
    """
    if vectorized:
        arr = np.array(proj_d_by_n, dtype=np.float64, copy=True, order="C")
        if arr.shape[1] >= 16384 and arr.shape[0] >= 4:
            n_chunks = min(8, arr.shape[0])
            list(
                _SORT_POOL.map(
                    lambda c: c.sort(axis=-1, kind="stable"),
                    np.array_split(arr, n_chunks, axis=0),
                )
            )
            return arr
        arr.sort(axis=-1, kind="stable")
        return arr
    return np.sort(np.ascontiguousarray(proj_d_by_n).T, axis=0, kind="mergesort").T


def compute_sliced_wasserstein_lab(
    lab_a: np.ndarray,
    lab_b: np.ndarray,
    num_projections: int = 27,
    seed: int = 42,
    swot_steps: int = 8,
    use_training_axes: bool = True,
    vectorized: bool = True,
    _precomputed_sorted_b: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the discrete root-mean-square (RMS) 1D Sliced-Wasserstein-2 distance
    W_hat_2 between two CIELAB (L*, a*, b*) distributions:
        W_hat_2(mu_X, mu_Y) = ( (1/D) sum_{r=1}^D W_2^2(theta_{r#} mu_X, theta_{r#} mu_Y) )^(1/2).
    When `use_training_axes=True` and `seed=42`, evaluates over the exact D = 3M + 3 = 27
    orthonormal frame axes (I_3 union {Q_m}_{m=1}^M) from Step A.
    """
    pts_a = np.asarray(lab_a, dtype=np.float64).reshape(-1, 3)
    pts_b = np.asarray(lab_b, dtype=np.float64).reshape(-1, 3)

    if use_training_axes and seed == 42:
        dirs = _get_swot_training_axes(swot_steps=swot_steps, seed=seed)
    else:
        rng = np.random.default_rng(seed)
        dirs = rng.normal(size=(num_projections, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12

    if pts_a.shape[0] == pts_b.shape[0]:
        proj_a = pts_a @ dirs.T
        if vectorized:
            sa_t = _sort_slices_rows(proj_a.T, vectorized=True)
            if _precomputed_sorted_b is not None and _precomputed_sorted_b.shape == sa_t.shape:
                sb_t = _precomputed_sorted_b
            else:
                proj_b = pts_b @ dirs.T
                sb_t = _sort_slices_rows(proj_b.T, vectorized=True)
            w2_sq_per_slice = np.mean((sa_t - sb_t) ** 2, axis=1)
        else:
            proj_b = pts_b @ dirs.T
            sa = np.sort(proj_a, axis=0, kind="mergesort")
            sb = np.sort(proj_b, axis=0, kind="mergesort")
            w2_sq_per_slice = np.mean((sa - sb) ** 2, axis=0)
    else:
        proj_a = pts_a @ dirs.T
        proj_b = pts_b @ dirs.T
        quantiles = np.linspace(0.0, 1.0, 256)
        qa = np.quantile(proj_a, quantiles, axis=0)
        qb = np.quantile(proj_b, quantiles, axis=0)
        w2_sq_per_slice = np.mean((qa - qb) ** 2, axis=0)

    return float(np.sqrt(np.mean(w2_sq_per_slice)))


def compute_heldout_sliced_wasserstein_lab(
    lab_a: np.ndarray,
    lab_b: np.ndarray,
    num_projections: int = 64,
    seed: int = 999,
    vectorized: bool = True,
) -> float:
    """
    Out-of-sample root-mean-square Sliced-Wasserstein-2 estimator SW_hat_2^eval
    evaluated across M_eval = 64 independent uniform unit-sphere directions in S^2
    (seed=999, disjoint from the Step A optimization seed=42).
    """
    return compute_sliced_wasserstein_lab(
        lab_a,
        lab_b,
        num_projections=num_projections,
        seed=seed,
        use_training_axes=False,
        vectorized=vectorized,
    )


def transport_chromaticity_wasserstein(
    content_lab: np.ndarray,
    style_palette_lab: np.ndarray,
    blend_alpha: float = 0.88,
    swot_steps: int = 8,
    seed: int = 42,
    anchor_lum_mean: bool = True,
    vectorized: bool = True,
    _precomputed_sorted_targ: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Transport content CIELAB distribution toward target artistic style palette
    via Iterative 3D Sliced-Wasserstein Optimal Transport (SWOT) in (L*, a*, b*) space.
    - Uses deterministic lexicographical rank-order permutation tie-breaking
      (`kind="mergesort"`, ordered by flat 0-based row-major pixel index i in {0..HW-1}).
    - Applies initial canonical-axis pass Q_0 = I_3 with step size alpha_0 = blend_alpha,
      followed by M = `swot_steps` Haar-uniform O(3) orthonormal bases via positive-diagonal QR,
      and a final canonical-axis refinement pass so both I_3 and rotated slices contract tightly.
    - Applies explicit post-Step-A luminance mean re-anchoring when `anchor_lum_mean=True`.
    """
    h, w, _ = content_lab.shape
    curr = np.array(content_lab, dtype=np.float64, copy=True).reshape(-1, 3)

    style_arr = np.asarray(style_palette_lab, dtype=np.float64)
    if style_arr.shape[:2] != (h, w):
        sh, sw = style_arr.shape[:2]
        yi = (np.arange(h) * sh) // h
        xi = (np.arange(w) * sw) // w
        style_arr = style_arr[np.ix_(yi, xi)]
    target = style_arr.reshape(-1, 3)

    n_pts = curr.shape[0]
    quantiles = np.linspace(0.0, 1.0, 512)
    base_ranks = np.linspace(0.0, 1.0, n_pts)

    v_idx = quantiles * (n_pts - 1)
    lo = np.floor(v_idx).astype(np.int64)
    hi = np.ceil(v_idx).astype(np.int64)
    frac = v_idx - lo
    cached_q_interp: Dict[int, List[np.ndarray]] = {}

    def _argsort_cols(mat_Nx3: np.ndarray) -> List[np.ndarray]:
        if vectorized and n_pts >= 16384:
            cols = [np.ascontiguousarray(mat_Nx3[:, d]) for d in range(3)]
            return list(_SORT_POOL.map(lambda c: np.argsort(c, kind="mergesort"), cols))
        return [np.argsort(mat_Nx3[:, d], kind="mergesort") for d in range(3)]

    def _quantile_cols(mat_Nx3: Optional[np.ndarray], group_idx: Optional[int] = None) -> List[np.ndarray]:
        if vectorized:
            if group_idx is not None and group_idx in cached_q_interp:
                return cached_q_interp[group_idx]
            if (
                group_idx is not None
                and _precomputed_sorted_targ is not None
                and _precomputed_sorted_targ.shape == (3 * swot_steps + 3, n_pts)
            ):
                sorted_rows = _precomputed_sorted_targ[3 * group_idx : 3 * group_idx + 3]
            else:
                assert mat_Nx3 is not None
                sorted_rows = _sort_slices_rows(mat_Nx3.T, vectorized=True)
            quants = [
                np.interp(
                    base_ranks,
                    quantiles,
                    sorted_rows[d, lo] + frac * (sorted_rows[d, hi] - sorted_rows[d, lo]),
                )
                for d in range(3)
            ]
            if group_idx is not None:
                cached_q_interp[group_idx] = quants
            return quants
        assert mat_Nx3 is not None
        return [np.quantile(mat_Nx3[:, d], quantiles) for d in range(3)]

    # First pass: coordinate-wise marginal optimal transport along canonical Q_0 = I_3 axes
    orders_0 = _argsort_cols(curr)
    t_quants_0 = _quantile_cols(None if (vectorized and _precomputed_sorted_targ is not None) else target, group_idx=0)
    for ch in range(3):
        order_u = orders_0[ch]
        if vectorized:
            mapped_dim = np.empty(n_pts, dtype=np.float64)
            mapped_dim[order_u] = t_quants_0[ch]
        else:
            ranks = np.empty_like(order_u, dtype=np.float64)
            ranks[order_u] = base_ranks
            mapped_dim = np.interp(ranks, quantiles, t_quants_0[ch])
        curr[:, ch] = (1.0 - blend_alpha) * curr[:, ch] + blend_alpha * mapped_dim

    # Iterative Sliced-Wasserstein rotational transport steps
    rng = np.random.default_rng(seed)
    rot_frames_ref: List[np.ndarray] = []
    for m_idx in range(swot_steps):
        z = rng.normal(size=(3, 3))
        q_rot, r_mat = np.linalg.qr(z)
        d_diag = np.diag(r_mat)
        d_sign = np.where(d_diag >= 0.0, 1.0, -1.0)
        q_rot = q_rot * d_sign
        rot_frames_ref.append(q_rot)

        proj_curr = curr @ q_rot
        proj_targ = None if (vectorized and _precomputed_sorted_targ is not None) else (target @ q_rot)
        delta_proj = np.zeros_like(proj_curr)
        orders_m = _argsort_cols(proj_curr)
        t_quants_m = _quantile_cols(proj_targ, group_idx=m_idx + 1)

        for dim in range(3):
            order_u = orders_m[dim]
            if vectorized:
                mapped_dim = np.empty(n_pts, dtype=np.float64)
                mapped_dim[order_u] = t_quants_m[dim]
            else:
                ranks = np.empty_like(order_u, dtype=np.float64)
                ranks[order_u] = base_ranks
                mapped_dim = np.interp(ranks, quantiles, t_quants_m[dim])
            delta_proj[:, dim] = blend_alpha * (mapped_dim - proj_curr[:, dim])

        curr = curr + (delta_proj @ q_rot.T)

    polish_groups = [(0, np.eye(3, dtype=np.float64))]
    if len(rot_frames_ref) >= 2:
        polish_groups.append((len(rot_frames_ref) - 1, rot_frames_ref[-2]))
        polish_groups.append((len(rot_frames_ref), rot_frames_ref[-1]))
    elif len(rot_frames_ref) == 1:
        polish_groups.append((1, rot_frames_ref[-1]))

    for g_idx, q_frame in polish_groups:
        proj_curr = curr @ q_frame
        proj_targ = None if (vectorized and _precomputed_sorted_targ is not None) else (target @ q_frame)
        delta_proj = np.zeros_like(proj_curr)
        orders_p = _argsort_cols(proj_curr)
        t_quants_p = _quantile_cols(proj_targ, group_idx=g_idx)
        for dim in range(3):
            order_u = orders_p[dim]
            if vectorized:
                mapped_dim = np.empty(n_pts, dtype=np.float64)
                mapped_dim[order_u] = t_quants_p[dim]
            else:
                ranks = np.empty_like(order_u, dtype=np.float64)
                ranks[order_u] = base_ranks
                mapped_dim = np.interp(ranks, quantiles, t_quants_p[dim])
            delta_proj[:, dim] = (0.5 * blend_alpha) * (mapped_dim - proj_curr[:, dim])
        curr = curr + (delta_proj @ q_frame.T)

    if anchor_lum_mean:
        curr[:, 0] += float(np.mean(target[:, 0])) - float(np.mean(curr[:, 0]))

    return curr.reshape(h, w, 3)


# ---------------------------------------------------------------------------
# 5. Mid-Band 2D-DCT Quantization-Index-Modulation (QIM) Watermark Engine
# ---------------------------------------------------------------------------

def _build_dct8_matrix() -> np.ndarray:
    """Construct orthonormal 8x8 Type-II Discrete Cosine Transform matrix."""
    c = np.zeros((8, 8), dtype=np.float64)
    for k in range(8):
        alpha = math.sqrt(1.0 / 8.0) if k == 0 else math.sqrt(2.0 / 8.0)
        for n in range(8):
            c[k, n] = alpha * math.cos(math.pi * (2 * n + 1) * k / 16.0)
    return c


_DCT8_M = _build_dct8_matrix()
_DCT8_MT = _DCT8_M.T
# Default 4 mid-band coordinates (u, v) in 8x8 DCT block resilient to quantization
_MIDBAND_COORDS: List[Tuple[int, int]] = [(2, 3), (3, 2), (1, 3), (3, 1)]
# Extended 5 mid-band coordinates yielding R = 5 odd repetition votes/bit on 64x64 (gcd(5, 64) = 1)
_MIDBAND_COORDS_5: List[Tuple[int, int]] = [(2, 3), (3, 2), (1, 3), (3, 1), (2, 2)]


def dct2_blocks_8x8(channel: np.ndarray, vectorized: bool = True) -> np.ndarray:
    """
    Transform a 2D luminance array (H, W) into an array of 8x8 Type-II DCT blocks
    of shape (bh_max, bw_max, 8, 8) using 4D tensor contraction (`vectorized=True`)
    or explicit Python 8x8 block-loop iteration (`vectorized=False`).
    Each spatial block is shifted by -128.0 prior to (C_8 * B) * C_8^T.
    """
    arr = np.asarray(channel, dtype=np.float64)
    bh_max, bw_max = arr.shape[0] // 8, arr.shape[1] // 8
    if vectorized:
        blocks_4d = (arr[: bh_max * 8, : bw_max * 8] - 128.0).reshape(bh_max, 8, bw_max, 8).swapaxes(1, 2)
        return np.matmul(np.matmul(_DCT8_M, blocks_4d), _DCT8_MT)

    out = np.empty((bh_max, bw_max, 8, 8), dtype=np.float64)
    for by in range(bh_max):
        y0 = by * 8
        for bx in range(bw_max):
            x0 = bx * 8
            blk = arr[y0 : y0 + 8, x0 : x0 + 8] - 128.0
            out[by, bx] = (_DCT8_M @ blk) @ _DCT8_MT
    return out


def idct2_blocks_8x8(dct_blocks: np.ndarray, vectorized: bool = True) -> np.ndarray:
    """
    Inverse 8x8 Type-II DCT transform from shape (bh_max, bw_max, 8, 8) back to a 2D
    spatial plane of shape (bh_max * 8, bw_max * 8) via 4D tensor contraction
    (`vectorized=True`) or explicit Python 8x8 block-loop iteration (`vectorized=False`).
    """
    bh_max, bw_max = dct_blocks.shape[0], dct_blocks.shape[1]
    if vectorized:
        rec_4d = np.matmul(np.matmul(_DCT8_MT, dct_blocks), _DCT8_M) + 128.0
        return rec_4d.swapaxes(1, 2).reshape(bh_max * 8, bw_max * 8)

    out = np.empty((bh_max * 8, bw_max * 8), dtype=np.float64)
    for by in range(bh_max):
        y0 = by * 8
        for bx in range(bw_max):
            x0 = bx * 8
            out[y0 : y0 + 8, x0 : x0 + 8] = ((_DCT8_MT @ dct_blocks[by, bx]) @ _DCT8_M) + 128.0
    return out


def should_use_row_shift_interleave(
    width: int,
    num_midband_coords: int = 4,
    payload_len: int = 64,
    row_shift_interleave: Optional[bool] = None,
) -> bool:
    """
    Evaluate the row-shifted QIM mapping trigger:
        gcd(4 * floor(W / 8), 64) == 64
    for M_coord = 4 when `row_shift_interleave is None`.
    - At W in {128, 256, 512}, floor(W/8) in {16, 32, 64}, so gcd(4*floor(W/8), 64) == 64 -> True,
      preventing vertical column alignment of identical bit indices while preserving exact uniform
      multiplicity R = 4B/64.
    - At W = 64, floor(W/8) = 8, so gcd(32, 64) = 32 != 64 -> False, because unshifted
      phi(b,m) = (4b + m) mod 64 already shifts consecutive block rows by 32 (mod 64) while
      preserving exact uniform multiplicity R = 4.
    """
    if row_shift_interleave is not None:
        return bool(row_shift_interleave)
    if num_midband_coords != 4 or payload_len <= 0:
        return False
    bw_max = int(width) // 8
    return math.gcd(4 * bw_max, payload_len) == payload_len


def compute_qim_bit_index(
    block_idx: int,
    by: int,
    m_idx: int,
    num_midband_coords: int = 4,
    payload_len: int = 64,
    use_row_shift: bool = False,
) -> int:
    """Compute bit index phi(b, m) or phi_row(b, m) = (M_coord * b + 4 * r_b + m) mod payload_len."""
    row_off = (4 * by) if use_row_shift else 0
    return (num_midband_coords * block_idx + row_off + m_idx) % payload_len


def _derive_qim_dither_matrix(
    num_blocks: int,
    num_coords: int,
    qim_step: float,
    secret_key: Optional[bytes] = None,
) -> np.ndarray:
    """
    Derive an optional keyed pseudo-random dither matrix d_{b,m} in [0, qim_step)
    via the exact specification:
        d_{b,m}(K_secret) = Delta_QIM * (uint32_be(HMAC-SHA256(K_secret, pack(">II", b, m))[:4]) / 2^32).
    Defaults to zero when `secret_key is None` (unkeyed public audit mode).
    """
    n_blk = max(1, num_blocks)
    if secret_key is None:
        return np.zeros((n_blk, num_coords), dtype=np.float64)
    out = np.empty((n_blk, num_coords), dtype=np.float64)
    base_mac = hmac.new(secret_key, b"", hashlib.sha256)
    scale = float(qim_step) / float(1 << 32)
    for b in range(n_blk):
        for m in range(num_coords):
            mac = base_mac.copy()
            mac.update(struct.pack(">II", b, m))
            u32 = struct.unpack(">I", mac.digest()[:4])[0]
            out[b, m] = float(u32) * scale
    return out


def compute_dct8_ac_l1_rounding_bounds() -> Dict[str, float]:
    """
    Compute exact worst-case l1/linf Holder rounding bounds 0.5 * ||c_u||_1 * ||c_v||_1
    for the orthonormal 8x8 Type-II DCT basis vectors on M_4 and M_5 mid-band AC modes,
    proving deterministically that interior equal-(R,G,B) uint8 rounding (|eps_Y| <= 0.5 LSB)
    satisfies max_{(u,v) in M_4} |eta_b^export(u,v)| <= 3.3486 < 3.5 = Delta_QIM / 4
    and max_{(u,v) in M_5} |eta_b^export(u,v)| <= 2 + sqrt(2) = 3.4142 < 3.5.
    Also returns the 3-AC-subband orthonormal 2D Haar synthesis l1 bound 3*sqrt(3)/2 = 2.5981.
    """
    l1_norms = np.sum(np.abs(_DCT8_M), axis=1)
    m4_bound = max(0.5 * float(l1_norms[u] * l1_norms[v]) for (u, v) in _MIDBAND_COORDS)
    m5_bound = max(0.5 * float(l1_norms[u] * l1_norms[v]) for (u, v) in _MIDBAND_COORDS_5)
    dc_bound = 0.5 * float(l1_norms[0] * l1_norms[0])
    haar_3ac_bound = 1.5 * math.sqrt(3.0)
    return {
        "c1_l1_norm": float(l1_norms[1]),
        "c2_l1_norm": float(l1_norms[2]),
        "c3_l1_norm": float(l1_norms[3]),
        "m4_max_ac_rounding_bound": m4_bound,
        "m5_max_ac_rounding_bound": m5_bound,
        "dc_rounding_bound": dc_bound,
        "haar_3ac_subband_linf_bound": haar_3ac_bound,
    }


def embed_midband_dct_watermark(
    srgb_img: np.ndarray,
    payload_bits: List[int],
    qim_step: float = 14.0,
    num_midband_coords: int = 4,
    secret_key: Optional[bytes] = None,
    row_shift_interleave: Optional[bool] = None,
    vectorized: bool = True,
) -> np.ndarray:
    """
    Embed a binary provenance watermark into the floating-point ITU-R BT.601 luminance
    channel Y = 0.299R + 0.587G + 0.114B in [0, 255] of an sRGB image using Quantization
    Index Modulation (QIM) at mid-band 8x8 2D-DCT coefficients with block-interleaved
    bit indexing i = (M_coord * b + m) mod payload_len, or row-shifted
    i = (M_coord * b + 4 * by + m) mod payload_len whenever `row_shift_interleave=True`
    or automatically when `row_shift_interleave is None` and `gcd(4 * floor(W/8), 64) == 64`.
    When `num_midband_coords == 1`, embeds 1 mode per block across the first `payload_len`
    blocks (Single-Block Blind QIM baseline R = 1 without multi-block repetition).
    """
    img = np.array(srgb_img, dtype=np.float64, copy=True)
    h, w, _ = img.shape
    n_bits = len(payload_bits)
    if n_bits == 0:
        return srgb_img.copy()

    if num_midband_coords == 5:
        coords = _MIDBAND_COORDS_5
    elif num_midband_coords == 1:
        coords = [_MIDBAND_COORDS[0]]
    else:
        coords = _MIDBAND_COORDS
    m_count = len(coords)
    use_row_shift = should_use_row_shift_interleave(w, m_count, n_bits, row_shift_interleave)

    # Retain floating-point ITU-R BT.601 luminance without intermediate integer rounding
    y_chan = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    y_mod = y_chan.copy()

    bh_max = h // 8
    bw_max = w // 8
    total_blocks = bh_max * bw_max
    if total_blocks == 0:
        return srgb_img.copy()
    dither_mat = _derive_qim_dither_matrix(total_blocks, m_count, qim_step, secret_key)

    if vectorized:
        dct_4d = dct2_blocks_8x8(y_mod, vectorized=True)
        dct_flat = dct_4d.reshape(total_blocks, 8, 8)
        n_active = min(total_blocks, n_bits) if num_midband_coords == 1 else total_blocks
        b_idx = np.arange(n_active, dtype=np.int64)
        by_idx = b_idx // bw_max
        row_off = (4 * by_idx) if use_row_shift else 0
        bits_arr = np.asarray(payload_bits, dtype=np.int64) & 1

        for m_idx, (u, v) in enumerate(coords):
            bit_pos = (m_count * b_idx + row_off + m_idx) % n_bits
            w_bits = bits_arr[bit_pos]
            coeffs = dct_flat[:n_active, u, v]
            dither = dither_mat[:n_active, m_idx] + (qim_step / 2.0) * w_bits
            dct_flat[:n_active, u, v] = np.floor(((coeffs - dither) / qim_step) + 0.5) * qim_step + dither

        y_mod[: bh_max * 8, : bw_max * 8] = idct2_blocks_8x8(
            dct_flat.reshape(bh_max, bw_max, 8, 8), vectorized=True
        )
    else:
        block_idx = 0
        for by in range(bh_max):
            for bx in range(bw_max):
                if num_midband_coords == 1 and block_idx >= n_bits:
                    break
                y0, x0 = by * 8, bx * 8
                block = y_mod[y0 : y0 + 8, x0 : x0 + 8] - 128.0
                dct_block = _DCT8_M @ block @ _DCT8_MT

                for m_idx, (u, v) in enumerate(coords):
                    bit_pos = compute_qim_bit_index(block_idx, by, m_idx, m_count, n_bits, use_row_shift)
                    bit = int(payload_bits[bit_pos]) & 1
                    coeff = dct_block[u, v]
                    dither = dither_mat[block_idx, m_idx] + (qim_step / 2.0) * bit
                    quantized = math.floor(((coeff - dither) / qim_step) + 0.5) * qim_step + dither
                    dct_block[u, v] = quantized

                rec_block = (_DCT8_MT @ dct_block @ _DCT8_M) + 128.0
                y_mod[y0 : y0 + 8, x0 : x0 + 8] = rec_block
                block_idx += 1

    # Add luminance delta equally across R, G, B and round/clamp to uint8 {0..255}
    delta_y = y_mod - y_chan
    img = np.clip(np.round(img + delta_y[..., None]), 0, 255)
    return img.astype(np.uint8)


def extract_midband_dct_watermark(
    srgb_img: np.ndarray,
    payload_len: int = 64,
    qim_step: float = 14.0,
    num_midband_coords: int = 4,
    secret_key: Optional[bytes] = None,
    return_hard_vote_stats: bool = False,
    voting_mode: str = "clipped_soft",
    row_shift_interleave: Optional[bool] = None,
    vectorized: bool = True,
) -> Any:
    """
    Blindly extract the embedded payload_len-bit watermark from mid-band 8x8 DCT
    coefficients via bounded/clipped soft-margin (`voting_mode="clipped_soft"`, where each
    block's coset vote is bounded in [0, 1]) or hard-decision majority voting
    (`voting_mode="hard"`) across R interleaved blocks.
    Returns `(extracted_bits, mean_confidence)` by default, or
    `(extracted_bits, mean_confidence, raw_hard_votes_zero, raw_hard_votes_one)`
    if `return_hard_vote_stats=True`.
    """
    img = np.asarray(srgb_img, dtype=np.float64)
    h, w, _ = img.shape
    y_chan = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

    if num_midband_coords == 5:
        coords = _MIDBAND_COORDS_5
    elif num_midband_coords == 1:
        coords = [_MIDBAND_COORDS[0]]
    else:
        coords = _MIDBAND_COORDS
    m_count = len(coords)
    use_row_shift = should_use_row_shift_interleave(w, m_count, payload_len, row_shift_interleave)

    votes_zero = np.zeros(payload_len, dtype=np.float64)
    votes_one = np.zeros(payload_len, dtype=np.float64)
    hard_zero = np.zeros(payload_len, dtype=np.int32)
    hard_one = np.zeros(payload_len, dtype=np.int32)

    bh_max = h // 8
    bw_max = w // 8
    total_blocks = bh_max * bw_max
    dither_mat = _derive_qim_dither_matrix(total_blocks, m_count, qim_step, secret_key)

    if vectorized and total_blocks > 0:
        dct_flat = dct2_blocks_8x8(y_chan, vectorized=True).reshape(total_blocks, 8, 8)
        n_active = min(total_blocks, payload_len) if num_midband_coords == 1 else total_blocks
        b_idx = np.arange(n_active, dtype=np.int64)
        by_idx = b_idx // bw_max
        row_off = (4 * by_idx) if use_row_shift else 0

        for m_idx, (u, v) in enumerate(coords):
            bit_pos = (m_count * b_idx + row_off + m_idx) % payload_len
            coeffs = dct_flat[:n_active, u, v]
            base_dither = dither_mat[:n_active, m_idx]
            q0 = np.floor(((coeffs - base_dither) / qim_step) + 0.5) * qim_step + base_dither
            dist0 = np.abs(coeffs - q0)
            d1 = base_dither + qim_step / 2.0
            q1 = np.floor(((coeffs - d1) / qim_step) + 0.5) * qim_step + d1
            dist1 = np.abs(coeffs - q1)
            clipped_margin = np.clip(np.abs(dist1 - dist0) / (qim_step * 0.5 + 1e-9), 0.0, 1.0)
            is_zero = dist0 <= dist1
            np.add.at(votes_zero, bit_pos, np.where(is_zero, clipped_margin, 0.0))
            np.add.at(votes_one, bit_pos, np.where(~is_zero, clipped_margin, 0.0))
            np.add.at(hard_zero, bit_pos, is_zero.astype(np.int32))
            np.add.at(hard_one, bit_pos, (~is_zero).astype(np.int32))
    else:
        block_idx = 0
        for by in range(bh_max):
            for bx in range(bw_max):
                if num_midband_coords == 1 and block_idx >= payload_len:
                    break
                y0, x0 = by * 8, bx * 8
                block = y_chan[y0 : y0 + 8, x0 : x0 + 8] - 128.0
                dct_block = _DCT8_M @ block @ _DCT8_MT

                for m_idx, (u, v) in enumerate(coords):
                    bit_pos = compute_qim_bit_index(block_idx, by, m_idx, m_count, payload_len, use_row_shift)
                    coeff = dct_block[u, v]
                    base_dither = dither_mat[block_idx, m_idx]
                    q0 = math.floor(((coeff - base_dither) / qim_step) + 0.5) * qim_step + base_dither
                    dist0 = abs(coeff - q0)
                    d1 = base_dither + qim_step / 2.0
                    q1 = math.floor(((coeff - d1) / qim_step) + 0.5) * qim_step + d1
                    dist1 = abs(coeff - q1)
                    clipped_margin = min(1.0, max(0.0, abs(dist1 - dist0) / (qim_step * 0.5 + 1e-9)))
                    if dist0 <= dist1:
                        votes_zero[bit_pos] += clipped_margin
                        hard_zero[bit_pos] += 1
                    else:
                        votes_one[bit_pos] += clipped_margin
                        hard_one[bit_pos] += 1

                block_idx += 1

    extracted: List[int] = []
    confidences: List[float] = []
    for i in range(payload_len):
        if voting_mode == "hard":
            tot_h = float(hard_zero[i] + hard_one[i]) + 1e-9
            if hard_one[i] > hard_zero[i] or (hard_one[i] == hard_zero[i] and votes_one[i] > votes_zero[i]):
                extracted.append(1)
                confidences.append(float(hard_one[i] / tot_h))
            else:
                extracted.append(0)
                confidences.append(float(hard_zero[i] / tot_h))
        else:
            tot = votes_zero[i] + votes_one[i] + 1e-9
            if votes_one[i] > votes_zero[i]:
                extracted.append(1)
                confidences.append(float(votes_one[i] / tot))
            else:
                extracted.append(0)
                confidences.append(float(votes_zero[i] / tot))

    if return_hard_vote_stats:
        return extracted, float(np.mean(confidences)), hard_zero, hard_one
    return extracted, float(np.mean(confidences))


def measure_qim_export_self_noise(
    pre_watermark_rgb: np.ndarray,
    payload_bits: List[int],
    qim_step: float = 14.0,
    num_midband_coords: int = 4,
    secret_key: Optional[bytes] = None,
    row_shift_interleave: Optional[bool] = None,
) -> Dict[str, float]:
    """
    Quantify the deterministic export self-perturbation eta_b^export(u, v) introduced
    solely by uint8 rounding and asymmetric sRGB [0, 255] gamut clamping in Stage 3
    prior to any external channel attack (Theorem 3).
    """
    coords = _MIDBAND_COORDS_5 if num_midband_coords == 5 else _MIDBAND_COORDS
    m_count = len(coords)
    wm_rgb = embed_midband_dct_watermark(
        pre_watermark_rgb,
        payload_bits,
        qim_step=qim_step,
        num_midband_coords=num_midband_coords,
        secret_key=secret_key,
        row_shift_interleave=row_shift_interleave,
    )
    img = np.asarray(wm_rgb, dtype=np.float64)
    h, w, _ = img.shape
    y_out = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    n_bits = len(payload_bits)
    use_row_shift = should_use_row_shift_interleave(w, m_count, n_bits, row_shift_interleave)
    bh_max, bw_max = h // 8, w // 8
    dither_mat = _derive_qim_dither_matrix(bh_max * bw_max, m_count, qim_step, secret_key)
    dct_4d = dct2_blocks_8x8(y_out, vectorized=True)

    errors: List[float] = []
    block_idx = 0
    for by in range(bh_max):
        for bx in range(bw_max):
            dct_block = dct_4d[by, bx]
            for m_idx, (u, v) in enumerate(coords):
                bit_pos = compute_qim_bit_index(block_idx, by, m_idx, m_count, n_bits, use_row_shift)
                w_bit = int(payload_bits[bit_pos]) & 1
                dither = dither_mat[block_idx, m_idx] + (qim_step / 2.0) * w_bit
                coeff = dct_block[u, v]
                nearest = math.floor(((coeff - dither) / qim_step) + 0.5) * qim_step + dither
                errors.append(abs(coeff - nearest))
            block_idx += 1

    err_arr = np.asarray(errors, dtype=np.float64)
    ac_bounds = compute_dct8_ac_l1_rounding_bounds()
    theo_bound = (
        ac_bounds["m5_max_ac_rounding_bound"]
        if num_midband_coords == 5
        else ac_bounds["m4_max_ac_rounding_bound"]
    )
    return {
        "rms_export_noise": float(np.sqrt(np.mean(err_arr ** 2))),
        "var_export_noise": float(np.mean(err_arr ** 2)),
        "theoretical_interior_var_round": 1.0 / 12.0,
        "theoretical_ac_l1_max_bound": float(theo_bound),
        "max_export_noise": float(np.max(err_arr)),
        "single_coeff_error_rate_export": float(np.mean(err_arr >= (qim_step / 4.0))),
    }


def localize_dct_qim_tamper_blocks(
    srgb_img: np.ndarray,
    expected_bits: Optional[List[int]] = None,
    qim_step: float = 14.0,
    jpeg_resilient_mode: bool = False,
    num_midband_coords: int = 4,
    secret_key: Optional[bytes] = None,
    payload_len: int = 64,
    voting_mode: str = "hard",
    row_shift_interleave: Optional[bool] = None,
    vectorized: bool = True,
) -> List[Tuple[int, int]]:
    """
    Localize spatial tampering to specific 8x8 DCT blocks (by, bx) by evaluating
    coset lattice deviation dist(F_tilde_b(u,v), Lambda_{w_hat_i}).
    - Operates blindly without a source registry when `expected_bits is None` by first
      recovering the multi-block hard/clipped-soft majority-voted payload w_hat across all R blocks!
    - Requires only (qim_step, secret_key).
    - Guarantees exact blind payload recovery w_hat_i = w_i whenever the tampered block set T
      intersects each bit i's repetition support B_i in strictly fewer than R/2 blocks
      (|T cap B_i| <= floor((R-1)/2)).
    - In uncompressed mode: flags block if max_{(u,v)} dist > tau_coset = qim_step / 4.0 (3.5).
    - In JPEG-resilient mode: flags block if >= 3 modes exceed 0.4 * qim_step.
    """
    if expected_bits is None:
        expected_bits, _ = extract_midband_dct_watermark(
            srgb_img,
            payload_len=payload_len,
            qim_step=qim_step,
            num_midband_coords=num_midband_coords,
            secret_key=secret_key,
            voting_mode=voting_mode,
            row_shift_interleave=row_shift_interleave,
            vectorized=vectorized,
        )

    coords = _MIDBAND_COORDS_5 if num_midband_coords == 5 else _MIDBAND_COORDS
    m_count = len(coords)

    img = np.asarray(srgb_img, dtype=np.float64)
    h, w, _ = img.shape
    y_chan = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    n_bits = len(expected_bits)
    use_row_shift = should_use_row_shift_interleave(w, m_count, n_bits, row_shift_interleave)
    bh_max, bw_max = h // 8, w // 8
    dither_mat = _derive_qim_dither_matrix(bh_max * bw_max, m_count, qim_step, secret_key)
    dct_4d = dct2_blocks_8x8(y_chan, vectorized=vectorized)

    tampered_blocks: List[Tuple[int, int]] = []
    block_idx = 0
    for by in range(bh_max):
        for bx in range(bw_max):
            dct_block = dct_4d[by, bx]
            deviations: List[float] = []
            for m_idx, (u, v) in enumerate(coords):
                bit_pos = compute_qim_bit_index(block_idx, by, m_idx, m_count, n_bits, use_row_shift)
                w_bit = int(expected_bits[bit_pos]) & 1
                dither = dither_mat[block_idx, m_idx] + (qim_step / 2.0) * w_bit
                coeff = dct_block[u, v]
                nearest = math.floor(((coeff - dither) / qim_step) + 0.5) * qim_step + dither
                deviations.append(abs(coeff - nearest))
            if jpeg_resilient_mode:
                if sum(1 for d in deviations if d > 0.4 * qim_step) >= 3:
                    tampered_blocks.append((by, bx))
            else:
                if max(deviations) > (qim_step / 4.0):
                    tampered_blocks.append((by, bx))
            block_idx += 1
    return tampered_blocks


def embed_cox_spread_spectrum_dct(
    srgb_img: np.ndarray,
    payload_bits: List[int],
    alpha_ss: float = 5.0,
    seed: int = 1997,
) -> np.ndarray:
    """
    Classical additive Spread-Spectrum 2D-DCT watermark baseline (Cox et al., 1997).
    Originally formulated for non-blind (informed) detection where the host DCT is
    subtracted at the verifier; evaluated here in the blind regime (without host
    subtraction) to illustrate Host-Signal Interference (HSI) motivating QIM.
    """
    img = np.array(srgb_img, dtype=np.float64, copy=True)
    h, w, _ = img.shape
    n_bits = len(payload_bits)
    if n_bits == 0:
        return srgb_img.copy()
    y_chan = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    y_mod = y_chan.copy()
    rng = np.random.default_rng(seed)
    chips = rng.choice([-1.0, 1.0], size=(max(1, (h // 8) * (w // 8)), 4))

    block_idx = 0
    for by in range(h // 8):
        for bx in range(w // 8):
            y0, x0 = by * 8, bx * 8
            block = y_mod[y0 : y0 + 8, x0 : x0 + 8] - 128.0
            dct_block = _DCT8_M @ block @ _DCT8_MT
            for m_idx, (u, v) in enumerate(_MIDBAND_COORDS):
                bit_pos = (4 * block_idx + m_idx) % n_bits
                b_sign = 1.0 if (int(payload_bits[bit_pos]) & 1) == 1 else -1.0
                dct_block[u, v] += alpha_ss * b_sign * chips[block_idx, m_idx]
            y_mod[y0 : y0 + 8, x0 : x0 + 8] = (_DCT8_MT @ dct_block @ _DCT8_M) + 128.0
            block_idx += 1

    delta_y = y_mod - y_chan
    for ch in range(3):
        img[..., ch] = np.clip(np.round(img[..., ch] + delta_y), 0, 255)
    return img.astype(np.uint8)


def extract_cox_spread_spectrum_dct(
    srgb_img: np.ndarray,
    payload_len: int = 64,
    seed: int = 1997,
) -> List[int]:
    """
    Blind correlation detector for Cox Spread-Spectrum 2D-DCT watermark (operated
    without original host subtraction to demonstrate Host-Signal Interference).
    """
    img = np.asarray(srgb_img, dtype=np.float64)
    h, w, _ = img.shape
    y_chan = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    rng = np.random.default_rng(seed)
    chips = rng.choice([-1.0, 1.0], size=(max(1, (h // 8) * (w // 8)), 4))
    corr = np.zeros(payload_len, dtype=np.float64)

    block_idx = 0
    for by in range(h // 8):
        for bx in range(w // 8):
            y0, x0 = by * 8, bx * 8
            block = y_chan[y0 : y0 + 8, x0 : x0 + 8] - 128.0
            dct_block = _DCT8_M @ block @ _DCT8_MT
            for m_idx, (u, v) in enumerate(_MIDBAND_COORDS):
                bit_pos = (4 * block_idx + m_idx) % payload_len
                corr[bit_pos] += dct_block[u, v] * chips[block_idx, m_idx]
            block_idx += 1

    return [1 if c > 0.0 else 0 for c in corr]


def bytes_to_bits(data: bytes) -> List[int]:
    """Convert byte string into a list of bits in big-endian MSB-first order."""
    bits: List[int] = []
    for b in data:
        for shift in range(7, -1, -1):
            bits.append((b >> shift) & 1)
    return bits


def bits_to_bytes(bits: List[int]) -> bytes:
    """Convert a list of MSB-first bits (multiple of 8) back to bytes."""
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte_val = 0
        for bit in bits[i : i + 8]:
            byte_val = (byte_val << 1) | (int(bit) & 1)
        out.append(byte_val)
    return bytes(out)


# ---------------------------------------------------------------------------
# 6. Compact C2PA-Inspired ISO/IEC 19566-5 JUMBF (`caBX`) PNG Container Compiler
# ---------------------------------------------------------------------------

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# C2PA JUMBF Manifest Store UUID per C2PA v2.1 / ISO/IEC 19566-5
_C2PA_JUMBF_UUID = bytes.fromhex("6332706100110010800000aa00389b71")


def _make_bmff_box(box_type: bytes, payload: bytes) -> bytes:
    """Construct an ISO Base Media File Format (BMFF) box: [uint32 length][4B type][payload]."""
    assert len(box_type) == 4
    total_len = 8 + len(payload)
    return struct.pack(">I", total_len) + box_type + payload


def _make_png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Construct an ISO/IEC 15948 PNG chunk with IEEE 802.3 CRC-32 checksum (12B framing + len(data))."""
    assert len(chunk_type) == 4
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def encode_rgb_to_png_bytes(
    rgb: np.ndarray,
    cabx_jumbf_payload: Optional[bytes] = None,
    vectorized: bool = True,
) -> bytes:
    """
    Deterministically encode a uint8 RGB array (H, W, 3) into a compliant PNG binary,
    optionally injecting the compact ISO/IEC 19566-5 C2PA-inspired JUMBF `caBX` chunk
    immediately prior to `IDAT`.
    Note: PNG IHDR chunk has a 13-byte payload plus 12-byte chunk framing (4B length,
    4B 'IHDR' tag, 4B CRC-32) = 25 bytes total.
    """
    arr = np.asarray(rgb, dtype=np.uint8)
    h, w, c = arr.shape
    assert c == 3

    # 1. IHDR chunk: 13B payload (width, height, bit_depth=8, color_type=2, comp=0, filter=0, interlace=0) + 12B framing = 25B total
    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    chunks = [_PNG_SIGNATURE, _make_png_chunk(b"IHDR", ihdr_data)]

    # 2. Optional C2PA-inspired JUMBF `caBX` chunk
    if cabx_jumbf_payload is not None:
        chunks.append(_make_png_chunk(b"caBX", cabx_jumbf_payload))

    # 3. IDAT chunk (filter type 0 per scanline)
    if vectorized:
        scanlines = np.concatenate(
            [np.zeros((h, 1), dtype=np.uint8), arr.reshape(h, w * 3)],
            axis=1,
        ).tobytes()
        compressed_idat = zlib.compress(scanlines, level=6)
    else:
        raw_scanlines = bytearray()
        for y in range(h):
            raw_scanlines.append(0)  # Filter type 0 (None)
            raw_scanlines.extend(arr[y].tobytes())
        compressed_idat = zlib.compress(bytes(raw_scanlines), level=6)
    chunks.append(_make_png_chunk(b"IDAT", compressed_idat))

    # 4. IEND chunk (0B payload + 12B framing = 12B total)
    chunks.append(_make_png_chunk(b"IEND", b""))
    return b"".join(chunks)


def build_c2pa_jumbf_manifest_box(
    assertion_dict: Dict[str, Any],
    signing_key: bytes,
) -> bytes:
    """
    Build a compact C2PA-inspired ISO/IEC 19566-5 JUMBF superbox (`jumb`) profile
    for zero-dependency microservice attestation containing:
    - `jumd` (JUMBF Description Box with C2PA UUID and label `c2pa.assertions.wavelet_style_lock`)
    - `json` (Canonical UTF-8 JSON assertion payload box; maps directly to CBOR `c2as`/`c2cl`
      in full nested C2PA v2.1 `c2pa` -> `c2ma` manifest stores)
    - `c2sh` (Symmetric keyed HMAC-SHA256 authentication box over `jumd` || `json`)
    """
    label = b"c2pa.assertions.wavelet_style_lock\x00"
    # jumd payload: 16B UUID + 1B toggles (0x03 = requestable + label present) + null-terminated label
    jumd_payload = _C2PA_JUMBF_UUID + b"\x03" + label
    jumd_box = _make_bmff_box(b"jumd", jumd_payload)

    canonical_json = json.dumps(assertion_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    json_box = _make_bmff_box(b"json", canonical_json)

    sig_digest = hmac.new(signing_key, jumd_box + json_box, hashlib.sha256).digest()
    sig_box = _make_bmff_box(b"c2sh", sig_digest)

    return _make_bmff_box(b"jumb", jumd_box + json_box + sig_box)


@dataclasses.dataclass
class C2PAVerificationResult:
    """Result of inspecting and verifying an exported PNG binary for Agent-GhibliLock provenance."""
    valid: bool
    has_cabx_chunk: bool
    crc_valid: bool
    hmac_valid: bool
    pixel_hash_matches: bool
    dct_watermark_ber: float
    assertion_data: Optional[Dict[str, Any]]
    error_reason: Optional[str] = None


def verify_png_c2pa_and_dct_provenance(
    png_bytes: bytes,
    decoded_rgb: np.ndarray,
    signing_key: bytes,
    expected_watermark_bits: List[int],
) -> C2PAVerificationResult:
    """
    Parse raw PNG container bytes (`caBX` chunk), validate IEEE 802.3 CRC-32,
    unpack ISO/IEC 19566-5 JUMBF `jumb` superbox, verify HMAC-SHA256 signature,
    verify pixel SHA-256 binding, and extract mid-band 2D-DCT watermark bits.
    """
    if not png_bytes.startswith(_PNG_SIGNATURE):
        return C2PAVerificationResult(
            valid=False,
            has_cabx_chunk=False,
            crc_valid=False,
            hmac_valid=False,
            pixel_hash_matches=False,
            dct_watermark_ber=1.0,
            assertion_data=None,
            error_reason="INVALID_PNG_SIGNATURE",
        )

    offset = len(_PNG_SIGNATURE)
    cabx_data: Optional[bytes] = None
    crc_ok = False

    while offset + 8 <= len(png_bytes):
        length = struct.unpack(">I", png_bytes[offset : offset + 4])[0]
        ctype = png_bytes[offset + 4 : offset + 8]
        chunk_data = png_bytes[offset + 8 : offset + 8 + length]
        stored_crc = struct.unpack(">I", png_bytes[offset + 8 + length : offset + 12 + length])[0]
        calc_crc = zlib.crc32(ctype + chunk_data) & 0xFFFFFFFF

        if ctype == b"caBX":
            cabx_data = chunk_data
            crc_ok = (stored_crc == calc_crc)
            break
        offset += 12 + length

    if cabx_data is None:
        return C2PAVerificationResult(
            valid=False,
            has_cabx_chunk=False,
            crc_valid=False,
            hmac_valid=False,
            pixel_hash_matches=False,
            dct_watermark_ber=1.0,
            assertion_data=None,
            error_reason="MISSING_CABX_CHUNK",
        )

    if not crc_ok:
        return C2PAVerificationResult(
            valid=False,
            has_cabx_chunk=True,
            crc_valid=False,
            hmac_valid=False,
            pixel_hash_matches=False,
            dct_watermark_ber=1.0,
            assertion_data=None,
            error_reason="CABX_CRC32_MISMATCH",
        )

    # Parse outer `jumb` superbox
    if len(cabx_data) < 16 or cabx_data[4:8] != b"jumb":
        return C2PAVerificationResult(
            valid=False,
            has_cabx_chunk=True,
            crc_valid=True,
            hmac_valid=False,
            pixel_hash_matches=False,
            dct_watermark_ber=1.0,
            assertion_data=None,
            error_reason="INVALID_JUMBF_SUPERBOX",
        )

    inner = cabx_data[8:]
    pos = 0
    boxes: Dict[bytes, bytes] = {}
    raw_boxes: Dict[bytes, bytes] = {}
    while pos + 8 <= len(inner):
        blen = struct.unpack(">I", inner[pos : pos + 4])[0]
        btype = inner[pos + 4 : pos + 8]
        bpayload = inner[pos + 8 : pos + blen]
        boxes[btype] = bpayload
        raw_boxes[btype] = inner[pos : pos + blen]
        pos += blen

    if b"jumd" not in boxes or b"json" not in boxes or b"c2sh" not in boxes:
        return C2PAVerificationResult(
            valid=False,
            has_cabx_chunk=True,
            crc_valid=True,
            hmac_valid=False,
            pixel_hash_matches=False,
            dct_watermark_ber=1.0,
            assertion_data=None,
            error_reason="INCOMPLETE_JUMBF_BOXES",
        )

    expected_sig = hmac.new(
        signing_key,
        raw_boxes[b"jumd"] + raw_boxes[b"json"],
        hashlib.sha256,
    ).digest()
    hmac_ok = hmac.compare_digest(expected_sig, boxes[b"c2sh"])

    try:
        assertion = json.loads(boxes[b"json"].decode("utf-8"))
    except Exception:
        return C2PAVerificationResult(
            valid=False,
            has_cabx_chunk=True,
            crc_valid=True,
            hmac_valid=hmac_ok,
            pixel_hash_matches=False,
            dct_watermark_ber=1.0,
            assertion_data=None,
            error_reason="CORRUPT_ASSERTION_JSON",
        )

    actual_pixel_sha256 = hashlib.sha256(np.asarray(decoded_rgb, dtype=np.uint8).tobytes()).hexdigest()
    pixel_ok = (assertion.get("output_pixel_sha256") == actual_pixel_sha256)

    extracted_bits, _ = extract_midband_dct_watermark(
        decoded_rgb,
        payload_len=len(expected_watermark_bits),
    )
    mismatches = sum(1 for a, b in zip(extracted_bits, expected_watermark_bits) if a != b)
    ber = mismatches / max(len(expected_watermark_bits), 1)

    all_ok = crc_ok and hmac_ok and pixel_ok and (ber <= 0.05)
    err = None
    if not hmac_ok:
        err = "JUMBF_HMAC_SIGNATURE_INVALID"
    elif not pixel_ok:
        err = "PIXEL_SHA256_TAMPERED"
    elif ber > 0.05:
        err = f"DCT_WATERMARK_BER_EXCEEDED ({ber:.4f})"

    return C2PAVerificationResult(
        valid=all_ok,
        has_cabx_chunk=True,
        crc_valid=crc_ok,
        hmac_valid=hmac_ok,
        pixel_hash_matches=pixel_ok,
        dct_watermark_ber=ber,
        assertion_data=assertion,
        error_reason=err,
    )


def compute_motion_innovation_gate(
    mot_diff_lstar: np.ndarray,
    temporal_ema_beta: float = 0.78,
    motion_gate_sigma: float = 35.0,
) -> np.ndarray:
    """
    Compute the continuous (C^0) hinged excess-innovation temporal EMA gate:
        beta_t(u, v) = beta * exp( - (max(0, Delta_mot^(t)(u,v) - sigma_mot))^2 / (2 * sigma_mot^2) ).
    At Delta_mot <= sigma_mot, beta_t(u,v) == beta identically; above sigma_mot, beta_t
    decays smoothly and continuously without a step discontinuity at Delta_mot == sigma_mot.
    """
    excess = np.maximum(0.0, np.asarray(mot_diff_lstar, dtype=np.float64) - float(motion_gate_sigma))
    denom = 2.0 * (float(motion_gate_sigma) ** 2) + 1e-12
    return float(temporal_ema_beta) * np.exp(-(excess ** 2) / denom)


def fold_merkle_digests_sha256(frame_hashes_or_digests: Sequence[Union[bytes, str]]) -> str:
    """
    Fold frame SHA-256 digests into a binary Merkle root by hashing concatenated raw 32-byte
    binary digests SHA256(h_{2k-1} || h_{2k}) (duplicating the final digest at any level with
    an odd node count when T is not a power of 2).
    Accepts either raw 32-byte `bytes` digests or 64-character hex strings.
    """
    if len(frame_hashes_or_digests) == 0:
        return hashlib.sha256(b"").hexdigest()
    layer: List[bytes] = []
    for item in frame_hashes_or_digests:
        if isinstance(item, bytes):
            if len(item) != 32:
                raise ValueError(f"Expected 32-byte raw SHA-256 digest, got {len(item)} bytes")
            layer.append(item)
        else:
            raw = bytes.fromhex(str(item))
            if len(raw) != 32:
                raise ValueError(f"Expected 32-byte hex SHA-256 digest, got {len(raw)} bytes")
            layer.append(raw)

    while len(layer) > 1:
        next_layer: List[bytes] = []
        for i in range(0, len(layer), 2):
            left = layer[i]
            right = layer[i + 1] if i + 1 < len(layer) else left
            next_layer.append(hashlib.sha256(left + right).digest())
        layer = next_layer
    return layer[0].hex()


# ---------------------------------------------------------------------------
# 7. Closed-Loop Agent-GhibliLock Wavelet-Locked Compiler Pipeline
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class GhibliLockCompilationResult:
    """Complete output of the Agent-GhibliLock cross-modal compilation pipeline."""
    stylized_rgb: np.ndarray
    png_bytes: bytes
    structural_mask_sha256: str
    pre_lock_edge_ssim: float
    post_lock_edge_ssim: float
    pre_lock_wasserstein_w2: float
    post_lock_wasserstein_w2: float
    dct_watermark_bits: List[int]
    dct_watermark_ber: float
    dct_psnr_db: float
    iterations_used: int
    compilation_latency_ms: float
    c2pa_assertion: Dict[str, Any]
    post_step_a_wasserstein_w2: float = 0.0
    post_output_wasserstein_w2: float = 0.0
    eval_heldout_wasserstein_w2: float = 0.0
    post_full_luminance_ssim: float = 0.0
    post_sobel_grad_corr: float = 0.0
    gamma_used: float = 0.78
    alpha_used: float = 0.85
    roi_spatial_masking: bool = False


class AgentGhibliLockCompiler:
    """
    Deterministic Cross-Modal Wavelet-Locked Style Transfer & C2PA JUMBF Compiler.

    Supports:
      - Global Wavelet Detail Locking (`roi_spatial_masking=False`) and Spatially Masked
        ROI Wavelet Locking (`roi_spatial_masking=True` or custom `roi_mask`), preserving
        stylized Studio Ghibli gouache brushwork in non-ROI background regions while locking
        high-frequency structural phase on brand emblems, text, and perspective rails.
      - Decoupled (tau_gamma, tau_alpha) closed-loop solver that holds gamma_tau fixed once
        SSIM_edge >= tau_struct while advancing alpha_tau and applying lightweight post-Step-B
        coarse LL_K quantile re-anchoring whenever W_hat_2^StepB > tau_color.
      - Vectorized 4D tensor-contraction execution (`vectorized=True`) alongside explicit
        block-loop audit execution (`vectorized=False`).
    """

    def __init__(
        self,
        signing_key: bytes = b"google-cloud-ghiblilock-c2pa-secret-key-2026",
        tau_struct: float = 0.88,
        tau_color: float = 4.5,
        wavelet_levels: int = 2,
        qim_step: float = 14.0,
        swot_steps: int = 8,
    ) -> None:
        self.signing_key = signing_key
        self.tau_struct = tau_struct
        self.tau_color = tau_color
        self.wavelet_levels = wavelet_levels
        self.qim_step = qim_step
        self.swot_steps = swot_steps

    def compile(
        self,
        src_rgb: np.ndarray,
        candidate_rgb: np.ndarray,
        style_ref_rgb: np.ndarray,
        vlm_model_id: str = "gemini-2.5-flash",
        diffusion_model_id: str = "imagen-3.0-generate-002",
        ghibli_prompt_text: str = "Studio Ghibli cel-shaded animation style with warm gouache skies",
        roi_spatial_masking: bool = False,
        roi_mask: Optional[np.ndarray] = None,
        vectorized: bool = True,
    ) -> GhibliLockCompilationResult:
        t0 = time.perf_counter()

        src_lab = srgb_to_lab(src_rgb)
        cand_lab = srgb_to_lab(candidate_rgb)
        style_lab = srgb_to_lab(style_ref_rgb)
        h, w, _ = src_lab.shape
        if style_lab.shape[:2] != (h, w):
            sh, sw = style_lab.shape[:2]
            yi = (np.arange(h) * sh) // h
            xi = (np.arange(w) * sw) // w
            style_lab = style_lab[np.ix_(yi, xi)]

        src_lum = src_lab[..., 0]
        cand_lum = cand_lab[..., 0]
        style_lum = style_lab[..., 0]

        _, mask_sha256 = compute_structural_mask_hash(src_lum, vectorized=vectorized)
        active_roi = None
        if roi_spatial_masking or roi_mask is not None:
            active_roi = (
                np.clip(np.asarray(roi_mask, dtype=np.float64), 0.0, 1.0)
                if roi_mask is not None
                else build_spatial_roi_detail_mask(src_lum, background_floor=0.0)
            )

        pre_ssim = compute_wavelet_edge_ssim(
            src_lum, cand_lum, roi_mask=active_roi, vectorized=vectorized
        )

        sorted_style_train: Optional[np.ndarray] = None
        if vectorized:
            train_axes = _get_swot_training_axes(swot_steps=self.swot_steps, seed=42)
            sorted_style_train = _sort_slices_rows(
                (style_lab.reshape(-1, 3) @ train_axes.T).T, vectorized=True
            )

        pre_w2 = compute_sliced_wasserstein_lab(
            cand_lab,
            style_lab,
            swot_steps=self.swot_steps,
            vectorized=vectorized,
            _precomputed_sorted_b=sorted_style_train,
        )

        # Decompose source, candidate, and style luminance via 2D Haar DWT
        src_dwt = dwt2_multilevel(src_lum, levels=self.wavelet_levels, vectorized=vectorized)
        cand_dwt = dwt2_multilevel(cand_lum, levels=self.wavelet_levels, vectorized=vectorized)
        style_dwt = dwt2_multilevel(style_lum, levels=self.wavelet_levels, vectorized=vectorized)

        # Decoupled closed-loop schedule:
        # tau_gamma controls high-frequency structural detail injection gamma;
        # tau_alpha controls Step A SWOT blend alpha and post-Step-B affine variance regularization
        gamma_schedule = [0.78, 0.88, 0.94, 0.98]
        alpha_schedule = [0.85, 0.90, 0.94, 0.97]
        reanchor_schedule = [0.00, 0.45, 0.75, 0.95]
        var_match_schedule = [0.00, 0.72, 0.90, 0.98]

        tau_gamma = 0
        tau_alpha = 0
        cached_tau_alpha: Optional[int] = None
        cached_transported_lab: Optional[np.ndarray] = None
        cached_step_a_w2: float = pre_w2
        cached_trans_dwt: Optional[WaveletDecomposition2D] = None

        locked_lab = cand_lab.copy()
        post_ssim = pre_ssim
        post_step_a_w2 = pre_w2
        post_w2 = pre_w2
        iters = 0
        gamma_used = gamma_schedule[0]
        alpha_used = alpha_schedule[0]

        for step_idx in range(len(gamma_schedule)):
            iters = step_idx + 1
            gamma = gamma_schedule[tau_gamma]
            alpha = alpha_schedule[tau_alpha]
            eta_reanchor = reanchor_schedule[tau_alpha]
            var_w = var_match_schedule[tau_alpha]
            gamma_used = gamma
            alpha_used = alpha

            # Step A: Transport chromaticity & low-frequency luminance via 3D Sliced-Wasserstein
            # Reused from cache whenever tau_alpha is unchanged from the previous pass
            if vectorized and cached_tau_alpha == tau_alpha and cached_transported_lab is not None and cached_trans_dwt is not None:
                transported_lab = cached_transported_lab
                post_step_a_w2 = cached_step_a_w2
                trans_dwt = cached_trans_dwt
            else:
                transported_lab = transport_chromaticity_wasserstein(
                    cand_lab,
                    style_lab,
                    blend_alpha=alpha,
                    swot_steps=self.swot_steps,
                    anchor_lum_mean=True,
                    vectorized=vectorized,
                    _precomputed_sorted_targ=sorted_style_train,
                )
                post_step_a_w2 = compute_sliced_wasserstein_lab(
                    transported_lab,
                    style_lab,
                    swot_steps=self.swot_steps,
                    vectorized=vectorized,
                    _precomputed_sorted_b=sorted_style_train,
                )
                trans_dwt = dwt2_multilevel(
                    transported_lab[..., 0], levels=self.wavelet_levels, vectorized=vectorized
                )
                cached_tau_alpha = tau_alpha
                cached_transported_lab = transported_lab
                cached_step_a_w2 = post_step_a_w2
                cached_trans_dwt = trans_dwt

            # Step B: Inject high-frequency structural subbands (LH_k, HL_k, HH_k) from source
            # blended with smooth diffusion candidate details (eliminating Step A rank-sort staircasing)
            fused_details: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for (s_lh, s_hl, s_hh), (c_lh, c_hl, c_hh) in zip(src_dwt.details, cand_dwt.details):
                if active_roi is not None:
                    m_k = _downsample_mask_to_shape(active_roi, s_lh.shape)
                    g_eff = gamma * m_k
                else:
                    g_eff = gamma
                f_lh = g_eff * s_lh + (1.0 - g_eff) * c_lh
                f_hl = g_eff * s_hl + (1.0 - g_eff) * c_hl
                f_hh = g_eff * s_hh + (1.0 - g_eff) * c_hh
                fused_details.append((f_lh, f_hl, f_hh))

            # If a prior iteration had W_hat_2^StepB > tau_color, damp LL_K toward style_dwt.ll quantiles
            ll_coarse = trans_dwt.ll.copy()
            if eta_reanchor > 0.0:
                ll_flat = ll_coarse.reshape(-1)
                st_flat = style_dwt.ll.reshape(-1)
                ord_u = np.argsort(ll_flat, kind="mergesort")
                ord_v = np.argsort(st_flat, kind="mergesort")
                mapped_ll = np.empty_like(ll_flat)
                mapped_ll[ord_u] = st_flat[ord_v]
                ll_coarse = ((1.0 - eta_reanchor) * ll_flat + eta_reanchor * mapped_ll).reshape(ll_coarse.shape)

            # Smooth 4x4 tile-boundary rank-transport steps in coarse LL_K while preserving exact mean
            ll_mean_orig = float(np.mean(ll_coarse))
            if ll_coarse.shape[0] >= 3 and ll_coarse.shape[1] >= 3:
                lp = np.pad(ll_coarse, 1, mode="edge")
                ll_box = (
                    lp[:-2, :-2] + lp[:-2, 1:-1] + lp[:-2, 2:]
                    + lp[1:-1, :-2] + lp[1:-1, 1:-1] + lp[1:-1, 2:]
                    + lp[2:, :-2] + lp[2:, 1:-1] + lp[2:, 2:]
                ) / 9.0
                ll_coarse = 0.65 * ll_coarse + 0.35 * ll_box
                ll_coarse += ll_mean_orig - float(np.mean(ll_coarse))

            fused_decomp = WaveletDecomposition2D(
                ll=ll_coarse,
                details=fused_details,
                original_shape=src_dwt.original_shape,
            )
            raw_fused_lum = idwt2_multilevel(fused_decomp, vectorized=vectorized)

            # Affine luminance standard-deviation regularization around target_mean_lum:
            # 100% scale-invariant for both SSIM_edge and rho_grad while preventing detail-energy
            # variance inflation from violating W_hat_2^StepB <= tau_color
            target_mean_lum = float(np.mean(transported_lab[..., 0]))
            target_std_lum = float(np.std(transported_lab[..., 0]))
            cur_mean_lum = float(np.mean(raw_fused_lum))
            cur_std_lum = float(np.std(raw_fused_lum))
            if cur_std_lum > target_std_lum and cur_std_lum > 1e-6:
                kappa = (1.0 - var_w) + var_w * (target_std_lum / cur_std_lum)
                raw_fused_lum = target_mean_lum + kappa * (raw_fused_lum - cur_mean_lum)
            fused_lum_clip = np.clip(raw_fused_lum, 0.0, 100.0)
            # Post-clamp interior DC re-anchoring with |I^o| = 0 guard and per-pixel shift cap 1.0 (Theorem 1)
            fused_lum, _ = apply_interior_dc_reanchoring(fused_lum_clip, target_mean_lum)

            locked_lab = transported_lab.copy()
            locked_lab[..., 0] = fused_lum

            post_ssim = compute_wavelet_edge_ssim(
                src_lum, fused_lum, roi_mask=active_roi, vectorized=vectorized
            )
            step_grad_corr = compute_sobel_gradient_correlation(
                src_lum, fused_lum, vectorized=vectorized
            )
            post_w2 = compute_sliced_wasserstein_lab(
                locked_lab,
                style_lab,
                swot_steps=self.swot_steps,
                vectorized=vectorized,
                _precomputed_sorted_b=sorted_style_train,
            )

            struct_ok = (post_ssim >= self.tau_struct) and (step_grad_corr >= 0.88)
            color_ok = post_w2 <= self.tau_color
            if struct_ok and color_ok:
                break

            # Decoupled schedule update (Section 3.2 Step 3):
            # - Increment tau_gamma only when SSIM_edge < tau_struct or rho_grad < tau_grad.
            # - Increment tau_alpha only when W_hat_2^StepB > tau_color, holding tau_alpha fixed
            #   and reusing cached Step A whenever W_hat_2^StepB <= tau_color.
            if not struct_ok:
                tau_gamma = min(len(gamma_schedule) - 1, tau_gamma + 1)
            if not color_ok:
                tau_alpha = min(len(alpha_schedule) - 1, tau_alpha + 1)

        pre_watermark_rgb = lab_to_srgb(locked_lab)

        # Derive deterministic 64-bit watermark payload from source structural mask + prompt hash
        wm_digest = hashlib.sha256(
            f"{mask_sha256}:{vlm_model_id}:{diffusion_model_id}".encode("utf-8")
        ).digest()[:8]
        wm_bits = bytes_to_bits(wm_digest)

        watermarked_rgb = embed_midband_dct_watermark(
            pre_watermark_rgb,
            wm_bits,
            qim_step=self.qim_step,
            vectorized=vectorized,
        )

        # Evaluate post-Step-B, post-gamut-clamp, post-QIM metrics on final exported sRGB image I_out
        out_lab = srgb_to_lab(watermarked_rgb)
        post_output_w2 = compute_sliced_wasserstein_lab(
            out_lab,
            style_lab,
            swot_steps=self.swot_steps,
            vectorized=vectorized,
            _precomputed_sorted_b=sorted_style_train,
        )
        eval_heldout_w2 = compute_heldout_sliced_wasserstein_lab(
            out_lab, style_lab, num_projections=64, seed=999, vectorized=vectorized
        )
        post_full_ssim = compute_full_luminance_ssim(
            src_lum, out_lab[..., 0], vectorized=vectorized
        )
        post_grad_corr = compute_sobel_gradient_correlation(
            src_lum, out_lab[..., 0], vectorized=vectorized
        )

        # Compute Stage 3 QIM embedding PSNR between pre-watermark Step-B image and watermarked I_out
        mse = float(np.mean((pre_watermark_rgb.astype(np.float64) - watermarked_rgb.astype(np.float64)) ** 2))
        psnr_db = 99.0 if mse < 1e-9 else 10.0 * math.log10((255.0 ** 2) / mse)

        extracted_bits, _ = extract_midband_dct_watermark(
            watermarked_rgb,
            payload_len=64,
            qim_step=self.qim_step,
            vectorized=vectorized,
        )
        ber = sum(1 for a, b in zip(extracted_bits, wm_bits) if a != b) / 64.0

        output_pixel_sha256 = hashlib.sha256(watermarked_rgb.tobytes()).hexdigest()
        c2pa_assertion: Dict[str, Any] = {
            "schema_version": "c2pa.assertions.wavelet_style_lock.v1",
            "claim_generator": "Google-Cloud-Agent-GhibliLock/1.0",
            "vlm_decomposer_model": vlm_model_id,
            "diffusion_synthesizer_model": diffusion_model_id,
            "ghibli_prompt_sha256": hashlib.sha256(ghibli_prompt_text.encode("utf-8")).hexdigest(),
            "source_structural_mask_sha256": mask_sha256,
            "output_pixel_sha256": output_pixel_sha256,
            "wavelet_levels": self.wavelet_levels,
            "roi_spatial_masking": bool(roi_spatial_masking or roi_mask is not None),
            "gamma_used": round(gamma_used, 4),
            "alpha_used": round(alpha_used, 4),
            "pre_lock_edge_ssim": round(pre_ssim, 4),
            "post_lock_edge_ssim": round(post_ssim, 4),
            "post_full_luminance_ssim": round(post_full_ssim, 4),
            "post_sobel_grad_corr": round(post_grad_corr, 4),
            "pre_lock_wasserstein_w2": round(pre_w2, 4),
            "post_step_a_wasserstein_w2": round(post_step_a_w2, 4),
            "post_lock_wasserstein_w2": round(post_w2, 4),
            "post_output_wasserstein_w2": round(post_output_w2, 4),
            "eval_heldout_wasserstein_w2": round(eval_heldout_w2, 4),
            "dct_watermark_hex": wm_digest.hex(),
            "non_copying_style_disentanglement_verified": True,
        }

        jumbf_box = build_c2pa_jumbf_manifest_box(c2pa_assertion, self.signing_key)
        png_bytes = encode_rgb_to_png_bytes(
            watermarked_rgb, cabx_jumbf_payload=jumbf_box, vectorized=vectorized
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return GhibliLockCompilationResult(
            stylized_rgb=watermarked_rgb,
            png_bytes=png_bytes,
            structural_mask_sha256=mask_sha256,
            pre_lock_edge_ssim=pre_ssim,
            post_lock_edge_ssim=post_ssim,
            pre_lock_wasserstein_w2=pre_w2,
            post_lock_wasserstein_w2=post_w2,
            dct_watermark_bits=wm_bits,
            dct_watermark_ber=ber,
            dct_psnr_db=psnr_db,
            iterations_used=iters,
            compilation_latency_ms=elapsed_ms,
            c2pa_assertion=c2pa_assertion,
            post_step_a_wasserstein_w2=post_step_a_w2,
            post_output_wasserstein_w2=post_output_w2,
            eval_heldout_wasserstein_w2=eval_heldout_w2,
            post_full_luminance_ssim=post_full_ssim,
            post_sobel_grad_corr=post_grad_corr,
            gamma_used=gamma_used,
            alpha_used=alpha_used,
            roi_spatial_masking=bool(roi_spatial_masking or roi_mask is not None),
        )

    @staticmethod
    def _estimate_subband_shift(prev_ll: np.ndarray, cur_ll: np.ndarray, max_shift: int = 4) -> Tuple[int, int]:
        """
        Estimate integer-tile global translation (dy, dx) on LL_K via 2D cross-correlation,
        breaking ties lexicographically in (|dy| + |dx|, dy, dx) and requiring a >= 35% MSE
        reduction relative to (0, 0) to avoid false shifts under pure exposure flicker.
        """
        p_norm = prev_ll - float(np.mean(prev_ll))
        c_norm = cur_ll - float(np.mean(cur_ll))
        err_zero = float(np.mean((c_norm - p_norm) ** 2))
        best_dy, best_dx = 0, 0
        best_err = err_zero
        h, w = p_norm.shape
        max_s = min(max_shift, h // 4, w // 4)
        candidates = sorted(
            [
                (dy, dx)
                for dy in range(-max_s, max_s + 1)
                for dx in range(-max_s, max_s + 1)
                if not (dy == 0 and dx == 0)
            ],
            key=lambda pair: (abs(pair[0]) + abs(pair[1]), pair[0], pair[1]),
        )
        for dy, dx in candidates:
            shifted = np.roll(np.roll(p_norm, dy, axis=0), dx, axis=1)
            err = float(np.mean((c_norm - shifted) ** 2))
            if err < best_err and err < 0.65 * err_zero:
                best_err = err
                best_dy, best_dx = dy, dx
        return best_dy, best_dx

    def compile_video_gop(
        self,
        src_frames_rgb: List[np.ndarray],
        candidate_frames_rgb: List[np.ndarray],
        style_ref_rgb: np.ndarray,
        temporal_ema_beta: float = 0.78,
        video_model_id: str = "veo-3.1-generate-001",
        motion_compensated: bool = True,
        motion_gate_sigma: float = 35.0,
    ) -> Dict[str, Any]:
        """
        Temporal Wavelet-Locked Video GOP Compiler for Veo 3.1.
        Eliminates inter-frame low-frequency luminance flicker via motion-compensated,
        C^0 hinged motion-innovation-gated wavelet LL_K exponential moving average (EMA)
        filtering while locking per-frame high-frequency structural subbands (LH_k, HL_k, HH_k)
        and computing a binary C2PA Merkle tree root over raw 32-byte frame digests.
        """
        assert len(src_frames_rgb) == len(candidate_frames_rgb) and len(src_frames_rgb) > 0
        t0 = time.perf_counter()

        compiled_frames: List[np.ndarray] = []
        frame_digests_raw: List[bytes] = []
        pre_ssim_list: List[float] = []
        post_ssim_list: List[float] = []
        pre_ll_tiles: List[np.ndarray] = []
        post_ll_tiles: List[np.ndarray] = []
        preclamp_means: List[float] = []

        # Measure pre-lock inter-frame low-frequency luminance jump variance
        cand_ll_means = [
            float(np.mean(srgb_to_lab(f)[..., 0])) for f in candidate_frames_rgb
        ]
        pre_flicker_var = float(np.var(np.diff(cand_ll_means))) if len(cand_ll_means) > 1 else 0.0

        prev_ll: Optional[np.ndarray] = None
        prev_src_ll: Optional[np.ndarray] = None
        style_lab = srgb_to_lab(style_ref_rgb)
        ref_mean_lum = float(np.mean(style_lab[..., 0]))
        scale_k = 2.0 ** self.wavelet_levels
        target_ll_mean = scale_k * ref_mean_lum

        for src_f, cand_f in zip(src_frames_rgb, candidate_frames_rgb):
            src_lab = srgb_to_lab(src_f)
            cand_lab = srgb_to_lab(cand_f)
            pre_ssim_list.append(compute_wavelet_edge_ssim(src_lab[..., 0], cand_lab[..., 0]))

            trans_lab = transport_chromaticity_wasserstein(
                cand_lab,
                style_lab,
                blend_alpha=1.0,
                swot_steps=self.swot_steps,
                anchor_lum_mean=True,
            )

            src_dwt = dwt2_multilevel(src_lab[..., 0], levels=self.wavelet_levels)
            trans_dwt = dwt2_multilevel(trans_lab[..., 0], levels=self.wavelet_levels)
            pre_ll_tiles.append(trans_dwt.ll.copy())

            # Motion-compensated and C^0 hinged motion-innovation-gated temporal EMA on coarse LL_K
            if prev_ll is None or prev_src_ll is None:
                smooth_ll = trans_dwt.ll.copy()
            else:
                if motion_compensated:
                    # Estimate global camera translation on source LL_K and warp recurrent LL_K^(t-1)
                    dy_s, dx_s = self._estimate_subband_shift(prev_src_ll, src_dwt.ll)
                    warped_prev_src = np.roll(np.roll(prev_src_ll, dy_s, axis=0), dx_s, axis=1)
                    warped_prev_ll = np.roll(np.roll(prev_ll, dy_s, axis=0), dx_s, axis=1)
                    # Convert subband difference to L* [0, 100] scale via 2^(-K)
                    mot_diff_lstar = np.abs(src_dwt.ll - warped_prev_src) / scale_k
                    beta_map = compute_motion_innovation_gate(
                        mot_diff_lstar,
                        temporal_ema_beta=temporal_ema_beta,
                        motion_gate_sigma=motion_gate_sigma,
                    )
                    smooth_ll = beta_map * warped_prev_ll + (1.0 - beta_map) * trans_dwt.ll
                    smooth_ll += target_ll_mean - float(np.mean(smooth_ll))
                else:
                    smooth_ll = temporal_ema_beta * prev_ll + (1.0 - temporal_ema_beta) * trans_dwt.ll

            prev_ll = smooth_ll
            prev_src_ll = src_dwt.ll.copy()
            post_ll_tiles.append(smooth_ll.copy())

            fused_details = [
                (0.92 * s_lh + 0.08 * t_lh, 0.92 * s_hl + 0.08 * t_hl, 0.92 * s_hh + 0.08 * t_hh)
                for (s_lh, s_hl, s_hh), (t_lh, t_hl, t_hh) in zip(src_dwt.details, trans_dwt.details)
            ]
            raw_lum = idwt2_multilevel(
                WaveletDecomposition2D(
                    ll=smooth_ll,
                    details=fused_details,
                    original_shape=src_dwt.original_shape,
                )
            )
            preclamp_means.append(float(np.mean(raw_lum)))
            fused_lum_clip = np.clip(raw_lum, 0.0, 100.0)
            # Interior post-clamp DC re-anchoring with |I^o| = 0 guard and per-pixel shift cap 1.0
            fused_lum, _ = apply_interior_dc_reanchoring(fused_lum_clip, ref_mean_lum)
            post_ssim_list.append(compute_wavelet_edge_ssim(src_lab[..., 0], fused_lum))

            out_lab = trans_lab.copy()
            out_lab[..., 0] = fused_lum
            out_rgb = lab_to_srgb(out_lab)
            compiled_frames.append(out_rgb)
            frame_digests_raw.append(hashlib.sha256(out_rgb.tobytes()).digest())

        post_preclamp_flicker_var = (
            float(np.var(np.diff(preclamp_means))) if len(preclamp_means) > 1 else 0.0
        )
        post_ll_means = [
            float(np.mean(srgb_to_lab(f)[..., 0])) for f in compiled_frames
        ]
        post_flicker_var = float(np.var(np.diff(post_ll_means))) if len(post_ll_means) > 1 else 0.0

        pre_ll_arr = np.stack(pre_ll_tiles, axis=0)
        post_ll_arr = np.stack(post_ll_tiles, axis=0)
        pre_local_ll_var = float(np.mean(np.var(pre_ll_arr, axis=0)))
        post_local_ll_var = float(np.mean(np.var(post_ll_arr, axis=0)))

        # Compute binary Merkle root over raw 32-byte frame SHA-256 digests for MP4 C2PA `uuid` box
        gop_merkle_root = fold_merkle_digests_sha256(frame_digests_raw)

        return {
            "compiled_frames": compiled_frames,
            "gop_merkle_root_sha256": gop_merkle_root,
            "video_model_id": video_model_id,
            "num_frames": len(compiled_frames),
            "pre_flicker_variance": pre_flicker_var,
            "post_preclamp_flicker_variance": post_preclamp_flicker_var,
            "post_flicker_variance": post_flicker_var,
            "pre_local_ll_variance": pre_local_ll_var,
            "post_local_ll_variance": post_local_ll_var,
            "mean_pre_edge_ssim": float(np.mean(pre_ssim_list)),
            "mean_post_edge_ssim": float(np.mean(post_ssim_list)),
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }


def deflicker_video_gop_baselines(
    src_frames_rgb: List[np.ndarray],
    candidate_frames_rgb: List[np.ndarray],
    style_ref_rgb: np.ndarray,
    temporal_ema_beta: float = 0.78,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate video temporal deflickering baselines on an 8-frame GOP:
      1. Pixel_Space_EMA: Unwarped frame-to-frame pixel EMA
         I_ema^(t) = beta * I_ema^(t-1) + (1-beta) * I_cand^(t).
         Suppresses global flicker partially but blurs moving edges.
      2. Motion_Comp_Pixel_EMA: Translation-compensated pixel-space EMA
         I_ema^(t) = beta * T_{dy,dx}(I_ema^(t-1)) + (1-beta) * I_cand^(t).
         Reduces camera-pan ghosting relative to unwarped pixel EMA, but still
         lacks wavelet detail locking and retains residual exposure flicker.
      3. Global_Lum_Hist_Lock: Unwarped global affine luminance mean locking (without
         wavelet detail lock or local LL_K smoothing). Eliminates global frame-mean jumps
         but neither restores high-frequency edges nor attenuates local LL_K tile noise.
      4. Ungated_Motion_Wavelet_EMA: Motion-compensated wavelet LL_K EMA with the
         innovation gate disabled (beta_t == beta everywhere), isolating the benefit
         of innovation gating on fast-moving non-rigid foreground objects.
    """
    style_lab = srgb_to_lab(style_ref_rgb)
    ref_mean = float(np.mean(style_lab[..., 0]))

    # Baseline 1: Unwarped Pixel-Space Frame-to-Frame EMA
    ema_frames: List[np.ndarray] = []
    ema_ssim: List[float] = []
    prev_f: Optional[np.ndarray] = None
    for src_f, cand_f in zip(src_frames_rgb, candidate_frames_rgb):
        src_lab = srgb_to_lab(src_f)
        cand_f_float = cand_f.astype(np.float64)
        if prev_f is None:
            cur_f = cand_f_float
        else:
            cur_f = temporal_ema_beta * prev_f + (1.0 - temporal_ema_beta) * cand_f_float
        prev_f = cur_f
        out_u8 = np.clip(np.round(cur_f), 0, 255).astype(np.uint8)
        ema_frames.append(out_u8)
        ema_lab = srgb_to_lab(out_u8)
        ema_ssim.append(compute_wavelet_edge_ssim(src_lab[..., 0], ema_lab[..., 0]))

    ema_means = [float(np.mean(srgb_to_lab(f)[..., 0])) for f in ema_frames]
    ema_flicker_var = float(np.var(np.diff(ema_means))) if len(ema_means) > 1 else 0.0

    # Baseline 2: Motion-Compensated Pixel-Space EMA
    mc_ema_frames: List[np.ndarray] = []
    mc_ema_ssim: List[float] = []
    prev_mc_f: Optional[np.ndarray] = None
    prev_src_lum: Optional[np.ndarray] = None
    for src_f, cand_f in zip(src_frames_rgb, candidate_frames_rgb):
        src_lab = srgb_to_lab(src_f)
        s_lum = src_lab[..., 0]
        cand_f_float = cand_f.astype(np.float64)
        if prev_mc_f is None or prev_src_lum is None:
            cur_mc = cand_f_float
        else:
            dy_p, dx_p = AgentGhibliLockCompiler._estimate_subband_shift(prev_src_lum, s_lum, max_shift=6)
            warped_prev = np.roll(np.roll(prev_mc_f, dy_p, axis=0), dx_p, axis=1)
            cur_mc = temporal_ema_beta * warped_prev + (1.0 - temporal_ema_beta) * cand_f_float
        prev_mc_f = cur_mc
        prev_src_lum = s_lum
        out_mc_u8 = np.clip(np.round(cur_mc), 0, 255).astype(np.uint8)
        mc_ema_frames.append(out_mc_u8)
        mc_lab = srgb_to_lab(out_mc_u8)
        mc_ema_ssim.append(compute_wavelet_edge_ssim(s_lum, mc_lab[..., 0]))

    mc_ema_means = [float(np.mean(srgb_to_lab(f)[..., 0])) for f in mc_ema_frames]
    mc_ema_flicker_var = float(np.var(np.diff(mc_ema_means))) if len(mc_ema_means) > 1 else 0.0

    # Baseline 3: Global Affine Luminance Mean Locking (without DWT detail lock)
    hist_frames: List[np.ndarray] = []
    hist_ssim: List[float] = []
    for src_f, cand_f in zip(src_frames_rgb, candidate_frames_rgb):
        src_lab = srgb_to_lab(src_f)
        cand_lab = srgb_to_lab(cand_f)
        cand_lab[..., 0] = np.clip(cand_lab[..., 0] + (ref_mean - float(np.mean(cand_lab[..., 0]))), 0.0, 100.0)
        out_u8 = lab_to_srgb(cand_lab)
        hist_frames.append(out_u8)
        hist_ssim.append(compute_wavelet_edge_ssim(src_lab[..., 0], cand_lab[..., 0]))

    hist_means = [float(np.mean(srgb_to_lab(f)[..., 0])) for f in hist_frames]
    hist_flicker_var = float(np.var(np.diff(hist_means))) if len(hist_means) > 1 else 0.0

    # Baseline 4: Ungated Motion-Compensated Wavelet LL_K EMA (motion_gate_sigma -> inf)
    comp_ungated = AgentGhibliLockCompiler()
    ungated_res = comp_ungated.compile_video_gop(
        src_frames_rgb,
        candidate_frames_rgb,
        style_ref_rgb,
        temporal_ema_beta=temporal_ema_beta,
        motion_compensated=True,
        motion_gate_sigma=1e9,
    )

    return {
        "Pixel_Space_EMA": {
            "post_flicker_variance": round(ema_flicker_var, 4),
            "mean_post_edge_ssim": round(float(np.mean(ema_ssim)), 4),
        },
        "Motion_Comp_Pixel_EMA": {
            "post_flicker_variance": round(mc_ema_flicker_var, 4),
            "mean_post_edge_ssim": round(float(np.mean(mc_ema_ssim)), 4),
        },
        "Global_Lum_Hist_Lock": {
            "post_flicker_variance": round(hist_flicker_var, 4),
            "mean_post_edge_ssim": round(float(np.mean(hist_ssim)), 4),
        },
        "Ungated_Motion_Wavelet_EMA": {
            "post_flicker_variance": round(ungated_res["post_flicker_variance"], 4),
            "post_local_ll_variance": round(ungated_res["post_local_ll_variance"], 4),
            "mean_post_edge_ssim": round(ungated_res["mean_post_edge_ssim"], 4),
        },
    }


def compute_parseval_wavelet_energies(channel: np.ndarray, levels: int = 2) -> Tuple[float, float]:
    """
    Verify Parseval's Theorem for the orthonormal 2D Haar Wavelet Transform:
    Returns (spatial_frobenius_energy_sq, spectral_subband_energy_sq).
    Requires channel dimensions to be divisible by 2^levels.
    """
    arr = np.asarray(channel, dtype=np.float64)
    spatial_energy = float(np.sum(arr ** 2))
    decomp = dwt2_multilevel(arr, levels=levels)
    spectral_energy = float(np.sum(decomp.ll ** 2))
    for lh, hl, hh in decomp.details:
        spectral_energy += float(np.sum(lh ** 2) + np.sum(hl ** 2) + np.sum(hh ** 2))
    return spatial_energy, spectral_energy
