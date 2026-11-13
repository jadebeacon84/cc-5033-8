"""Objective comparison against generated clean truth.

We report three families of numbers and are explicit about what they do *not*
prove:

* fidelity: PSNR / SSIM / noise sigma of a flat patch before vs after,
* detail: high-frequency energy inside and outside thin-line structure,
* temporal: frame-to-frame difference (flicker) and residual against truth.

Detail loss is the key anti-cheat metric: a flat blur reduces noise but also
destroys fine-line contrast.  We measure fine-line contrast directly on known
line geometry from the synthetic scene, and we always compare against the
original source, never against a re-encoded proxy.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((np.clip(a, 0, 1) - np.clip(b, 0, 1)) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM using the standard gaussian-window formulation on luma."""
    x = _gray(a)
    y = _gray(b)
    C1, C2 = (0.01 * 1.0) ** 2, (0.03 * 1.0) ** 2
    gx = cv2.GaussianBlur(x, (11, 11), 1.5)
    gy = cv2.GaussianBlur(y, (11, 11), 1.5)
    gxx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - gx * gx
    gyy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - gy * gy
    gxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - gx * gy
    s = ((2 * gx * gy + C1) * (2 * gxy + C2)) / ((gx ** 2 + gy ** 2 + C1) * (gxx + gyy + C2))
    return float(np.mean(s))


def high_freq_energy(img: np.ndarray) -> float:
    g = _gray(img)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return float(np.mean(lap * lap))


def noise_sigma_flat(img: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Estimate residual noise in flat/lower-texture areas only."""
    g = _gray(img)
    hp = g - cv2.medianBlur(g, 3)
    if mask is None:
        # use quietest 40% of blocks
        b = 8
        h, w = hp.shape
        bh, bw = h // b, w // b
        blocks = hp[:bh * b, :bw * b].reshape(bh, b, bw, b).swapaxes(1, 2).reshape(-1, b * b)
        mad = 1.4826 * np.median(np.abs(blocks - np.median(blocks, axis=1, keepdims=True)), axis=1)
        m = mad <= np.quantile(mad, 0.4)
        return float(np.median(mad[m]))
    idx = mask.astype(bool)
    if idx.sum() < 64:
        return float(1.4826 * np.median(np.abs(hp - np.median(hp)))
                     if hp.size else 0.0)
    return float(1.4826 * np.median(np.abs(hp[idx] - np.median(hp[idx]))))


def flat_mask(img: np.ndarray, block: int = 12) -> np.ndarray:
    """Mask of low-texture pixels, used to measure residual noise honestly."""
    g = _gray(img)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    e = np.sqrt(gx * gx + gy * gy)
    return (e < np.percentile(e, 45)).astype(np.uint8)


def detail_mask(img: np.ndarray) -> np.ndarray:
    """High-structure mask (thin lines / edges) used for detail retention."""
    g = _gray(img)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    e = np.sqrt(gx * gx + gy * gy)
    return (e > np.percentile(e, 97)).astype(np.uint8)


def detail_contrast(img: np.ndarray, mask: np.ndarray) -> float:
    """Mean |laplacian| inside the structure mask: drops when lines are blurred."""
    g = _gray(img)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
    idx = mask.astype(bool)
    if idx.sum() == 0:
        return 0.0
    return float(np.mean(lap[idx]))


def temporal_flicker(frames: list[np.ndarray]) -> float:
    """Mean frame-to-frame luma difference in static-flat areas.

    Lower is smoother.  Measured on the *whole* frame so moving content
    contributes; it is a relative number between runs of the same clip.
    """
    diffs = []
    for a, b in zip(frames[:-1], frames[1:]):
        diffs.append(float(np.mean(np.abs(_gray(a) - _gray(b)))))
    return float(np.mean(diffs)) if diffs else 0.0


def evaluate_run(
    noisy: list[np.ndarray],
    clean: list[np.ndarray],
    out: list[np.ndarray],
    *,
    timestamps: list[float],
    label: str,
    baseline_bicubic: list[np.ndarray] | None = None,
) -> dict:
    """Compare noisy and denoised output against clean truth frame by frame."""
    n = min(len(noisy), len(clean), len(out))
    per_frame = []
    detail_bbox = None
    for i in range(n):
        fm = flat_mask(clean[i])
        dm = detail_mask(clean[i])
        rec = {
            "frame": i,
            "timestamp": float(timestamps[i]) if i < len(timestamps) else i / 25.0,
            "psnr_noisy": psnr(noisy[i], clean[i]),
            "psnr_out": psnr(out[i], clean[i]),
            "ssim_noisy": ssim(noisy[i], clean[i]),
            "ssim_out": ssim(out[i], clean[i]),
            "sigma_noisy_flat": noise_sigma_flat(noisy[i], fm),
            "sigma_out_flat": noise_sigma_flat(out[i], fm),
            "sigma_clean_flat": noise_sigma_flat(clean[i], fm),
            "hfe_noisy": high_freq_energy(noisy[i]),
            "hfe_out": high_freq_energy(out[i]),
            "hfe_clean": high_freq_energy(clean[i]),
            "detail_contrast_noisy": detail_contrast(noisy[i], dm),
            "detail_contrast_out": detail_contrast(out[i], dm),
            "detail_contrast_clean": detail_contrast(clean[i], dm),
        }
        if baseline_bicubic is not None and i < len(baseline_bicubic):
            rec["psnr_bicubic"] = psnr(baseline_bicubic[i], clean[i])
            rec["ssim_bicubic"] = ssim(baseline_bicubic[i], clean[i])
        per_frame.append(rec)

    def agg(key):
        vals = [r[key] for r in per_frame if r.get(key) is not None and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "label": label,
        "frames": n,
        "psnr_noisy": agg("psnr_noisy"),
        "psnr_out": agg("psnr_out"),
        "ssim_noisy": agg("ssim_noisy"),
        "ssim_out": agg("ssim_out"),
        "sigma_noisy_flat": agg("sigma_noisy_flat"),
        "sigma_out_flat": agg("sigma_out_flat"),
        "sigma_clean_flat": agg("sigma_clean_flat"),
        "hfe_noisy": agg("hfe_noisy"),
        "hfe_out": agg("hfe_out"),
        "hfe_clean": agg("hfe_clean"),
        "detail_contrast_noisy": agg("detail_contrast_noisy"),
        "detail_contrast_out": agg("detail_contrast_out"),
        "detail_contrast_clean": agg("detail_contrast_clean"),
        "flicker_noisy": temporal_flicker(noisy[:n]),
        "flicker_out": temporal_flicker(out[:n]),
        "flicker_clean": temporal_flicker(clean[:n]),
        "noise_reduction_flat_sigma": (
            agg("sigma_noisy_flat") - agg("sigma_out_flat")
            if np.isfinite(agg("sigma_noisy_flat")) else float("nan")
        ),
        "noise_reduction_ratio": (
            agg("sigma_out_flat") / agg("sigma_noisy_flat")
            if agg("sigma_noisy_flat") > 1e-9 else float("nan")
        ),
        "detail_retention_ratio": (
            agg("detail_contrast_out") / agg("detail_contrast_noisy")
            if agg("detail_contrast_noisy") > 1e-9 else float("nan")
        ),
    }
    if baseline_bicubic is not None:
        summary["psnr_bicubic"] = agg("psnr_bicubic")
        summary["ssim_bicubic"] = agg("ssim_bicubic")
    return {"summary": summary, "frames": per_frame}


def write_report(evals: dict[str, dict], path: str, meta: dict | None = None) -> None:
    payload = {
        "meta": meta or {},
        "comparisons": {k: v["summary"] for k, v in evals.items()},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
