"""Noise level estimation.

The temporal weights need a noise scale, so we must be able to say where that
number came from.  Two sources are supported:

* ``manifest``: the synthetic generator recorded the sigma it injected.
* ``flat_patch_estimate``: robust blockwise MAD of a high-pass image.

The estimator intentionally returns a conservative (low) estimate: textured
blocks inflate the high-pass energy, so we take a low quantile of block
activity rather than the global average.
"""
from __future__ import annotations

import numpy as np

from .types import NoiseModel

# 0.6745 = MAD -> sigma for a zero-mean Gaussian.
MAD_TO_SIGMA = 1.4826


def _high_pass(channel: np.ndarray) -> np.ndarray:
    """3x3 median minus the pixel: cheap, edge-sparse, robust to impulse."""
    med = channel.copy()
    # cv2 is a hard dependency of the project; import locally to keep the
    # module importable for pure-numpy tests.
    import cv2

    med = cv2.medianBlur(channel.astype(np.float32), 3)
    return channel - med


def estimate_noise_sigma(
    image: np.ndarray,
    block: int = 8,
    quantile: float = 0.25,
    sample_cap: int = 3,
) -> tuple[float, float, float]:
    """Estimate per-channel additive noise sigma from one RGB frame.

    Returns (sigma_r, sigma_g, sigma_b).  ``image`` is float RGB in [0, 1].
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    sigmas = []
    for c in range(3):
        ch = image[..., c]
        hp = _high_pass(ch)
        h, w = hp.shape
        bh, bw = h // block, w // block
        if bh < 1 or bw < 1:
            sigmas.append(float(MAD_TO_SIGMA * np.median(np.abs(hp - np.median(hp)))))
            continue
        blocks = (
            hp[: bh * block, : bw * block]
            .reshape(bh, block, bw, block)
            .swapaxes(1, 2)
            .reshape(bh * bw, block * block)
        )
        mad = MAD_TO_SIGMA * np.median(np.abs(blocks - np.median(blocks, axis=1, keepdims=True)), axis=1)
        # Reject blocks that contain structure: take the quiet quantile.
        cutoff = np.quantile(mad, quantile)
        quiet = mad[mad <= cutoff]
        if quiet.size == 0:
            quiet = mad
        sigmas.append(float(np.median(quiet)))
    return tuple(sigmas)  # type: ignore[return-value]


def noise_model_from_frames(
    frames: list[np.ndarray],
    manifest_sigma: tuple[float, float, float] | None = None,
    manifest_detail: str = "",
) -> NoiseModel:
    """Prefer the recorded injection level, otherwise measure flat blocks."""
    if manifest_sigma is not None and all(s > 0 for s in manifest_sigma):
        return NoiseModel(
            sigma_rgb=tuple(float(s) for s in manifest_sigma),
            origin="manifest",
            detail=manifest_detail
            or "synthetic generator recorded the added noise sigma in the clip manifest",
        )
    cap = frames[: max(1, min(3, len(frames)))]
    per_frame = [estimate_noise_sigma(f) for f in cap]
    med = np.median(np.asarray(per_frame), axis=0)
    return NoiseModel(
        sigma_rgb=tuple(float(s) for s in med),
        origin="flat_patch_estimate",
        detail=(
            "robust 8x8 flat-block MAD over "
            f"{len(cap)} sampled frame(s); quiet quartile of block activity"
        ),
    )
