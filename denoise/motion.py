"""Motion estimation between the current frame and its temporal neighbours.

Design choices that matter for the "no ghosting" requirement:

* We warp each neighbour *onto* the current frame with an explicit forward
  field and then measure the residual there.  Averaging happens only where the
  warp is demonstrably accurate.
* Dense Farneback fields are refined and sanity-checked.  If the dense
  estimator is unsure (flat texture, large motion, bad residual) we fall back to
  global translation, then to identity, then drop the candidate.  A candidate
  that fails is *never* averaged in unregistered.
* Matching residual and validity both gate the fusion weight, and the caller
  can reduce validity further (e.g. clipping at clip boundaries).
"""
from __future__ import annotations

import cv2
import numpy as np

from .types import MotionEstimate


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)


def _shift_global(gray_a: np.ndarray, gray_b: np.ndarray, max_shift: int = 48) -> tuple[float, float, float]:
    """Phase-correlation global shift a->b plus its response strength."""
    h, w = gray_a.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(gray_a, gray_b, win)
    if not np.isfinite(dx) or not np.isfinite(dy) or abs(dx) > max_shift or abs(dy) > max_shift:
        return 0.0, 0.0, 0.0
    return float(dx), float(dy), float(max(0.0, response))


def _warp(img: np.ndarray, flow_x: np.ndarray, flow_y: np.ndarray) -> np.ndarray:
    """Sample `img` at (x + flow_x, y + flow_y), bilinear, border constant."""
    h, w = flow_x.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = xx + flow_x
    map_y = yy + flow_y
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _in_bounds(flow_x: np.ndarray, flow_y: np.ndarray) -> np.ndarray:
    h, w = flow_x.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    inside = (xx + flow_x >= 0) & (xx + flow_x <= w - 1) & (yy + flow_y >= 0) & (yy + flow_y <= h - 1)
    return inside.astype(np.float32)


def _colour_residual(a: np.ndarray, b_w: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(a - b_w), axis=2)


def _local_mean(img: np.ndarray, k: int) -> np.ndarray:
    return cv2.blur(img.astype(np.float32), (k, k))


def _confidence_map(
    ref_gray: np.ndarray,
    warped_gray: np.ndarray,
    residual: np.ndarray,
    noise_sigma: float,
) -> np.ndarray:
    """Combine photometric agreement with edge-direction consistency.

    Flat water is not penalised just for being flat: a mis-registration there
    produces almost no visible error, so temporal averaging is safe.  What we
    must catch is half-occluded or independently moving content, where dense
    flow can land on a plausible colour by luck while the local edge
    orientation disagrees.  That check is only meaningful where structure
    exists, hence the texture weighting of the edge term.
    """
    gx = cv2.Sobel(ref_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ref_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    texture = np.clip(grad / (np.percentile(grad, 95) + 1e-6), 0.0, 1.0)
    texture_w = np.clip(2.0 * texture, 0.0, 1.0)

    # Residual normalised by expected noise floor: residual <= sigma is roughly
    # pure noise, residual >= 4*sigma is a genuine mismatch.
    sigma = max(noise_sigma, 1e-4)
    resid_score = np.clip(1.0 - (residual - sigma) / (3.0 * sigma), 0.0, 1.0)

    wgx = cv2.Sobel(warped_gray, cv2.CV_32F, 1, 0, ksize=3)
    wgy = cv2.Sobel(warped_gray, cv2.CV_32F, 0, 1, ksize=3)
    e1 = np.sqrt(gx * gx + gy * gy) + 1e-4
    e2 = np.sqrt(wgx * wgx + wgy * wgy) + 1e-4
    cos = (gx * wgx + gy * wgy) / (e1 * e2)
    edge_align = np.clip((cos + 1.0) * 0.5, 0.0, 1.0)
    consistency = 1.0 - texture_w * (1.0 - edge_align)

    conf = _local_mean(resid_score * consistency, 5)
    return np.clip(conf, 0.0, 1.0)


def _dense_flow(ref_gray: np.ndarray, nb_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Farneback estimates displacements in *pixel* units and assumes integer
    # intensities.  Feeding it 0..1 floats silently yields a ~zero field, so
    # the images must be scaled to 0..255 uint8 first.
    a = np.clip(ref_gray * 255.0 + 0.5, 0, 255).astype(np.uint8)
    b = np.clip(nb_gray * 255.0 + 0.5, 0, 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(
        a, b, None,
        pyr_scale=0.5, levels=4, winsize=21, iterations=3,
        poly_n=7, poly_sigma=1.5, flags=0,
    )
    return flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32)


def estimate_motion(
    ref: np.ndarray,
    neighbor: np.ndarray,
    ref_index: int,
    neighbor_index: int,
    noise_sigma: float,
    max_residual: float,
    min_confidence: float,
    model: str = "dense",
    reference_usable: bool = True,
    neighbor_usable: bool = True,
    is_boundary: bool = False,
) -> MotionEstimate:
    h, w = ref.shape[:2]
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    kind = "dense"
    detail = ""
    ref_gray = _to_gray(ref)
    nb_gray = _to_gray(neighbor)

    if not (reference_usable and neighbor_usable):
        kind = "dropped"
        detail = "source frame missing or unusable"
        residual = np.ones((h, w), np.float32)
        valid = np.zeros((h, w), np.float32)
        return MotionEstimate(ref_index, neighbor_index, dx, dy, residual, valid,
                              np.zeros((h, w), np.float32), kind, detail)

    if model == "off":
        # Identity is only safe when the camera is actually static; treat any
        # structured motion as a failed registration.
        kind = "fallback_identity"
        detail = "motion compensation disabled; residual gate decides"
    elif model == "translation":
        gdx, gdy, response = _shift_global(ref_gray, nb_gray)
        dx = np.full((h, w), gdx, np.float32)
        dy = np.full((h, w), gdy, np.float32)
        kind = "translation" if response > 0.05 else "fallback_identity"
        detail = f"phase correlation response {response:.3f}"
    else:
        try:
            dx, dy = _dense_flow(ref_gray, nb_gray)
            kind = "dense"
        except Exception as exc:  # pragma: no cover - defensive around cv2
            gdx, gdy, response = _shift_global(ref_gray, nb_gray)
            dx = np.full((h, w), gdx, np.float32)
            dy = np.full((h, w), gdy, np.float32)
            kind = "translation" if response > 0.05 else "fallback_identity"
            detail = f"dense flow failed ({exc}); phase correlation response {response:.3f}"

    warped = _warp(neighbor, dx, dy)
    inside = _in_bounds(dx, dy)
    residual = _colour_residual(ref, warped)
    confidence = _confidence_map(ref_gray, _warp(nb_gray, dx, dy), residual, noise_sigma)
    valid = (
        (inside > 0)
        & (residual <= max_residual)
        & (confidence >= min_confidence)
    ).astype(np.float32)

    # Edge-preserving cleanup of the validity mask (remove speckles, fill
    # holes that are surrounded by agreement).
    valid = cv2.medianBlur((valid * 255).astype(np.uint8), 3).astype(np.float32) / 255.0

    # Sanity check: if almost nothing is valid, the dense field is not
    # trustworthy.  Try global translation; if that is also bad, declare
    # failure rather than allowing an unregistered average.
    valid_ratio = float(valid.mean())
    if kind == "dense" and valid_ratio < 0.15:
        gdx, gdy, response = _shift_global(ref_gray, nb_gray)
        tdx = np.full((h, w), gdx, np.float32)
        tdy = np.full((h, w), gdy, np.float32)
        tw = _warp(neighbor, tdx, tdy)
        tr = _colour_residual(ref, tw)
        tin = _in_bounds(tdx, tdy)
        tc = _confidence_map(ref_gray, _warp(nb_gray, tdx, tdy), tr, noise_sigma)
        tv = ((tin > 0) & (tr <= max_residual) & (tc >= min_confidence)).astype(np.float32)
        tv = cv2.medianBlur((tv * 255).astype(np.uint8), 3).astype(np.float32) / 255.0
        if tv.mean() > valid_ratio + 0.02 and response > 0.05:
            dx, dy, kind = tdx, tdy, "translation"
            residual, confidence, valid = tr, tc, tv
            detail = f"dense registration weak ({valid_ratio:.0%}); global shift response {response:.3f}"
        elif tv.mean() <= 0.02:
            kind = "fallback_identity"
            detail = f"motion compensation failed (dense valid {valid_ratio:.0%}, global {tv.mean():.0%})"

    if is_boundary:
        detail = (detail + "; " if detail else "") + "clip boundary: shorter temporal window"

    return MotionEstimate(ref_index, neighbor_index, dx.astype(np.float32), dy.astype(np.float32),
                          residual.astype(np.float32), valid, confidence.astype(np.float32),
                          kind, detail)
