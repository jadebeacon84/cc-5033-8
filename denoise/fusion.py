"""Temporal weighted fusion followed by a separate spatial stage.

The temporal stage answers one question per pixel: *how much neighbour
evidence do we accept here?*  The spatial stage is a deliberately separate,
edge-preserving filter whose strength is controlled independently, so the
operator can tell "the time window cleaned the flicker but left grain" apart
from "the spatial stage smoothed texture".

Ghosting control is explicit:

* The current frame always carries self weight 1.0, so the result is anchored
  to what was actually observed.
* A neighbour's per-pixel weight is ``temporal_strength * distance_decay *
  registration_validity * residual_soft_gate`` and is capped at 0.85.
* Registration failure (``valid`` ~ 0) therefore yields zero neighbour weight;
  a failed candidate is dropped from the table rather than averaged.
"""
from __future__ import annotations

import cv2
import numpy as np

from .motion import _warp
from .types import Candidate, DenoiseResult, FrameRef, MotionEstimate

MAX_NEIGHBOR_WEIGHT = 0.85


def _edge_preserve_map(img: np.ndarray) -> np.ndarray:
    """0 in flat water, 1 on crisp structure.  Drives spatial blend gating."""
    gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    # Smooth first so isolated noise does not masquerade as structure.
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    # Scale-free threshold: use a robust high quantile of the frame's structure.
    hi = float(np.percentile(edge, 98)) + 1e-5
    lo = float(np.percentile(edge, 55))
    p = np.clip((edge - lo) / (hi - lo + 1e-5), 0.0, 1.0)
    return cv2.GaussianBlur(p.astype(np.float32), (5, 5), 0)


def spatial_stage(
    img: np.ndarray,
    strength: float,
    noise_sigma: float,
    preserve_hint: np.ndarray | None = None,
) -> np.ndarray:
    """Edge-preserving spatial blend, controlled separately from the window.

    strength=0 -> passthrough.  strength is *not* a fixed blur radius: the
    blend is attenuated strongly on detected structure so thin lines, coral
    edges and slow-moving object boundaries are retained.  This stage does not
    invent detail; it only reduces residual noise.
    """
    if strength <= 0.0:
        return img.astype(np.float32)
    strength = float(np.clip(strength, 0.0, 1.0))

    preserve = _edge_preserve_map(img) if preserve_hint is None else preserve_hint
    amount = strength * (1.0 - 0.9 * preserve)          # 0..1 per pixel
    if float(amount.max()) < 1e-4:
        return img.astype(np.float32)

    sigma = max(noise_sigma, 1e-4)
    base = cv2.GaussianBlur(img.astype(np.float32), (5, 5), 1.0)
    # Joint bilateral against the Gaussian base: the range term keeps colours
    # that are genuinely different (coral vs. water) from mixing.
    smooth = cv2.bilateralFilter(
        base,
        d=7,
        sigmaColor=max(8.0, sigma * 255.0 * 2.5),
        sigmaSpace=6.0,
    )
    out = img * (1.0 - amount[..., None]) + smooth * amount[..., None]
    # Do not let the spatial stage overshoot ringing into new artefacts.
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _candidate_note(m: MotionEstimate | None, usable: bool) -> str:
    if not usable:
        return "source frame unusable/missing"
    if m is None:
        return "no registration against this frame"
    if m.kind == "dropped":
        return "registration dropped"
    if m.kind == "fallback_identity":
        return "compensation failed; residual gate only, no blind average"
    return ""


def fuse_frame(
    frames: list[FrameRef],
    current_pos: int,
    motion: dict[int, MotionEstimate],
    params,
    noise_sigma: float,
) -> DenoiseResult:
    """Fuse one frame from its temporal window.

    ``frames`` is the decoded sequence (may contain unusable frames);
    ``current_pos`` indexes into it; ``motion`` maps neighbour position to the
    estimate against the current frame.
    """
    ref = frames[current_pos]
    h, w = ref.image.shape[:2]
    radius = max(0, int(params.temporal_radius))
    temporal_strength = float(np.clip(params.temporal_strength, 0.0, 1.0))
    sigma = max(noise_sigma, 1e-4)

    acc = ref.image.astype(np.float32).copy()
    weights = np.ones((h, w), np.float32)      # self weight 1.0 everywhere
    candidates: list[Candidate] = []
    used_offsets: list[int] = []

    for off in range(-radius, radius + 1):
        if off == 0:
            continue
        pos = current_pos + off
        if pos < 0 or pos >= len(frames):
            candidates.append(Candidate(off, 0.0, 0.0, None, "dropped",
                                        "outside clip window"))
            continue
        nb = frames[pos]
        if not nb.usable:
            candidates.append(Candidate(off, 0.0, 0.0, None, "dropped",
                                        "source frame unusable/missing"))
            continue
        m = motion.get(pos)
        if m is None or m.kind == "dropped":
            candidates.append(Candidate(off, 0.0, 0.0, None, "dropped",
                                        _candidate_note(m, nb.usable)))
            continue

        warped = _warp(nb.image, m.dx, m.dy)
        # Soft residual falloff: residual <= 1 sigma is noise-only (keep),
        # residual >= 5 sigma is a real mismatch (reject smoothly).
        soft = np.exp(-0.5 * (m.residual / (2.5 * sigma)) ** 2).astype(np.float32)
        soft = np.clip(soft, 0.0, 1.0)
        dist_decay = float(np.exp(-0.5 * (abs(off) / max(1.0, radius)) ** 2))
        pixel_w = temporal_strength * dist_decay * m.valid * soft
        pixel_w = np.clip(pixel_w, 0.0, MAX_NEIGHBOR_WEIGHT).astype(np.float32)

        vr = float(m.valid.mean())
        med_resid = float(np.nanmedian(m.residual)) if m.residual.size else float("inf")
        # Majority misregistered (e.g. across a hard cut the dense field can
        # invent a plausible warp): drop the whole candidate. Averaging in the
        # few percent of pixels that happen to pass would ghost them.
        if vr < 0.02 or float(pixel_w.max()) < 0.02 or med_resid > params.max_residual:
            candidates.append(Candidate(off, 0.0, 0.0, med_resid,
                                        "dropped",
                                        _candidate_note(m, True) or
                                        (f"median residual {med_resid:.3f} exceeds gate "
                                         f"{params.max_residual:.3f}; registration failed")))
            continue

        acc += warped * pixel_w[..., None]
        weights += pixel_w
        used_offsets.append(off)
        candidates.append(Candidate(
            offset=off,
            weight_mean=float(pixel_w.mean()),
            overlap_ratio=float((pixel_w > 0.05).mean()),
            residual_median=float(np.nanmedian(m.residual)),
            kind=m.kind,
            note=m.detail,
        ))

    fused = acc / np.maximum(weights, 1e-6)[..., None]
    fused = np.clip(fused, 0.0, 1.0).astype(np.float32)

    # Fraction of neighbour evidence relative to the maximum an ideal window
    # could supply (all neighbours at full trust and full validity).
    ideal = max(1e-6, sum(
        temporal_strength * MAX_NEIGHBOR_WEIGHT * float(np.exp(-0.5 * (abs(o) / max(1.0, radius)) ** 2))
        for o in range(-radius, radius + 1) if o != 0
    ))
    temporal_evidence = float(np.clip((weights.mean() - 1.0) / ideal, 0.0, 1.0))

    preserve = _edge_preserve_map(fused)
    out = spatial_stage(fused, params.spatial_strength, sigma, preserve_hint=preserve)

    fallback = "none"
    if not used_offsets:
        fallback = "no temporal evidence; returned self frame" if radius > 0 else "temporal window disabled"
    elif all(c.kind == "fallback_identity" for c in candidates if c.weight_mean > 0):
        fallback = "identity fallback: weights limited to residual-consistent areas"
    elif any(c.note and "clip boundary" in c.note for c in candidates):
        fallback = "shortened window at clip boundary"

    return DenoiseResult(
        image=out,
        timestamp=ref.t,
        source_index=ref.index,
        noise_sigma=sigma,
        temporal_weight=weights,
        candidates=candidates,
        diagnostics={
            "fallback": fallback,
            "used_offsets": used_offsets,
            "temporal_evidence": temporal_evidence,
        },
    )
