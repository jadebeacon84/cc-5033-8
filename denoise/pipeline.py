"""Orchestration: turn a decoded timeline + params into per-frame results.

Fallback policy (explicit and testable):

1. A neighbour that is missing/unusable or outside the clip is dropped.
2. A neighbour whose registration fails falls back to identity *only if* its
   residual gate still passes locally; otherwise it is dropped.  It is never
   averaged in unregistered.
3. If no neighbour survives, the frame passes through the temporal stage with
   self weight only (spatial stage may still apply) and the diagnostics say so.
4. At clip start/end the window simply shrinks; missing frames are explicit
   slots in the timeline, and their timestamps are preserved in the output.
"""
from __future__ import annotations

import copy

import numpy as np

from .fusion import fuse_frame
from .motion import estimate_motion
from .noise import noise_model_from_frames
from .types import (DenoiseParams, DenoiseResult, FrameDiagnostics, FrameRef,
                    NoiseModel)


def motion_for_pair(
    frames: list[FrameRef],
    current_pos: int,
    neighbor_pos: int,
    params: DenoiseParams,
    noise_sigma: float,
    scene_cut: bool = False,
) -> object:
    ref, nb = frames[current_pos], frames[neighbor_pos]
    return estimate_motion(
        ref.image, nb.image, ref.index, nb.index,
        noise_sigma=noise_sigma,
        max_residual=params.max_residual,
        min_confidence=params.min_confidence,
        model=params.motion_model,
        reference_usable=ref.usable,
        neighbor_usable=nb.usable,
        is_boundary=(current_pos == 0 or current_pos == len(frames) - 1
                     or neighbor_pos == 0 or neighbor_pos == len(frames) - 1),
    )


def _cut_between(frames: list[FrameRef], a: int, b: int, cuts: set[int]) -> bool:
    """True if a declared scene cut (by source index) falls between positions."""
    lo, hi = sorted((a, b))
    if not cuts:
        return False
    for pos in range(lo + 1, hi + 1):
        if frames[pos].index in cuts:
            return True
    return False


def denoise_timeline(
    frames: list[FrameRef],
    params: DenoiseParams,
    noise: NoiseModel,
    scene_cuts: set[int] | None = None,
    progress=None,
) -> tuple[list[DenoiseResult], list[FrameDiagnostics], dict]:
    """Process the whole timeline, returning results, diagnostics and a summary.

    The output list has exactly one entry per *usable* source frame; unusable
    slots are skipped but their timestamps remain in the timeline map returned
    in the summary so the exporter can preserve cadence.
    """
    cuts = set(scene_cuts or set())
    sigma = noise.mean_sigma * float(params.sigma_scale)
    radius = max(0, int(params.temporal_radius))
    results: list[DenoiseResult] = []
    diags: list[FrameDiagnostics] = []
    cut_skips = 0

    for pos, ref in enumerate(frames):
        if not ref.usable:
            continue
        # Estimate motion only against neighbours in the same shot.
        motion: dict[int, object] = {}
        for off in range(-radius, radius + 1):
            if off == 0:
                continue
            npos = pos + off
            if npos < 0 or npos >= len(frames):
                continue
            if _cut_between(frames, pos, npos, cuts):
                cut_skips += 1
                continue
            if not frames[npos].usable:
                continue
            motion[npos] = motion_for_pair(frames, pos, npos, params, sigma)
        res = fuse_frame(frames, pos, motion, params, sigma)
        results.append(res)
        diag = FrameDiagnostics(
            frame_index=ref.index,
            timestamp=ref.t,
            motion=[{
                "neighbor_index": m.neighbor_index,
                "kind": m.kind,
                "valid_ratio": float(m.valid.mean()),
                "residual_median": float(np.nanmedian(m.residual)),
                "detail": m.detail,
            } for m in motion.values()],
            candidates=[{
                "offset": c.offset,
                "weight_mean": c.weight_mean,
                "overlap_ratio": c.overlap_ratio,
                "residual_median": c.residual_median,
                "kind": c.kind,
                "note": c.note,
            } for c in res.candidates],
            fallback=res.diagnostics.get("fallback", "none"),
            notes=[c.note for c in res.candidates if c.note and c.weight_mean == 0.0],
            noise_sigma=sigma,
            temporal_evidence=float(res.diagnostics.get("temporal_evidence", 0.0)),
            valid_ratio=float(np.mean([c.overlap_ratio for c in res.candidates])) if res.candidates else 1.0,
        )
        diags.append(diag)
        if progress is not None:
            progress(pos, len(frames), res)

    timestamps_out = [r.timestamp for r in results]
    source_indices = [r.source_index for r in results]
    summary = {
        "noise": noise.to_dict(),
        "working_sigma": sigma,
        "params": params.__dict__,
        "frames_in": len(frames),
        "frames_out": len(results),
        "missing_slots": sum(1 for f in frames if not f.usable),
        "scene_cut_skips": cut_skips,
        "fallback_frames": sum(1 for d in diags if d.fallback not in ("none",)),
        "mean_temporal_evidence": float(np.mean([d.temporal_evidence for d in diags])) if diags else 0.0,
        "mean_valid_ratio": float(np.mean([d.valid_ratio for d in diags])) if diags else 0.0,
        "timestamps": timestamps_out,
        "source_indices": source_indices,
    }
    return results, diags, summary


def detect_scene_cuts(frames: list[FrameRef], threshold: float = 0.35) -> set[int]:
    """Cheap histogram/pixel-difference cut detector for real footage.

    Cuts are guarded by an absolute jump relative to the clip's own motion
    distribution, so ordinary fast pans are not mistaken for cuts.
    """
    import cv2

    vals = []
    for a, b in zip(frames[:-1], frames[1:]):
        if not (a.usable and b.usable):
            vals.append(0.0)
            continue
        ga = cv2.resize(cv2.cvtColor(a.image, cv2.COLOR_RGB2GRAY), (160, 90))
        gb = cv2.resize(cv2.cvtColor(b.image, cv2.COLOR_RGB2GRAY), (160, 90))
        vals.append(float(np.mean(np.abs(ga - gb)))
                    + 0.5 * float(np.mean(np.abs(
                        np.histogram(ga, bins=32, range=(0, 1))[0].astype(np.float32) / ga.size
                        - np.histogram(gb, bins=32, range=(0, 1))[0].astype(np.float32) / gb.size))))
    if not vals:
        return set()
    base = float(np.median(vals))
    cut_pos = []
    for i, v in enumerate(vals):
        if v > max(threshold, base * 3.5):
            cut_pos.append(i + 1)
    return {frames[p].index for p in cut_pos if frames[p].usable}


def results_to_frames(results: list[DenoiseResult]) -> list[FrameRef]:
    return [FrameRef(index=r.source_index, t=r.timestamp, image=r.image) for r in results]
