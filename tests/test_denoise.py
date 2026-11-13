"""End-to-end checks for the underwater temporal denoiser.

Run with:  PYTHONPATH=$(pwd)/.tools/pylib:. python3 -m pytest tests -q
(.tools/pylib comes from `python3 tools/fetch_deps.py --test`.)

Each test builds a tiny deterministic clip in a temp directory and asserts
behaviour that the brief requires: noise falls, detail survives, misregistered
neighbours are rejected (no ghosting), boundaries/missing frames fall back
explicitly, and output timestamps match the source cadence.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from denoise import (DenoiseParams, denoise_timeline, noise_model_from_frames,
                     read_frames)
from denoise.fusion import fuse_frame
from denoise.media import make_synthetic_clip
from denoise.metrics import evaluate_run
from denoise.motion import estimate_motion
from denoise.pipeline import detect_scene_cuts
from denoise.types import FrameRef, MotionEstimate, NoiseModel

ROOT = Path(__file__).resolve().parent.parent


def _clean_depth(h=96, w=128):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 0.10 + 0.05 * (yy / h)
    img[..., 1] = 0.28 + 0.10 * (yy / h)
    img[..., 2] = 0.42 + 0.12 * (yy / h)
    return img


def _pan_sequence(n=9, shift=1.0, h=96, w=128):
    base = _clean_depth(h, w + int(shift * (n - 1)) + 2)
    # fine structure so registration has something to lock onto
    for k in range(6):
        x = int(10 + k * 19)
        cv2.line(base, (x, 12), (x + 12, h - 16), (0.95, 0.9, 0.8), 1)
    cv2.circle(base, (w // 2, h // 2), 9, (0.9, 0.9, 0.6), -1)
    frames = []
    for i in range(n):
        x = int(round(i * shift))
        frames.append(base[:, x:x + w].copy())
    return frames


# --------------------------------------------------------------------- motion --
def test_dense_flow_matches_known_translation():
    """Farneback must report the real pixel displacement, not ~0.

    Regression: feeding 0..1 floats to OpenCV yields an all-zero field, so the
    whole temporal stage silently degrades to identity registration.
    """
    seq = _pan_sequence(n=2, shift=3.0)
    m = estimate_motion(seq[0], seq[1], 0, 1, noise_sigma=0.01,
                        max_residual=0.2, min_confidence=0.0, model="dense")
    mag = np.sqrt(m.dx ** 2 + m.dy ** 2)
    # The scene slices move left, so sampling frame 1 at (x + dx) recovers frame
    # 0 with dx = -shift.
    assert abs(float(np.median(m.dx)) + 3.0) < 0.6, np.median(m.dx)
    assert float(np.median(mag)) > 2.0
    assert float(m.valid.mean()) > 0.5


def test_translation_model_used_for_uniform_shift():
    seq = _pan_sequence(n=2, shift=4.0)
    m = estimate_motion(seq[0], seq[1], 0, 1, noise_sigma=0.01,
                        max_residual=0.2, min_confidence=0.0, model="translation")
    assert abs(float(np.mean(m.dx)) + 4.0) < 0.3
    assert m.kind == "translation"


def test_cut_candidate_is_dropped_not_averaged():
    """Across a hard cut the whole candidate must be rejected.

    A dense field can invent a warp that matches a few percent of pixels; fusing
    those is exactly how ghosting appears.
    """
    rng = np.random.default_rng(0)
    a = _pan_sequence(n=1)[0]
    b = rng.random(a.shape, dtype=np.float32) * 0.6 + 0.2  # unrelated shot
    frames = [FrameRef(0, 0.0, a, True), FrameRef(1, 1.0, b, True)]
    sigma = 0.03
    m = estimate_motion(a, b, 0, 1, sigma, 0.12, 0.25)
    # spatial_strength=0 isolates the temporal decision.
    res = fuse_frame(frames, 0, {1: m}, DenoiseParams(temporal_radius=1,
                                                      spatial_strength=0.0), sigma)
    pos = [c for c in res.candidates if c.offset == 1]
    assert len(pos) == 1
    assert pos[0].weight_mean == 0.0
    assert pos[0].kind == "dropped"
    assert "residual" in pos[0].note.lower()
    # No temporal evidence: output is the self frame, not a cross-cut average.
    assert np.allclose(res.image, a, atol=1e-5)


def test_aligned_neighbour_does_get_weight():
    seq = _pan_sequence(n=3, shift=1.0)
    rng = np.random.default_rng(1)
    sigma = 0.02
    noisy = [np.clip(f + rng.normal(0, sigma, f.shape), 0, 1).astype(np.float32) for f in seq]
    frames = [FrameRef(i, i / 24.0, img, True) for i, img in enumerate(noisy)]
    m = estimate_motion(noisy[1], noisy[0], 1, 0, sigma, 0.12, 0.25)
    res = fuse_frame(frames, 1, {0: m}, DenoiseParams(temporal_radius=1), sigma)
    assert res.candidates[0].weight_mean > 0.1
    assert res.diagnostics["temporal_evidence"] > 0.1


# -------------------------------------------------------------------- pipeline --
def _noisy(frames, sigma, rng):
    return [np.clip(f + rng.normal(0, sigma, f.shape), 0, 1).astype(np.float32)
            for f in frames]


def the_rest_finite(summary):
    return all(np.isfinite(v) for v in summary.values() if isinstance(v, (int, float)))


def test_noise_reduction_and_detail_retention():
    rng = np.random.default_rng(2)
    sigma = 0.05
    clean = _pan_sequence(n=16, shift=1.0)
    noisy = _noisy(clean, sigma, rng)
    frames = [FrameRef(i, i / 24.0, noisy[i], True) for i in range(len(noisy))]
    noise = NoiseModel(sigma_rgb=(sigma,) * 3, origin="test",
                       detail="test injected gaussian", model="additive_gaussian_rgb")
    results, diags, _ = denoise_timeline(frames, DenoiseParams(temporal_radius=2), noise)
    out = [r.image for r in results]
    ev = evaluate_run(noisy=noisy, clean=clean, out=out,
                      timestamps=[r.timestamp for r in results], label="synth")
    s = ev["summary"]
    # Objective, truth-based: noise down and thin-line contrast kept.
    assert s["sigma_out_flat"] < s["sigma_noisy_flat"] * 0.5
    assert s["psnr_out"] > s["psnr_noisy"] + 5
    assert s["ssim_out"] > s["ssim_noisy"] + 0.15
    assert s["detail_retention_ratio"] > 0.55
    assert the_rest_finite(s)


def test_low_texture_scene_still_denoises():
    rng = np.random.default_rng(3)
    sigma = 0.06
    clean = [_clean_depth() for _ in range(10)]  # nearly static, flat water
    noisy = _noisy(clean, sigma, rng)
    frames = [FrameRef(i, i / 24.0, noisy[i], True) for i in range(len(noisy))]
    noise = NoiseModel(sigma_rgb=(sigma,) * 3, origin="test", detail="test",
                       model="additive_gaussian_rgb")
    results, _, _ = denoise_timeline(frames, DenoiseParams(temporal_radius=2), noise)
    out = [r.image for r in results]
    ev = evaluate_run(noisy=noisy, clean=clean, out=out,
                      timestamps=[r.timestamp for r in results], label="flat")
    s = ev["summary"]
    assert s["sigma_out_flat"] < s["sigma_noisy_flat"] * 0.5
    # A flat scene must not be smeared into a different flat scene.
    assert s["psnr_out"] > s["psnr_noisy"] + 4


def test_missing_frame_slot_is_skipped_and_timestamp_kept():
    rng = np.random.default_rng(4)
    sigma = 0.02
    clean = _pan_sequence(n=8, shift=1.0)
    noisy = _noisy(clean, sigma, rng)
    # Simulate read_frames' reconstruction: frames 3 and 4 never decoded, so a
    # 2-slot gap is inserted at the expected cadence.
    seq, stamps = [], []
    nominal = 1 / 24.0
    prev = None
    for i, img in enumerate(noisy):
        if i in (3, 4):
            continue
        t = i * nominal
        if prev is not None and t - prev > 1.6 * nominal:
            slots = int(round((t - prev) / nominal)) - 1
            for k in range(slots):
                seq.append(FrameRef(len(seq), prev + (k + 1) * nominal,
                                    np.zeros_like(img), False))
                stamps.append(prev + (k + 1) * nominal)
        seq.append(FrameRef(len(seq), t, img, True))
        stamps.append(t)
        prev = t
    noise = NoiseModel(sigma_rgb=(sigma,) * 3, origin="test", detail="test",
                       model="additive_gaussian_rgb")
    results, diags, summar = denoise_timeline(seq, DenoiseParams(temporal_radius=2), noise)
    assert summar["missing_slots"] == 2
    assert len(results) == 6
    # Output timestamps are the *source* ones; the gap is preserved.
    ts = [r.timestamp for r in results]
    assert abs(ts[3] - 5 / 24.0) < 1e-6  # 2/24 -> 5/24, slots 3 and 4 skipped
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    assert abs(gaps[2] - 3 / 24.0) < 1e-6      # the hole is 3 intervals wide
    assert all(abs(g - 1 / 24.0) < 1e-6 for g in gaps[:2] + gaps[3:])


def test_boundary_window_shrinks_and_is_reported():
    rng = np.random.default_rng(5)
    sigma = 0.02
    clean = _pan_sequence(n=9, shift=1.0)
    noisy = _noisy(clean, sigma, rng)
    frames = [FrameRef(i, i / 24.0, noisy[i], True) for i in range(len(noisy))]
    noise = NoiseModel(sigma_rgb=(sigma,) * 3, origin="test", detail="test",
                       model="additive_gaussian_rgb")
    results, diags, _ = denoise_timeline(frames, DenoiseParams(temporal_radius=3), noise)
    # First/last frames cannot use a full window and say so.
    assert any("boundary" in d.fallback for d in diags[:1]) or \
           len(diags[0].candidates) < 6
    assert len(diags[0].candidates) == 6  # slots are listed, even if unusable
    dropped = [c for c in diags[0].candidates if c["weight_mean"] == 0]
    assert len(dropped) >= 3             # the negative offsets do not exist


def test_scene_cut_blocks_temporal_window():
    rng = np.random.default_rng(6)
    sigma = 0.02
    clean = _pan_sequence(n=10, shift=1.0)
    noisy = _noisy(clean, sigma, rng)
    cut_clean = np.clip(1.0 - clean[-1], 0, 1)  # inverted shot: a real hard cut
    clean = clean + [cut_clean] * 4
    noisy = _noisy(clean, sigma, rng)
    frames = [FrameRef(i, i / 24.0, noisy[i], True) for i in range(len(noisy))]
    cuts = detect_scene_cuts(frames, threshold=0.3)
    assert 10 in cuts
    noise = NoiseModel(sigma_rgb=(sigma,) * 3, origin="test", detail="test",
                       model="additive_gaussian_rgb")
    _, diags, summar = denoise_timeline(frames, DenoiseParams(temporal_radius=2),
                                        noise, scene_cuts=cuts)
    assert summar["scene_cut_skips"] > 0
    # the first post-cut frame cannot fuse with pre-cut frames
    d = diags[10]
    for c in d.candidates:
        if c["offset"] < 0:
            assert c["weight_mean"] == 0.0


def test_timestamps_preserved_through_pipeline():
    rng = np.random.default_rng(7)
    sigma = 0.02
    clean = _pan_sequence(n=8, shift=1.0)
    noisy = _noisy(clean, sigma, rng)
    frames = [FrameRef(i, 0.01 + i / 30.0, noisy[i], True) for i in range(len(noisy))]
    noise = NoiseModel(sigma_rgb=(sigma,) * 3, origin="test", detail="test",
                       model="additive_gaussian_rgb")
    results, _, _ = denoise_timeline(frames, DenoiseParams(temporal_radius=1), noise)
    assert [r.timestamp for r in results] == pytest.approx(
        [f.t for f in frames], abs=1e-9)


def test_json_safe_diagnostics():
    """Dropped candidates must never emit NaN: JSON.parse rejects it."""
    rng = np.random.default_rng(8)
    sigma = 0.02
    clean = _pan_sequence(n=4, shift=1.0)
    noisy = _noisy(clean, sigma, rng)
    frames = [FrameRef(0, 0.0, noisy[0], True), FrameRef(1, 0.1, noisy[1], True)]
    m = estimate_motion(noisy[0], noisy[1], 0, 1, sigma, 0.12, 0.25)
    res = fuse_frame(frames, 0, {1: m}, DenoiseParams(temporal_radius=2), sigma)
    json.dumps([c.__dict__ for c in res.candidates], allow_nan=False)


# ----------------------------------------------------------------- media IO ----
def test_synthetic_clip_roundtrip_and_gap(tmp_path):
    man = make_synthetic_clip(
        str(tmp_path), name="roundtrip", width=160, height=96, frames_n=8,
        fps=24.0, sigma_rgb=(0.02, 0.025, 0.03), cut_at=5,
        dropped_frames=(2, 3), write_mp4=False)
    assert man["noise"]["origin"] == "manifest"
    assert man["dropped_frames"] == [2, 3]
    frames, meta = read_frames(os.path.join(str(tmp_path), man["source"]))
    assert sum(1 for f in frames if not f.usable) == 2
    assert meta["missing_slots"] == 2
    # Cadence is not re-quantised: the hole survives at the source's own pts.
    usable_ts = [f.t for f in frames if f.usable]
    hole = [b - a for a, b in zip(usable_ts, usable_ts[1:]) if b - a > 1.5 / 24.0]
    assert hole and abs(hole[0] - 3 / 24.0) < 1e-3


def test_cfr_output_keeps_frame_count_and_rate(tmp_path):
    """Source pts are millisecond-rounded; export must not re-quantise cadence."""
    from denoise.media import probe_video, write_video

    fps = 24.0
    ts = [round(i / fps, 3) for i in range(24)]   # 0.042/0.041 alternation
    rng = np.random.default_rng(3)
    imgs = [rng.random((96, 160, 3), dtype=np.float32) for _ in ts]
    out = str(tmp_path / "cfr.mp4")
    write_video(out, imgs, ts, fps, crf=12)
    info = probe_video(out)
    assert len(info["pts"]) == len(ts)
    assert abs(info["fps"] - fps) < 0.05
    assert abs(info["pts"][-1] - ts[-1]) < 2.0 / fps


def test_vfr_output_keeps_gap(tmp_path):
    """A missing-frame hole must survive export instead of being closed."""
    from denoise.media import probe_video, write_video

    fps = 24.0
    ts = [i / fps for i in range(10)]
    ts = [t for i, t in enumerate(ts) if i not in (4, 5, 6)]  # 3-frame hole
    rng = np.random.default_rng(4)
    imgs = [rng.random((96, 160, 3), dtype=np.float32) for _ in ts]
    out = str(tmp_path / "vfr.mp4")
    write_video(out, imgs, ts, fps, crf=12)
    info = probe_video(out)
    assert len(info["pts"]) == len(ts)
    gaps = [b - a for a, b in zip(info["pts"], info["pts"][1:])]
    assert max(gaps) > 3.0 / fps - 0.01
