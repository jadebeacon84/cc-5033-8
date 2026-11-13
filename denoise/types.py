"""Shared data model for the underwater temporal denoiser.

Everything here is plain dataclasses + numpy arrays so the core can be unit
tested without any IO or UI dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


@dataclass
class NoiseModel:
    """Additive noise assumption, always traceable to a measured source.

    The capture pipeline estimates a per-channel variance from flat patches.
    When the clip is synthetic the generator stores the *injected* sigma in the
    media manifest; both routes end up here so the UI can show where the
    number came from.
    """

    sigma_rgb: tuple[float, float, float]
    origin: str            # "manifest" | "flat_patch_estimate" | "calibration"
    detail: str = ""
    # Noise is assumed = shot-ish/gaussian radiance noise added in the digital
    # domain after tone mapping.  We deliberately do not model compression
    # ringing; the tests keep the assumption honest.
    model: str = "additive_gaussian_rgb"

    @property
    def mean_sigma(self) -> float:
        return float(np.mean(self.sigma_rgb))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sigma_rgb"] = [float(x) for x in self.sigma_rgb]
        d["mean_sigma"] = self.mean_sigma
        return d


@dataclass
class FrameRef:
    """A source frame and its nominal presentation timestamp in seconds."""

    index: int            # index within the decoded source sequence
    t: float              # timestamp in seconds (presentation time)
    image: np.ndarray     # float32 RGB in [0, 1], shape (H, W, 3)
    usable: bool = True   # False for decoded-but-unusable (missing/corrupt)


@dataclass
class MotionEstimate:
    """Motion from the current frame to one neighbour.

    `flow` is a forward displacement field (dx, dy) at frame `index`:
    sample neighbour at (x + dx, y + dy) using warp-inverse sampling.
    `valid` is 0 where the registration is unreliable (occlusion, residual,
    out-of-bounds, texture starvation).
    """

    frame_index: int
    neighbor_index: int
    dx: np.ndarray
    dy: np.ndarray
    residual: np.ndarray        # per-pixel colour residual after warping
    valid: np.ndarray           # 0..1 reliability mask
    confidence: np.ndarray      # 0..1 local match confidence before residual gate
    kind: str = "dense"         # "dense" | "fallback_identity" | "dropped"
    detail: str = ""


@dataclass
class Candidate:
    """One neighbour contributing to the fused frame."""

    offset: int                 # signed distance in frames
    weight_mean: float          # mean applied weight, for the UI table
    overlap_ratio: float        # fraction of frame with weight > threshold
    residual_median: float | None  # None when the candidate was never measured
    kind: str
    note: str = ""


@dataclass
class DenoiseParams:
    temporal_radius: int = 2          # +/- frames in the temporal window
    spatial_strength: float = 0.35    # 0..1 edge-preserving spatial blend
    temporal_strength: float = 0.85   # 0..1 trust in neighbour evidence
    motion_model: str = "dense"       # "dense" | "translation" | "off"
    max_residual: float = 0.12        # residual gate (mean abs per channel)
    min_confidence: float = 0.25      # motion confidence gate
    sigma_scale: float = 1.0          # multiplier on measured sigma


@dataclass
class DenoiseResult:
    image: np.ndarray
    timestamp: float
    source_index: int
    noise_sigma: float
    temporal_weight: np.ndarray
    candidates: list[Candidate] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


@dataclass
class FrameDiagnostics:
    frame_index: int
    timestamp: float
    motion: list[dict]
    candidates: list[dict]
    fallback: str
    notes: list[str]
    noise_sigma: float
    temporal_evidence: float      # 0..1 amount of neighbour evidence used
    valid_ratio: float
