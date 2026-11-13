"""Regenerate the three verification clips under media/clips/.

Run:  python3 tools/make_clips.py

Each clip ships a clean truth, a lossless noisy processing input (FFV1) and a
manifest that records the noise provenance.  The manifest is the source of the
noise-strength assumption used by the denoiser and shown in the UI.
"""
from __future__ import annotations

import json
from pathlib import Path

from denoise.media import make_synthetic_clip

OUT = Path(__file__).resolve().parent.parent / "media" / "clips"

CLIPS = [
    {
        "name": "reef_sway_24fps",
        "width": 960, "height": 540, "frames_n": 96, "fps": 24.0,
        "sigma_rgb": (0.035, 0.040, 0.045), "seed": 11,
        "cut_at": None, "dropped_frames": (), "moving_object": True,
        "pan_speed": 0.55,
        "notes": "slow swaying reef: fine structure, slow object, sub-pixel pan",
    },
    {
        "name": "grading_timeline",
        "width": 960, "height": 540, "frames_n": 72, "fps": 24.0,
        "sigma_rgb": (0.050, 0.055, 0.060), "seed": 21,
        "cut_at": 40, "dropped_frames": (17, 18, 19), "moving_object": True,
        "pan_speed": 0.9,
        "notes": "timeline stress: hard cut at frame 40, three missing frames",
    },
    {
        "name": "low_texture_silt",
        "width": 960, "height": 540, "frames_n": 64, "fps": 30.0,
        "sigma_rgb": (0.055, 0.060, 0.065), "seed": 31,
        "cut_at": None, "dropped_frames": (), "moving_object": False,
        "pan_speed": 0.35,
        "notes": "low-texture silty water: no moving object, noise-dominated",
    },
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for spec in CLIPS:
        print(f"[clip] {spec['name']} …")
        man = make_synthetic_clip(str(OUT), **spec)
        (OUT / f"{spec['name']}.json").write_text(json.dumps(man, indent=2))
        print(f"       source={man['source']} truth={man['clean_truth']} "
              f"cut={man['scene_cuts']} dropped={man['dropped_frames']}")
    print("done")


if __name__ == "__main__":
    main()
