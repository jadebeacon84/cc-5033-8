"""Video/media IO and synthetic underwater clip generation.

Timestamps are first-class: we decode the real presentation timestamps, build a
timeline that keeps gaps as explicit unusable slots, and write the output with
exactly those timestamps (constant-rate rawvideo when the source is CFR,
per-frame durations through the concat demuxer when it is VFR).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .types import FrameRef


# ---------------------------------------------------------------- ffmpeg lookup
def ffmpeg_exe() -> str:
    env = os.environ.get("FFMPEG_BIN")
    if env and Path(env).exists():
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`, "
            "or set FFMPEG_BIN."
        ) from exc


def ffprobe_exe() -> str | None:
    env = os.environ.get("FFPROBE_BIN")
    if env and Path(env).exists():
        return env
    for cand in (
        shutil.which("ffprobe"),
        str(Path(ffmpeg_exe()).with_name("ffprobe")),
        str(Path(__file__).resolve().parent.parent / ".tools" / "ffprobe"),
    ):
        if cand and Path(cand).exists():
            return cand
    return None


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, **kw)


def probe_video(path: str) -> dict:
    """Return width, height, fps, frame count and per-frame pts (if probeable)."""
    ff = ffprobe_exe()
    info = {"path": str(path), "width": None, "height": None, "fps": None,
            "pts": None, "codec": None}
    if ff:
        try:
            out = _run([ff, "-v", "error", "-select_streams", "v:0",
                        "-show_entries",
                        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name:"
                        "frame=pts_time",
                        "-show_frames", "-show_streams", "-of", "json", str(path)]).stdout
            data = json.loads(out)
            st = data.get("streams", [{}])[0]
            info["width"] = int(st.get("width") or 0) or None
            info["height"] = int(st.get("height") or 0) or None
            rate = st.get("avg_frame_rate") or st.get("r_frame_rate") or "0/1"
            if "/" in rate:
                num, den = rate.split("/")
                info["fps"] = float(num) / float(den) if float(den) else None
            info["codec"] = st.get("codec_name")
            pts = [float(f["pts_time"]) for f in data.get("frames", []) if "pts_time" in f]
            info["pts"] = pts if pts else None
        except Exception:
            pass
    if not info["width"]:
        cap = cv2.VideoCapture(str(path))
        info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        if not info["fps"]:
            info["fps"] = cap.get(cv2.CAP_PROP_FPS) or None
        cap.release()
    return info


def read_frames(path: str, max_frames: int | None = None) -> tuple[list[FrameRef], dict]:
    """Decode frames as float RGB plus a timeline with explicit missing slots.

    A gap in presentation timestamps larger than 1.6 nominal intervals inserts
    unusable placeholder slots at the expected times.  Those slots are kept in
    the timeline so output timing matches the source.
    """
    meta = probe_video(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    raw = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        raw.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        if max_frames is not None and len(raw) >= max_frames:
            break
    cap.release()
    if not raw:
        raise RuntimeError(f"no frames decoded from {path}")

    h, w = raw[0].shape[:2]
    fps = meta.get("fps") or 25.0
    pts = meta.get("pts")
    if pts and len(pts) >= len(raw):
        pts = pts[: len(raw)]
    else:
        pts = [i / fps for i in range(len(raw))]

    # Normalise so the first frame is at 0 but keep relative spacing exactly.
    t0 = pts[0]
    pts = [p - t0 for p in pts]
    nominal = float(np.median(np.diff(pts))) if len(pts) > 1 else 1.0 / fps
    nominal = nominal if nominal > 0 else 1.0 / fps

    frames: list[FrameRef] = []
    timeline_pts: list[float] = []
    for i, (p, img) in enumerate(zip(pts, raw)):
        if i > 0 and p - pts[i - 1] > 1.6 * nominal:
            # One or more missing slots; fill the expected cadence.
            gap = p - pts[i - 1]
            slots = int(round(gap / nominal)) - 1
            for k in range(slots):
                t = pts[i - 1] + (k + 1) * nominal
                frames.append(FrameRef(index=len(frames), t=t, image=np.zeros_like(img), usable=False))
                timeline_pts.append(t)
        frames.append(FrameRef(index=len(frames), t=p, image=img, usable=True))
        timeline_pts.append(p)

    meta.update({
        "width": w, "height": h, "fps": fps, "nominal_dt": nominal,
        "decoded": len(raw), "timeline_frames": len(frames),
        "missing_slots": sum(1 for f in frames if not f.usable),
    })
    return frames, meta


# ------------------------------------------------------------------ video write
def write_video(
    path: str,
    images: list[np.ndarray],
    timestamps: list[float],
    fps: float,
    crf: int = 14,
    codec: str = "libx264",
) -> str:
    """Encode float RGB images at exact timestamps.

    CFR sources use a rawvideo pipe at the nominal rate.  If the timestamps are
    not evenly spaced we use a PNG sequence + concat demuxer with per-frame
    durations, so playback cadence matches the source rather than being forced
    onto a fixed grid.
    """
    if not images:
        raise ValueError("no images to write")
    h, w = images[0].shape[:2]
    # yuv420p needs even dimensions; crop one pixel rather than fail or pad
    # with black (padding would change framing relative to the source).
    ew, eh = (w // 2) * 2, (h // 2) * 2
    if (ew, eh) != (w, h):
        images = [img[:eh, :ew] for img in images]
        h, w = eh, ew
    ff = ffmpeg_exe()
    ts = np.asarray(timestamps, np.float64)
    # Containers often store pts at millisecond precision (`42 ms` at 24 fps),
    # so identical cadence arrives as 0.042/0.041 alternation.  Judge uniformity
    # by the max relative deviation from the median interval instead of atol on
    # consecutive diffs, otherwise every CFR clip falls into the VFR path and
    # playback is re-quantised (24 -> 23.81 fps, N+1 frames).
    dts = np.diff(ts)
    uniform = len(ts) < 2 or (
        abs(float(np.median(dts))) > 1e-9
        and float(np.max(np.abs(dts - np.median(dts)))) <= 0.02 * float(np.median(dts))
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if uniform:
        rate = fps if fps and fps > 1 else (1.0 / float(np.median(dts)) if len(ts) > 1 else 25.0)
        cmd = [
            ff, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", f"{rate:.8f}", "-i", "-",
            "-c:v", codec, "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)
        assert proc.stdin is not None
        for img in images:
            proc.stdin.write((np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8).tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode() if proc.stderr else ""
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg encode failed: {err[-2000:]}")
        return path

    # VFR path: explicit per-frame durations.  Last frame needs a duration too,
    # so the concat list repeats its filename; `-frames:v N` then drops the
    # duplicate that repeat would otherwise add.  `option framerate` must follow
    # each `file` line; without it the concat demuxer quantises timestamps to its
    # guessed 25 fps grid and silently moves the cadence.
    dts = np.diff(ts)
    last_dur = float(np.median(dts)) if len(dts) else 1.0 / max(fps, 1)
    nominal = float(np.median(dts)) if len(dts) else 1.0 / max(fps, 1)
    base_rate = fps if fps and fps > 1 else 1.0 / nominal
    tmp = tempfile.mkdtemp(prefix="denoise_vfr_")
    try:
        concat_lines = ["ffconcat version 1.0"]
        for i, img in enumerate(images):
            p = os.path.join(tmp, f"f{i:06d}.png")
            bgr = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)[..., ::-1]
            cv2.imwrite(p, bgr)
            concat_lines.append(f"file '{p}'")
            concat_lines.append(f"option framerate {base_rate:.8f}")
            dur = float(dts[i]) if i < len(dts) else last_dur
            concat_lines.append(f"duration {dur:.8f}")
        concat_lines.append(f"file '{os.path.join(tmp, f'f{len(images)-1:06d}.png')}'")
        concat_lines.append(f"option framerate {base_rate:.8f}")
        concat_path = os.path.join(tmp, "concat.txt")
        with open(concat_path, "w") as fh:
            fh.write("\n".join(concat_lines) + "\n")
        cmd = [ff, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", concat_path,
               "-fps_mode", "vfr", "-frames:v", str(len(images)),
               "-c:v", codec, "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        res = _run(cmd)
        if res.returncode != 0:  # pragma: no cover
            raise RuntimeError(res.stderr.decode()[-2000:])
        return path
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------- synthetic underwater --
def _water_background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Depth gradient + slow haze bands, low texture so noise dominates."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    vy = yy / max(1, h - 1)
    vx = xx / max(1, w - 1)
    base = np.zeros((h, w, 3), np.float32)
    # Blue-green water column, darker with depth, slight vertical light shaft.
    base[..., 0] = 0.05 + 0.06 * (1.0 - vy) + 0.20 * vx * 0
    base[..., 1] = 0.16 + 0.18 * (1.0 - vy)
    base[..., 2] = 0.22 + 0.30 * (1.0 - vy)
    haze = 0.5 + 0.5 * np.sin(2 * np.pi * (vy * 2.4 + 0.13))
    base *= (0.86 + 0.14 * haze)[..., None]
    # Gentle large-scale variation only: no fine texture.
    phase = rng.uniform(0, 2 * np.pi, size=2)
    base += (0.025 * np.sin((vx * 1.7 + vy * 1.1) * 2 * np.pi + phase[0]))[..., None]
    base += (0.020 * np.cos((vx * 2.2 - vy * 1.6) * 2 * np.pi + phase[1]))[..., None]
    return np.clip(base, 0, 1)


def _draw_thin_lines(img: np.ndarray, rng: np.random.Generator, n: int = 9, offset=(0.0, 0.0)) -> None:
    """Fine 1px-ish lines: the detail-retention reference."""
    h, w = img.shape[:2]
    offx, offy = offset
    for i in range(n):
        x0 = int(rng.integers(8, w - 8) + offx)
        y0 = int(rng.integers(8, h - 8) + offy)
        length = int(rng.integers(20, min(h, w) // 3))
        ang = rng.uniform(-0.7, 0.7)
        col = rng.uniform(0.55, 0.95, size=3).astype(np.float32)
        p0 = (x0, y0)
        p1 = (int(x0 + np.cos(ang) * length), int(y0 + np.sin(ang) * length))
        cv2.line(img, p0, p1, tuple(float(c) for c in col), 1, cv2.LINE_AA)


def _paste_texture(img: np.ndarray, region: tuple[int, int, int, int], rng: np.random.Generator,
                    kind: str = "coral") -> tuple[int, int, int, int]:
    """Paste a textured object.  Returns its bbox for metrics."""
    x, y, ww, hh = region
    h, w = img.shape[:2]
    patch = np.zeros((hh, ww, 3), np.float32)
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    yy /= max(1, hh - 1)
    xx /= max(1, ww - 1)
    if kind == "coral":
        phase = rng.uniform(0, 2 * np.pi, 3)
        for f in (5.0, 11.0, 23.0):
            patch += (0.08 / (f / 5.0)) * np.sin(2 * np.pi * f * xx + phase[0])[..., None]
            patch += (0.07 / (f / 5.0)) * np.cos(2 * np.pi * f * yy * 1.2 + phase[1])[..., None]
        rim = 1.0 - np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) * 5.0)
        patch *= (0.3 + 0.7 * rim)[..., None]
        patch[..., 0] += 0.15 * rim
        patch[..., 1] += 0.11 * rim
    else:  # "rock": low-frequency gradient with one sharp fine-texture band
        patch[..., 0] += 0.18 * (0.4 + 0.6 * yy)
        patch[..., 1] += 0.15 * (0.4 + 0.6 * yy)
        patch[..., 2] += 0.12 * (0.4 + 0.6 * yy)
        band = (np.abs(yy - 0.55) < 0.12).astype(np.float32)
        patch += band[..., None] * 0.2 * np.sin(40 * np.pi * xx)[..., None]
    full = np.zeros_like(img)
    full[y:y + hh, x:x + ww] = patch
    mask = np.zeros(img.shape[:2], np.float32)
    mask[y:y + hh, x:x + ww] = 1.0
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    img *= 1.0 - mask[..., None]
    img += _water_background(h, w, rng) * 0.35 * mask[..., None]
    img += full
    return (x, y, ww, hh)


def make_synthetic_clip(
    out_dir: str,
    name: str = "underwater_demo",
    width: int = 960,
    height: int = 540,
    frames_n: int = 96,
    fps: float = 24.0,
    sigma_rgb: tuple[float, float, float] = (0.040, 0.045, 0.050),
    seed: int = 7,
    cut_at: int | None = None,
    dropped_frames: tuple[int, ...] = (),
    moving_object: bool = True,
    pan_speed: float = 0.55,   # pixels/frame, sub-pixel
    write_mp4: bool = True,
) -> dict:
    """Generate clean truth + noisy source and a manifest.

    The noise is additive Gaussian in RGB radiance, with per-channel sigma,
    exactly as the manifest records.  Compression of the deliverable preview is
    high-quality (CRF 12) and the lossless FFV1 source is the processing input.
    """
    rng = np.random.default_rng(seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    H, W = height, width
    sigma = np.asarray(sigma_rgb, np.float32)
    clean_frames: list[np.ndarray] = []
    noisy_frames: list[np.ndarray] = []
    line_bboxes = []

    # Base scene with a few "rocks" so panning has structure without turning
    # the frame into a texture test chart.
    base = _water_background(H, W + 240, rng)
    rock = _paste_texture(base, (int(W * 0.52), int(H * 0.58), int(W * 0.20), int(H * 0.22)), rng, "coral")
    rock2 = _paste_texture(base, (int(W * 0.10), int(H * 0.30), int(W * 0.13), int(H * 0.16)), rng, "rock")
    for _ in range(3):
        _draw_thin_lines(base, rng, n=4)

    obj_w, obj_h = int(W * 0.10), int(H * 0.12)
    obj_patch = np.zeros((obj_h, obj_w, 3), np.float32)
    yy, xx = np.mgrid[0:obj_h, 0:obj_w].astype(np.float32)
    yy /= max(1, obj_h - 1); xx /= max(1, obj_w - 1)
    body = (xx - 0.5) ** 2 / 0.16 + (yy - 0.5) ** 2 / 0.30
    obj_mask = np.clip(1.0 - body * 2.2, 0.0, 1.0)
    obj_patch += obj_mask[..., None] * np.array([0.55, 0.42, 0.18], np.float32)
    obj_patch += (np.abs(xx - 0.5) < 0.06)[..., None] * 0.25  # fine stripe on the object
    obj_t0 = 7
    obj_v = 2.2  # px/frame, slow enough to need motion compensation

    cuts = set() if cut_at is None else {int(cut_at)}
    jitter = np.zeros((2, frames_n), np.float32)
    jitter[0] = np.cumsum(rng.normal(0, 0.06, size=frames_n))  # tiny frame jitter
    jitter[1] = np.cumsum(rng.normal(0, 0.04, size=frames_n))

    gt_face = None
    cut_base = None
    if cut_at is not None:
        crng = np.random.default_rng(seed + 999)
        cut_base = _water_background(H, W + 240, crng)
        # A genuinely different shot: warmer palette, different structure, and
        # plenty of fine lines so an unregistered average would visibly ghost.
        cut_base[..., 0] *= 2.3
        cut_base[..., 1] *= 1.25
        cut_base[..., 2] *= 0.55
        cut_base = 0.55 + 0.45 * cut_base
        _paste_texture(cut_base, (int(W * 0.62), int(H * 0.18), int(W * 0.20), int(H * 0.22)),
                       np.random.default_rng(seed + 1001), "coral")
        _paste_texture(cut_base, (int(W * 0.20), int(H * 0.62), int(W * 0.24), int(H * 0.20)),
                       np.random.default_rng(seed + 1003), "rock")
        _draw_thin_lines(cut_base, np.random.default_rng(seed + 1002), n=12)

    for i in range(frames_n):
        shift_x = i * pan_speed + jitter[0, i]
        shift_y = 0.25 * np.sin(i * 0.09) + jitter[1, i]
        # Overscan equals W+240 so we have material for a full pan of ~90 px.
        sx = int(np.clip(shift_x, 0, 240))
        sy = int(np.round(np.clip(shift_y, 0, 0)))
        frame = base[sy:sy + H, sx:sx + W].copy()
        if cut_at is not None and i >= cut_at:
            # Hard cut to a different framing/pan: the temporal window must not
            # bridge it, and only the first post-cut frame is a declared cut.
            csx = int(np.clip((i - cut_at) * pan_speed, 0, 240))
            frame = cut_base[0:H, csx:csx + W].copy()
            if i == cut_at:
                cuts.add(i)
        # Keep the fine-structure overlay fixed in scene coordinates (it pans
        # with the frame but does not move relative to the scene).
        if i == 0:
            line_bboxes.append(rock)
        if moving_object and obj_t0 <= i < obj_t0 + int(0.35 * frames_n):
            cxp = int(W * 0.30 + obj_v * (i - obj_t0))
            cyp = int(H * 0.70 - 0.9 * (i - obj_t0))
            x0 = max(0, cxp - obj_w // 2); y0 = max(0, cyp - obj_h // 2)
            x1 = min(W, x0 + obj_w); y1 = min(H, y0 + obj_h)
            if x1 > x0 and y1 > y0:
                m = obj_mask[: y1 - y0, : x1 - x0, None]
                frame[y0:y1, x0:x1] = frame[y0:y1, x0:x1] * (1 - m) + obj_patch[: y1 - y0, : x1 - x0] * m
        clean_frames.append(np.clip(frame, 0, 1).astype(np.float32))
        n = np.clip(frame + rng.normal(0, sigma, size=frame.shape), 0, 1)
        noisy_frames.append(n.astype(np.float32))

    # Lossless source: FFV1 mkv.  Preview mp4 for the UI.
    import tempfile
    src_mkv = os.path.join(out_dir, f"{name}_noisy.mkv")
    gt_mkv = os.path.join(out_dir, f"{name}_truth.mkv")
    _write_lossless(src_mkv, noisy_frames, fps)
    _write_lossless(gt_mkv, clean_frames, fps)
    ts = [i / fps for i in range(frames_n)]
    out_mp4 = os.path.join(out_dir, f"{name}_source.mp4")
    gt_mp4 = os.path.join(out_dir, f"{name}_truth.mp4")
    if write_mp4:
        write_video(out_mp4, noisy_frames, ts, fps, crf=12)
        write_video(gt_mp4, clean_frames, ts, fps, crf=12)

    # Encode absence: dropped frames are simply not in the media, so the
    # timeline reconstruction must infer them.  We record them for verification.
    if dropped_frames:
        dropped = set(int(x) for x in dropped_frames)
        keep_noisy = [f for i, f in enumerate(noisy_frames) if i not in dropped]
        keep_clean = [f for i, f in enumerate(clean_frames) if i not in dropped]
        ts_keep = [ts[i] for i in range(frames_n) if i not in dropped]
        # Lossless VFR sources that keep the original gap in presentation time;
        # the missing slots are inferred at read time, not invented content.
        _write_lossless_timed(src_mkv, keep_noisy, ts_keep, last_duration=1.0 / fps)
        _write_lossless_timed(gt_mkv, keep_clean, ts_keep, last_duration=1.0 / fps)
        if write_mp4:
            write_video(out_mp4, keep_noisy, ts_keep, fps, crf=12)
            write_video(gt_mp4, keep_clean, ts_keep, fps, crf=12)

    manifest = {
        "name": name,
        "width": W, "height": H, "frames": frames_n, "fps": fps,
        "timestamps": ts,
        "noise": {
            "model": "additive_gaussian_rgb",
            "sigma_rgb": [float(x) for x in sigma],
            "origin": "manifest",
            "detail": "generator injected i.i.d. Gaussian radiance noise per RGB channel",
        },
        "clean_truth": os.path.relpath(gt_mkv, out_dir),
        "clean_truth_mp4": os.path.relpath(gt_mp4, out_dir),
        "source": os.path.relpath(src_mkv, out_dir),
        "source_mp4": os.path.relpath(out_mp4, out_dir),
        "scene_cuts": sorted(cuts),
        "dropped_frames": sorted(int(x) for x in dropped_frames),
        "moving_object": bool(moving_object),
        "notes": [
            "synthetic: thin lines, low-texture water, slow moving object",
            "processing input is lossless FFV1 so the measured sigma is the injected sigma",
            "does not claim recovery of detail absent from the source",
        ],
    }
    with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def _write_lossless(path: str, images: list[np.ndarray], fps: float) -> None:
    h, w = images[0].shape[:2]
    ff = ffmpeg_exe()
    cmd = [ff, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", f"{fps:.8f}", "-i", "-",
           "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", "-f", "matroska", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for img in images:
        proc.stdin.write((np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8).tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode() if proc.stderr else ""
    if proc.wait() != 0:
        raise RuntimeError(f"lossless encode failed: {err[-1500:]}")


def _write_lossless_timed(
    path: str, images: list[np.ndarray], timestamps: list[float], last_duration: float
) -> None:
    """Lossless FFV1 Matroska with real per-frame timestamps (VFR)."""
    ff = ffmpeg_exe()
    h, w = images[0].shape[:2]
    ts = np.asarray(timestamps, np.float64)
    dts = np.diff(ts)
    durations = list(dts) + [float(np.median(dts)) if len(dts) else last_duration]
    tmp = tempfile.mkdtemp(prefix="lossless_vfr_")
    try:
        lines = ["ffconcat version 1.0"]
        for i, img in enumerate(images):
            p = os.path.join(tmp, f"f{i:06d}.png")
            cv2.imwrite(p, (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)[..., ::-1])
            lines.append(f"file '{p}'")
            lines.append(f"option framerate {1.0/float(np.median(dts)):.8f}" if len(dts) else "")
            lines.append(f"duration {float(durations[i]):.8f}")
        lines.append(f"file '{os.path.join(tmp, f'f{len(images)-1:06d}.png')}'")
        lines.append(f"option framerate {1.0/float(np.median(dts)):.8f}" if len(dts) else "")
        concat = os.path.join(tmp, "concat.txt")
        with open(concat, "w") as fh:
            fh.write("\n".join(l for l in lines if l) + "\n")
        _run([ff, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", concat,
              "-fps_mode", "vfr", "-frames:v", str(len(images)),
              "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0",
              "-f", "matroska", path])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
