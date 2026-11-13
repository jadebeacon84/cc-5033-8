"""HTTP API + static UI for the underwater temporal denoiser.

Stdlib-only server (no web framework).  A single background worker thread runs
one job at a time so the UI can poll progress while a clip renders.

Endpoints:
  GET  /                             UI
  GET  /static/<file>                UI assets
  GET  /api/library                  clips, manifests and parameter presets
  POST /api/jobs                     start a job -> {job_id}
  GET  /api/jobs                     recent jobs
  GET  /api/jobs/<id>                status, diagnostics, summary, metrics
  POST /api/jobs/<id>/cancel
  GET  /api/jobs/<id>/frames/<k>?kind=source|output|truth|motion|valid|weights
  GET  /api/jobs/<id>/video/<kind>   denoised.mp4 | lossless | input | truth
  GET  /api/jobs/<id>/metrics        per-frame objective comparison
"""
from __future__ import annotations

import json
import math
import queue
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "media" / "clips"
JOBS = ROOT / "outputs" / "jobs"
WEB = ROOT / "web"
WEB.mkdir(parents=True, exist_ok=True)
JOBS.mkdir(parents=True, exist_ok=True)

from denoise import (DenoiseParams, denoise_timeline, noise_model_from_frames,
                     read_frames, write_video)
from denoise.types import FrameRef
from denoise.metrics import evaluate_run
from denoise.pipeline import detect_scene_cuts, motion_for_pair

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_work_q: queue.Queue = queue.Queue()

PREVIEW_WIDTH = 720
_NO_REG = object()          # sentinel: no neighbour survived registration
_frames_cache = None        # (job_id, (frames8, meta)) for the active job
_frames_cache_lock = threading.Lock()


def _cache_frames8(job_id: str, frames: list[FrameRef]) -> None:
    """Convert once to uint8 and keep only the newest job's copy."""
    global _frames_cache
    frames8 = [FrameRef(
        index=f.index, t=f.t,
        image=np.round(f.image * 255.0).astype(np.uint8) if f.usable else f.image,
        usable=f.usable) for f in frames]
    with _frames_cache_lock:
        _frames_cache = (job_id, frames8)


# ------------------------------------------------------------------- preview --
def _store_rgb(img: np.ndarray, width: int = PREVIEW_WIDTH) -> bytes:
    """Downscale + JPEG encode.  Keeping encoded bytes bounds memory; these
    clips have hundreds of frames and float32 previews would not fit."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, max(1, int(round(h * width / w)))),
                         interpolation=cv2.INTER_AREA)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _store_heat(field: np.ndarray, vmax: float) -> bytes:
    """8-bit heat map for weights (one channel of float weight, imshow-style)."""
    f = np.clip(field.astype(np.float32) / max(vmax, 1e-6), 0, 1)
    f = (f * 255).astype(np.uint8)
    h, w, _ = 0, 0, 0
    h, w = f.shape[:2]
    if w > PREVIEW_WIDTH // 2:
        f = cv2.resize(f, (PREVIEW_WIDTH // 2, max(1, int(round(h * (PREVIEW_WIDTH // 2) / w)))))
    ok, buf = cv2.imencode(".png", f)
    return buf.tobytes()


def _encode_field_png(field: np.ndarray, cmap_scale: float | None = None) -> bytes:
    """Colourise a scalar field for display (weights/validity)."""
    f = np.clip(field.astype(np.float32) * (cmap_scale or 1.0), 0, 1)
    u8 = (f * 255).astype(np.uint8)
    col = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    ok, buf = cv2.imencode(".jpg", col, [cv2.IMWRITE_JPEG_QUALITY, 86])
    return buf.tobytes()


def _motion_png(dx: np.ndarray, dy: np.ndarray, valid: np.ndarray, px_full_scale: float = 4.0) -> bytes:
    mag = np.sqrt(dx * dx + dy * dy)
    # Fixed scale, not per-frame percentile: these clips move well under a
    # pixel per frame, so auto-normalising would paint sub-pixel noise as a
    # full-saturation rainbow and overstate the motion.
    norm = np.clip(mag / px_full_scale, 0, 1)
    ang = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * 180.0
    hsv = np.zeros((*mag.shape, 3), np.float32)
    hsv[..., 0] = ang
    hsv[..., 1] = 0.95
    hsv[..., 2] = norm * (0.25 + 0.75 * np.clip(valid, 0, 1))
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.resize(rgb, (PREVIEW_WIDTH, max(1, int(round(rgb.shape[0] * PREVIEW_WIDTH / rgb.shape[1])))))
    ok, buf = cv2.imencode(".jpg", (rgb * 255).astype(np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()


def _valid_png(valid: np.ndarray) -> bytes:
    return _encode_field_png(valid, 1.0)


def _residual_png(resid: np.ndarray, sigma: float) -> bytes:
    """Residual magnified relative to the measured noise floor.

    Values around sigma are noise-only; >= 4 sigma is a real mis-registration.
    Colour scale is fixed by the model, not auto-normalised, so two frames can
    be compared honestly.
    """
    f = np.clip(resid / (max(1e-4, sigma) * 4.0), 0, 1)
    u8 = (f * 255).astype(np.uint8)
    col = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    col = cv2.resize(col, (PREVIEW_WIDTH, max(1, int(round(col.shape[0] * PREVIEW_WIDTH / col.shape[1])))),
                     interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".jpg", col, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()


def _clip_library() -> list[dict]:
    out = []
    if not CLIPS.exists():
        return out
    for man in sorted(CLIPS.glob("*.json")):
        try:
            data = json.loads(man.read_text())
        except Exception:
            continue
        src = man.parent / data["source"] if data.get("source") else None
        truth = man.parent / data["clean_truth"] if data.get("clean_truth") else None
        out.append({
            "id": man.stem,
            "name": data.get("name", man.stem),
            "width": data.get("width"), "height": data.get("height"),
            "frames": data.get("frames"), "fps": data.get("fps"),
            "source": str(src.relative_to(ROOT)) if src and src.exists() else None,
            "truth": str(truth.relative_to(ROOT)) if truth and truth.exists() else None,
            "source_mp4": str((man.parent / data["source_mp4"]).relative_to(ROOT))
            if data.get("source_mp4") and (man.parent / data["source_mp4"]).exists() else None,
            "truth_mp4": str((man.parent / data["truth_mp4"]).relative_to(ROOT))
            if data.get("truth_mp4") and (man.parent / data["truth_mp4"]).exists() else None,
            "noise": data.get("noise"),
            "dropped_frames": data.get("dropped_frames", []),
            "scene_cuts": data.get("scene_cuts", []),
            "notes": data.get("notes", []),
        })
    return out


# ----------------------------------------------------------------------- job --
def _process_job(job: dict) -> None:
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        job["status"] = "loading"
        frames, meta = read_frames(job["source"])
        usable = [f for f in frames if f.usable]
        if not usable:
            raise RuntimeError("clip contains no usable frames")

        manifest_sigma, origin_detail = None, ""
        if job.get("clip_id"):
            mpath = CLIPS / f"{job['clip_id']}.json"
            if mpath.exists():
                m = json.loads(mpath.read_text())
                s = m.get("noise", {}).get("sigma_rgb")
                if s:
                    manifest_sigma = tuple(float(x) for x in s)
                    origin_detail = m.get("noise", {}).get("detail", "")
        noise = noise_model_from_frames(
            [f.image for f in usable[: min(6, len(usable))]], manifest_sigma, origin_detail)
        params = DenoiseParams(**job["params"])

        cuts = set(job.get("scene_cuts") or [])
        if job.get("auto_cuts", True) and not cuts:
            cuts = detect_scene_cuts(frames)
        job["meta"] = meta
        job["detected_cuts"] = sorted(cuts)
        job["noise"] = noise.to_dict()

        # Cache uint8 frames for the motion viewer now that we already have
        # them decoded; otherwise the first motion-tab request re-decodes the
        # whole clip (~3 s) and the panel sits blank.
        _cache_frames8(job["id"], frames)

        truth = None
        if job.get("truth"):
            tf, _ = read_frames(job["truth"])
            truth = {f.index: f.image for f in tf if f.usable}

        job["status"] = "denoising"
        job["progress"] = {"done": 0, "total": len(usable)}

        preview = {"source": {}, "output": {}, "truth": {}, "weights": {}}
        job["preview"] = preview
        job["live_timeline"] = []
        t_start = time.time()

        def on_frame(step, total, res):
            if job.get("cancel"):
                return
            k = len(preview["source"])
            pos = step
            job["live_timeline"].append({"index": res.source_index, "t": res.timestamp})
            preview["source"][k] = _store_rgb(frames[pos].image)
            preview["output"][k] = _store_rgb(res.image)
            if truth and res.source_index in truth:
                preview["truth"][k] = _store_rgb(truth[res.source_index])
            preview["weights"][k] = _store_heat(res.temporal_weight, 2.5)
            job["progress"] = {"done": k + 1, "total": max(1, len(usable))}

        results, diags, run_summary = denoise_timeline(
            frames, params, noise, scene_cuts=cuts, progress=on_frame)
        if job.get("cancel"):
            job["status"] = "cancelled"
            return
        job["preview"] = preview

        job["status"] = "rendering"
        timestamps = [r.timestamp for r in results]
        imgs = [r.image for r in results]
        fps = meta.get("fps") or 25.0
        in_imgs = [frames[step].image for step in range(len(frames)) if frames[step].usable]
        render_tasks = {
            "denoised.mp4": lambda: write_video(
                str(out_dir / "denoised.mp4"), imgs, timestamps, fps, crf=12),
            "denoised_lossless.mkv": lambda: write_video(
                str(out_dir / "denoised_lossless.mkv"), imgs, timestamps, fps,
                crf=10, codec="ffv1"),
            "input_proxy.mp4": lambda: write_video(
                str(out_dir / "input_proxy.mp4"), in_imgs, timestamps, fps, crf=12),
        }
        if truth:
            render_tasks["truth_proxy.mp4"] = lambda: write_video(
                str(out_dir / "truth_proxy.mp4"),
                [truth[r.source_index] for r in results], timestamps, fps, crf=12)
        # Encode independently in parallel: each write is a separate ffmpeg
        # process, so this is wall-clock reduction, not extra CPU on one encoder.
        with ThreadPoolExecutor(max_workers=len(render_tasks)) as pool:
            futures = {name: pool.submit(fn) for name, fn in render_tasks.items()}
            for name, fut in futures.items():
                fut.result()

        metrics = None
        if truth and all(r.source_index in truth for r in results):
            metrics = evaluate_run(
                noisy=in_imgs,
                clean=[truth[r.source_index] for r in results],
                out=imgs, timestamps=timestamps,
                label=job.get("clip_id") or "custom")
            (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        summary = {
            "job_id": job["id"],
            "clip_id": job.get("clip_id"),
            "params": job["params"],
            "noise": job["noise"],
            "meta": meta,
            "detected_cuts": sorted(cuts),
            "timeline": [{"index": r.source_index, "t": r.timestamp} for r in results],
            "frames_out": len(results),
            "frames_in": len(frames),
            "missing_slots": run_summary.get("missing_slots", 0),
            "fallback_frames": run_summary.get("fallback_frames", 0),
            "mean_temporal_evidence": run_summary.get("mean_temporal_evidence", 0.0),
            "mean_valid_ratio": run_summary.get("mean_valid_ratio", 0.0),
            "elapsed_s": round(time.time() - t_start, 2),
            "metrics": metrics["summary"] if metrics else None,
            "warnings": _warnings(job, diags, metrics, run_summary),
        }
        job["summary"] = summary
        job["diags"] = [asdict(d) for d in diags]
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        job["status"] = "done"
        job["finished"] = time.time()
    except Exception as exc:
        job["status"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
        job["traceback"] = traceback.format_exc()[-4000:]
        try:
            (out_dir / "error.txt").write_text(job["traceback"])
        except Exception:
            pass


def _warnings(job, diags, metrics, run_summary) -> list[str]:
    w = []
    if job.get("meta", {}).get("missing_slots"):
        w.append(f"时间轴含 {job['meta']['missing_slots']} 个缺帧槽位：保留为空档，不做插值。")
    fb = [d for d in diags if d.fallback != "none"]
    if fb:
        kinds = sorted({d.fallback for d in fb})
        w.append(f"{len(fb)} 帧发生回退（{', '.join(kinds)}）；逐帧诊断可见原因，未做无对齐平均。")
    if metrics:
        s = metrics["summary"]
        if s["detail_retention_ratio"] < 0.8:
            w.append(f"细节对比保留率 {s['detail_retention_ratio']:.2f}，空间强度可能压平了细线。")
        if s["psnr_out"] <= s["psnr_noisy"]:
            w.append("本片段输出 PSNR 未优于输入：请检查噪声模型与参数。")
        if s["noise_reduction_ratio"] > 0.98:
            w.append("残留噪声几乎未下降：可提高时窗/时间强度。")
    return w


# -------------------------------------------------------------------- worker --
def _worker() -> None:
    while True:
        jid = _work_q.get()
        with _jobs_lock:
            job = _jobs.get(jid)
        if job is None:
            continue
        try:
            _process_job(job)
        except Exception:
            job["status"] = "error"
            job["error"] = "worker crashed"
            job["traceback"] = traceback.format_exc()[-4000:]


def start_worker() -> None:
    threading.Thread(target=_worker, name="denoise-worker", daemon=True).start()


# ------------------------------------------------------------------- handler --
def _public(job: dict) -> dict:
    out = {k: job.get(k) for k in ("id", "clip_id", "status", "progress", "error",
                                   "params", "noise", "meta", "detected_cuts",
                                   "summary", "traceback")}
    out["diags"] = job.get("diags")
    out["has_preview"] = "preview" in job
    out["timeline"] = job.get("live_timeline") or (job.get("summary") or {}).get("timeline")
    if job.get("preview"):
        out["preview_frames"] = sorted(job["preview"]["source"].keys())
        out["has_truth_preview"] = bool(job["preview"].get("truth"))
        out["preview_bytes"] = msize(job["preview"])
    if job.get("summary") and (Path(job["out_dir"]) / "denoised.mp4").exists():
        out["downloads"] = {
            "mp4": f"/api/jobs/{job['id']}/video/denoised.mp4",
            "lossless": f"/api/jobs/{job['id']}/video/lossless",
            "input": f"/api/jobs/{job['id']}/video/input",
            "truth": f"/api/jobs/{job['id']}/video/truth" if job.get("truth") else None,
            "metrics": f"/api/jobs/{job['id']}/metrics",
        }
    return out


def msize(preview: dict) -> int:
    return sum(len(v) for group in ("source", "output", "truth", "weights")
               for v in preview.get(group, {}).values())


def _finite(obj):
    """Replace non-finite floats with None.

    `json.dumps` happily writes bare NaN/Infinity, which is not valid JSON and
    makes the browser's JSON.parse throw — a dropped candidate with an
    unmeasured residual once froze the whole progress panel.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


class Handler(BaseHTTPRequestHandler):
    server_version = "UnderwaterDenoise/1.0"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body: bytes, ctype="application/octet-stream", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(_finite(obj), ensure_ascii=False).encode(),
                   "application/json")

    def do_GET(self):
        try:
            u = urlparse(self.path)
            p = u.path.rstrip("/") or "/"
            if p in ("/", "/index.html"):
                return self._send(200, (WEB / "index.html").read_bytes(),
                                  "text/html; charset=utf-8")
            if p.startswith("/static/"):
                f = WEB / p.split("/static/", 1)[1]
                if f.is_file():
                    ct = {".js": "text/javascript", ".css": "text/css",
                          ".svg": "image/svg+xml", ".woff2": "font/woff2",
                          ".otf": "font/otf"}.get(f.suffix, "application/octet-stream")
                    return self._send(200, f.read_bytes(), ct)
                return self._json({"error": "not found"}, 404)
            if p == "/api/library":
                return self._json({"clips": _clip_library(), "presets": _presets()})
            if p == "/api/jobs":
                with _jobs_lock:
                    js = sorted(_jobs.values(), key=lambda x: x.get("created", 0), reverse=True)
                return self._json({"jobs": [{"id": j["id"], "clip_id": j.get("clip_id"),
                                             "status": j["status"]} for j in js[:30]]})
            parts = [x for x in p.split("/") if x]
            if parts[:2] == ["api", "jobs"] and len(parts) >= 3:
                jid = parts[2]
                with _jobs_lock:
                    job = _jobs.get(jid)
                if job is None:
                    return self._json({"error": "unknown job"}, 404)
                if len(parts) == 3:
                    return self._json(_public(job))
                if parts[3] == "frames" and len(parts) >= 5:
                    return self._frame(job, int(parts[4]),
                                       parse_qs(u.query).get("kind", ["output"])[0])
                if parts[3] == "video" and len(parts) >= 5:
                    return self._video(job, parts[4])
                if parts[3] == "metrics":
                    return self._file(job, "metrics.json", "application/json")
            return self._json({"error": "not found"}, 404)
        except BrokenPipeError:
            pass
        except Exception as exc:
            try:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            except Exception:
                pass

    def _frame(self, job, k: int, kind: str):
        if "preview" not in job:
            return self._json({"error": "previews not ready"}, 404)
        pv = job["preview"]
        if kind in ("source", "output", "truth"):
            blob = pv.get(kind, {}).get(k)
            if blob is None:
                return self._json({"error": f"no {kind} preview for frame {k}"}, 404)
            return self._send(200, blob, "image/jpeg")
        if kind == "weights":
            blob = pv["weights"].get(k)
            if blob is None:
                return self._json({"error": "no weight preview"}, 404)
            f = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE)
            col = cv2.applyColorMap(f, cv2.COLORMAP_INFERNO)
            ok, buf = cv2.imencode(".jpg", col, [cv2.IMWRITE_JPEG_QUALITY, 88])
            return self._send(200, buf.tobytes(), "image/jpeg")
        if kind in ("motion", "valid", "residual"):
            if not (0 <= k < len(job.get("live_timeline") or
                                 (job.get("summary") or {}).get("timeline") or [])):
                return self._json({"error": f"frame {k} out of range"}, 404)
            return self._send(200, self._motion_field(job, k, kind), "image/jpeg")
        return self._json({"error": "unknown kind"}, 404)

    def _frames(self, job):
        """Decoded frames for on-demand motion previews.

        Kept as uint8: a 960x540 clip is ~150 MB that way vs ~600 MB float32,
        and the preview path only promotes the handful of frames it needs.
        Without this cache every slider move re-decoded the whole video.
        Only one clip is cached at a time — retaining one per job would grow
        unbounded as jobs accumulate.
        """
        global _frames_cache
        with _frames_cache_lock:
            cached = _frames_cache
            if cached is not None and cached[0] == job["id"]:
                return cached[1], job.get("meta") or {}
        frames, meta = read_frames(job["source"])
        _cache_frames8(job["id"], frames)
        with _frames_cache_lock:
            return _frames_cache[1], meta

    @staticmethod
    def _float_frames(frames8: list[FrameRef], positions: set[int]) -> list[FrameRef]:
        """Shallow frame list with float32 images only at `positions`."""
        out = list(frames8)
        for p in positions:
            f = frames8[p]
            if f.usable:
                out[p] = FrameRef(index=f.index, t=f.t,
                                  image=f.image.astype(np.float32) / 255.0,
                                  usable=True)
        return out

    def _motion_field(self, job, result_index: int, kind: str) -> bytes:
        """Recompute the motion field for one frame on demand.

        Storing dense flow for every frame would dwarf the source; one
        Farneback pass for the selected frame is cheap and keeps memory flat.
        The neighbour selection mirrors the job's params and cuts.
        """
        frames8, meta = self._frames(job)
        tl = job.get("live_timeline") or (job.get("summary", {}).get("timeline"))
        idx = tl[result_index]["index"]
        pos = next(p for p, f in enumerate(frames8) if f.index == idx)
        params = DenoiseParams(**job["params"])
        noise = job["noise"]
        sigma = float(noise["mean_sigma"])
        cuts = set(job.get("detected_cuts") or [])
        from denoise.pipeline import _cut_between
        # The motion/valid/residual views all describe the same chosen
        # neighbour; compute the registration once per frame and derive the
        # three renderings from it.
        cache = job.setdefault("_field_cache", {})
        if len(cache) > 16:
            cache.clear()
        best = cache.get(result_index)
        if best is None:
            best = _NO_REG
            for off in range(-params.temporal_radius, params.temporal_radius + 1):
                if off == 0:
                    continue
                npos = pos + off
                if not (0 <= npos < len(frames8)) or not frames8[npos].usable:
                    continue
                if _cut_between(frames8, pos, npos, cuts):
                    continue
                probe = self._float_frames(frames8, {pos, npos})
                m = motion_for_pair(probe, pos, npos, params, sigma)
                if best is _NO_REG or float(m.valid.mean()) > float(best.valid.mean()):
                    best = m
            cache[result_index] = best
        if best is _NO_REG:
            h, w = frames8[pos].image.shape[:2]
            dx = np.zeros((h, w), np.float32); dy = np.zeros((h, w), np.float32)
            valid = np.zeros((h, w), np.float32)
        else:
            dx, dy, valid = best.dx, best.dy, best.valid
        if kind == "valid":
            return _valid_png(valid)
        if kind == "residual":
            from denoise.motion import _warp
            if best is _NO_REG:
                h, w = frames8[pos].image.shape[:2]
                return _residual_png(np.zeros((h, w), np.float32), sigma)
            nb_pos = next(p for p, f in enumerate(frames8) if f.index == best.neighbor_index)
            fl = self._float_frames(frames8, {pos, nb_pos})
            warped = _warp(fl[nb_pos].image, best.dx, best.dy)
            resid = np.mean(np.abs(fl[pos].image - warped), axis=2)
            return _residual_png(resid, sigma)
        return _motion_png(dx, dy, valid)

    def _file(self, job, name, ctype):
        f = Path(job["out_dir"]) / name
        if not f.exists():
            return self._json({"error": f"{name} not ready"}, 404)
        return self._send_file(f, ctype)

    def _send_file(self, f: Path, ctype: str):
        """Serve a file with single-range support.

        Browsers seek in <video> by requesting a byte range; without it the
        player either refuses to scrub or re-downloads the whole file.
        """
        size = f.stat().st_size
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            try:
                spec = rng.split("=", 1)[1].split(",")[0].strip()
                start_s, _, end_s = spec.partition("-")
                start = int(start_s) if start_s else max(0, size - int(end_s))
                end = int(end_s) if end_s else size - 1
                start = max(0, min(start, size - 1))
                end = max(start, min(end, size - 1))
            except ValueError:
                start, end = 0, size - 1
            data = f.read_bytes()[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        else:
            self._send(200, f.read_bytes(), ctype,
                       {"Accept-Ranges": "bytes"})

    def _video(self, job, kind):
        names = {"denoised.mp4": "denoised.mp4", "out": "denoised.mp4",
                 "lossless": "denoised_lossless.mkv",
                 "input": "input_proxy.mp4", "truth": "truth_proxy.mp4"}
        name = names.get(kind)
        if not name:
            return self._json({"error": "unknown video kind"}, 404)
        ctype = "video/mp4" if name.endswith(".mp4") else "video/x-matroska"
        return self._send_file(Path(job["out_dir"]) / name, ctype)

    def do_POST(self):
        try:
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if u.path == "/api/jobs":
                return self._json(self._create_job(body), 201)
            parts = [x for x in u.path.split("/") if x]
            if len(parts) == 4 and parts[:2] == ["api", "jobs"]:
                with _jobs_lock:
                    job = _jobs.get(parts[2])
                if job is None:
                    return self._json({"error": "unknown job"}, 404)
                if parts[3] == "cancel":
                    job["cancel"] = True
                    return self._json({"ok": True})
            return self._json({"error": "not found"}, 404)
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def _create_job(self, body: dict):
        params = DenoiseParams()
        for k, v in (body.get("params") or {}).items():
            if hasattr(params, k):
                old = getattr(params, k)
                setattr(params, k, type(old)(v))
        clip_id = body.get("clip_id")
        if clip_id:
            man = CLIPS / f"{clip_id}.json"
            if not man.exists():
                raise RuntimeError("unknown clip")
            d = json.loads(man.read_text())
            src = str((man.parent / d["source"]).resolve())
            truth = str((man.parent / d["clean_truth"]).resolve()) if d.get("clean_truth") else None
            cuts = d.get("scene_cuts", [])
        elif body.get("source"):
            src, truth, cuts = body["source"], body.get("truth"), body.get("scene_cuts", [])
        else:
            raise RuntimeError("clip_id or source required")
        jid = uuid.uuid4().hex[:10]
        job = {
            "id": jid, "clip_id": clip_id,
            "source": src, "truth": truth,
            "params": asdict(params), "out_dir": str(JOBS / jid),
            "scene_cuts": cuts, "auto_cuts": bool(body.get("auto_cuts", True)),
            "status": "queued", "created": time.time(),
            "progress": {"done": 0, "total": 0},
        }
        with _jobs_lock:
            _jobs[jid] = job
        _work_q.put(jid)
        return {"job_id": jid, "status": "queued"}


def _presets() -> list[dict]:
    return [
        {"name": "均衡", "params": asdict(DenoiseParams())},
        {"name": "温和（护纹理）",
         "params": asdict(DenoiseParams(temporal_radius=2, spatial_strength=0.12,
                                        temporal_strength=0.75, max_residual=0.10))},
        {"name": "强时域",
         "params": asdict(DenoiseParams(temporal_radius=3, spatial_strength=0.2,
                                        temporal_strength=0.95, max_residual=0.14))},
        {"name": "仅空间（诊断）",
         "params": asdict(DenoiseParams(temporal_radius=0, spatial_strength=0.5,
                                        temporal_strength=0.0))},
        {"name": "关闭运动补偿（风险演示）",
         "params": asdict(DenoiseParams(temporal_radius=2, spatial_strength=0.0,
                                        temporal_strength=0.85, motion_model="off"))},
    ]


def main(port: int = 8765) -> None:
    start_worker()
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"underwater denoise UI: http://127.0.0.1:{port}/")
    srv.serve_forever()


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8765)
