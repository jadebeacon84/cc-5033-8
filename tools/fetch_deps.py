"""Vendor runtime dependencies into .tools/ using only the standard library.

Some containers ship a python3 with no pip and no numpy/opencv.  This script
downloads the matching manylinux wheels from PyPI, checks the recorded SHA-256
from the PyPI JSON API, and unpacks them into `.tools/pylib/`.  `run_ui.sh`
adds that directory to PYTHONPATH, so the UI starts without touching the
system interpreter.

    python3 tools/fetch_deps.py              # detect interpreter + platform
    python3 tools/fetch_deps.py --force      # ignore existing pylib

Wheels are the only thing downloaded; nothing here executes code from them.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import sysconfig
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIREMENTS = ["numpy", "opencv-python-headless", "imageio-ffmpeg"]

ARCH_TAGS = {
    "aarch64": ("aarch64", "arm64"),
    "x86_64": ("x86_64", "amd64"),
}


def _target_interpreter() -> tuple[int, int]:
    if sys.implementation.name != "cpython":
        raise SystemExit("only CPython is supported")
    return sys.version_info.major, sys.version_info.minor


def _matches(fn: str, abi: str, arch: tuple[str, ...]) -> bool:
    if not fn.endswith(".whl"):
        return False
    parts = fn[:-4].split("-")
    if len(parts) < 5:
        return False
    py_tag, abi_tag = parts[2], parts[3]
    if "none-any" in fn:
        return py_tag in (f"cp{abi}", "py3")
    if "linux" not in fn:
        return False
    if py_tag not in (f"cp{abi}", "py3") and abi_tag != "abi3":
        return False
    return any(t in parts[-1] for t in arch)


def _wheel_candidates(name: str, abi: str, machine: str) -> dict:
    """Newest release that actually ships a wheel for this interpreter."""
    arch = ARCH_TAGS.get(machine)
    if arch is None:
        raise SystemExit(f"unsupported platform: {machine}")
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout=60) as r:
        data = json.load(r)
    for u in data["urls"]:
        if _matches(u["filename"], abi, arch):
            return u
    # The latest release may have dropped this ABI (numpy 2.5 is cp312+), so
    # walk the per-version file lists newest-first.
    releases = sorted(data["releases"].items(),
                      key=lambda kv: kv[1][0]["upload_time"] if kv[1] else "", reverse=True)
    for _, urls in releases:
        for u in urls:
            if _matches(u["filename"], abi, arch):
                return u
    raise SystemExit(f"no linux wheel for {name} on cp{abi} {machine}")


def _download(url: str, want_sha: str) -> bytes:
    with urllib.request.urlopen(url, timeout=600) as r:
        blob = r.read()
    got = hashlib.sha256(blob).hexdigest()
    if got != want_sha:
        raise RuntimeError(f"checksum mismatch for {url}: {got}")
    return blob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--test", action="store_true",
                    help="also vendor pytest (for `python3 -m pytest tests`)")
    ap.add_argument("--dest", default=str(ROOT / ".tools" / "pylib"))
    args = ap.parse_args()

    major, minor = _target_interpreter()
    abi = f"{major}{minor}"
    machine = platform.machine()
    dest = Path(args.dest)
    wanted = REQUIREMENTS + (["pytest", "pluggy", "iniconfig", "packaging", "pygments"]
                             if args.test else [])

    wanted_dirs = ["numpy", "cv2", "imageio_ffmpeg"]
    if args.test:
        wanted_dirs += ["_pytest", "pygments", "pluggy"]
    if not args.force and all((dest / m).exists() for m in wanted_dirs):
        print(f"already vendored: {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        pkg_dir = {"opencv-python-headless": "cv2", "pytest": "_pytest"}.get(
            name, name.replace("-", "_"))
        if not args.force and (dest / pkg_dir).exists():
            continue
        chosen = _wheel_candidates(name, abi, machine)
        print(f"{name}: {chosen['filename']}")
        blob = _download(chosen["url"], chosen["digests"]["sha256"])
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            z.extractall(dest)
            # Wheel archives store the bundled ffmpeg without its +x bit.
            for member in z.namelist():
                if member.startswith("imageio_ffmpeg/binaries/"):
                    p = dest / member
                    if p.is_file():
                        p.chmod(p.stat().st_mode | 0o755)
    print(f"vendored into {dest}")


if __name__ == "__main__":
    main()
