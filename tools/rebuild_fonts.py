"""Rebuild the WOFF2 subsets embedded in web/fonts/.

The UI must render Chinese in containers that ship no CJK fonts.  We subset
Noto Sans SC to just the glyphs used by web/ and denoise/, so the page carries
~170 KB instead of ~18 MB.

Usage:
    python3 tools/rebuild_fonts.py /path/to/NotoSansSC-VF.ttf

Download the variable source once with:
    curl -L -o /tmp/NotoSansSC-VF.ttf \
      https://github.com/googlefonts/noto-cjk/raw/main/Sans/Variable/TTF/Subset/NotoSansSC-VF.ttf
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

ROOT = Path(__file__).resolve().parent.parent
CJK = re.compile(r"[　-〿一-鿿＀-￯]")
ASCII = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ " \
        ".,:;!?()[]{}<>/\\|_-+=%×·—…“”‘’@#&*^~$"


def used_chars() -> str:
    chars = set(ASCII)
    for folder in ("web", "denoise"):
        for p in (ROOT / folder).glob("*"):
            if p.suffix in {".html", ".js", ".css", ".py", ".json"}:
                chars |= set(CJK.findall(p.read_text(errors="ignore")))
    for p in (ROOT / "README.md",):
        if p.exists():
            chars |= set(CJK.findall(p.read_text(errors="ignore")))
    return "".join(sorted(chars))


def main(src: str) -> None:
    text = used_chars()
    for weight, name in ((400, "NotoSansSC-Regular.woff2"),
                         (600, "NotoSansSC-Bold.woff2")):
        f = TTFont(src)
        if "fvar" in f:
            f = instancer.instantiateVariableFont(f, {"wght": weight}, inplace=True)
        opts = subset.Options()
        opts.flavor = "woff2"
        opts.layout_features = ["kern", "liga", "locl", "ccmp"]
        ss = subset.Subsetter(options=opts)
        ss.populate(text=text)
        ss.subset(f)
        for tag in ("gasp", "hdmx", "LTSH", "VDMX", "prep", "fpgm", "cvt "):
            if tag in f:
                del f[tag]
        out = ROOT / "web" / "fonts" / name
        f.save(out)
        print(f"{out.relative_to(ROOT)}: {out.stat().st_size / 1024:.1f} KB "
              f"({len(text)} chars)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    main(sys.argv[1])
