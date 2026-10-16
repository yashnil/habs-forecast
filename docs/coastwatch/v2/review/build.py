#!/usr/bin/env python3
"""Build the v2 review page: index.html (images by relative path, for the repo) and, with --inline,
a single-file copy with every image embedded as a data URI (for a private artifact)."""
import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
IMG = HERE / "img"
TEMPLATE = (HERE / "template.html").read_text()


def build(inline: bool) -> str:
    out = TEMPLATE
    for f in sorted(IMG.glob("*")):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".svg"):
            continue
        rel = f"img/{f.name}"
        if inline:
            mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".svg": "image/svg+xml"}[f.suffix.lower()]
            out = out.replace(f'src="{rel}"', f'src="data:{mime};base64,{base64.b64encode(f.read_bytes()).decode()}"')
    return out


if __name__ == "__main__":
    inline = "--inline" in sys.argv
    target = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else HERE / "index.html"
    html = build(inline)
    target.write_text(html)
    print(json.dumps({"out": str(target), "bytes": len(html.encode()), "inline": inline}))
