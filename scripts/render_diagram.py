#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Render the mermaid block from docs/architecture.md to docs/architecture.png.

Uses `npx mermaid-cli` (mmdc) if available. Falls back gracefully if Node /
mermaid-cli is not installed — prints instructions and leaves the markdown
intact so README embedding still works via inline mermaid or ASCII.

Usage:
    python scripts/render_diagram.py
    python scripts/render_diagram.py --background transparent --width 900

Requires (optional):
    Node.js + `npm install -g @mermaid-js/mermaid-cli`
    or it will fall back to one-shot `npx -p @mermaid-js/mermaid-cli mmdc`.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = REPO_ROOT / "docs" / "architecture.md"
OUT_PATH = REPO_ROOT / "docs" / "architecture.png"


def extract_mermaid(md_text: str) -> str | None:
    """Pull the FIRST ```mermaid ... ``` fenced block from md_text."""
    match = re.search(r"```mermaid\s*\n(.*?)```", md_text, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def find_mmdc() -> list[str] | None:
    """Return the command vector for invoking mermaid-cli, or None."""
    # 1) Direct binary on PATH
    if shutil.which("mmdc"):
        return ["mmdc"]
    # 2) npx (one-shot)
    if shutil.which("npx"):
        return ["npx", "-y", "-p", "@mermaid-js/mermaid-cli", "mmdc"]
    return None


def render(width: int, background: str) -> int:
    if not DOC_PATH.exists():
        print(f"[render_diagram] missing source file: {DOC_PATH}", file=sys.stderr)
        return 1

    md_text = DOC_PATH.read_text(encoding="utf-8")
    mermaid_src = extract_mermaid(md_text)
    if not mermaid_src:
        print(
            "[render_diagram] no ```mermaid``` block found in "
            f"{DOC_PATH}. Nothing to render.",
            file=sys.stderr,
        )
        return 1

    mmdc = find_mmdc()
    if mmdc is None:
        print(
            "[render_diagram] mermaid-cli not found.\n"
            "  Install one of:\n"
            "    npm install -g @mermaid-js/mermaid-cli   # global mmdc\n"
            "    npm install -D @mermaid-js/mermaid-cli   # local devDep\n"
            "  Or rely on inline mermaid in the README — GitHub renders it natively.\n"
            "  Skipping PNG render. Markdown source at "
            f"{DOC_PATH.relative_to(REPO_ROOT)} is unchanged.",
            file=sys.stderr,
        )
        return 0  # not an error — graceful fallback

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "diagram.mmd"
        src.write_text(mermaid_src, encoding="utf-8")
        cmd = [
            *mmdc,
            "-i", str(src),
            "-o", str(OUT_PATH),
            "-w", str(width),
            "-b", background,
        ]
        print(f"[render_diagram] running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=False)
        except FileNotFoundError as exc:
            print(f"[render_diagram] failed to launch mermaid-cli: {exc}", file=sys.stderr)
            return 1
        if result.returncode != 0:
            print(
                f"[render_diagram] mermaid-cli exited with {result.returncode}. "
                "PNG was not produced.",
                file=sys.stderr,
            )
            return result.returncode

    print(f"[render_diagram] wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=900, help="Output width in px (default: 900)")
    parser.add_argument(
        "--background",
        default="white",
        help="Background colour (e.g. white, transparent, '#0f172a'). Default: white",
    )
    args = parser.parse_args()
    return render(args.width, args.background)


if __name__ == "__main__":
    raise SystemExit(main())
