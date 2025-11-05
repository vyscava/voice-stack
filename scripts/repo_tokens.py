#!/usr/bin/env python3
"""
Approximate token stats per top-level module/folder.

Heuristic:
    approx_tokens ~= bytes / 4
"""

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
EXCLUDE = {"node_modules", "dist", "build", ".venv", "venv", "__pycache__", ".git", "coverage", ".next", ".cache"}

mods = {}
for p in ROOT.rglob("*"):
    parts = p.relative_to(ROOT).parts
    if not parts or parts[0] in EXCLUDE:
        continue
    if any(x in EXCLUDE for x in parts):
        continue
    if p.is_file():
        top = parts[0]
        size = p.stat().st_size
        d = mods.setdefault(top, {"files": 0, "bytes": 0})
        d["files"] += 1
        d["bytes"] += size

print("module,files,bytes,approx_tokens")
for m, d in sorted(mods.items()):
    toks = round(d["bytes"] / 4)
    print(f"{m},{d['files']},{d['bytes']},{toks}")
