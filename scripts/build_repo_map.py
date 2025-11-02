#!/usr/bin/env python3
"""
Cross-platform repo map builder (pure stdlib).

Writes .docs/repo_map.json using a compact schema.

Usage:
    python scripts/build_repo_map.py [--root PATH] [--max-bytes 12000]

Heuristics:
- approx_tokens ~= bytes / 4
- Truncates long lists to fit under max-bytes
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any

# Try Python 3.11+ tomllib; otherwise use tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

EXCLUDE_DIRS = {
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    "coverage",
    ".next",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    "Saved",
    "DerivedDataCache",
    "Intermediate",
    "Binaries",
}

LANG_BY_EXT = {
    ".py": "Python",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".md": "Markdown",
    ".json": "JSON",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".sh": "Shell",
    ".toml": "TOML",
}

PY_PUBLIC_RE = re.compile(r"^(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
FASTAPI_RE = re.compile(r"\b(FastAPI|APIRouter|Flask)\b")
CLI_RE = re.compile(r"\b(import\s+(typer|click)|Typer\(|@app\.command|@click\.command)\b")


def git_root_or_cwd(start: pathlib.Path) -> pathlib.Path:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=start, stderr=subprocess.DEVNULL)
        return pathlib.Path(out.decode("utf-8", errors="ignore").strip())
    except Exception:
        return start


def iter_files(root: pathlib.Path) -> Any:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".tox")]

        for fn in filenames:
            if fn.endswith((".log", ".tmp", ".parquet", ".csv", ".sqlite", ".db", ".bin")):
                continue
            p = pathlib.Path(dirpath) / fn
            yield p


def read_text(p: pathlib.Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def file_loc_bytes(p: pathlib.Path) -> tuple[int, int]:
    try:
        data = p.read_bytes()
        text = data.decode("utf-8", errors="ignore")
        loc = sum(1 for line in text.splitlines() if line.strip())
        return loc, len(data)
    except Exception:
        return 0, 0


def summarize_languages(files: list[pathlib.Path]) -> tuple[dict[str, int], int]:
    langs: dict[str, int] = {}
    count = 0
    for p in files:
        count += 1
        lang = LANG_BY_EXT.get(p.suffix.lower())
        langs[lang or "Other"] = langs.get(lang or "Other", 0) + 1
    return langs, count


def parse_pyproject(path: pathlib.Path) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "name": "",
        "description": "",
        "entry_points": {"cli": [], "web": [], "jobs": []},
        "dependencies": {"runtime": [], "dev": []},
    }
    if not path.exists():
        return meta

    txt = path.read_text(encoding="utf-8", errors="ignore")

    if tomllib:
        try:
            data = tomllib.loads(txt)
            project = data.get("project", {})
            meta["name"] = project.get("name", "")
            meta["description"] = project.get("description", "")

            # Parse dependencies
            deps = project.get("dependencies", [])
            if isinstance(deps, list):
                meta["dependencies"]["runtime"] = deps[:25]

            # Parse optional dependencies
            opt = project.get("optional-dependencies", {})
            if isinstance(opt, dict):
                for values in opt.values():
                    if isinstance(values, list):
                        meta["dependencies"]["dev"] = values[:25]
                        break

            # Parse entry points
            eps = project.get("entry-points", {})
            if isinstance(eps, dict):
                # Handle console scripts
                cs = eps.get("console_scripts", [])
                if isinstance(cs, list):
                    meta["entry_points"]["cli"] = cs[:10]
                elif isinstance(cs, dict):
                    meta["entry_points"]["cli"] = [f"{k}={v}" for k, v in list(cs.items())[:10]]

            # Handle scripts section
            scripts = project.get("scripts", {})
            if isinstance(scripts, dict):
                for k, v in list(scripts.items())[:10]:
                    meta["entry_points"]["cli"].append(f"{k}={v}")
        except Exception as e:
            print(f"Error parsing TOML: {e}", file=sys.stderr)
    else:
        # Fallback to regex parsing
        mname = re.search(r'^\s*name\s*=\s*["\']([^"\']+)["\']', txt, re.MULTILINE)
        mdesc = re.search(r'^\s*description\s*=\s*["\']([^"\']+)["\']', txt, re.MULTILINE)
        if mname:
            meta["name"] = mname.group(1)
        if mdesc:
            meta["description"] = mdesc.group(1)

    return meta


def py_symbols(text: str) -> tuple[list[str], list[str]]:
    classes, funcs = [], []
    for m in PY_PUBLIC_RE.finditer(text):
        name = m.group(1)
        if text[m.start() :].startswith("class"):
            classes.append(name)
        else:
            funcs.append(name)
    return classes[:30], funcs[:40]


def build_modules(root: pathlib.Path, files: Any) -> list[dict[str, Any]]:
    modules: dict[str, Any] = {}
    for p in files:
        rel = p.relative_to(root)
        parts = rel.parts
        if not parts or parts[0] != "src" or len(parts) < 2:
            continue
        mod = parts[1]
        key = f"src/{mod}"
        d = modules.setdefault(
            key,
            {
                "path": key,
                "purpose": "",
                "public_api": {"classes": [], "functions": [], "constants": []},
                "notable_files": [],
                "metrics": {"loc": 0, "files": 0, "approx_tokens": 0},
            },
        )
        d["metrics"]["files"] += 1
        loc, b = file_loc_bytes(p)
        d["metrics"]["loc"] += loc
        d["metrics"]["approx_tokens"] += b // 4

        if p.suffix == ".py":
            txt = read_text(p)
            c, f = py_symbols(txt)
            d["public_api"]["classes"].extend(c)
            d["public_api"]["functions"].extend(f)
            if FASTAPI_RE.search(txt):
                d.setdefault("_web_hint", True)
            if CLI_RE.search(txt):
                d.setdefault("_cli_hint", True)

        if p.name in {"__init__.py", "main.py", "app.py", "cli.py"}:
            d["notable_files"].append(str(rel))

    for d in modules.values():
        d["public_api"]["classes"] = sorted(set(d["public_api"]["classes"]))[:30]
        d["public_api"]["functions"] = sorted(set(d["public_api"]["functions"]))[:40]
        d["notable_files"] = sorted(set(d["notable_files"]))[:10]

        if not d["purpose"]:
            leaf = d["path"].split("/")[-1]
            if "api" in leaf.lower():
                d["purpose"] = "Web/API endpoints and routing."
            elif "core" in leaf.lower():
                d["purpose"] = "Core domain logic and shared abstractions."
            elif "utils" in leaf.lower() or "common" in leaf.lower():
                d["purpose"] = "Utility helpers and cross-cutting concerns."
            else:
                d["purpose"] = f"Module {leaf}."

    return sorted(modules.values(), key=lambda x: x["path"])


def detect_tests(root: pathlib.Path) -> dict[str, Any]:
    test_dir = root / "tests"
    count, patterns = 0, []
    if test_dir.exists():
        for p in test_dir.rglob("*.py"):
            if p.name.startswith("test_") or p.name.endswith("_test.py"):
                count += 1
        patterns = ["tests/test_*.py", "tests/**/*_test.py"]
    return {"framework": "pytest" if count > 0 else "", "count": count, "patterns": patterns}


def parse_ci_stages(root: pathlib.Path) -> dict[str, Any]:
    yml = root / ".gitlab-ci.yml"
    system, stages = "", []
    if yml.exists():
        system = "gitlab"
        txt = read_text(yml)
        m = re.search(r"stages:\s*\[([^\]]+)\]", txt, re.MULTILINE)
        if m:
            stages = [s.strip() for s in m.group(1).split(",") if s.strip()]
        else:
            block = re.search(r"stages:\s*\n((?:\s*-\s*[A-Za-z0-9_-]+\s*\n)+)", txt, re.MULTILINE)
            if block:
                stages = [s.strip("- ").strip() for s in block.group(1).splitlines() if s.strip().startswith("-")]
    return {"system": system, "stages": stages}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None, help="Repo root (defaults to git root or CWD)")
    ap.add_argument("--max-bytes", type=int, default=12000, help="Target max JSON bytes")
    args = ap.parse_args()

    start = pathlib.Path.cwd()
    root = pathlib.Path(args.root) if args.root else git_root_or_cwd(start)
    root = root.resolve()

    (root / ".docs").mkdir(parents=True, exist_ok=True)

    files = list(iter_files(root))
    langs, files_count = summarize_languages(files)

    meta = parse_pyproject(root / "pyproject.toml")
    primary_language = max(langs.items(), key=lambda kv: kv[1])[0] if langs else ""
    project = {
        "name": meta.get("name", "") or root.name,
        "description": meta.get("description", ""),
        "primary_language": primary_language,
        "entry_points": meta.get("entry_points", {"cli": [], "web": [], "jobs": []}),
        "dependencies": meta.get("dependencies", {"runtime": [], "dev": []}),
    }

    modules = build_modules(root, files)

    web = list(project["entry_points"].get("web", []))
    cli = list(project["entry_points"].get("cli", []))
    for m in modules:
        if m.pop("_web_hint", False):
            web.append(m["path"] + "/...")
        if m.pop("_cli_hint", False):
            cli.append(m["path"] + "/...")
    project["entry_points"]["web"] = sorted(set(web))[:10]
    project["entry_points"]["cli"] = sorted(set(cli))[:10]

    ci = parse_ci_stages(root)

    total_loc = 0
    for p in files:
        loc, _ = file_loc_bytes(p)
        total_loc += loc
    metrics = {"loc": total_loc, "langs": sorted([f"{k}:{v}" for k, v in langs.items()])[:12], "files": files_count}

    data = {
        "project": project,
        "layout": {
            "packages": [],
            "apps": [],
            "scripts": ["scripts/"] if (root / "scripts").exists() else [],
            "tests": detect_tests(root),
        },
        "modules": modules[:40],
        "ci": ci,
        "metrics": metrics,
    }

    def packed(d: dict[str, Any]) -> bytes:
        return json.dumps(d, ensure_ascii=False, indent=2).encode("utf-8")

    while len(packed(data)) > args.max_bytes:
        if data["modules"] and any(m["public_api"]["functions"] for m in data["modules"]):
            for m in data["modules"]:
                if m["public_api"]["functions"]:
                    m["public_api"]["functions"] = m["public_api"]["functions"][:-5]
        elif len(data["modules"]) > 20:
            data["modules"] = data["modules"][:-1]
        else:
            break

    out = root / ".docs" / "repo_map.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
