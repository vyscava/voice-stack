#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import pathlib
import re
import subprocess

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
FASTAPI_METHODS = {"get", "post", "put", "delete", "patch", "options", "head"}


def git_root_or_cwd(start: pathlib.Path) -> pathlib.Path:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            stderr=subprocess.DEVNULL,
        )
        return pathlib.Path(out.decode().strip())
    except Exception:
        return start


def iter_files(root: pathlib.Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".tox")]
        for fn in filenames:
            if fn in {".DS_Store"}:  # ignore mac junk
                continue
            if fn.endswith((".log", ".tmp", ".parquet", ".sqlite", ".db", ".bin", ".lock")):
                continue
            yield pathlib.Path(dirpath) / fn


def read_text(p: pathlib.Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def file_loc_bytes(p: pathlib.Path):
    try:
        b = p.read_bytes()
        text = b.decode("utf-8", errors="ignore")
        loc = sum(1 for line in text.splitlines() if line.strip())
        return loc, len(b)
    except Exception:
        return 0, 0


def summarize_languages(files: list[pathlib.Path]):
    langs: dict[str, int] = {}
    count = 0
    for p in files:
        count += 1
        lang = LANG_BY_EXT.get(p.suffix.lower())
        langs[lang or "Other"] = langs.get(lang or "Other", 0) + 1
    return langs, count


def parse_requirements_file(path: pathlib.Path) -> list[str]:
    out: list[str] = []
    if not path.exists():
        return out
    for line in read_text(path).splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-r ", "--")):
            continue
        out.append(line)
    return out


def gather_dependencies(root: pathlib.Path) -> dict[str, list[str]]:
    runtime, dev = [], []
    data = None
    try:
        import tomllib  # py3.11+
        txt = read_text(root / "pyproject.toml")
        if txt:
            data = tomllib.loads(txt)
    except Exception:
        data = None

    if data:
        proj = data.get("project", {})
        if isinstance(proj.get("dependencies"), list):
            runtime += proj["dependencies"]
        opt = proj.get("optional-dependencies", {})
        if isinstance(opt, dict):
            for lst in opt.values():
                if isinstance(lst, list):
                    runtime += lst
        poetry = data.get("tool", {}).get("poetry", {})
        if poetry:
            if isinstance(poetry.get("dependencies"), dict):
                runtime += list(poetry["dependencies"].keys())
            if isinstance(poetry.get("group"), dict):
                for _spec in poetry["group"].items():
                    if isinstance(_spec.get("dependencies"), dict):
                        dev += list(_spec["dependencies"].keys())

    for fname in [
        "requirements.txt",
        "requirements-dev.txt",
        "dev-requirements.txt",
        "requirements-dev.in",
    ]:
        lst = parse_requirements_file(root / fname)
        if lst:
            (dev if "dev" in fname else runtime).extend(lst)

    def norm(x: str) -> str:
        return re.split(r"[<>=;#\[]", x, maxsplit=1)[0].strip().lower()

    def dedupe(seq: list[str]) -> list[str]:
        seen, out = set(), []
        for s in seq:
            k = norm(s)
            if k and k not in seen:
                seen.add(k)
                out.append(k)
        return out

    return {"runtime": dedupe(runtime)[:80], "dev": dedupe(dev)[:80]}


class TopLevelVisitor(ast.NodeVisitor):
    def __init__(self):
        self.classes: list[str] = []
        self.functions: list[str] = []
        self.routes: list[dict] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.append(node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node.name)
        self._inspect_route(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.functions.append(node.name)
        self._inspect_route(node)  # type: ignore

    def _inspect_route(self, node: ast.FunctionDef) -> None:
        for dec in getattr(node, "decorator_list", []):
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                method = dec.func.attr
                if method in FASTAPI_METHODS:
                    path = ""
                    if dec.args and isinstance(dec.args[0], (ast.Str, ast.Constant)):
                        path = dec.args[0].s if isinstance(dec.args[0], ast.Str) else (
                            dec.args[0].value if isinstance(dec.args[0].value, str) else ""
                        )
                    self.routes.append({"method": method.upper(), "path": path})


def py_symbols_and_routes(text: str):
    try:
        tree = ast.parse(text)
    except Exception:
        return [], [], []
    v = TopLevelVisitor()
    v.visit(tree)
    classes = sorted(set(v.classes))[:50]
    functions = sorted(set(v.functions))[:80]
    seen, routes = set(), []
    for r in v.routes:
        k = (r["method"], r["path"])
        if k not in seen:
            seen.add(k)
            routes.append({"method": r["method"], "path": r["path"]})
    return classes, functions, routes[:80]


def build_modules(root: pathlib.Path, files: list[pathlib.Path]):
    modules: dict[str, dict] = {}
    src = root / "src"
    top_dirs = []
    if src.exists():
        top_dirs = [
            child.name
            for child in src.iterdir()
            if (
                child.is_dir()
                and child.name not in EXCLUDE_DIRS
                and not child.name.startswith(".")
            )
        ]

    for p in files:
        rel = p.relative_to(root)
        if rel.parts and rel.parts[0] == "src":
            if len(rel.parts) < 2 or rel.parts[1] not in top_dirs:
                continue
            key = f"src/{rel.parts[1]}"
        else:
            continue

        d = modules.setdefault(
            key,
            {
                "path": key,
                "purpose": "",
                "public_api": {"classes": [], "functions": [], "constants": []},
                "notable_files": [],
                "http_routes": [],
                "metrics": {"loc": 0, "files": 0, "approx_tokens": 0},
            },
        )
        d["metrics"]["files"] += 1
        loc, b = file_loc_bytes(p)
        d["metrics"]["loc"] += loc
        d["metrics"]["approx_tokens"] += b // 4

        if p.suffix == ".py":
            c, f, routes = py_symbols_and_routes(read_text(p))
            if c:
                d["public_api"]["classes"] += c
            if f:
                d["public_api"]["functions"] += f
            if routes:
                d["http_routes"] += routes
        if p.name in {"__init__.py", "main.py", "app.py", "cli.py"}:
            d["notable_files"].append(str(rel))

    for d in modules.values():
        d["public_api"]["classes"] = sorted(set(d["public_api"]["classes"]))[:50]
        d["public_api"]["functions"] = sorted(set(d["public_api"]["functions"]))[:80]
        d["notable_files"] = sorted(set(d["notable_files"]))[:12]
        d["http_routes"] = [
            {"method": m, "path": p}
            for (m, p) in sorted({(r["method"], r["path"]) for r in d["http_routes"]})
        ][:60]
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


def detect_tests(root: pathlib.Path):
    frameworks, count = [], 0
    tdir = root / "tests"
    if tdir.exists():
        for p in tdir.rglob("*.py"):
            if p.name.startswith("test_") or p.name.endswith("_test.py"):
                count += 1
    try:
        import tomllib
        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        devdeps = []
        proj = data.get("project", {})
        opt = proj.get("optional-dependencies", {})
        for lst in opt.values():
            if isinstance(lst, list):
                devdeps += lst
        poetry = data.get("tool", {}).get("poetry", {})
        if isinstance(poetry.get("group"), dict):
            for _spec in poetry["group"].items():
                pass
        txt = read_text(root / "requirements-dev.txt") + " " + " ".join(devdeps)
        low = txt.lower()
        if "pytest" in low:
            frameworks.append("pytest")
        if "nose" in low:
            frameworks.append("nose")
        if "unittest" in low:
            frameworks.append("unittest")
    except Exception:
        pass
    frameworks = sorted(set(frameworks))
    return {
        "framework": (frameworks[0] if frameworks else ""),
        "count": count,
        "patterns": ["tests/test_*.py", "tests/**/*_test.py"],
    }


def parse_ci_stages(root: pathlib.Path):
    yml = root / ".gitlab-ci.yml"
    system, stages = "", []
    if yml.exists():
        system = "gitlab"
        txt = read_text(yml)
        m = re.search(r"stages:\s*\[([^\]]+)\]", txt, re.MULTILINE)
        if m:
            stages = [s.strip() for s in m.group(1).split(",") if s.strip()]
        else:
            block = re.search(
                r"stages:\s*\n((?:\s*-\s*[A-Za-z0-9_-]+\s*\n)+)", txt, re.MULTILINE
            )
            if block:
                stages = [
                    s.strip("- ").strip()
                    for s in block.group(1).splitlines()
                    if s.strip().startswith("-")
                ]
    return {"system": system, "stages": stages}


def parse_pyproject_meta(root: pathlib.Path) -> dict:
    meta = {"name": "", "description": "", "entry_points": {"cli": [], "web": [], "jobs": []}}
    try:
        import tomllib
        txt = read_text(root / "pyproject.toml")
        if not txt:
            return meta
        data = tomllib.loads(txt)
        project = data.get("project", {})
        meta["name"] = project.get("name", "") or root.name
        meta["description"] = project.get("description", "")
        eps = project.get("entry-points", {})
        if isinstance(eps, dict):
            cs = eps.get("console_scripts", [])
            if isinstance(cs, list):
                meta["entry_points"]["cli"] += cs[:12]
            elif isinstance(cs, dict):
                meta["entry_points"]["cli"] += [
                    f"{k}={v}" for k, v in list(cs.items())[:12]
                ]
        scripts = project.get("scripts", {})
        if isinstance(scripts, dict):
            meta["entry_points"]["cli"] += [
                f"{k}={v}" for k, v in list(scripts.items())[:12]
            ]
    except Exception:
        meta["name"] = root.name
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", type=str, default=None, help="Repo root (defaults to git root or CWD)"
    )
    ap.add_argument(
        "--max-bytes", type=int, default=24000, help="Target max JSON bytes"
    )
    args = ap.parse_args()

    start = pathlib.Path.cwd()
    root = pathlib.Path(args.root) if args.root else git_root_or_cwd(start)
    root = root.resolve()

    (root / ".docs").mkdir(parents=True, exist_ok=True)

    files = list(iter_files(root))
    langs, files_count = summarize_languages(files)
    meta = parse_pyproject_meta(root)
    deps = gather_dependencies(root)

    primary_language = max(langs.items(), key=lambda kv: kv[1])[0] if langs else ""
    project = {
        "name": meta.get("name", "") or root.name,
        "description": meta.get("description", ""),
        "primary_language": primary_language,
        "entry_points": meta.get("entry_points", {"cli": [], "web": [], "jobs": []}),
        "dependencies": deps,
    }

    modules = build_modules(root, files)
    ci = parse_ci_stages(root)

    total_loc = 0
    for p in files:
        loc, _ = file_loc_bytes(p)
        total_loc += loc
    metrics = {
        "loc": total_loc,
        "langs": sorted([f"{k}:{v}" for k, v in langs.items()])[:16],
        "files": files_count,
    }

    data = {
        "project": project,
        "layout": {
            "packages": [],
            "apps": [],
            "scripts": ["scripts/"] if (root / "scripts").exists() else [],
            "tests": detect_tests(root),
        },
        "modules": modules[:60],
        "ci": ci,
        "metrics": metrics,
    }

    def packed(d):
        return json.dumps(d, ensure_ascii=False, indent=2).encode("utf-8")

    while len(packed(data)) > args.max_bytes:
        if data["modules"] and any(m["public_api"]["functions"] for m in data["modules"]):
            for m in data["modules"]:
                if m["public_api"]["functions"]:
                    m["public_api"]["functions"] = m["public_api"]["functions"][:-10]
        elif len(data["modules"]) > 25:
            data["modules"] = data["modules"][:-1]
        else:
            break

    out = root / ".docs" / "repo_map.json"
    out.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out} (bytes={len(packed(data))})")


if __name__ == "__main__":
    main()
