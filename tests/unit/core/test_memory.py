"""
core.memory: the memory guard measures the container's cgroup limit, not the host.

Regression: on a host running a large LLM beside voice-stack, host
RAM sat at 96.8% while the voice-stack container had headroom under its own
limit, and the guard (which read psutil.virtual_memory(), i.e. the host)
answered every request with 503.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import core.memory
from core.memory import memory_snapshot, read_cgroup_memory
from core.middleware import ResourceGuardMiddleware

GiB = 1024**3

# A shared host: 128 GiB, 96.8% used, ~4 GiB available.
SHARED_HOST = SimpleNamespace(total=128 * GiB, available=int(4.1 * GiB), percent=96.8)


def _v2(root: Path, max_value: str, current: int, inactive_file: int = 0) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "memory.max").write_text(f"{max_value}\n")
    (root / "memory.current").write_text(f"{current}\n")
    (root / "memory.stat").write_text(f"anon 1\nfile 2\ninactive_file {inactive_file}\nactive_file 3\n")
    return root


def _v1(root: Path, limit: int, usage: int, inactive_file: int = 0) -> Path:
    mem = root / "memory"
    mem.mkdir(parents=True, exist_ok=True)
    (mem / "memory.limit_in_bytes").write_text(f"{limit}\n")
    (mem / "memory.usage_in_bytes").write_text(f"{usage}\n")
    (mem / "memory.stat").write_text(f"cache 5\ntotal_inactive_file {inactive_file}\n")
    return root


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch):
    def use(ns: SimpleNamespace) -> None:
        monkeypatch.setattr("psutil.virtual_memory", lambda: ns)

    use(SHARED_HOST)
    return use


@pytest.fixture
def cgroup_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "cgroup"
    monkeypatch.setattr(core.memory, "CGROUP_ROOT", root)
    return root


# ---------------------------------------------------------------------------
# read_cgroup_memory
# ---------------------------------------------------------------------------


def test_v2_limit_and_usage_net_of_inactive_file(cgroup_root: Path) -> None:
    _v2(cgroup_root, str(4 * GiB), current=3 * GiB, inactive_file=1 * GiB)
    assert read_cgroup_memory() == (4 * GiB, 2 * GiB)


def test_v2_unlimited_is_none(cgroup_root: Path) -> None:
    _v2(cgroup_root, "max", current=3 * GiB)
    assert read_cgroup_memory() is None


def test_v1_limit_and_usage(cgroup_root: Path) -> None:
    _v1(cgroup_root, 4 * GiB, usage=3 * GiB, inactive_file=GiB // 2)
    assert read_cgroup_memory() == (4 * GiB, 3 * GiB - GiB // 2)


def test_no_cgroup_files_is_none(cgroup_root: Path) -> None:
    assert read_cgroup_memory() is None


def test_usage_below_inactive_file_clamps_to_zero(cgroup_root: Path) -> None:
    _v2(cgroup_root, str(4 * GiB), current=GiB, inactive_file=2 * GiB)
    assert read_cgroup_memory() == (4 * GiB, 0)


# ---------------------------------------------------------------------------
# memory_snapshot
# ---------------------------------------------------------------------------


def test_shared_host_with_container_limit_uses_the_limit(host, cgroup_root: Path) -> None:
    """The shared-host case: host 96.8% used, container 2 of 4 GiB used -> 50%."""
    _v2(cgroup_root, str(4 * GiB), current=3 * GiB, inactive_file=1 * GiB)
    snap = memory_snapshot()
    assert snap.source == "cgroup"
    assert snap.total == 4 * GiB
    assert snap.available == 2 * GiB
    assert snap.percent == pytest.approx(50.0)


def test_no_limit_falls_back_to_host(host, cgroup_root: Path) -> None:
    _v2(cgroup_root, "max", current=3 * GiB)
    snap = memory_snapshot()
    assert (snap.source, snap.percent, snap.total) == ("host", 96.8, 128 * GiB)


def test_limit_not_below_host_ram_is_treated_as_unlimited(host, cgroup_root: Path) -> None:
    # cgroup v1 reports "unlimited" as a huge page-rounded number.
    _v1(cgroup_root, 9223372036854771712, usage=3 * GiB)
    assert memory_snapshot().source == "host"


def test_host_shortage_still_counts_under_a_limit(host, cgroup_root: Path) -> None:
    """Headroom under the limit is capped by what the host can still give."""
    host(SimpleNamespace(total=128 * GiB, available=GiB // 5, percent=99.8))
    _v2(cgroup_root, str(4 * GiB), current=GiB)  # 3 GiB left under the limit
    snap = memory_snapshot()
    assert snap.source == "cgroup"
    assert snap.available == GiB // 5
    assert snap.percent == pytest.approx(95.0)


# ---------------------------------------------------------------------------
# The guard itself: same host, the cgroup decides
# ---------------------------------------------------------------------------


@pytest.fixture
def guarded_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr("psutil.swap_memory", lambda: SimpleNamespace(percent=0.0))
    app = FastAPI()
    app.add_middleware(ResourceGuardMiddleware)

    @app.get("/ping")
    def ping() -> dict[str, str]:
        return {"ok": "yes"}

    return TestClient(app)


def test_guard_rejects_on_shared_host_without_a_limit(host, cgroup_root: Path, guarded_client: TestClient) -> None:
    """Control: with no limit the host figure applies, and 96.8% > 90% is a 503."""
    resp = guarded_client.get("/ping")
    assert resp.status_code == 503
    assert resp.json()["memory_source"] == "host"


def test_guard_admits_on_shared_host_with_headroom_under_the_limit(
    host, cgroup_root: Path, guarded_client: TestClient
) -> None:
    """Same host, but the container has room under its limit -> served."""
    _v2(cgroup_root, str(4 * GiB), current=3 * GiB, inactive_file=1 * GiB)
    resp = guarded_client.get("/ping")
    assert resp.status_code == 200


def test_guard_rejects_when_the_container_is_near_its_limit(
    host, cgroup_root: Path, guarded_client: TestClient
) -> None:
    host(SimpleNamespace(total=128 * GiB, available=100 * GiB, percent=20.0))
    _v2(cgroup_root, str(4 * GiB), current=int(3.8 * GiB))
    resp = guarded_client.get("/ping")
    assert resp.status_code == 503
    assert resp.json()["memory_source"] == "cgroup"
