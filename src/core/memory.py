"""
Memory headroom as the service actually experiences it.

``psutil.virtual_memory()`` inside a container reports the HOST's memory, not
the container's. On a host where another workload holds most of the RAM (a
large LLM beside voice-stack on a shared GPU box), host usage sits above the
guard's threshold permanently and every request got a 503, even though the
container's own memory limit had plenty of room.

When the process runs under a cgroup memory limit (``docker run --memory``,
compose ``deploy.resources.limits.memory``), the percentage is taken against
that limit instead. The headroom is the SMALLER of what is left under the
limit and what the host still has available, so a host that is genuinely out
of memory still trips the guard:

    available = min(limit - used, host_available)
    percent   = 100 * (1 - available / limit)

``used`` excludes inactive page cache (``inactive_file``), the same figure
``docker stats`` shows: the kernel reclaims that cache before it OOM-kills.

Without a limit (no cgroup, cgroup v2 ``memory.max`` = ``max``, or a limit no
smaller than host RAM), the host figures are used, as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import psutil

# Where the cgroup filesystem is mounted. A module attribute so tests can point
# it at a fake tree; inside a container with a private cgroup namespace (the
# Docker default on cgroup v2) the root here IS the container's own cgroup.
CGROUP_ROOT = Path("/sys/fs/cgroup")


@dataclass(frozen=True)
class MemorySnapshot:
    """Memory figures the guard and the health endpoints report."""

    percent: float
    available: int
    total: int
    source: str  # "cgroup" (measured against the cgroup limit) or "host"


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _stat_value(path: Path, key: str) -> int:
    try:
        for line in path.read_text().splitlines():
            name, _, value = line.partition(" ")
            if name == key:
                return int(value)
    except (OSError, ValueError):
        pass
    return 0


def read_cgroup_memory(root: Path | None = None) -> tuple[int, int] | None:
    """Return ``(limit, used)`` in bytes for the current cgroup, or None.

    None means no limit is in force (or no cgroup memory controller is
    visible), and the caller should use host figures.
    """
    root = CGROUP_ROOT if root is None else root

    # cgroup v2: memory.max is a byte count, or the literal "max" for no limit.
    max_file = root / "memory.max"
    if max_file.exists():
        limit = _read_int(max_file)  # "max" fails int() and reads as None
        current = _read_int(root / "memory.current")
        if limit is None or current is None:
            return None
        inactive = _stat_value(root / "memory.stat", "inactive_file")
        return limit, max(current - inactive, 0)

    # cgroup v1: an unlimited group reports a huge page-rounded number, which
    # the caller's "limit no smaller than host RAM" test turns into None.
    v1 = root / "memory"
    limit = _read_int(v1 / "memory.limit_in_bytes")
    usage = _read_int(v1 / "memory.usage_in_bytes")
    if limit is None or usage is None:
        return None
    inactive = _stat_value(v1 / "memory.stat", "total_inactive_file")
    return limit, max(usage - inactive, 0)


def memory_snapshot() -> MemorySnapshot:
    """Current memory headroom, against the cgroup limit when there is one."""
    host = psutil.virtual_memory()
    cgroup = read_cgroup_memory()
    if cgroup is None or cgroup[0] <= 0 or cgroup[0] >= host.total:
        return MemorySnapshot(percent=host.percent, available=host.available, total=host.total, source="host")

    limit, used = cgroup
    available = max(min(limit - used, host.available), 0)
    percent = 100.0 * (1 - available / limit)
    return MemorySnapshot(percent=percent, available=available, total=limit, source="cgroup")
