# src/utils/debugpy_helper.py
from __future__ import annotations


def maybe_enable_debugpy(*, host: str, port: int, wait_for_client: bool) -> None:
    """
    Start debugpy listener if not already active.
    Safe to call multiple times (in parent or reloader child).
    """
    try:
        import debugpy
    except Exception:
        return  # debugpy not installed in this env

    try:
        # debugpy.listen is idempotent; if already bound, it’s a no-op.
        debugpy.listen((host, port))
    except OSError:
        # Already listening or port taken — ignore to avoid crashing dev.
        return

    if wait_for_client:
        try:
            debugpy.wait_for_client()
        except Exception:
            # Don’t block forever if something’s off
            pass
