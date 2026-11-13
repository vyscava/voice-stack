"""Entry point for SERVICE_MODE=agent: the web app in a thread, the LiveKit worker in front.

The worker is started by LiveKit's CLI (``python -m voice_agent.main start``),
which owns the main thread and signal handling. The web app (page, tokens,
health, metrics) runs alongside it, so health stays green while LiveKit is
unreachable: the worker keeps retrying, and restarting would not help.
"""

from __future__ import annotations

import logging
import threading

import uvicorn
from livekit.agents import cli

from core.settings import get_settings
from voice_agent.web import app
from voice_agent.worker import worker_options

logger = logging.getLogger("voice_agent")


def _serve(host: str, port: int) -> None:
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        # uvicorn returns (or raises) only when it could not serve or was stopped.
        logger.error(f"voice agent web app on {host}:{port} stopped; /health is now down")


def start_web(host: str, port: int) -> threading.Thread:
    thread = threading.Thread(target=_serve, args=(host, port), name="voice-agent-web", daemon=True)
    thread.start()
    return thread


def main() -> None:
    settings = get_settings()
    start_web(settings.HOST, settings.AGENT_WEB_PORT)
    cli.run_app(worker_options(settings))


if __name__ == "__main__":
    main()
