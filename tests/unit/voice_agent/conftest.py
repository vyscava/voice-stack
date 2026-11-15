"""Fixtures for the voice agent tests: a local fake of the ASR /v1/stream WebSocket."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

Handler = Callable[[web.WebSocketResponse, list[Any]], Awaitable[None]]


class FakeAsr:
    """A /v1/stream server whose behaviour is a per-test coroutine; records what it receives."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.received: list[Any] = []  # dicts for JSON frames, bytes for audio
        self.connections = 0

    async def ws(self, request: web.Request) -> web.WebSocketResponse:
        self.connections += 1
        ws = web.WebSocketResponse(timeout=0.5)  # a fake that never answers a close must not slow teardown
        await ws.prepare(request)
        await self.handler(ws, self.received)
        return ws


@pytest.fixture
async def serve() -> AsyncIterator[Callable[[Handler], Awaitable[tuple[FakeAsr, str]]]]:
    servers: list[TestServer] = []

    async def start(handler: Handler) -> tuple[FakeAsr, str]:
        fake = FakeAsr(handler)
        app = web.Application()
        app.router.add_get("/v1/stream", fake.ws)
        server = TestServer(app)
        await server.start_server()
        servers.append(server)
        return fake, str(server.make_url("/v1/stream")).replace("http://", "ws://")

    yield start
    for server in servers:
        await server.close()
