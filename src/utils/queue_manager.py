"""
queue_manager.py — Minimal thread-safe task/result registry.

This is *not* a background worker. It just gives you a small in-memory
queue + status registry so you can:
  - assign IDs to long operations,
  - mark them processing/done/failed,
  - fetch results by ID,
  - and report how many are pending.

You can extend this later into a real worker pool if needed.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from queue import Queue
from threading import Lock
from typing import Any


class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class QueueManager:
    def __init__(self, *, max_size: int = 32) -> None:
        self.queue: Queue[str] = Queue(maxsize=max_size)
        self.results: dict[str, dict[str, Any]] = {}
        self.lock = Lock()

    def submit(self) -> str:
        """Create a new task ID and mark as queued."""
        task_id = str(uuid.uuid4())
        with self.lock:
            self.results[task_id] = {
                "status": TaskStatus.QUEUED,
                "result": None,
                "error": None,
                "created_at": time.time(),
                "updated_at": time.time(),
            }
        self.queue.put(task_id)
        return task_id

    def mark_processing(self, *, task_id: str) -> None:
        with self.lock:
            if task_id in self.results:
                self.results[task_id]["status"] = TaskStatus.PROCESSING
                self.results[task_id]["updated_at"] = time.time()

    def complete(self, *, task_id: str, result: Any) -> None:
        with self.lock:
            if task_id in self.results:
                self.results[task_id]["status"] = TaskStatus.DONE
                self.results[task_id]["result"] = result
                self.results[task_id]["updated_at"] = time.time()

    def fail(self, *, task_id: str, error: str) -> None:
        with self.lock:
            if task_id in self.results:
                self.results[task_id]["status"] = TaskStatus.FAILED
                self.results[task_id]["error"] = error
                self.results[task_id]["updated_at"] = time.time()

    def get_status(self, *, task_id: str) -> dict[str, Any] | None:
        with self.lock:
            return self.results.get(task_id)

    def pending(self) -> int:
        return self.queue.qsize()
